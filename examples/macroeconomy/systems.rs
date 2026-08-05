use std::sync::{Arc, Mutex};

use abm_framework::environment::{EnvKey, EnvironmentBoundary};
use abm_framework::messaging::{Capacity, MessageBufferSet, MessageHandle};
use abm_framework::model::ModelBuilder;
use abm_framework::{AccessSets, ComponentID, ECSReference, ECSResult, FnSystem, Read, Write};

use super::accounting::{negative_abs, positive_part, AccountingReport, GdpIdentity};
use super::components::*;
use super::equations::{
    bank_liabilities_a42, bank_reserves_a43, buy_probability_a110, cost_push_inflation_a76,
    constrained_goods_target_a83_a84, firm_predicted_profit_a61, firm_target_demand_a60,
    firm_target_production_a62, idiosyncratic_growth_a59, log_growth,
    min_input_constraint_a63_a64, ppi_a3, price_a73, price_or_rent_reduction_a113_a115,
    purchase_cost_a109_literal_pdf, ratio, rent_cost_a108, sector_price_a5, target_capital_a79,
    target_intermediate_a78, unit_cost_a77, work_effort_a66_a67,
};
use super::forecasting::{
    ardl_error_correction_delta_rate, fit_ar1_log_level_forecast, ArdlErrorCorrectionInput,
};
use super::messages::*;
use super::state::{
    rng_salt, FirmProbe, MacroAggregates, MacroEnvironment, MacroRng, MACRO_ENV_KEY,
};

/// Fraction of a household's saving that goes to deposits rather than other
/// financial assets (A.119/A.120).
///
/// The paper says the surplus is "shared among deposits and other financial
/// assets in fixed fractions depending on current household income, wealth and
/// debt" but never gives those fractions. This is a replication blocker, not a
/// calibrated value: it is stated here so the gap is visible, and it must not
/// be tuned to change the trajectory.

/// Quarters in a year. A.30's loan-to-income multiple is an annual figure
/// applied to a quarterly income series.
const QUARTERS_PER_YEAR: f64 = 4.0;

const DEPOSIT_SHARE_OF_SAVING: f64 = 0.6;

/// A loan the lending bank loses when its borrower fails (A.41).
#[derive(Clone, Copy, Debug)]
struct BadDebt {
    bank_id: u32,
    borrower_kind: u8,
    borrower_id: u32,
    deposits_seized: f64,
    loans_lost: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MacroComponentIds {
    pub firm: ComponentID,
    pub individual: ComponentID,
    pub household: ComponentID,
    pub bank: ComponentID,
    pub government_entity: ComponentID,
    pub government_account: ComponentID,
    pub central_bank: ComponentID,
    pub property: ComponentID,
    pub rest_of_world: ComponentID,
}

#[derive(Clone, Copy)]
pub struct MacroMessageHandles {
    pub labour_offer: MessageHandle<LabourOffer>,
    pub wage_payment: MessageHandle<WagePayment>,
    pub goods_demand: MessageHandle<GoodsDemand>,
    pub goods_receipt: MessageHandle<GoodsReceipt>,
    pub excess_demand: MessageHandle<ExcessDemand>,
    pub credit_application: MessageHandle<CreditApplication>,
    pub credit_grant: MessageHandle<CreditGrant>,
    pub credit_failure: MessageHandle<CreditFailure>,
    pub mortgage_need: MessageHandle<MortgageNeed>,
    pub tentative_purchase: MessageHandle<TentativePurchase>,
    pub tentative_rental: MessageHandle<TentativeRental>,
    pub property_transfer: MessageHandle<PropertyTransfer>,
}

#[derive(Clone, Copy)]
pub struct PhaseKeys {
    pub aggregate_done: EnvKey<u64>,
    pub expectations_done: EnvKey<u64>,
    pub targets_done: EnvKey<u64>,
    pub labour_done: EnvKey<u64>,
    pub planning_done: EnvKey<u64>,
    pub housing_preclear_done: EnvKey<u64>,
    pub credit_done: EnvKey<u64>,
    pub housing_completion_done: EnvKey<u64>,
    pub goods_done: EnvKey<u64>,
    pub accounting_done: EnvKey<u64>,
}

pub fn register_message_handles(
    builder: &mut ModelBuilder,
) -> Result<MacroMessageHandles, abm_framework::model::ModelError> {
    Ok(MacroMessageHandles {
        labour_offer: builder
            .register_brute_force_message::<LabourOffer>(Capacity::unbounded(64))?,
        wage_payment: builder
            .register_brute_force_message::<WagePayment>(Capacity::unbounded(64))?,
        goods_demand: builder
            .register_brute_force_message::<GoodsDemand>(Capacity::unbounded(256))?,
        goods_receipt: builder
            .register_brute_force_message::<GoodsReceipt>(Capacity::unbounded(256))?,
        excess_demand: builder
            .register_brute_force_message::<ExcessDemand>(Capacity::unbounded(128))?,
        credit_application: builder
            .register_brute_force_message::<CreditApplication>(Capacity::unbounded(128))?,
        credit_grant: builder
            .register_brute_force_message::<CreditGrant>(Capacity::unbounded(128))?,
        credit_failure: builder
            .register_brute_force_message::<CreditFailure>(Capacity::unbounded(128))?,
        mortgage_need: builder
            .register_brute_force_message::<MortgageNeed>(Capacity::unbounded(64))?,
        tentative_purchase: builder
            .register_brute_force_message::<TentativePurchase>(Capacity::unbounded(64))?,
        tentative_rental: builder
            .register_brute_force_message::<TentativeRental>(Capacity::unbounded(64))?,
        property_transfer: builder
            .register_brute_force_message::<PropertyTransfer>(Capacity::unbounded(64))?,
    })
}

pub fn add_macroeconomy_systems(
    builder: ModelBuilder,
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
) -> ModelBuilder {
    let env_boundary = builder.environment_boundary_id();
    let message_boundary = builder.message_boundary_id();

    builder
        .with_system(aggregate_previous_state_system(ids, phases, env_boundary))
        .with_system(refit_expectations_system(phases, env_boundary))
        .with_system(target_setting_system(ids, phases, env_boundary))
        .with_system(labour_market_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(planning_and_production_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(housing_preclear_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(credit_market_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(housing_completion_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(goods_market_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
        .with_system(realised_accounting_system(
            ids,
            messages,
            phases,
            env_boundary,
            message_boundary,
        ))
}

fn aggregate_previous_state_system(
    ids: MacroComponentIds,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.read.set(ids.firm);
    access.read.set(ids.household);
    access.read.set(ids.bank);
    access.read.set(ids.government_entity);
    access.read.set(ids.government_account);
    access.read.set(ids.property);
    access.read.set(ids.rest_of_world);
    access.produces.insert(phases.aggregate_done.channel_id());
    FnSystem::new(10, "macro::aggregate_previous_state", access, move |ecs| {
        // Eqs. A.1-A.15: previous-quarter aggregates and GDP identity inputs.
        let firms = collect_rows::<Firm>(ecs)?;
        let households = collect_rows::<Household>(ecs)?;
        let banks = collect_rows::<Bank>(ecs)?;
        let governments = collect_rows::<GovernmentEntity>(ecs)?;
        let accounts = collect_rows::<GovernmentAccount>(ecs)?;
        let properties = collect_rows::<Property>(ecs)?;
        let row = collect_rows::<RestOfWorld>(ecs)?;
        let mut state = macro_state(ecs, env_boundary)?;
        let aggregates = compute_aggregates(
            &state,
            &firms,
            &households,
            &banks,
            &governments,
            &accounts,
            &properties,
            &row,
        );
        state.previous_aggregates = state.aggregates;
        state.aggregates = aggregates;
        state.push_phase("aggregate_previous_state");
        set_phase_and_state(ecs, env_boundary, phases.aggregate_done, state)
    })
}

fn refit_expectations_system(
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.consumes.insert(phases.aggregate_done.channel_id());
    access
        .produces
        .insert(phases.expectations_done.channel_id());
    FnSystem::new(20, "macro::refit_expectations", access, move |ecs| {
        // Eqs. A.16-A.21 / thesis Ch. 6 pp. 156-157: deterministic AR(1) on log levels through t-1.
        let mut state = macro_state(ecs, env_boundary)?;
        let y_fit = fit_ar1_log_level_forecast(
            &state.history.production,
            state.aggregates.production.max(1.0),
        );
        let ppi_fit =
            fit_ar1_log_level_forecast(&state.history.ppi, state.aggregates.ppi.max(1e-9));
        let cpi_fit =
            fit_ar1_log_level_forecast(&state.history.cpi, state.aggregates.cpi.max(1e-9));
        let hpi_fit =
            fit_ar1_log_level_forecast(&state.history.hpi, state.aggregates.hpi.max(1e-9));
        let rpi_fit =
            fit_ar1_log_level_forecast(&state.history.rpi, state.aggregates.rpi.max(1e-9));

        state.forecast.predicted_growth =
            log_growth(y_fit.forecast_level, state.aggregates.production);
        state.forecast.predicted_ppi = ppi_fit.forecast_level;
        state.forecast.predicted_cpi = cpi_fit.forecast_level;
        state.forecast.predicted_hpi = hpi_fit.forecast_level;
        state.forecast.predicted_rpi = rpi_fit.forecast_level;
        state.forecast.predicted_ppi_inflation =
            log_growth(ppi_fit.forecast_level, state.aggregates.ppi);
        state.forecast.predicted_cpi_inflation =
            log_growth(cpi_fit.forecast_level, state.aggregates.cpi);
        state.forecast.predicted_hpi_inflation =
            log_growth(hpi_fit.forecast_level, state.aggregates.hpi);
        state.forecast.predicted_rpi_inflation =
            log_growth(rpi_fit.forecast_level, state.aggregates.rpi);
        // A.95: total target government consumption from an AR(1) on the
        // realised series, which `realised_accounting` now records.
        let government_fit = fit_ar1_log_level_forecast(
            &state.history.government_consumption,
            state.aggregates.government_consumption.max(1e-9),
        );
        state.forecast.predicted_government_consumption = government_fit.forecast_level;
        state.forecast.ar1_observations = y_fit.observations;
        state.audit.last_expectation_fit_observations = y_fit.observations;

        if let Some(last_sector) = state.history.sector_production.last().copied() {
            for s in 0..SECTORS {
                let history: Vec<f64> = state
                    .history
                    .sector_production
                    .iter()
                    .map(|row| row[s])
                    .collect();
                let fit = fit_ar1_log_level_forecast(&history, last_sector[s].max(1.0));
                state.forecast.predicted_sector_growth[s] =
                    log_growth(fit.forecast_level, last_sector[s]);
            }
        }
        state.push_phase("refit_expectations");
        set_phase_and_state(ecs, env_boundary, phases.expectations_done, state)
    })
}

fn target_setting_system(
    ids: MacroComponentIds,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.firm);
    access.write.set(ids.individual);
    access.read.set(ids.government_account);
    access
        .consumes
        .insert(phases.expectations_done.channel_id());
    access.produces.insert(phases.targets_done.channel_id());
    FnSystem::new(30, "macro::firm_individual_targets", access, move |ecs| {
        // Eqs. A.59-A.68 and A.129-A.132: firm targets and individual supply/income targets.
        let mut state = macro_state(ecs, env_boundary)?;
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut individuals = collect_rows::<Individual>(ecs)?;
        let accounts = collect_rows::<GovernmentAccount>(ecs)?;
        let account = accounts.first().copied().unwrap_or_default();
        // `P_s(t-1)` for A.59's relative-price test.
        let sector_prices_previous = previous_sector_prices(&firms);

        for firm in &mut firms {
            let sector = firm.sector as usize;
            // A.59's case condition, which A.74 shares: apply the idiosyncratic
            // growth term only when the firm faced excess demand *and* priced
            // at or above its sector average, or faced excess supply *and*
            // priced at or below it. Otherwise it is zero.
            //
            // The old test was `excess_demand.abs() > 1e-9`, which is neither
            // case. Inert while phi^Q_F = phi^DP = 0, but wrong, and a tripwire
            // the moment cost-push pricing is switched on -- which is the whole
            // point of the ignored regression test.
            let supply_offered = firm.previous_production + firm.inventory_two_periods_ago;
            let sector_average_price = sector_prices_previous[firm.sector as usize];
            let excess_demand_case =
                supply_offered <= firm.previous_demand && firm.previous_price >= sector_average_price;
            let excess_supply_case =
                supply_offered >= firm.previous_demand && firm.previous_price <= sector_average_price;
            let gamma_applies = excess_demand_case || excess_supply_case;
            let gamma_f = idiosyncratic_growth_a59(
                firm.previous_demand.max(1e-9),
                firm.previous_production.max(1e-9),
                firm.inventory_two_periods_ago.max(0.0),
                gamma_applies,
            );
            // A.94: "A bankrupt firm is replaced by a new firm that enters the
            // same sector. That new firm keeps the same stock and inventory as
            // the bankrupt firm and is initialised with D_f(t+1) = L_f(t+1) =
            // 0." The deposits and debt were already zeroed when the failure
            // was recorded; what was missing was the *replacement* -- the flag
            // was never cleared, so the firm population decayed monotonically
            // and every failed firm stayed failed for the rest of the run.
            if firm.bankrupt {
                firm.bankrupt = false;
                firm.overdraft = 0.0;
                // A new entrant has no loss history to extrapolate from.
                firm.profits = 0.0;
                firm.predicted_profits = 0.0;
            }
            firm.target_demand = firm_target_demand_a60(
                state.forecast.predicted_sector_growth[sector],
                state.calibration.phi_f_q,
                gamma_f,
                firm.previous_demand.max(1.0),
            );
            firm.predicted_profits = firm_predicted_profit_a61(
                state.forecast.predicted_ppi_inflation,
                gamma_f,
                firm.profits,
            );
            let intermediate_constraint = min_input_constraint_a63_a64(
                &firm.intermediate_stock,
                &state.params.io_matrix[sector],
            );
            let capital_constraint = min_input_constraint_a63_a64(
                &firm.capital_stock,
                &state.params.net_fixed_assets_matrix[sector],
            );
            // A.128 fixes individual labour inputs at H_i = 1, and A.8.1 sets
            // h^U = h^E = 0, so they never move: sum_i H_i is simply the
            // headcount. Productivity lives in the work-effort factor, whose
            // baseline h_f(0) is the sector's labour productivity h_s = output
            // per employee (Table A.11, A.5.1).
            let labour_supply = firm.employees as f64;
            // A.66-A.67, evaluated *before* it is used. This previously ran
            // after `labour_constraint`, so production was constrained by the
            // previous quarter's work effort.
            firm.work_effort = work_effort_a66_a67(
                state.params.work_effort_max,
                firm.initial_work_effort,
                labour_supply,
                firm.target_demand,
                intermediate_constraint,
                capital_constraint,
            );
            // A.65: H_f(t) = h_f(t) * sum_i H_i(t), in output units -- which is
            // what makes A.68 (`target labour = target production`) and the
            // A.141/A.142 hire/fire comparisons dimensionally sound.
            firm.labour = (firm.work_effort * labour_supply).max(0.0);
            let labour_constraint = firm.labour;
            firm.target_production = firm_target_production_a62(
                firm.target_demand,
                state.calibration.phi_st_y,
                firm.previous_production,
                firm.previous_inventory,
                state.calibration.chi_h,
                labour_constraint,
                state.calibration.chi_m,
                intermediate_constraint,
                state.calibration.chi_k,
                capital_constraint,
            );
            firm.target_labour = firm.target_production.max(0.0);
            // A.69: w_i(t) = (1 + pi^PPI)(1 + mu^WN) * phi^WE * w_i(t-1).
            //
            // The work-effort factor was missing entirely, so a firm running
            // overtime paid the same as one running short time. phi^WE(1) is
            // 1.176 at the A.55 opening stocks, so this is a live nominal
            // channel, not a rounding term. Poledna A.25/A.26 corroborate it
            // independently: the same factor scales productivity *and* the wage.
            //
            // mu^WN is zero because A.5.1 sets phi^WN = 0 -- A.70 is
            // deliberately inert, not unimplemented.
            //
            // KNOWN SPECIFICATION WEAKNESS -- do not "fix" this.
            //
            // A.69 multiplies the *previous* wage, so phi^WE compounds. And
            // phi^WE does not decay back to 1: A.55 opens every firm at
            // `M_f(0) = Y_f(0)/omega^M`, A.78 with `phi^M = 1` restores that
            // same stock ratio every quarter, so A.66 holds
            // `phi^WE = 1/omega^M = 1/0.85 = 1.176` indefinitely. Wages
            // therefore compound at ~17.6% a quarter for as long as the model
            // runs. Every step of that is a paper parameter (A.5.1 pins
            // omega^M and phi^M), so there is no synthetic-data freedom to
            // avoid it.
            //
            // The thesis confirms the form verbatim (Eq. 6.70, "w_i(t-1) is the
            // previous wage paid to employee i") and its footnote 57 concedes
            // the problem: "The functional form (6.70) does not include wage
            // stickiness. This should be improved in further iterations, as
            // this is an important driver for inflation."
            //
            // We implement it as specified. The consequence is that nominal
            // aggregates are not stationary; see `docs/limitations.md`.
            let work_effort_factor = if firm.initial_work_effort.abs() > 1e-12 {
                firm.work_effort / firm.initial_work_effort
            } else {
                1.0
            };
            let tightness_markup = state.params.wage_tightness_sensitivity * 0.0;
            if state.params.wage_effort_on_base {
                // Poledna A.26: the work-effort factor scales a *base* wage,
                // so overtime raises the wage by a level and never compounds.
                // Indexation still accumulates on the base, exactly as A.69
                // applies it to the previous wage.
                firm.base_wage *= ((1.0 + state.forecast.predicted_ppi_inflation)
                    * (1.0 + tightness_markup))
                    .max(0.0);
                firm.wage = firm.base_wage * work_effort_factor.max(0.0);
            } else {
                // Wiese A.69 verbatim. Left bit-identical to the form this
                // branch had before the Poledna alternative was added.
                firm.wage *= ((1.0 + state.forecast.predicted_ppi_inflation)
                    * (1.0 + tightness_markup)
                    * work_effort_factor)
                    .max(0.0);
                firm.base_wage = firm.wage;
            }
            firm.sales_quantity = 0.0;
            firm.sales_revenue = 0.0;
            firm.excess_demand = 0.0;
        }

        for individual in &mut individuals {
            match individual.labour_status {
                LABOUR_UNEMPLOYED => {
                    individual.labour_input /= 1.0 + state.params.unemployment_growth_h;
                }
                LABOUR_EMPLOYED => {
                    individual.labour_input *= 1.0 + state.params.employed_growth_h;
                }
                _ => individual.labour_input = 0.0,
            }
            let average_wage =
                individual.wage_history.iter().sum::<f64>() / individual.wage_history.len() as f64;
            individual.reservation_wage =
                (state.forecast.predicted_cpi * account.unemployment_benefit).max(average_wage);
            individual.predicted_income = if individual.labour_status == LABOUR_EMPLOYED {
                state.forecast.predicted_cpi
                    * individual.wage
                    * (1.0
                        - account.social_insurance_worker_rate
                        - account.income_tax_rate * (1.0 - account.social_insurance_worker_rate))
            } else if individual.labour_status == LABOUR_UNEMPLOYED {
                state.forecast.predicted_cpi * account.unemployment_benefit
            } else {
                0.0
            };
        }

        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, individuals, |individual: &Individual| individual.id)?;
        state.push_phase("target_setting");
        set_phase_and_state(ecs, env_boundary, phases.targets_done, state)
    })
}

fn labour_market_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.firm);
    access.write.set(ids.individual);
    access.consumes.insert(phases.targets_done.channel_id());
    access.produces.insert(messages.labour_offer.channel_id());
    access.produces.insert(messages.wage_payment.channel_id());
    access.produces.insert(phases.labour_done.channel_id());
    FnSystem::new(40, "macro::labour_market", access, move |ecs| {
        // Eqs. A.141-A.142: all firing before all hiring.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let mut state = macro_state(ecs, env_boundary)?;
        let mut rng = state.rng(ecs.run_context(), rng_salt::LABOUR_MARKET);
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut individuals = collect_rows::<Individual>(ecs)?;

        for firm in &mut firms {
            let mut employees: Vec<usize> = individuals
                .iter()
                .enumerate()
                .filter(|(_, worker)| {
                    worker.labour_status == LABOUR_EMPLOYED && worker.employer_firm_id == firm.id
                })
                .map(|(idx, _)| idx)
                .collect();
            rng.shuffle(&mut employees);
            // A.141: fire until any further firing would take H_f below the
            // target. `firm.labour` is H_f in output units (A.65), so losing an
            // employee costs `h_f * H_i`, not the raw `H_i = 1`. Moving it by
            // 1.0 made firing effectively inert against a target of ~34.
            while firm.labour > firm.target_labour && !employees.is_empty() {
                let idx = employees.pop().unwrap();
                let contribution = firm.work_effort * individuals[idx].labour_input;
                if firm.labour - contribution < firm.target_labour {
                    break;
                }
                firm.labour -= contribution;
                firm.employees = firm.employees.saturating_sub(1);
                individuals[idx].labour_status = LABOUR_UNEMPLOYED;
                individuals[idx].employer_firm_id = NOT_LINKED;
                individuals[idx].wage = 0.0;
            }
        }

        let mut firm_order: Vec<usize> = (0..firms.len()).collect();
        rng.shuffle(&mut firm_order);
        let mut job_seekers: Vec<usize> = individuals
            .iter()
            .enumerate()
            .filter(|(_, worker)| worker.labour_status == LABOUR_UNEMPLOYED)
            .map(|(idx, _)| idx)
            .collect();
        rng.shuffle(&mut job_seekers);

        for firm_idx in firm_order {
            let firm = &mut firms[firm_idx];
            // A.142: hire until H_f reaches the target. Each hire adds
            // `h_f * H_i` output units, so the number of vacancies is the gap
            // divided by one worker's contribution.
            // H_i = 1 for every individual (A.128), so one hire is worth h_f.
            let per_hire = firm.work_effort.max(1e-9);
            let needed = ((firm.target_labour - firm.labour) / per_hire)
                .ceil()
                .max(0.0) as u32;
            if needed == 0 {
                continue;
            }
            buffers.emit(
                messages.labour_offer,
                LabourOffer {
                    firm_id: firm.id,
                    wage: firm.wage,
                    slots: needed,
                },
            )?;
            let mut hired = 0;
            let mut remaining_seekers = Vec::new();
            for worker_idx in job_seekers.drain(..) {
                if hired >= needed {
                    remaining_seekers.push(worker_idx);
                    continue;
                }
                if firm.wage >= individuals[worker_idx].reservation_wage {
                    individuals[worker_idx].labour_status = LABOUR_EMPLOYED;
                    individuals[worker_idx].employer_firm_id = firm.id;
                    individuals[worker_idx].industry = firm.sector;
                    individuals[worker_idx].wage = firm.wage;
                    firm.labour += firm.work_effort * individuals[worker_idx].labour_input;
                    firm.employees += 1;
                    hired += 1;
                } else {
                    remaining_seekers.push(worker_idx);
                }
            }
            job_seekers = remaining_seekers;
        }

        // A.66/A.67 depend on `sum_i H_i`, which the firing and hiring above
        // just changed. Leaving `phi^WE` at the value computed in
        // `firm_individual_targets` let `H_f` drift above `min(M_f, K_f)` --
        // the `labour_over_materials` counter ran 4.2 to 16.3 in quarters 4-8.
        for firm in &mut firms {
            let sector = firm.sector as usize;
            let intermediate_constraint = min_input_constraint_a63_a64(
                &firm.intermediate_stock,
                &state.params.io_matrix[sector],
            );
            let capital_constraint = min_input_constraint_a63_a64(
                &firm.capital_stock,
                &state.params.net_fixed_assets_matrix[sector],
            );
            let labour_supply = firm.employees as f64;
            firm.work_effort = work_effort_a66_a67(
                state.params.work_effort_max,
                firm.initial_work_effort,
                labour_supply,
                firm.target_demand,
                intermediate_constraint,
                capital_constraint,
            );
            firm.labour = (firm.work_effort * labour_supply).max(0.0);
        }

        for individual in &individuals {
            if individual.labour_status == LABOUR_EMPLOYED {
                buffers.emit(
                    messages.wage_payment,
                    WagePayment {
                        firm_id: individual.employer_firm_id,
                        household_id: individual.household_id,
                        individual_id: individual.id,
                        amount: individual.wage,
                    },
                )?;
            }
        }

        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, individuals, |individual: &Individual| individual.id)?;
        state.audit.labour_fired_before_hiring = true;
        state.push_phase("labour_market");
        set_phase_and_state(ecs, env_boundary, phases.labour_done, state)
    })
}

fn planning_and_production_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.firm);
    access.write.set(ids.individual);
    access.write.set(ids.household);
    access.write.set(ids.government_entity);
    access.write.set(ids.government_account);
    access.write.set(ids.central_bank);
    access.write.set(ids.rest_of_world);
    access.read.set(ids.property);
    access.consumes.insert(phases.labour_done.channel_id());
    access.consumes.insert(messages.wage_payment.channel_id());
    access.produces.insert(messages.goods_demand.channel_id());
    access
        .produces
        .insert(messages.credit_application.channel_id());
    access.produces.insert(phases.planning_done.channel_id());
    FnSystem::new(50, "macro::planning_and_production", access, move |ecs| {
        // Eqs. A.45, A.72-A.82, A.95-A.106, A.134-A.139.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let wages: Vec<WagePayment> = buffers.brute_force(messages.wage_payment)?.collect();
        let mut state = macro_state(ecs, env_boundary)?;
        let mut rng = state.rng(ecs.run_context(), rng_salt::PLANNING);
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut individuals = collect_rows::<Individual>(ecs)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let mut governments = collect_rows::<GovernmentEntity>(ecs)?;
        let mut accounts = collect_rows::<GovernmentAccount>(ecs)?;
        let mut central_banks = collect_rows::<CentralBank>(ecs)?;
        let mut rows = collect_rows::<RestOfWorld>(ecs)?;
        let properties = collect_rows::<Property>(ecs)?;

        let account = accounts.first().copied().unwrap_or_default();
        for central_bank in &mut central_banks {
            // A.45 takes *realised* `pi^CPI(t)` and `gamma(t)`, and A.46's
            // estimating equation is fitted on the same realised series. The
            // code fed it the AR(1) *forecasts* instead. On an oscillating
            // history the forecast overshoots badly -- measured 2.39 (a 239%
            // growth prediction) against a realised swing of +47% -- and
            // `xi_gamma * 2.39` alone moved the policy rate from 0.5% to 24% in
            // one quarter. Banks then paid ~30% a quarter on their whole
            // deposit base, equity was destroyed, and A.32 closed the credit
            // market. The Taylor coefficients were never the problem.
            // Taken from the recorded history, not from `previous_aggregates`:
            // that field is assigned at priority 10 from `aggregates`, which
            // was last written at priority 100 of the same tick from the same
            // production, so the two are always equal here and realised growth
            // would read a constant zero.
            //
            // The central bank sets the rate before this quarter's production
            // exists, so the freshest realised growth available is t-1 over
            // t-2 -- which is what A.46's estimating equation is fitted on.
            let realised_cpi_inflation = last_two_growth(&state.history.cpi);
            let realised_growth = last_two_growth(&state.history.production);
            central_bank.predicted_policy_rate = positive_part(
                central_bank.rho * central_bank.policy_rate
                    + (1.0 - central_bank.rho)
                        * (central_bank.natural_rate
                            + central_bank.inflation_target
                            + central_bank.xi_pi
                                * (realised_cpi_inflation - central_bank.inflation_target)
                            + central_bank.xi_gamma * realised_growth),
            );
            central_bank.policy_rate = central_bank.predicted_policy_rate;
            state.audit.policy_rate = central_bank.policy_rate;
            state.audit.taylor_cpi_inflation = realised_cpi_inflation;
            state.audit.taylor_growth = realised_growth;
        }
        // `r(t)` for A.80's interest terms.
        let central_bank_rate = central_banks
            .first()
            .map(|central_bank| central_bank.policy_rate)
            .unwrap_or_default();

        for individual in &mut individuals {
            individual.income = if individual.labour_status == LABOUR_EMPLOYED {
                state.aggregates.cpi
                    * individual.wage
                    * (1.0
                        - account.social_insurance_worker_rate
                        - account.income_tax_rate * (1.0 - account.social_insurance_worker_rate))
            } else if individual.labour_status == LABOUR_UNEMPLOYED {
                state.aggregates.cpi * account.unemployment_benefit
            } else {
                0.0
            };
            individual.wage_history.rotate_left(1);
            individual.wage_history[7] = individual.wage;
        }

        // `P_s'(t-1)` for the A.81/A.82 financing needs.
        let previous_sector_prices = previous_sector_prices(&firms);
        // Quantity invariants, recomputed from scratch each quarter. Recorded
        // rather than asserted so a violation surfaces as a number in the CLI
        // output instead of a panic mid-run. See
        // `MarketAudit::quantity_invariants_hold`.
        state.audit.max_production_over_labour = 0.0;
        state.audit.max_labour_over_materials = 0.0;
        state.audit.individual_headcount = individuals.len() as u64;
        state.audit.employed_headcount = firms.iter().map(|firm| u64::from(firm.employees)).sum();

        for firm in &mut firms {
            let sector = firm.sector as usize;
            // A.72's labour argument is H_f(t) itself. `firm.labour` already
            // holds it: A.65 is applied in `firm_individual_targets`, and the
            // labour market maintains it in the same units. Multiplying by
            // `work_effort` here applied h_f a second time -- the constraint
            // then scaled as h_f^2 * headcount, roughly an order of magnitude
            // too loose and quadratic in output. (The earlier comment here was
            // correct for the code that preceded A.65 being applied upstream,
            // when `firm.labour` really was a raw headcount; both fixes landed
            // and compounded.)
            let labour_constraint = firm.labour.max(0.0);
            let intermediate_constraint = min_input_constraint_a63_a64(
                &firm.intermediate_stock,
                &state.params.io_matrix[sector],
            );
            let capital_constraint = min_input_constraint_a63_a64(
                &firm.capital_stock,
                &state.params.net_fixed_assets_matrix[sector],
            );
            firm.production = firm
                .target_production
                .min(labour_constraint)
                .min(intermediate_constraint)
                .min(capital_constraint)
                .max(0.0);
            // A.72 mins over H_f, so production can never exceed the firm's
            // labour inputs. Substituting A.66 into A.65 additionally pins
            // H_f = min(h^max * h_f(0) * sum_i H_i, min(M_f, K_f)), so labour
            // weakly dominates the material constraints.
            state.audit.max_production_over_labour = state
                .audit
                .max_production_over_labour
                .max(firm.production - firm.labour);
            state.audit.max_labour_over_materials = state
                .audit
                .max_labour_over_materials
                .max(firm.labour - intermediate_constraint.min(capital_constraint));
            // A.74/A.75 share A.59's case condition -- see the note there.
            let supply_offered = firm.previous_production + firm.inventory_two_periods_ago;
            let sector_average_price = previous_sector_prices[firm.sector as usize];
            let demand_pull_applies = (supply_offered <= firm.previous_demand
                && firm.previous_price >= sector_average_price)
                || (supply_offered >= firm.previous_demand
                    && firm.previous_price <= sector_average_price);
            let demand_pull = idiosyncratic_growth_a59(
                firm.previous_demand,
                firm.previous_production,
                firm.inventory_two_periods_ago,
                demand_pull_applies,
            )
            .max(0.0);
            // A.76 has no floor: costs falling should pull prices down.
            let cost_push = cost_push_inflation_a76(firm.unit_cost, firm.previous_price);
            firm.price = price_a73(
                firm.previous_price,
                state.forecast.predicted_ppi_inflation,
                state.calibration.phi_dp,
                demand_pull,
                state.calibration.phi_cp,
                cost_push,
            )
            .max(0.01);
            if state.policy.debug_firm_id == Some(firm.id) {
                // `unit_cost` and `demand` are still last quarter's -- both are
                // written by `realised_accounting` at the end of the tick.
                state.audit.firm_probe = Some(FirmProbe {
                    id: firm.id,
                    employees: firm.employees,
                    work_effort: firm.work_effort,
                    initial_work_effort: firm.initial_work_effort,
                    labour: firm.labour,
                    intermediate_constraint,
                    capital_constraint,
                    target_production: firm.target_production,
                    production: firm.production,
                    price: firm.price,
                    unit_cost: firm.unit_cost,
                    demand: firm.demand,
                    excess_demand: firm.excess_demand,
                    ..FirmProbe::default()
                });
            }
            for s in 0..SECTORS {
                firm.target_intermediate[s] = target_intermediate_a78(
                    state.params.io_matrix[sector][s],
                    firm.target_production,
                    state.params.firm_input_adjustment,
                    firm.intermediate_stock[s],
                    firm.initial_intermediate_stock[s],
                    firm.production,
                    firm.initial_production,
                );
                firm.target_capital[s] = target_capital_a79(
                    state.params.capital_compensation_matrix[sector][s],
                    firm.target_production,
                    state.params.firm_capital_adjustment,
                    firm.capital_stock[s],
                    firm.initial_capital_stock[s],
                    firm.production,
                    firm.initial_production,
                );
            }
            // A.80: the predicted change in deposits, excluding new loans and
            // the purchase of new inputs.
            //
            // This equation did not exist. A.81/A.82 were built from an ad-hoc
            // "need minus deposits", which is not what the paper asks for: the
            // financing gap is measured against *predicted* deposits, so a firm
            // expecting a profitable quarter borrows less than one expecting a
            // loss even at identical current balances.
            //
            // PAPER ERRATUM (docs/errata.md (A.80)): A.80 prints
            // `- r(t)[D_f(t-1)]^+`, which *subtracts* the interest a firm earns
            // on positive deposits from its own deposit change. The sign is
            // corrected here on three independent grounds:
            //
            //   1. Poledna's online appendix A.33 -- the cash-flow equation
            //      A.80 derives from -- carries this term as
            //      `+ r_bar(t) max(0, D_i(t-1))`, labelled "Interest received".
            //   2. A.89 in this same paper carries the identical pair with the
            //      correct sign inside its cost block.
            //   3. As printed, holding cash makes a firm poorer.
            //
            // The overdraft term keeps its printed sign, which is correct: a
            // firm *pays* on a negative balance.
            let production_value = firm.price * firm.production;
            let (loan_interest_due, loan_instalment_due, short_rate) =
                firm_loan_obligations(&state, firm.id);
            // A.3.3: the firm overdraft rate equals the short-term firm-loan
            // rate.
            let overdraft_rate = if short_rate > 0.0 {
                short_rate
            } else {
                central_bank_rate
            };
            let predicted_deposit_change = production_value
                - firm_wage_bill(firm)
                - account.corporate_tax_rate * positive_part(firm.predicted_profits)
                - overdraft_rate * negative_abs(firm.deposits)
                + central_bank_rate * positive_part(firm.deposits)
                - loan_interest_due
                - account_production_tax(&accounts, sector) * production_value
                - loan_instalment_due;
            let predicted_deposits = firm.deposits + predicted_deposit_change;
            // A.81/A.82: `[predicted deposits - cost of the inputs]^-`, valued
            // at `P_s'(t-1)`. `[x]^-` is a non-negative shortfall magnitude.
            let intermediate_cost = (0..SECTORS)
                .map(|s| previous_sector_prices[s] * firm.target_intermediate[s])
                .sum::<f64>();
            let capital_cost = (0..SECTORS)
                .map(|s| previous_sector_prices[s] * firm.target_capital[s])
                .sum::<f64>();
            firm.target_short_loan = negative_abs(predicted_deposits - intermediate_cost);
            firm.target_long_loan =
                negative_abs(predicted_deposits - intermediate_cost - capital_cost);
            firm.granted_short_loan = 0.0;
            firm.granted_long_loan = 0.0;
            if firm.target_short_loan > 0.0 {
                buffers.emit(
                    messages.credit_application,
                    CreditApplication {
                        borrower_kind: BUYER_FIRM,
                        borrower_id: firm.id,
                        loan_class: LOAN_FIRM_SHORT,
                        sector: firm.sector,
                        amount: firm.target_short_loan,
                        collateral: firm.equity,
                        income: firm.predicted_profits,
                    },
                )?;
            }
            if firm.target_long_loan > 0.0 {
                buffers.emit(
                    messages.credit_application,
                    CreditApplication {
                        borrower_kind: BUYER_FIRM,
                        borrower_id: firm.id,
                        loan_class: LOAN_FIRM_LONG,
                        sector: firm.sector,
                        amount: firm.target_long_loan,
                        collateral: firm.capital_stock.iter().sum(),
                        income: firm.predicted_profits,
                    },
                )?;
            }
        }

        // A.95 distributes the sector's government consumption across the
        // entities in that sector. Poledna A.56 divides by `J` explicitly; the
        // recompute below was assigning each entity the *whole* sectoral
        // amount, multiplying government demand by the entity count.
        let mut entities_in_sector = [0u32; SECTORS];
        for government in governments.iter() {
            entities_in_sector[government.sector as usize] += 1;
        }
        for government in &mut governments {
            // Realised consumption is a quarterly flow, accumulated by the
            // goods market during this tick.
            government.realised_consumption = 0.0;
            // A.95: C_hat^CG_s(t) = c_s^CG (1 + pi^PPI) P_s(t-1) C_hat^CG(t),
            // where the total `C_hat^CG(t)` comes from an AR(1) on historical
            // government consumption and is then distributed evenly among
            // entities.
            //
            // This was a fixed share of aggregate production, with production
            // standing in for the sector price. The model produces the series
            // itself, so nothing external was missing -- it just was not being
            // recorded or fitted.
            government.target_consumption = state.params.government_consumption_weights
                [government.sector as usize]
                * (1.0 + state.forecast.predicted_ppi_inflation)
                * previous_sector_prices[government.sector as usize]
                * state.forecast.predicted_government_consumption
                / entities_in_sector[government.sector as usize].max(1) as f64;
            buffers.emit(
                messages.goods_demand,
                GoodsDemand {
                    buyer_kind: BUYER_GOVERNMENT,
                    buyer_id: government.id,
                    purpose: GOODS_GOVERNMENT,
                    sector: government.sector,
                    quantity: government.target_consumption,
                    max_spend: government.target_consumption * state.forecast.predicted_ppi,
                },
            )?;
        }

        for row in &mut rows {
            let production_index = ratio(
                state.aggregates.production,
                state.history.production.first().copied().unwrap_or(1.0),
            );
            let price_index = state.aggregates.ppi.max(0.0);
            for s in 0..SECTORS {
                row.sector_prices[s] =
                    positive_part(1.0 + row.adjustment_speed * (price_index - 1.0));
            }
            // A.137 and A.138 both index to the *initial* trade levels
            // Y^ROW(0) / C^ROW(0), scaled by the A.136 production index and the
            // A.134 price index. Chaining off `row.exports` / `row.imports`
            // instead made each quarter multiply the previous level again, so
            // external demand compounded geometrically -- and because
            // `row.exports` was itself a running total that was never reset,
            // it compounded off a cumulative base.
            row.target_exports = positive_part(
                (1.0 + row.adjustment_speed * (production_index - 1.0)) * row.initial_exports,
            );
            row.target_imports = positive_part(
                (1.0 + row.adjustment_speed * (price_index - 1.0))
                    * (1.0 + row.adjustment_speed * (production_index - 1.0))
                    * row.initial_imports,
            );
            row.imports = row.target_imports;
            // Realised exports are a quarterly flow, accumulated by the goods
            // market during this tick.
            row.exports = 0.0;
            for s in 0..SECTORS {
                row.import_nominal_by_sector[s] = row.imports * row.import_weights[s];
                row.import_real_by_sector[s] =
                    row.import_nominal_by_sector[s] / row.sector_prices[s].max(1e-9);
                let export_quantity = row.target_exports * row.export_weights[s];
                if export_quantity > 0.0 {
                    buffers.emit(
                        messages.goods_demand,
                        GoodsDemand {
                            buyer_kind: BUYER_ROW,
                            buyer_id: row.id,
                            purpose: GOODS_EXPORT,
                            sector: s as u8,
                            quantity: export_quantity,
                            max_spend: export_quantity * row.sector_prices[s],
                        },
                    )?;
                }
            }
        }

        let mut wage_income_by_household = vec![0.0; households.len()];
        for wage in wages {
            if let Some(slot) = wage_income_by_household.get_mut(wage.household_id as usize) {
                *slot += wage.amount;
            }
        }
        // A.103/A.104 sum over each household's individuals and rented-out
        // properties. Written as a filter inside the household loop that is
        // O(households x individuals) + O(households x properties) -- 23M
        // comparisons a tick at 595 firms, and quadratic in population.
        //
        // One indexed pass instead. Both the accumulate and the read are keyed
        // on `household.id`, never on row position: `collect_rows` pushes from
        // a parallel `for_each`, so row order is whatever order the workers
        // happened to push in and `households[i].id == i` does not hold in
        // general.
        let mut predicted_labour_by_household = vec![0.0; households.len()];
        let mut labour_by_household = vec![0.0; households.len()];
        for individual in &individuals {
            let slot = individual.household_id as usize;
            if let Some(value) = predicted_labour_by_household.get_mut(slot) {
                *value += individual.predicted_income;
            }
            if let Some(value) = labour_by_household.get_mut(slot) {
                *value += individual.income;
            }
        }
        let mut rent_by_household = vec![0.0; households.len()];
        for property in &properties {
            if property.owner_household_id == property.occupant_household_id {
                continue;
            }
            if let Some(value) = rent_by_household.get_mut(property.owner_household_id as usize) {
                *value += property.rent;
            }
        }
        for household in households.iter_mut() {
            let slot = household.id as usize;
            let labour_income = predicted_labour_by_household.get(slot).copied().unwrap_or(0.0);
            let rent_income = rent_by_household.get(slot).copied().unwrap_or(0.0);
            household.predicted_income = labour_income
                + state.forecast.predicted_cpi * household.social_benefits_other
                + rent_income
                + state.params.financial_asset_income_phi * household.other_financial_assets
                + household.dividend_income;
            let financial_asset_epsilon =
                rng.normal_f64(0.0, state.params.financial_asset_income_sigma);
            household.income = labour_by_household.get(slot).copied().unwrap_or(0.0)
                + state.aggregates.cpi * household.social_benefits_other
                + rent_income
                + (1.0 + financial_asset_epsilon)
                    * state.params.financial_asset_income_phi
                    * household.other_financial_assets
                + household.dividend_income;
            // A.104 income is the sum of member *individual* incomes, which are
            // already CPI-scaled and net of tax (A.133). Adding the raw
            // `WagePayment` amounts on top counted every wage twice -- once
            // taxed, once gross.
            let history_consumption = household.consumption_history.iter().sum::<f64>()
                / household.consumption_history.len() as f64;
            let target_total = ((1.0 - household.saving_rate)
                * state.forecast.predicted_cpi
                * account.unemployment_benefit)
                .max((1.0 - household.saving_rate) * household.predicted_income)
                .max(state.params.phi_consumption_history * history_consumption);
            for s in 0..SECTORS {
                household.consumption_target[s] =
                    state.params.cpi_weights[s] / (1.0 + account.vat_rate) * target_total;
                household.investment_target[s] = state.params.household_investment_weights[s]
                    / (1.0 + account.capital_tax_rate)
                    * household.investment_rate
                    * household.predicted_income;
            }
            let desired_consumption = household.consumption_target.iter().sum::<f64>();
            let quarterly_rent = properties
                .iter()
                .filter(|property| {
                    property.occupant_household_id == household.id
                        && property.owner_household_id != household.id
                })
                .map(|property| property.rent)
                .sum::<f64>();
            // `property.rent` is a quarterly rent -- A.108 annualises it as
            // `4(1+mu^PS) r`. The tenant was paying a quarter of it while the
            // landlord received all of it (A.103/A.104 sum `r_p` undivided), so
            // rent created money at four times the rate it destroyed it.
            household.disposable_income_after_rent =
                positive_part(household.income - quarterly_rent);
            // A.117: L^C_h = [C_hat_h - Y^-r_h - W^FA_h(t-1)]^+ , where A.120
            // makes `W^FA` the *whole* of financial wealth -- deposits plus
            // other financial assets. Netting only the latter overstated the
            // financing gap for every household holding deposits.
            let consumption_gap =
                positive_part(desired_consumption - household.disposable_income_after_rent);
            let financial_wealth =
                positive_part(household.deposits) + positive_part(household.other_financial_assets);
            let financial_assets_used = consumption_gap.min(financial_wealth);
            household.consumption_gap_after_financial_assets =
                positive_part(consumption_gap - financial_assets_used);
            household.desired_consumption_loan = household.consumption_gap_after_financial_assets;
            household.granted_consumption_loan = 0.0;
            household.granted_mortgage = 0.0;
            if household.desired_consumption_loan > 0.0 {
                buffers.emit(
                    messages.credit_application,
                    CreditApplication {
                        borrower_kind: BUYER_HOUSEHOLD,
                        borrower_id: household.id,
                        loan_class: LOAN_HOUSEHOLD_CONSUMPTION,
                        sector: 0,
                        amount: household.desired_consumption_loan,
                        collateral: household.net_wealth.max(0.0),
                        income: household.predicted_income,
                    },
                )?;
            }
        }

        for account in &mut accounts {
            // Eq. 6.97 as printed is `wU(t) = wU(t-1) / (1 + growth)`, which
            // *shrinks* the benefit when the economy grows -- while the very
            // next line indexes other benefits *up* by the same growth rate.
            // A `.max(1.0)` clamp used to sit here, which silently inverted the
            // rule whenever growth was positive (the common case), pinning the
            // benefit instead of moving it.
            //
            // Both transfers are indexed to nominal growth. This departs from
            // the printed form of 6.97 and is a deliberate modelling choice:
            // a deflating unemployment benefit alongside an inflating
            // "other benefits" line has no economic reading we can defend, and
            // with no calibration target there is nothing to prefer the
            // printed asymmetry.
            // A.96: w^U(t) = max(1, 1/(1 + gamma_bar(t))) * w^U(t-1).
            //
            // This is *countercyclical by design*: the benefit is held flat
            // when the economy grows and raised when it shrinks. The previous
            // code multiplied by `(1 + growth)` -- procyclical -- with a comment
            // arguing the printed form "has no economic reading we can defend".
            // That reasoning was wrong: §3.5 Table 3 lists "countercyclical
            // unemployment benefits" as one of the headline differences between
            // this model and the IIASA model it extends.
            account.unemployment_benefit *=
                (1.0 / (1.0 + state.forecast.predicted_growth)).max(1.0);
            // A.97 is the asymmetric one, and deliberately so: other benefits
            // *do* grow with the economy.
            account.other_benefits *= 1.0 + state.forecast.predicted_growth;
        }

        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, individuals, |individual: &Individual| individual.id)?;
        write_rows(ecs, households, |household: &Household| household.id)?;
        write_rows(ecs, governments, |government: &GovernmentEntity| {
            government.id
        })?;
        write_rows(ecs, accounts, |account: &GovernmentAccount| account.id)?;
        write_rows(ecs, central_banks, |central_bank: &CentralBank| {
            central_bank.id
        })?;
        write_rows(ecs, rows, |row: &RestOfWorld| row.id)?;
        state.push_phase("planning_and_production");
        set_phase_and_state(ecs, env_boundary, phases.planning_done, state)
    })
}

fn housing_preclear_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.household);
    access.write.set(ids.property);
    access.read.set(ids.bank);
    access.consumes.insert(phases.planning_done.channel_id());
    access
        .produces
        .insert(messages.tentative_purchase.channel_id());
    access
        .produces
        .insert(messages.tentative_rental.channel_id());
    access.produces.insert(messages.mortgage_need.channel_id());
    access
        .produces
        .insert(phases.housing_preclear_done.channel_id());
    FnSystem::new(60, "macro::housing_preclear", access, move |ecs| {
        // Eqs. A.107-A.116 plus Appendix A.13 purchase-before-rental clearing.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let mut state = macro_state(ecs, env_boundary)?;
        let mut rng = state.rng(ecs.run_context(), rng_salt::HOUSING_PRECLEAR);
        let mut households = collect_rows::<Household>(ecs)?;
        let mut properties = collect_rows::<Property>(ecs)?;
        let banks = collect_rows::<Bank>(ecs)?;

        let cpi_lag = lagged_cpi_inflation(&state);
        // A.108/A.109 need `V*`, the value of housing a household can afford,
        // "estimated by regressing the value of previously sold properties on
        // corresponding prices", and `r_V*`, the predicted rent of a property of
        // that value, regressed the same way. Neither existed: the code used the
        // candidate property's own price and value.
        //
        // With a synthetic stock these are regressions through the origin --
        // one ratio each -- fitted across the standing stock rather than a
        // transaction history, which a fresh run does not yet have.
        let value_per_price = {
            let price_total: f64 = properties.iter().map(|p| p.price).sum();
            let value_total: f64 = properties.iter().map(|p| p.value).sum();
            if price_total > 1e-9 {
                value_total / price_total
            } else {
                1.0
            }
        };
        let rent_per_value = {
            let value_total: f64 = properties.iter().map(|p| p.value).sum();
            let rent_total: f64 = properties.iter().map(|p| p.rent).sum();
            if value_total > 1e-9 {
                rent_total / value_total
            } else {
                0.0
            }
        };
        for property in &mut properties {
            property.previous_price = property.price;
            property.previous_rent = property.rent;
            // Property value is *not* marked to market by predicted HPI
            // inflation. That rule is nowhere in the paper, and it is circular:
            // A.6 defines `HPI = sum_p V_p(t) / sum_p V_p(0)`, and predicted
            // HPI inflation is forecast from that index -- so value grew by a
            // forecast of an index defined from value, with no transaction
            // anchor. A flat history froze it (hence `hpi` reading exactly
            // 1.000000); a history with variation would have made it drift
            // instead. `V_p` moves when a property actually sells (A.121 and
            // the housing completion system), which is the paper's mechanism.
            if property.market_status == PROPERTY_FOR_SALE {
                property.quarters_on_sale += 1;
                if rng.unit_f64() < state.params.sale_price_reduction_probability {
                    let epsilon = rng.normal_f64(
                        state.params.sale_price_reduction_mu,
                        state.params.sale_price_reduction_sigma,
                    );
                    property.price =
                        price_or_rent_reduction_a113_a115(property.previous_price, epsilon);
                }
            }
            if property.market_status == PROPERTY_FOR_RENT {
                property.quarters_on_rent_market += 1;
                if rng.unit_f64() < state.params.rent_reduction_probability {
                    let epsilon = rng.normal_f64(
                        state.params.rent_reduction_mu,
                        state.params.rent_reduction_sigma,
                    );
                    property.rent =
                        price_or_rent_reduction_a113_a115(property.previous_rent, epsilon);
                }
            }
            if property.market_status == PROPERTY_RENTAL {
                // A.116: rent on an occupied let is partially indexed to lagged
                // CPI inflation.
                property.rent *= 1.0 + state.params.rent_partial_indexation_phi * cpi_lag;
            }
            // A.114: a buy-to-let investor putting a vacant property on the
            // rental market asks `(1 + pi^RPI) * r_V*` -- the predicted rent for
            // a property of that value, marked up by predicted rental
            // inflation. This was absent: a property entering the rental market
            // simply kept whatever rent it last carried.
            if property.market_status == PROPERTY_FOR_RENT && property.quarters_on_rent_market <= 1
            {
                property.rent = (1.0 + state.forecast.predicted_rpi_inflation)
                    * property.value
                    * rent_per_value;
            }
            property.predicted_annual_rent_price =
                rent_cost_a108(state.params.housing_mu_ps, property.rent);
            property.predicted_annual_buy_price = purchase_cost_a109_literal_pdf(
                property.price,
                0.0,
                banks
                    .first()
                    .map(|bank| bank.mortgage_rate)
                    .unwrap_or_default(),
                state.params.mortgage_maturity_quarters,
                state.forecast.predicted_hpi_inflation,
                property.value,
            );
            property.predicted_rental_yield = ratio(4.0 * property.rent, property.value.max(1e-9));
        }

        let mut household_order: Vec<usize> = (0..households.len()).collect();
        rng.shuffle(&mut household_order);
        let mut buyers = Vec::new();
        let mut renters = Vec::new();
        for household_idx in household_order {
            let household = &mut households[household_idx];
            let needs_home = household.residence_property_id == NOT_LINKED;
            let stay_probability = if household.owns_residence {
                state.params.owner_stay_probability
            } else {
                state.params.renter_stay_probability
            };
            let considers_move = needs_home || rng.unit_f64() >= stay_probability;
            if !considers_move {
                continue;
            }
            if household.owns_residence && household.residence_property_id != NOT_LINKED {
                if let Some(property) = properties
                    .iter_mut()
                    .find(|property| property.id == household.residence_property_id)
                {
                    property.price =
                        (1.0 + state.forecast.predicted_hpi_inflation) * property.value;
                    property.market_status = PROPERTY_FOR_SALE;
                    property.occupant_household_id = NOT_LINKED;
                    state.audit.housing_listings += 1;
                }
            }
            let epsilon =
                rng.normal_f64(state.params.housing_mu_hp, state.params.housing_sigma_hp);
            household.desired_house_price = state.params.housing_phi_hp
                * household
                    .predicted_income
                    .max(1.0)
                    .powf(state.params.housing_beta_hp)
                * epsilon.exp();
            household.desired_rent = state.params.housing_phi_hr
                * household.income.max(1.0).powf(state.params.housing_beta_hr);
            let nearest_sale = closest_property(
                &properties,
                PROPERTY_FOR_SALE,
                household.desired_house_price,
            );
            let nearest_rent =
                closest_property(&properties, PROPERTY_FOR_RENT, household.desired_rent);
            let buy_probability = if let Some(property_idx) = nearest_sale {
                let property = properties[property_idx];
                let bank_rate = banks
                    .iter()
                    .find(|bank| bank.id == household.bank_id)
                    .or_else(|| banks.first())
                    .map(|bank| bank.mortgage_rate)
                    .unwrap_or_default();
                // A.108/A.109 compare the annual cost of *renting a property of
                // value V\** against the cost of buying at the household's own
                // desired price -- not the costs of one specific listing.
                let affordable_value = household.desired_house_price * value_per_price;
                let predicted_rent = affordable_value * rent_per_value;
                let rent_cost = rent_cost_a108(state.params.housing_mu_ps, predicted_rent);
                let purchase_cost = purchase_cost_a109_literal_pdf(
                    household.desired_house_price,
                    household.deposits + household.other_financial_assets,
                    bank_rate,
                    state.params.mortgage_maturity_quarters,
                    state.forecast.predicted_hpi_inflation,
                    affordable_value,
                );
                buy_probability_a110(state.params.housing_phi_b, rent_cost, purchase_cost)
            } else {
                0.0
            };
            if nearest_sale.is_some() && rng.unit_f64() < buy_probability {
                buyers.push(household_idx);
            } else if nearest_rent.is_some() {
                renters.push(household_idx);
            }
        }

        for household_idx in buyers {
            let household = &mut households[household_idx];
            if let Some(property_idx) = closest_property(
                &properties,
                PROPERTY_FOR_SALE,
                household.desired_house_price,
            ) {
                let property = &mut properties[property_idx];
                let financial_wealth =
                    (household.deposits + household.other_financial_assets).max(0.0);
                let mortgage_required = positive_part(property.price - financial_wealth);
                household.desired_property_id = property.id;
                household.desired_mortgage = mortgage_required;
                property.market_status = PROPERTY_TENTATIVE_SALE;
                buffers.emit(
                    messages.tentative_purchase,
                    TentativePurchase {
                        household_id: household.id,
                        seller_household_id: property.owner_household_id,
                        property_id: property.id,
                        price: property.price,
                        mortgage_required,
                    },
                )?;
                if mortgage_required > 0.0 {
                    buffers.emit(
                        messages.mortgage_need,
                        MortgageNeed {
                            household_id: household.id,
                            property_id: property.id,
                            desired_price: property.price,
                            amount: mortgage_required,
                        },
                    )?;
                }
            }
        }

        for household_idx in renters {
            let household = &mut households[household_idx];
            if let Some(property_idx) =
                closest_property(&properties, PROPERTY_FOR_RENT, household.desired_rent)
            {
                let property = &mut properties[property_idx];
                household.desired_property_id = property.id;
                property.market_status = PROPERTY_TENTATIVE_RENT;
                buffers.emit(
                    messages.tentative_rental,
                    TentativeRental {
                        household_id: household.id,
                        owner_household_id: property.owner_household_id,
                        property_id: property.id,
                        annual_rent: property.rent,
                    },
                )?;
            }
        }

        write_rows(ecs, households, |household: &Household| household.id)?;
        write_rows(ecs, properties, |property: &Property| property.id)?;
        state.push_phase("housing_preclear");
        set_phase_and_state(ecs, env_boundary, phases.housing_preclear_done, state)
    })
}

fn credit_market_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.bank);
    access.write.set(ids.firm);
    access.write.set(ids.household);
    access.read.set(ids.central_bank);
    access
        .consumes
        .insert(phases.housing_preclear_done.channel_id());
    access
        .consumes
        .insert(messages.credit_application.channel_id());
    access.consumes.insert(messages.mortgage_need.channel_id());
    access.produces.insert(messages.goods_demand.channel_id());
    access.produces.insert(messages.credit_grant.channel_id());
    access.produces.insert(messages.credit_failure.channel_id());
    access.produces.insert(phases.credit_done.channel_id());
    FnSystem::new(70, "macro::credit_market", access, move |ecs| {
        // Eqs. A.25-A.39 and A.117-A.118: credit caps, supply, and post-credit demand.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let mut state = macro_state(ecs, env_boundary)?;
        let mut rng = state.rng(ecs.run_context(), rng_salt::CREDIT_MARKET);
        let mut banks = collect_rows::<Bank>(ecs)?;
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let central_banks = collect_rows::<CentralBank>(ecs)?;
        let policy_rate = central_banks
            .first()
            .map(|central_bank| central_bank.policy_rate)
            .unwrap_or_default();
        for bank in &mut banks {
            bank.deposit_rate = policy_rate;
            bank.household_overdraft_rate = bank.household_rate;
            bank.firm_overdraft_rate = bank.short_firm_rate;
        }
        let mut applications: Vec<CreditApplication> =
            buffers.brute_force(messages.credit_application)?.collect();
        for need in buffers.brute_force(messages.mortgage_need)? {
            let household_income = households
                .iter()
                .find(|household| household.id == need.household_id)
                .map(|household| household.predicted_income)
                .unwrap_or(0.0);
            applications.push(CreditApplication {
                borrower_kind: BUYER_HOUSEHOLD,
                borrower_id: need.household_id,
                loan_class: LOAN_MORTGAGE,
                sector: 0,
                amount: need.amount,
                collateral: need.desired_price,
                income: household_income,
            });
        }

        state.audit.credit_clearing_order.clear();
        state.audit.firm_credit_applications = 0;
        state.audit.firm_credit_roa_failures = 0;
        state.audit.firm_credit_requested = 0.0;
        state.audit.firm_credit_granted = 0.0;
        state.audit.firm_roa_max = f64::NEG_INFINITY;
        state.audit.credit_blocked_by_roa = 0;
        state.audit.credit_blocked_by_cap = 0;
        state.audit.credit_blocked_by_supply = 0;
        state.audit.credit_envelope_total = 0.0;
        state.audit.credit_cap_total = 0.0;
        state.audit.cap_dte_total = 0.0;
        state.audit.cap_roe_total = 0.0;
        state.audit.cap_dte_zero = 0;
        state.audit.cap_roe_zero = 0;
        state.audit.credit_visits_ordered_by_rate = true;
        for loan_class in [
            LOAN_FIRM_SHORT,
            LOAN_FIRM_LONG,
            LOAN_HOUSEHOLD_CONSUMPTION,
            LOAN_MORTGAGE,
        ] {
            state.audit.credit_clearing_order.push(loan_class);
            let mut class_apps: Vec<CreditApplication> = applications
                .iter()
                .copied()
                .filter(|app| app.loan_class == loan_class)
                .collect();
            rng.shuffle(&mut class_apps);
            for app in class_apps {
                // Diagnostic for the A.27 return-on-assets screen. It gates
                // *all* firm credit -- if the ratio fails, both the A.25 and
                // A.26 caps are discarded and the firm is refused outright --
                // so a screen that no firm can clear starves the whole economy
                // of working capital without appearing anywhere in the CSV.
                if app.borrower_kind == BUYER_FIRM {
                    if let Some(firm) = firms.iter().find(|firm| firm.id == app.borrower_id) {
                        let assets =
                            firm.short_debt + firm.long_debt + firm.overdraft + firm.equity;
                        let roa = ratio(firm.predicted_profits, assets);
                        state.audit.firm_credit_applications += 1;
                        state.audit.firm_credit_requested += app.amount;
                        state.audit.firm_roa_max = state.audit.firm_roa_max.max(roa);
                        if roa < state.params.return_on_assets {
                            state.audit.firm_credit_roa_failures += 1;
                        }
                    }
                }
                if app.loan_class == LOAN_MORTGAGE {
                    if let Some(h) = households.iter().find(|h| h.id == app.borrower_id) {
                        let (ltv, lti, dsti) = mortgage_caps(h, &state.params, 0.0);
                        let cap = ltv.min(lti).min(dsti).max(0.0);
                        state.audit.mortgage_cap_sum += cap;
                        state.audit.mortgage_req_sum += app.amount;
                        if ltv <= lti && ltv <= dsti {
                            state.audit.mortgage_bind_ltv += 1;
                        } else if lti <= dsti {
                            state.audit.mortgage_bind_lti += 1;
                        } else {
                            state.audit.mortgage_bind_dsti += 1;
                        }
                    }
                }
                let visits = if app.borrower_kind == BUYER_FIRM {
                    state.policy.firm_bank_visits
                } else {
                    state.policy.household_bank_visits
                }
                .max(1) as usize;
                let mut bank_order: Vec<usize> = (0..banks.len()).collect();
                rng.shuffle(&mut bank_order);
                bank_order.truncate(visits.min(bank_order.len()));
                bank_order.sort_by(|&a, &b| {
                    offered_rate(&banks[a], app.loan_class)
                        .total_cmp(&offered_rate(&banks[b], app.loan_class))
                });
                if !bank_order.windows(2).all(|window| {
                    offered_rate(&banks[window[0]], app.loan_class)
                        <= offered_rate(&banks[window[1]], app.loan_class)
                }) {
                    state.audit.credit_visits_ordered_by_rate = false;
                }
                let mut granted = None;
                for bank_idx in bank_order {
                    let rate = offered_rate(&banks[bank_idx], app.loan_class);
                    let allowed = borrower_credit_cap(&state, &firms, &households, app, rate);
                    if app.borrower_kind == BUYER_FIRM {
                        // Recompute A.25 and A.26 separately so a zero joint cap
                        // can be attributed to one or the other.
                        if let Some(firm) = firms.iter().find(|f| f.id == app.borrower_id) {
                            let sp = previous_sector_prices(&firms);
                            let cv: f64 = (0..SECTORS)
                                .map(|s| sp[s] * firm.capital_stock[s])
                                .sum();
                            let dbt = firm.short_debt + firm.long_debt + firm.overdraft;
                            let od = negative_abs(firm.deposits);
                            let dte = positive_part(
                                state.params.debt_to_equity * cv - dbt + od + rate * od
                                    - rate * dbt,
                            );
                            let roe = positive_part(
                                cv + firm.deposits
                                    - dbt
                                    - firm.predicted_profits
                                        / state.params.return_on_equity.max(1e-9),
                            );
                            state.audit.cap_dte_total += dte;
                            state.audit.cap_roe_total += roe;
                            if dte <= 1e-9 {
                                state.audit.cap_dte_zero += 1;
                            }
                            if roe <= 1e-9 {
                                state.audit.cap_roe_zero += 1;
                            }
                        }
                        state.audit.credit_cap_total += allowed;
                        if allowed <= 1e-9 {
                            // A.27 zeroes the cap outright; A.25/A.26 merely
                            // bound it. Separating the two says whether the
                            // screen or the caps are binding.
                            let roa_failed = firms
                                .iter()
                                .find(|firm| firm.id == app.borrower_id)
                                .map(|firm| {
                                    ratio(
                                        firm.predicted_profits,
                                        firm.short_debt + firm.long_debt + firm.overdraft
                                            + firm.equity,
                                    ) < state.params.return_on_assets
                                })
                                .unwrap_or(false);
                            if roa_failed {
                                state.audit.credit_blocked_by_roa += 1;
                            } else {
                                state.audit.credit_blocked_by_cap += 1;
                            }
                        }
                    }
                    // A.36: the envelope is allocated across loan classes by
                    // the A.33-A.35 NPL weights, not offered whole to whoever
                    // asks first.
                    let supply_probe = bank_class_credit_supply(
                        &banks[bank_idx],
                        &state,
                        app.loan_class,
                        app.sector,
                    );
                    if app.borrower_kind == BUYER_FIRM {
                        state.audit.credit_envelope_total += supply_probe;
                        if supply_probe <= 1e-9 && allowed > 1e-9 {
                            state.audit.credit_blocked_by_supply += 1;
                        }
                    }
                    let supply = bank_class_credit_supply(
                        &banks[bank_idx],
                        &state,
                        app.loan_class,
                        app.sector,
                    );
                    let amount = app.amount.min(allowed).min(supply);
                    if amount > 1e-9 {
                        apply_loan(
                            &mut banks[bank_idx],
                            &mut firms,
                            &mut households,
                            app,
                            amount,
                        );
                        state.loan_book.add(
                            banks[bank_idx].id,
                            app.borrower_kind,
                            app.borrower_id,
                            app.sector,
                            app.loan_class,
                            amount,
                            rate,
                            loan_maturity_quarters(&state, app.loan_class),
                            state.quarter,
                        );
                        if app.borrower_kind == BUYER_FIRM {
                            state.audit.firm_credit_granted += amount;
                        }
                        granted = Some(CreditGrant {
                            borrower_kind: app.borrower_kind,
                            borrower_id: app.borrower_id,
                            loan_class: app.loan_class,
                            bank_id: banks[bank_idx].id,
                            amount,
                            rate,
                        });
                        break;
                    }
                }
                if let Some(grant) = granted {
                    buffers.emit(messages.credit_grant, grant)?;
                } else {
                    buffers.emit(
                        messages.credit_failure,
                        CreditFailure {
                            borrower_kind: app.borrower_kind,
                            borrower_id: app.borrower_id,
                            loan_class: app.loan_class,
                            requested: app.amount,
                            reason_code: 1,
                        },
                    )?;
                }
            }
        }

        let previous_sector_prices = previous_sector_prices(&firms);
        for firm in &mut firms {
            for s in 0..SECTORS {
                let previous_sector_price = previous_sector_prices[s];
                firm.target_intermediate[s] = constrained_goods_target_a83_a84(
                    firm.target_intermediate[s],
                    state.params.firm_credit_shortfall_intermediate_sensitivity,
                    firm.target_short_loan,
                    firm.granted_short_loan,
                    state.forecast.predicted_ppi_inflation,
                    previous_sector_price,
                );
                firm.target_capital[s] = constrained_goods_target_a83_a84(
                    firm.target_capital[s],
                    state.params.firm_credit_shortfall_capital_sensitivity,
                    firm.target_long_loan,
                    firm.granted_long_loan,
                    state.forecast.predicted_ppi_inflation,
                    previous_sector_price,
                );
                if firm.target_intermediate[s] > 0.0 {
                    buffers.emit(
                        messages.goods_demand,
                        GoodsDemand {
                            buyer_kind: BUYER_FIRM,
                            buyer_id: firm.id,
                            purpose: GOODS_INTERMEDIATE,
                            sector: s as u8,
                            quantity: firm.target_intermediate[s],
                            max_spend: firm.deposits.max(0.0),
                        },
                    )?;
                }
                if firm.target_capital[s] > 0.0 {
                    buffers.emit(
                        messages.goods_demand,
                        GoodsDemand {
                            buyer_kind: BUYER_FIRM,
                            buyer_id: firm.id,
                            purpose: GOODS_CAPITAL,
                            sector: s as u8,
                            quantity: firm.target_capital[s],
                            max_spend: firm.deposits.max(0.0),
                        },
                    )?;
                }
            }
        }

        for household in &households {
            for s in 0..SECTORS {
                if household.consumption_target[s] > 0.0 {
                    buffers.emit(
                        messages.goods_demand,
                        GoodsDemand {
                            buyer_kind: BUYER_HOUSEHOLD,
                            buyer_id: household.id,
                            purpose: GOODS_CONSUMPTION,
                            sector: s as u8,
                            quantity: household.consumption_target[s],
                            max_spend: household.deposits.max(0.0),
                        },
                    )?;
                }
                if household.investment_target[s] > 0.0 {
                    buffers.emit(
                        messages.goods_demand,
                        GoodsDemand {
                            buyer_kind: BUYER_HOUSEHOLD,
                            buyer_id: household.id,
                            purpose: GOODS_CAPITAL,
                            sector: s as u8,
                            quantity: household.investment_target[s],
                            max_spend: household.deposits.max(0.0),
                        },
                    )?;
                }
            }
        }

        write_rows(ecs, banks, |bank: &Bank| bank.id)?;
        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, households, |household: &Household| household.id)?;
        state.push_phase("credit_market");
        set_phase_and_state(ecs, env_boundary, phases.credit_done, state)
    })
}

fn housing_completion_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.household);
    access.write.set(ids.property);
    access.consumes.insert(phases.credit_done.channel_id());
    access
        .consumes
        .insert(messages.tentative_purchase.channel_id());
    access
        .consumes
        .insert(messages.tentative_rental.channel_id());
    access.consumes.insert(messages.credit_grant.channel_id());
    access
        .produces
        .insert(messages.property_transfer.channel_id());
    access
        .produces
        .insert(phases.housing_completion_done.channel_id());
    FnSystem::new(80, "macro::housing_completion", access, move |ecs| {
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let purchases: Vec<TentativePurchase> =
            buffers.brute_force(messages.tentative_purchase)?.collect();
        let rentals: Vec<TentativeRental> =
            buffers.brute_force(messages.tentative_rental)?.collect();
        let grants: Vec<CreditGrant> = buffers.brute_force(messages.credit_grant)?.collect();
        let mut state = macro_state(ecs, env_boundary)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let mut properties = collect_rows::<Property>(ecs)?;

        for purchase in purchases {
            let has_mortgage = purchase.mortgage_required <= 1e-9
                || grants.iter().any(|grant| {
                    grant.borrower_kind == BUYER_HOUSEHOLD
                        && grant.borrower_id == purchase.household_id
                        && grant.loan_class == LOAN_MORTGAGE
                        && grant.amount + 1e-9 >= purchase.mortgage_required
                });
            let Some(property_idx) = properties.iter().position(|p| p.id == purchase.property_id)
            else {
                continue;
            };
            if !has_mortgage {
                properties[property_idx].market_status = PROPERTY_FOR_SALE;
                state.audit.mortgage_blocked_purchases += 1;
                continue;
            }
            let buyer_idx = households
                .iter()
                .position(|h| h.id == purchase.household_id);
            let seller_idx = households
                .iter()
                .position(|h| h.id == purchase.seller_household_id);
            if let Some(idx) = buyer_idx {
                let mut remaining_payment = purchase.price;
                let deposit_payment = households[idx].deposits.min(remaining_payment);
                households[idx].deposits -= deposit_payment;
                remaining_payment -= deposit_payment;
                let other_financial_payment = households[idx]
                    .other_financial_assets
                    .min(remaining_payment);
                households[idx].other_financial_assets -= other_financial_payment;
                households[idx].residence_property_id = purchase.property_id;
                households[idx].owns_residence = true;
            }
            if let Some(idx) = seller_idx {
                households[idx].deposits += purchase.price;
            }
            properties[property_idx].owner_household_id = purchase.household_id;
            properties[property_idx].occupant_household_id = purchase.household_id;
            properties[property_idx].market_status = PROPERTY_OWNER_OCCUPIED;
            properties[property_idx].value = purchase.price;
            state.audit.housing_sales += 1;
            buffers.emit(
                messages.property_transfer,
                PropertyTransfer {
                    household_id: purchase.household_id,
                    seller_household_id: purchase.seller_household_id,
                    property_id: purchase.property_id,
                    price: purchase.price,
                    mortgage_amount: purchase.mortgage_required,
                },
            )?;
        }

        for rental in rentals {
            let Some(property_idx) = properties.iter().position(|p| p.id == rental.property_id)
            else {
                continue;
            };
            if let Some(idx) = households.iter().position(|h| h.id == rental.household_id) {
                households[idx].residence_property_id = rental.property_id;
                households[idx].owns_residence = false;
                households[idx].deposits -= rental.annual_rent / 4.0;
            }
            if let Some(idx) = households
                .iter()
                .position(|h| h.id == rental.owner_household_id)
            {
                households[idx].deposits += rental.annual_rent / 4.0;
            }
            properties[property_idx].occupant_household_id = rental.household_id;
            properties[property_idx].market_status = PROPERTY_RENTAL;
        }

        write_rows(ecs, households, |household: &Household| household.id)?;
        write_rows(ecs, properties, |property: &Property| property.id)?;
        state.push_phase("housing_completion");
        set_phase_and_state(ecs, env_boundary, phases.housing_completion_done, state)
    })
}

fn goods_market_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.firm);
    access.write.set(ids.household);
    access.write.set(ids.government_entity);
    access.write.set(ids.rest_of_world);
    access
        .consumes
        .insert(phases.housing_completion_done.channel_id());
    access.consumes.insert(messages.goods_demand.channel_id());
    access.produces.insert(messages.goods_receipt.channel_id());
    access.produces.insert(messages.excess_demand.channel_id());
    access.produces.insert(phases.goods_done.channel_id());
    FnSystem::new(90, "macro::goods_market", access, move |ecs| {
        // Eq. A.140 / thesis Ch. 6 pp. 210-211, with the Poledna et al.
        // Online Appendix A.1.1 source algorithm for the inherited ABM
        // lineage: random seller search weighted by price and firm size.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let raw_demands: Vec<GoodsDemand> = buffers.brute_force(messages.goods_demand)?.collect();
        let mut state = macro_state(ecs, env_boundary)?;
        let mut rng = state.rng(ecs.run_context(), rng_salt::GOODS_MARKET);
        // A.10.2: "Firms are always prioritised as buyers." Demands are grouped
        // by buyer type and the firm block is matched to completion first;
        // within a block the order is randomised as before.
        //
        // A single shuffled queue -- what this used to be -- lets household,
        // government and rest-of-world consumption crowd out firms' intermediate
        // purchases. A firm that loses the draw cannot restock, and A.63 then
        // caps its next-quarter output at the fraction of the input buffer it
        // still holds, which cuts its sales and its demand in turn.
        let demands = order_demands_firms_first(raw_demands, &mut rng);
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let mut governments = collect_rows::<GovernmentEntity>(ecs)?;
        let mut rows = collect_rows::<RestOfWorld>(ecs)?;

        // Real supply and demand offered to the market this quarter. See
        // `MarketAudit::goods_demand_quantity`.
        state.audit.goods_demand_quantity = demands.iter().map(|d| d.quantity).sum();
        // A.60 feeds Q_f(t-1) forward multiplicatively, so a t=1 shortfall of
        // realised demand against production never washes out. Attribute the
        // demand side by buyer and purpose to locate any missing component.
        let mut sum_of = |kind: u8, purpose: u8| -> f64 {
            demands
                .iter()
                .filter(|d| d.buyer_kind == kind && d.purpose == purpose)
                .map(|d| d.quantity)
                .sum()
        };
        state.audit.demand_firm_intermediate = sum_of(BUYER_FIRM, GOODS_INTERMEDIATE);
        state.audit.demand_firm_capital = sum_of(BUYER_FIRM, GOODS_CAPITAL);
        state.audit.demand_household_consumption = sum_of(BUYER_HOUSEHOLD, GOODS_CONSUMPTION);
        state.audit.demand_household_capital = sum_of(BUYER_HOUSEHOLD, GOODS_CAPITAL);
        state.audit.demand_government = demands
            .iter()
            .filter(|d| d.buyer_kind == BUYER_GOVERNMENT)
            .map(|d| d.quantity)
            .sum();
        state.audit.demand_export = demands
            .iter()
            .filter(|d| d.buyer_kind == BUYER_ROW)
            .map(|d| d.quantity)
            .sum();
        state.audit.goods_supply_quantity = firms
            .iter()
            .map(|firm| positive_part(firm.production + firm.inventory))
            .sum();

        for demand in demands {
            let sector = demand.sector as usize;
            let sellers: Vec<usize> = firms
                .iter()
                .enumerate()
                .filter(|(_, firm)| firm.sector as usize == sector)
                .filter(|(idx, firm)| {
                    positive_part(firm.production + firm.inventory - firms[*idx].sales_quantity)
                        > 0.0
                })
                .map(|(idx, _)| idx)
                .collect();
            if sellers.len() >= 2 {
                let min_price_idx = sellers
                    .iter()
                    .copied()
                    .min_by(|&a, &b| firms[a].price.total_cmp(&firms[b].price))
                    .unwrap_or(sellers[0]);
                let max_price_idx = sellers
                    .iter()
                    .copied()
                    .max_by(|&a, &b| firms[a].price.total_cmp(&firms[b].price))
                    .unwrap_or(sellers[0]);
                let min_component =
                    (-state.params.goods_market_phi * firms[min_price_idx].price).exp();
                let max_component =
                    (-state.params.goods_market_phi * firms[max_price_idx].price).exp();
                state.audit.lower_price_seller_priority_seen |= firms[min_price_idx].price
                    < firms[max_price_idx].price
                    && min_component > max_component;
            }

            let mut remaining = demand.quantity;
            let mut remaining_budget = demand.max_spend;
            // Sellers this buyer has already found unusable this round --
            // either sold out or too dear for the budget left. Without it the
            // `quantity <= 1e-9` arm below re-entered the loop having changed
            // nothing and spun forever, which is reachable as soon as a firm's
            // price exceeds the buyer's remaining budget by a factor of 1e9.
            let mut exhausted: Vec<usize> = Vec::new();
            while remaining > 1e-9 && remaining_budget > 1e-9 {
                let available_sellers: Vec<usize> = sellers
                    .iter()
                    .copied()
                    .filter(|idx| {
                        !exhausted.contains(idx)
                            && positive_part(
                                firms[*idx].production + firms[*idx].inventory
                                    - firms[*idx].sales_quantity,
                            ) > 1e-9
                    })
                    .collect();
                if available_sellers.is_empty() {
                    break;
                }
                let weights = seller_priority_weights(
                    &firms,
                    &available_sellers,
                    state.params.goods_market_phi,
                );
                let firm_idx = available_sellers[weighted_choice(&mut rng, &weights)];
                let available = positive_part(
                    firms[firm_idx].production + firms[firm_idx].inventory
                        - firms[firm_idx].sales_quantity,
                );
                let affordable = remaining_budget / firms[firm_idx].price.max(1e-9);
                let quantity = remaining.min(available).min(affordable);
                if quantity <= 1e-9 {
                    // Unusable seller for this buyer: retire it and try the
                    // rest. Each pass either moves goods or shrinks the
                    // candidate set, so the loop terminates.
                    exhausted.push(firm_idx);
                    continue;
                }
                let payment = quantity * firms[firm_idx].price;
                // Revenue is recorded, not banked: A.91 credits `P_f * Q~_f`
                // once at the end of the quarter. Crediting here as well made
                // every sale hit deposits twice.
                firms[firm_idx].sales_quantity += quantity;
                firms[firm_idx].sales_revenue += payment;
                apply_buyer_goods(
                    &mut firms,
                    &mut households,
                    &mut governments,
                    &mut rows,
                    demand,
                    quantity,
                    payment,
                );
                buffers.emit(
                    messages.goods_receipt,
                    GoodsReceipt {
                        buyer_kind: demand.buyer_kind,
                        buyer_id: demand.buyer_id,
                        seller_kind: BUYER_FIRM,
                        seller_id: firms[firm_idx].id,
                        purpose: demand.purpose,
                        sector: demand.sector,
                        quantity,
                        payment,
                    },
                )?;
                remaining -= quantity;
                remaining_budget -= payment;
            }
            if remaining > 1e-9 {
                state.audit.goods_excess_demand += remaining;
                distribute_excess_demand(
                    &mut firms,
                    sector,
                    remaining,
                    state.params.goods_market_phi,
                );
                buffers.emit(
                    messages.excess_demand,
                    ExcessDemand {
                        buyer_kind: demand.buyer_kind,
                        buyer_id: demand.buyer_id,
                        sector: demand.sector,
                        quantity: remaining,
                    },
                )?;
            }
        }

        // A.139: NX^ROW(t) = NX^ROW(t-1) + exports - imports. Cumulative by
        // definition, so unlike `exports` this is deliberately never reset --
        // but it has to net off imports, which it previously ignored.
        for row in &mut rows {
            row.net_exports += row.exports - row.imports;
        }

        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, households, |household: &Household| household.id)?;
        write_rows(ecs, governments, |government: &GovernmentEntity| {
            government.id
        })?;
        write_rows(ecs, rows, |row: &RestOfWorld| row.id)?;
        state.push_phase("goods_market");
        set_phase_and_state(ecs, env_boundary, phases.goods_done, state)
    })
}

fn realised_accounting_system(
    ids: MacroComponentIds,
    messages: MacroMessageHandles,
    phases: PhaseKeys,
    env_boundary: abm_framework::BoundaryID,
    message_boundary: abm_framework::BoundaryID,
) -> FnSystem<impl Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static> {
    let mut access = AccessSets::default();
    access.write.set(ids.firm);
    access.write.set(ids.household);
    access.write.set(ids.bank);
    access.write.set(ids.government_account);
    access.read.set(ids.individual);
    access.read.set(ids.government_entity);
    access.read.set(ids.central_bank);
    access.read.set(ids.property);
    access.read.set(ids.rest_of_world);
    access.consumes.insert(phases.goods_done.channel_id());
    access.consumes.insert(messages.goods_receipt.channel_id());
    access.consumes.insert(messages.excess_demand.channel_id());
    access.consumes.insert(messages.credit_grant.channel_id());
    access
        .consumes
        .insert(messages.property_transfer.channel_id());
    access.produces.insert(phases.accounting_done.channel_id());
    FnSystem::new(100, "macro::realised_accounting", access, move |ecs| {
        // Eqs. A.40-A.44, A.85-A.100, and A.119-A.127: realised stock-flow accounting.
        let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
        let receipts: Vec<GoodsReceipt> = buffers.brute_force(messages.goods_receipt)?.collect();
        // A.121: property wealth follows completed transfers. This channel was
        // emitted, declared consumed by this system, and never actually read,
        // so housing transactions never reached the accounting pass at all.
        let property_transfers: Vec<PropertyTransfer> =
            buffers.brute_force(messages.property_transfer)?.collect();
        // Drained to keep the channel from carrying into the next quarter. The
        // per-firm shares these aggregate to are read off `firm.excess_demand`,
        // which the goods market has already apportioned (A.88, A.140).
        // `ExcessDemand` is consumed in the goods market, where
        // `distribute_excess_demand` apportions each sector total across its
        // sellers. Nothing in this system reads it, so it is not collected --
        // materialising the whole buffer here allocated once per tick for a
        // value that was immediately dropped.
        let mut state = macro_state(ecs, env_boundary)?;
        let mut firms = collect_rows::<Firm>(ecs)?;
        let individuals = collect_rows::<Individual>(ecs)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let mut banks = collect_rows::<Bank>(ecs)?;
        let mut accounts = collect_rows::<GovernmentAccount>(ecs)?;
        let governments = collect_rows::<GovernmentEntity>(ecs)?;
        let central_banks = collect_rows::<CentralBank>(ecs)?;
        let properties = collect_rows::<Property>(ecs)?;
        let rows = collect_rows::<RestOfWorld>(ecs)?;
        let policy_rate = central_banks
            .first()
            .map(|central_bank| central_bank.policy_rate)
            .unwrap_or_default();
        // Tax rates for A.91's corporate tax and A.123's capital-formation tax.
        let account = accounts.first().copied().unwrap_or_default();
        // A.41 bad debt, collected as borrowers fail and applied to their banks
        // once every borrower has been settled.
        let mut write_offs: Vec<BadDebt> = Vec::new();
        let mut dividend_pool = 0.0f64;
        state.audit.profit_sales_revenue = 0.0;
        state.audit.profit_inventory_change = 0.0;
        state.audit.profit_costs = 0.0;
        // A.89's intermediate and capital terms are each firm's own receipt
        // payments. Filtering the whole receipt buffer inside the firm loop is
        // O(firms x receipts); one indexed pass gives the same totals. Firm ids
        // are dense, so the accumulators are Vecs.
        let mut intermediate_by_firm = vec![0.0; firms.len()];
        let mut capital_by_firm = vec![0.0; firms.len()];
        for receipt in &receipts {
            if receipt.buyer_kind != BUYER_FIRM {
                continue;
            }
            let slot = receipt.buyer_id as usize;
            let target = match receipt.purpose {
                GOODS_INTERMEDIATE => intermediate_by_firm.get_mut(slot),
                GOODS_CAPITAL => capital_by_firm.get_mut(slot),
                _ => None,
            };
            if let Some(value) = target {
                *value += receipt.payment;
            }
        }
        state.audit.cost_wages = 0.0;
        state.audit.cost_intermediate = 0.0;
        state.audit.cost_capital = 0.0;
        state.audit.cost_production_tax = 0.0;
        state.audit.cost_interest = 0.0;
        state.audit.bank_loan_interest = 0.0;
        state.audit.bank_reserve_income = 0.0;
        state.audit.bank_reserve_cost = 0.0;
        state.audit.bank_deposit_interest = 0.0;
        state.audit.bank_corporate_tax = 0.0;
        state.audit.bank_writeoff_seized = 0.0;
        state.audit.bank_writeoff_lost = 0.0;
        let loan_settlement = settle_loan_book(
            state.quarter,
            &mut state,
            &mut banks,
            &mut firms,
            &mut households,
        );

        // P_s'(t-1) for the A.77 unit-cost terms.
        let previous_prices = previous_sector_prices(&firms);

        for firm in &mut firms {
            let sector = firm.sector as usize;
            let previous_inventory = firm.inventory;
            let previous_production = firm.production;
            firm.inventory = positive_part(firm.inventory + firm.production - firm.sales_quantity);
            for s in 0..SECTORS {
                // A.86: stock is drawn down by what production *consumed*,
                // `m_{s's} * Y_f(t)`, not by what the firm sold. Using
                // `sales_quantity` decouples input usage from output and lets
                // stocks drift arbitrarily against production.
                firm.intermediate_stock[s] = positive_part(
                    firm.intermediate_stock[s]
                        - state.params.io_matrix[sector][s] * firm.production
                        + firm.realised_intermediate[s],
                );
                let installed_capital = firm.capital_to_install[s];
                // A.87: K_fs'(t) = [K_fs'(t-1) - d_{s's} Y_f(t) + K~(t-T^KD)]^+
                //
                // Capital is consumed in proportion to *production*, through
                // the capital-compensation matrix -- not as a flat rate on the
                // stock. The stock-proportional form is a different model: it
                // decays capital a firm is not using and leaves a producing
                // firm's capital untouched by how hard it produces.
                firm.capital_stock[s] = positive_part(
                    firm.capital_stock[s]
                        - state.params.capital_compensation_matrix[sector][s] * firm.production
                        + installed_capital,
                );
                firm.capital_to_install[s] = firm.realised_capital[s];
                firm.realised_intermediate[s] = 0.0;
                firm.realised_capital[s] = 0.0;
            }
            // A.88: Q_f(t) = Q~_f(t) + Y~^E_f(t) -- units sold plus *this
            // firm's* share of what buyers could not source. The `ExcessDemand`
            // messages are per buyer, so summing them by sector gives the
            // sector total; booking that against every firm in the sector
            // multiplied demand by the number of sellers and fed straight into
            // `previous_demand` -> A.60 -> A.62. `distribute_excess_demand`
            // already apportions the sector total across sellers by the A.140
            // weights, and leaves each firm's share in `excess_demand`.
            firm.demand = firm.sales_quantity + firm.excess_demand;
            let intermediate_purchases =
                intermediate_by_firm.get(firm.id as usize).copied().unwrap_or(0.0);
            let production_tax =
                account_production_tax(&accounts, sector) * firm.price * firm.production;
            // A.89's capital term is what the firm *bought*,
            // `P_s'(K(t) - K(t-1) + d_{s's} Y_f)`, which by A.87 is exactly the
            // realised capital purchases -- the same identity that makes the
            // intermediate term equal the receipt payments. The previous code
            // charged a notional depreciation of the stock instead, which is
            // neither a payment nor a price.
            let capital_purchases =
                capital_by_firm.get(firm.id as usize).copied().unwrap_or(0.0);
            let loan_interest_cost = loan_settlement.firm_interest(firm.id);
            firm.costs = firm_wage_bill(firm)
                + intermediate_purchases
                + production_tax
                + capital_purchases
                + loan_interest_cost;
            // A.90/A.89 term breakdown -- diagnostic only, see MarketAudit.
            state.audit.cost_wages += firm_wage_bill(firm);
            state.audit.cost_intermediate += intermediate_purchases;
            state.audit.cost_capital += capital_purchases;
            state.audit.cost_production_tax += production_tax;
            state.audit.cost_interest += loan_interest_cost;
            state.audit.profit_costs += firm.costs;
            // A.77 -- per-unit, from technology coefficients and prices. Not
            // `firm.costs / production`: that numerator includes restocking
            // purchases and loan interest, and divides by a denominator that
            // collapses when output falls.
            firm.unit_cost = unit_cost_a77(
                firm_wage_bill(firm),
                firm.production,
                &state.params.io_matrix[sector],
                &state.params.capital_compensation_matrix[sector],
                &previous_prices,
                account_production_tax(&accounts, sector),
                firm.previous_price,
            );
            let delta_inventory = firm.inventory - previous_inventory;
            // A.90.
            firm.profits =
                firm.price * firm.sales_quantity + firm.price * delta_inventory - firm.costs;
            state.audit.profit_sales_revenue += firm.price * firm.sales_quantity;
            state.audit.profit_inventory_change += firm.price * delta_inventory;
            // A.91: D_f(t) = D_f(t-1) + P_f*Q~_f - C_f - tau^CORP[Pi_f]^+ , with
            // loan instalments and new credit already applied by
            // `settle_loan_book` / `apply_loan`.
            //
            // Two defects here. Sales revenue was credited to deposits twice:
            // once per transaction in the goods market and again inside
            // `profits`, with purchases double-deducted the same way -- so the
            // firm sector was creating money out of its own turnover. And the
            // inventory change belongs in profit but not in cash: unsold goods
            // are a real asset, not a deposit. Separately, corporate tax was
            // collected into government revenue but never left any firm.
            let corporate_tax = account.corporate_tax_rate * positive_part(firm.profits);
            // Poledna A.33's `- theta^DIV (1 - tau^FIRM) max(0, Pi_i)`.
            // Wiese's A.80 omits this term; it is restored by default at
            // Poledna's Austrian value. Without it firm profits pile up in
            // deposits and never reach households as income. See
            // `CountryParameters::theta_dividend`.
            let dividend = state.params.theta_dividend
                * (1.0 - account.corporate_tax_rate)
                * positive_part(firm.profits);
            dividend_pool += dividend;
            firm.deposits +=
                firm.price * firm.sales_quantity - firm.costs - corporate_tax - dividend;
            // A negative deposit balance *is* an overdraft. This field was read
            // by A.25 and A.93 and incremented by nothing, so the overdraft
            // facility the paper assumes (A.25, A.40, A.80, A.89, A.123) did
            // not exist: a firm could run its deposits arbitrarily negative
            // free of charge.
            firm.overdraft = negative_abs(firm.deposits);
            // A.93: equity is valued at prices, and includes the intermediate
            // stock. It previously counted inventory and capital as bare
            // quantities and omitted M entirely.
            let stock_value: f64 = (0..SECTORS)
                .map(|s| previous_prices[s] * (firm.intermediate_stock[s] + firm.capital_stock[s]))
                .sum();
            firm.equity = firm.deposits + firm.price * firm.inventory + stock_value
                - firm.short_debt
                - firm.long_debt
                - firm.overdraft;
            if firm.deposits < 0.0 && firm.equity < 0.0 {
                firm.bankrupt = true;
                // A.41: the lending bank appropriates whatever is left as a
                // deposit and loses the loan outright. Previously the borrower's
                // debt was simply zeroed and the loss landed nowhere -- and the
                // loan-book entry survived, so a bankrupt firm went on being
                // debited and went on crediting its bank with interest forever.
                write_offs.push(BadDebt {
                    bank_id: firm.bank_id,
                    borrower_kind: BUYER_FIRM,
                    borrower_id: firm.id,
                    deposits_seized: positive_part(firm.deposits),
                    loans_lost: firm.short_debt + firm.long_debt + firm.overdraft,
                });
                firm.deposits = 0.0;
                firm.short_debt = 0.0;
                firm.long_debt = 0.0;
                firm.overdraft = 0.0;
                firm.equity = firm.price * firm.inventory + stock_value;
            }
            firm.inventory_two_periods_ago = firm.previous_inventory;
            firm.previous_inventory = firm.inventory;
            firm.previous_demand = firm.demand;
            firm.previous_production = previous_production;
            firm.previous_price = firm.price;
        }

        let household_income_total: f64 =
            households.iter().map(|h| positive_part(h.income)).sum();
        let household_count_f = households.len().max(1) as f64;
        for household in &mut households {
            let consumed = receipts
                .iter()
                .filter(|receipt| {
                    receipt.buyer_kind == BUYER_HOUSEHOLD
                        && receipt.buyer_id == household.id
                        && receipt.purpose == GOODS_CONSUMPTION
                })
                .map(|receipt| receipt.payment)
                .sum::<f64>();
            household.consumption_history.rotate_left(1);
            household.consumption_history[11] = consumed;
            // A.123. Household deposits previously received *nothing*: there
            // was no income credit anywhere in the model, so wages were paid by
            // firms and vanished. Loan instalments and new credit are already
            // applied by `settle_loan_book` / `apply_loan`; what is added here
            // is the income-less-spending flow, deposit interest, and the
            // capital-formation tax.
            //
            // A.119/A.120: a positive surplus is shared between deposits and
            // other financial assets "in fixed fractions"; a negative one draws
            // down other financial assets first, then deposits. The paper does
            // not give the fractions -- `DEPOSIT_SHARE_OF_SAVING` records that
            // silence rather than hiding it.
            let capital_formation = receipts
                .iter()
                .filter(|receipt| {
                    receipt.buyer_kind == BUYER_HOUSEHOLD
                        && receipt.buyer_id == household.id
                        && receipt.purpose == GOODS_CAPITAL
                })
                .map(|receipt| receipt.payment)
                .sum::<f64>();
            let deposit_interest = policy_rate * positive_part(household.deposits);
            let surplus = household.disposable_income_after_rent
                - consumed
                - capital_formation
                - account.capital_tax_rate * capital_formation
                + deposit_interest;
            if surplus >= 0.0 {
                household.deposits += DEPOSIT_SHARE_OF_SAVING * surplus;
                household.other_financial_assets += (1.0 - DEPOSIT_SHARE_OF_SAVING) * surplus;
            } else {
                let shortfall = -surplus;
                let from_other = shortfall.min(positive_part(household.other_financial_assets));
                household.other_financial_assets -= from_other;
                household.deposits -= shortfall - from_other;
            }
            household.other_real_assets = positive_part(
                household.other_real_assets
                    * (1.0 - state.params.other_real_asset_depreciation_rate)
                    + receipts
                        .iter()
                        .filter(|receipt| {
                            receipt.buyer_kind == BUYER_HOUSEHOLD
                                && receipt.buyer_id == household.id
                                && receipt.purpose == GOODS_CAPITAL
                        })
                        .map(|receipt| receipt.payment)
                        .sum::<f64>(),
            );
            // A.121, with the completed transfers of this quarter reflected: a
            // buyer's wealth rises by what it paid, a seller's falls by the
            // same, on top of the mark-to-market of what each still owns.
            household.property_wealth = properties
                .iter()
                .filter(|property| property.owner_household_id == household.id)
                .map(|property| property.value)
                .sum();
            let transferred_in: f64 = property_transfers
                .iter()
                .filter(|transfer| transfer.household_id == household.id)
                .map(|transfer| transfer.price)
                .sum();
            state.audit.housing_transfer_value += transferred_in;
            household.net_wealth = household.property_wealth
                + household.other_real_assets
                + household.deposits
                + household.other_financial_assets
                - household.consumption_debt
                - household.mortgage_debt;
            if household.net_wealth < 0.0 && household.deposits < 0.0 {
                household.bankrupt = true;
                // A.41 / A.7.3, same treatment as a bankrupt firm.
                write_offs.push(BadDebt {
                    bank_id: household.bank_id,
                    borrower_kind: BUYER_HOUSEHOLD,
                    borrower_id: household.id,
                    deposits_seized: positive_part(household.deposits),
                    loans_lost: household.consumption_debt + household.mortgage_debt,
                });
                // A.7.3: "The banks receive all of the households' financial
                // wealth, as well as its other owned properties. The remaining
                // debt and deposit overdrafts are written off. If the household
                // is in the process of paying off a mortgage for its current
                // residence, the bank takes the residence, and the household
                // will seek to rent or buy in the next iteration."
                household.consumption_debt = 0.0;
                household.mortgage_debt = 0.0;
                household.deposits = 0.0;
                household.other_financial_assets = 0.0;
                household.property_wealth = 0.0;
                household.owns_residence = false;
                household.residence_property_id = NOT_LINKED;
            }
            // Dividends are allocated by income share, following Poledna C.2's
            // use of income as the proxy for a household's ownership stake.
            let share = if household_income_total > 1e-9 {
                positive_part(household.income) / household_income_total
            } else {
                1.0 / household_count_f
            };
            household.dividend_income = dividend_pool * share;
            household.deposits += household.dividend_income;
            household.income_history.rotate_left(1);
            household.income_history[1] = household.income;
            household.previous_income = household.income;
        }

        let total_wages = firms
            .iter()
            .map(|firm| firm_wage_bill(firm))
            .sum::<f64>();
        let total_profits = firms.iter().map(|firm| firm.profits).sum::<f64>();
        // A.98's VAT base is household *consumption*. Filtering on buyer kind
        // alone swept in `GOODS_CAPITAL` receipts, which A.98 already taxes
        // separately at `tau^CF` -- so household capital formation was taxed
        // twice.
        let total_consumption = receipts
            .iter()
            .filter(|receipt| {
                receipt.buyer_kind == BUYER_HOUSEHOLD && receipt.purpose == GOODS_CONSUMPTION
            })
            .map(|receipt| receipt.payment)
            .sum::<f64>();
        let total_gov = receipts
            .iter()
            .filter(|receipt| receipt.buyer_kind == BUYER_GOVERNMENT)
            .map(|receipt| receipt.payment)
            .sum::<f64>();
        let total_investment = receipts
            .iter()
            .filter(|receipt| receipt.purpose == GOODS_CAPITAL)
            .map(|receipt| receipt.payment)
            .sum::<f64>();
        // Retire the loans of failed borrowers. Without this the loan book kept
        // debiting deposits and crediting bank interest for borrowers that no
        // longer exist -- forever, since nothing else ever removed an entry.
        if !write_offs.is_empty() {
            state.loan_book.loans.retain(|loan| {
                !write_offs.iter().any(|debt| {
                    debt.borrower_kind == loan.borrower_kind
                        && debt.borrower_id == loan.borrower_id
                })
            });
        }

        let previous_policy_rate = central_banks
            .first()
            .map(|central_bank| central_bank.predicted_policy_rate)
            .unwrap_or(policy_rate);
        for bank in &mut banks {
            let previous_reserves = bank.reserves;
            let firm_deposits = firms
                .iter()
                .filter(|firm| firm.bank_id == bank.id)
                .map(|firm| firm.deposits)
                .sum::<f64>();
            let household_deposits = households
                .iter()
                .filter(|household| household.bank_id == bank.id)
                .map(|household| household.deposits)
                .sum::<f64>();
            let positive_firm_deposits = firms
                .iter()
                .filter(|firm| firm.bank_id == bank.id)
                .map(|firm| positive_part(firm.deposits))
                .sum::<f64>();
            let positive_household_deposits = households
                .iter()
                .filter(|household| household.bank_id == bank.id)
                .map(|household| positive_part(household.deposits))
                .sum::<f64>();
            bank.deposits = positive_firm_deposits + positive_household_deposits;
            // A.24: bank lending rates follow a single-equation error-correction
            // model derived from an ARDL, one per loan type.
            //
            // `ardl_error_correction_delta_rate` has existed and been tested
            // since the beginning and was never called by any system, so
            // `short_firm_rate`, `long_firm_rate`, `household_rate` and
            // `mortgage_rate` never moved after initialisation. The Taylor rule
            // reached `deposit_rate` and stopped: there was no monetary
            // transmission in the model at all.
            //
            // The paper estimates the coefficients from historic real data at
            // initialisation. With synthetic data there is nothing to estimate
            // against, so they are stated in `CountryParameters` and the lag
            // structure is reduced to the contemporaneous terms A.24 always
            // carries. That is a synthetic-data deviation, not an equation
            // change.
            let npl_firm = bank.npl_firm_by_sector.iter().sum::<f64>() / SECTORS as f64;
            let policy_delta = policy_rate - previous_policy_rate;
            let ppi_delta = state.aggregates.ppi - state.previous_aggregates.ppi;
            let params = &state.params;
            let mut ardl_step = |rate: f64, npl: f64| -> f64 {
                let input = ArdlErrorCorrectionInput {
                    previous_loan_rate: rate,
                    current_policy_rate: policy_rate,
                    error_correction_phi: params.ardl_error_correction_phi,
                    long_run_pass_through_phi: params.ardl_long_run_pass_through,
                    lagged_loan_rate_deltas: &[],
                    alpha: &[],
                    lagged_policy_rate_deltas: &[policy_delta],
                    beta: &[params.ardl_policy_beta],
                    lagged_ppi_inflation_deltas: &[ppi_delta],
                    gamma: &[params.ardl_inflation_gamma],
                    lagged_npl_ratio_deltas: &[npl],
                    delta: &[params.ardl_npl_delta],
                    mu: 0.0,
                };
                (rate + ardl_error_correction_delta_rate(&input)).clamp(0.0, 1.0)
            };
            bank.short_firm_rate = ardl_step(bank.short_firm_rate, npl_firm);
            bank.long_firm_rate = ardl_step(bank.long_firm_rate, npl_firm);
            bank.household_rate = ardl_step(bank.household_rate, bank.npl_consumption);
            bank.mortgage_rate = ardl_step(bank.mortgage_rate, bank.npl_mortgage);
            // A.3.3: deposit rate equals the policy rate, the household
            // overdraft rate equals the consumption-loan rate, and the firm
            // overdraft rate equals the short-term firm-loan rate.
            bank.deposit_rate = policy_rate;
            bank.household_overdraft_rate = bank.household_rate;
            bank.firm_overdraft_rate = bank.short_firm_rate;
            let positive_reserve_income = positive_part(bank.reserves) * policy_rate;
            let negative_reserve_cost = negative_abs(bank.reserves) * bank.short_firm_rate;
            bank.profit = loan_settlement.bank_interest(bank.id) + positive_reserve_income
                - negative_reserve_cost
                - bank.deposit_rate * bank.deposits;
            // A.41 in full: previous equity, plus profit net of corporate tax,
            // -- see the loan retirement immediately before this loop.
            // plus appropriated deposits, less the loans written off.
            let seized: f64 = write_offs
                .iter()
                .filter(|debt| debt.bank_id == bank.id)
                .map(|debt| debt.deposits_seized)
                .sum();
            let lost: f64 = write_offs
                .iter()
                .filter(|debt| debt.bank_id == bank.id)
                .map(|debt| debt.loans_lost)
                .sum();
            let bank_tax = account.corporate_tax_rate * positive_part(bank.profit);
            state.audit.bank_loan_interest += loan_settlement.bank_interest(bank.id);
            state.audit.bank_reserve_income += positive_reserve_income;
            state.audit.bank_reserve_cost += negative_reserve_cost;
            state.audit.bank_deposit_interest += bank.deposit_rate * bank.deposits;
            state.audit.bank_corporate_tax += bank_tax;
            state.audit.bank_writeoff_seized += seized;
            state.audit.bank_writeoff_lost += lost;
            bank.equity += bank.profit - bank_tax + seized - lost;
            bank.liabilities = bank_liabilities_a42(
                bank.equity,
                positive_firm_deposits,
                positive_household_deposits,
                previous_reserves,
            );
            bank.reserves = bank_reserves_a43(
                firm_deposits,
                household_deposits,
                bank.equity,
                bank.firm_loan_volume_by_sector.iter().sum::<f64>()
                    + bank.consumption_loan_volume
                    + bank.mortgage_volume,
            );
            // A.44, first half: detect insolvency. The bail-in itself runs
            // after the loop, since it moves equity *between* banks.
            bank.insolvent = bank.equity
                / (bank.liabilities + positive_part(bank.reserves)).max(1e-9)
                < state.params.solvency_ratio;
        }

        // A.44, second half: "then the bank is bailed-in by all other
        // non-insolvent banks cancelling a fixed fraction of their debt until
        // the equity of the insolvent bank is equal to the average equity of
        // non-insolvent banks."
        //
        // §3.5 Table 3 lists the bail-in mechanism as one of the headline
        // differences from the IIASA model, where banks cannot fail. It was
        // entirely absent: `insolvent` was computed and acted on by nothing,
        // and A.40's `I_b(t)` equity-injection term did not exist.
        let solvent_equity: Vec<f64> = banks
            .iter()
            .filter(|bank| !bank.insolvent)
            .map(|bank| bank.equity)
            .collect();
        if !solvent_equity.is_empty() && banks.iter().any(|bank| bank.insolvent) {
            let average_equity = solvent_equity.iter().sum::<f64>() / solvent_equity.len() as f64;
            let positive_total: f64 = solvent_equity.iter().map(|e| positive_part(*e)).sum();
            let injections: Vec<(u32, f64)> = banks
                .iter()
                .filter(|bank| bank.insolvent)
                .map(|bank| (bank.id, positive_part(average_equity - bank.equity)))
                .collect();
            let required: f64 = injections.iter().map(|(_, amount)| *amount).sum();
            if required > 1e-12 && positive_total > 1e-12 {
                for bank in banks.iter_mut() {
                    if bank.insolvent {
                        if let Some((_, amount)) =
                            injections.iter().find(|(id, _)| *id == bank.id)
                        {
                            bank.equity += amount;
                        }
                    } else {
                        // Each solvent bank contributes in proportion to its
                        // own equity. `I_b(t)` in A.40 is this contribution.
                        let share = positive_part(bank.equity) / positive_total;
                        bank.equity -= required * share;
                        bank.profit -= required * share;
                    }
                }
                state.audit.bank_bail_ins += injections.len() as u32;
                state.audit.bank_bail_in_amount += required;
            }
        }

        // Sector balances and, when tracing, every firm's end-of-quarter state.
        // A collapse has to show up as one sector losing what another gains, so
        // the aggregates are always recorded; the per-firm rows are opt-in
        // because they are O(firms) per quarter.
        state.audit.firm_deposits_total = firms.iter().map(|f| f.deposits).sum();
        state.audit.firm_debt_total = firms
            .iter()
            .map(|f| f.short_debt + f.long_debt + f.overdraft)
            .sum();
        state.audit.firm_equity_total = firms.iter().map(|f| f.equity).sum();
        state.audit.firms_bankrupt = firms.iter().filter(|f| f.bankrupt).count() as u32;
        state.audit.household_deposits_total = households.iter().map(|h| h.deposits).sum();
        state.audit.household_ofa_total =
            households.iter().map(|h| h.other_financial_assets).sum();
        state.audit.household_income_total = households.iter().map(|h| h.income).sum();
        state.audit.household_net_wealth_total = households.iter().map(|h| h.net_wealth).sum();
        state.audit.households_bankrupt =
            households.iter().filter(|h| h.bankrupt).count() as u32;
        state.audit.bank_equity_total = banks.iter().map(|b| b.equity).sum();
        state.audit.bank_reserves_total = banks.iter().map(|b| b.reserves).sum();
        state.audit.bank_deposits_total = banks.iter().map(|b| b.deposits).sum();
        state.audit.household_consumption_total = state.aggregates.household_consumption;
        state.audit.unemployment_benefit = account.unemployment_benefit;
        let employed_now = individuals
            .iter()
            .filter(|i| i.labour_status == LABOUR_EMPLOYED)
            .count()
            .max(1) as f64;
        state.audit.average_wage =
            firms.iter().map(|f| firm_wage_bill(f)).sum::<f64>() / employed_now;
        state.audit.firm_trace.clear();
        if state.policy.trace {
            for firm in &firms {
                state.audit.firm_trace.push(FirmProbe {
                    id: firm.id,
                    employees: firm.employees,
                    work_effort: firm.work_effort,
                    initial_work_effort: firm.initial_work_effort,
                    labour: firm.labour,
                    intermediate_constraint: min_input_constraint_a63_a64(
                        &firm.intermediate_stock,
                        &state.params.io_matrix[firm.sector as usize],
                    ),
                    capital_constraint: min_input_constraint_a63_a64(
                        &firm.capital_stock,
                        &state.params.net_fixed_assets_matrix[firm.sector as usize],
                    ),
                    target_production: firm.target_production,
                    production: firm.production,
                    price: firm.price,
                    unit_cost: firm.unit_cost,
                    demand: firm.demand,
                    excess_demand: firm.excess_demand,
                    wage: firm.wage,
                    deposits: firm.deposits,
                    debt: firm.short_debt + firm.long_debt + firm.overdraft,
                    equity: firm.equity,
                    profits: firm.profits,
                    sales_quantity: firm.sales_quantity,
                    inventory: firm.inventory,
                    target_short_loan: firm.target_short_loan,
                    granted_short_loan: firm.granted_short_loan,
                });
            }
        }

        update_government_accounts(
            &mut accounts,
            &individuals,
            &firms,
            &banks,
            &properties,
            &rows,
            &receipts,
            total_wages,
            total_profits,
            total_consumption,
            total_gov,
            policy_rate,
            state.aggregates.cpi,
        );
        if let Some(updated) = accounts.first() {
            state.audit.government_revenue = updated.revenue;
            state.audit.government_deficit = updated.deficit;
            state.audit.government_debt = updated.debt;
        }

        let aggregates = compute_aggregates(
            &state,
            &firms,
            &households,
            &banks,
            &governments,
            &accounts,
            &properties,
            &rows,
        );
        state.aggregates = MacroAggregates {
            household_consumption: total_consumption,
            government_consumption: total_gov,
            investment: total_investment,
            wage_income: total_wages,
            profit_income: total_profits,
            ..aggregates
        };
        // A.15, in nominal terms on all three legs.
        //
        // `output` was previously *real* production while `expenditure` summed
        // nominal payments, so the identity could not close even in principle;
        // and the expenditure leg omitted the whole "changes in stocks and
        // inventories" block that A.15 carries.
        let production_value: f64 = firms.iter().map(|firm| firm.price * firm.production).sum();
        let intermediate_use: f64 = firms
            .iter()
            .map(|firm| {
                let sector = firm.sector as usize;
                (0..SECTORS)
                    .map(|s| {
                        previous_prices[s] * state.params.io_matrix[sector][s] * firm.production
                    })
                    .sum::<f64>()
            })
            .sum();
        let production_tax_total: f64 = firms
            .iter()
            .map(|firm| {
                account_production_tax(&accounts, firm.sector as usize)
                    * firm.price
                    * firm.production
            })
            .sum();
        let vat_total = account.vat_rate * state.aggregates.household_consumption;
        let capital_formation_tax = account.capital_tax_rate * state.aggregates.investment;
        let export_tax_total = account.export_tax_rate * state.aggregates.exports;
        let taxes_on_products =
            production_tax_total + vat_total + capital_formation_tax + export_tax_total;
        // Firm capital purchases are gross fixed capital formation too.
        let firm_capital_formation: f64 = receipts
            .iter()
            .filter(|receipt| {
                receipt.buyer_kind == BUYER_FIRM && receipt.purpose == GOODS_CAPITAL
            })
            .map(|receipt| receipt.payment)
            .sum();
        let inventory_change: f64 = firms
            .iter()
            .map(|firm| firm.price * (firm.inventory - firm.previous_inventory))
            .sum();

        state.aggregates.gdp = GdpIdentity {
            // taxes on products + (1 - tau^PROD) * P_f Y_f - intermediate inputs
            output: taxes_on_products + production_value - production_tax_total
                - intermediate_use,
            expenditure: (1.0 + account.vat_rate) * state.aggregates.household_consumption
                + state.aggregates.government_consumption
                + (1.0 + account.export_tax_rate) * state.aggregates.exports
                - state.aggregates.imports_nominal
                + (1.0 + account.capital_tax_rate) * state.aggregates.investment
                + firm_capital_formation
                + inventory_change,
            // taxes on products + gross operating surplus + compensation of
            // employees, which collapses to the output leg by construction.
            income: taxes_on_products + production_value
                - production_tax_total
                - intermediate_use,
        };
        state.accounting = AccountingReport {
            gdp: state.aggregates.gdp,
            bank_equity: banks.iter().map(|bank| bank.equity).sum(),
            firm_equity: firms.iter().map(|firm| firm.equity).sum(),
            household_net_wealth: households
                .iter()
                .map(|household| household.net_wealth)
                .sum(),
            government_debt: accounts.iter().map(|account| account.debt).sum(),
            // Relative, not absolute: a 1e-6 tolerance reads `true` for any
            // economy of non-trivial size regardless of whether the identity
            // actually holds.
            failed_gdp_identity: !state
                .aggregates
                .gdp
                .holds(1e-6 * state.aggregates.gdp.output.abs().max(1.0)),
        };
        state
            .history
            .production
            .push(state.aggregates.production.max(1e-9));
        // A.95's AR(1) needs the realised series to fit against.
        state
            .history
            .government_consumption
            .push(state.aggregates.government_consumption.max(1e-9));
        state.history.ppi.push(state.aggregates.ppi.max(1e-9));
        state.history.cpi.push(state.aggregates.cpi.max(1e-9));
        state.history.hpi.push(state.aggregates.hpi.max(1e-9));
        state.history.rpi.push(state.aggregates.rpi.max(1e-9));
        state
            .history
            .sector_production
            .push(state.aggregates.sector_production);
        state.quarter += 1;

        write_rows(ecs, firms, |firm: &Firm| firm.id)?;
        write_rows(ecs, households, |household: &Household| household.id)?;
        write_rows(ecs, banks, |bank: &Bank| bank.id)?;
        write_rows(ecs, accounts, |account: &GovernmentAccount| account.id)?;
        state.push_phase("realised_accounting");
        set_phase_and_state(ecs, env_boundary, phases.accounting_done, state)
    })
}

fn collect_rows<T>(ecs: ECSReference<'_>) -> ECSResult<Vec<T>>
where
    T: Copy + Send + Sync + 'static,
{
    let rows = Arc::new(Mutex::new(Vec::new()));
    let rows_for_query = Arc::clone(&rows);
    let q = ecs.query()?.read::<T>()?.build()?;
    ecs.for_each::<(Read<T>,), _>(q, move |row| {
        rows_for_query.lock().unwrap().push(*row.0);
    })?;
    let out = rows.lock().unwrap().clone();
    Ok(out)
}

fn write_rows<T, F>(ecs: ECSReference<'_>, rows: Vec<T>, id: F) -> ECSResult<()>
where
    T: Copy + Send + Sync + 'static,
    F: Fn(&T) -> u32 + Copy + Send + Sync + 'static,
{
    let q = ecs.query()?.write::<T>()?.build()?;
    ecs.for_each::<(Write<T>,), _>(q, move |slot| {
        let slot_id = id(slot.0);
        if let Some(updated) = rows.iter().find(|row| id(row) == slot_id) {
            *slot.0 = *updated;
        }
    })
}

fn macro_state(
    ecs: ECSReference<'_>,
    env_boundary: abm_framework::BoundaryID,
) -> ECSResult<MacroEnvironment> {
    ecs.boundary::<EnvironmentBoundary>(env_boundary)?
        .environment()
        .get(MACRO_ENV_KEY)
        .map_err(Into::into)
}

fn set_phase_and_state(
    ecs: ECSReference<'_>,
    env_boundary: abm_framework::BoundaryID,
    key: EnvKey<u64>,
    state: MacroEnvironment,
) -> ECSResult<()> {
    let env = ecs.boundary::<EnvironmentBoundary>(env_boundary)?;
    env.environment().set(MACRO_ENV_KEY, state)?;
    let current = env.environment().get::<u64>(key.name()).unwrap_or(0);
    env.environment().set(key.name(), current + 1)?;
    Ok(())
}

fn compute_aggregates(
    state: &MacroEnvironment,
    firms: &[Firm],
    households: &[Household],
    banks: &[Bank],
    governments: &[GovernmentEntity],
    _accounts: &[GovernmentAccount],
    properties: &[Property],
    rows: &[RestOfWorld],
) -> MacroAggregates {
    let mut aggregate = MacroAggregates::default();
    aggregate.production = firms.iter().map(|firm| firm.production).sum();
    for firm in firms {
        aggregate.sector_production[firm.sector as usize] += firm.production;
        aggregate.firm_loans_by_sector[firm.sector as usize] += firm.short_debt + firm.long_debt;
        if firm.bankrupt {
            aggregate.firm_npl_by_sector[firm.sector as usize] += firm.short_debt + firm.long_debt;
        }
    }
    let firm_quantity = firms
        .iter()
        .map(|firm| firm.production + firm.previous_inventory)
        .sum::<f64>();
    let firm_value = firms
        .iter()
        .map(|firm| firm.price * (firm.production + firm.previous_inventory))
        .sum::<f64>();
    aggregate.imports_nominal = rows
        .iter()
        .flat_map(|row| row.import_nominal_by_sector)
        .sum::<f64>();
    aggregate.imports_real = rows
        .iter()
        .flat_map(|row| row.import_real_by_sector)
        .sum::<f64>();
    aggregate.ppi = ppi_a3(
        firm_value,
        aggregate.imports_nominal,
        firm_quantity,
        aggregate.imports_real,
    );
    let mut sector_prices = [1.0; SECTORS];
    for (sector, slot) in sector_prices.iter_mut().enumerate() {
        let firm_quantity = firms
            .iter()
            .filter(|firm| firm.sector as usize == sector)
            .map(|firm| firm.production + firm.previous_inventory)
            .sum::<f64>();
        let firm_value = firms
            .iter()
            .filter(|firm| firm.sector as usize == sector)
            .map(|firm| firm.price * (firm.production + firm.previous_inventory))
            .sum::<f64>();
        let imports_nominal = rows
            .iter()
            .map(|row| row.import_nominal_by_sector[sector])
            .sum::<f64>();
        let imports_real = rows
            .iter()
            .map(|row| row.import_real_by_sector[sector])
            .sum::<f64>();
        *slot = sector_price_a5(firm_value, imports_nominal, firm_quantity, imports_real);
    }
    aggregate.cpi = state
        .params
        .cpi_weights
        .iter()
        .zip(sector_prices.iter())
        .map(|(weight, price)| weight * price)
        .sum::<f64>()
        .max(1e-9);
    aggregate.hpi = ratio(
        properties.iter().map(|property| property.value).sum(),
        properties
            .iter()
            .map(|property| property.initial_value)
            .sum(),
    )
    .max(1e-9);
    aggregate.rpi = ratio(
        properties.iter().map(|property| property.rent).sum(),
        properties
            .iter()
            .map(|property| property.initial_rent)
            .sum(),
    )
    .max(1e-9);
    aggregate.consumption_loans = households.iter().map(|h| h.consumption_debt).sum();
    aggregate.mortgages = households.iter().map(|h| h.mortgage_debt).sum();
    aggregate.total_loans = banks
        .iter()
        .map(|bank| {
            bank.firm_loan_volume_by_sector.iter().sum::<f64>()
                + bank.consumption_loan_volume
                + bank.mortgage_volume
        })
        .sum();
    let bad_consumption = households
        .iter()
        .filter(|household| household.bankrupt)
        .map(|household| household.consumption_debt)
        .sum::<f64>();
    let bad_mortgage = households
        .iter()
        .filter(|household| household.bankrupt)
        .map(|household| household.mortgage_debt)
        .sum::<f64>();
    aggregate.consumption_npl = ratio(bad_consumption, aggregate.consumption_loans);
    aggregate.mortgage_npl = ratio(bad_mortgage, aggregate.mortgages);
    for s in 0..SECTORS {
        aggregate.firm_npl_by_sector[s] = ratio(
            aggregate.firm_npl_by_sector[s],
            aggregate.firm_loans_by_sector[s],
        );
    }
    aggregate.exports = rows.iter().map(|row| row.exports).sum();
    aggregate.government_consumption = governments.iter().map(|gov| gov.realised_consumption).sum();
    aggregate.wage_income = firms.iter().map(|firm| firm_wage_bill(firm)).sum();
    aggregate.profit_income = firms.iter().map(|firm| firm.profits).sum();
    aggregate.gdp = GdpIdentity {
        output: aggregate.production,
        expenditure: aggregate.household_consumption
            + aggregate.government_consumption
            + aggregate.investment
            + aggregate.exports
            - aggregate.imports_nominal,
        income: aggregate.wage_income + aggregate.profit_income,
    };
    aggregate
}

fn closest_property(properties: &[Property], status: u8, desired: f64) -> Option<usize> {
    properties
        .iter()
        .enumerate()
        .filter(|(_, property)| property.market_status == status)
        .min_by(|(_, a), (_, b)| {
            let a_price = if status == PROPERTY_FOR_RENT {
                a.rent
            } else {
                a.price
            };
            let b_price = if status == PROPERTY_FOR_RENT {
                b.rent
            } else {
                b.price
            };
            (a_price - desired)
                .abs()
                .total_cmp(&(b_price - desired).abs())
        })
        .map(|(idx, _)| idx)
}

fn lagged_cpi_inflation(state: &MacroEnvironment) -> f64 {
    let lag = state.params.rent_partial_indexation_lag.max(1);
    if state.history.cpi.len() <= lag {
        return state.forecast.predicted_cpi_inflation;
    }
    let current_idx = state.history.cpi.len() - lag;
    let previous_idx = current_idx.saturating_sub(1);
    log_growth(
        state.history.cpi[current_idx],
        state.history.cpi[previous_idx],
    )
}

/// Orders goods demands per A.10.2: firms first, then the remaining buyer
/// types, with each block internally randomised.
///
/// The buyer-type order after firms is fixed (households, government, rest of
/// world) rather than randomised, so a run is reproducible; A.10.2 only
/// prescribes precedence for firms.
fn order_demands_firms_first(demands: Vec<GoodsDemand>, rng: &mut MacroRng) -> Vec<GoodsDemand> {
    let mut ordered = Vec::with_capacity(demands.len());
    for kind in [BUYER_FIRM, BUYER_HOUSEHOLD, BUYER_GOVERNMENT, BUYER_ROW] {
        let mut block: Vec<GoodsDemand> = demands
            .iter()
            .copied()
            .filter(|demand| demand.buyer_kind == kind)
            .collect();
        rng.shuffle(&mut block);
        ordered.append(&mut block);
    }
    // Anything with an unrecognised buyer kind still has to clear.
    let mut rest: Vec<GoodsDemand> = demands
        .into_iter()
        .filter(|demand| {
            !matches!(
                demand.buyer_kind,
                BUYER_FIRM | BUYER_HOUSEHOLD | BUYER_GOVERNMENT | BUYER_ROW
            )
        })
        .collect();
    rng.shuffle(&mut rest);
    ordered.append(&mut rest);
    ordered
}

/// `w_f(t)`: the firm's **total** wage bill.
///
/// A.50, A.77 and A.89 all treat `w_f` as a total, while `firm.wage` holds the
/// per-worker rate that A.69/A.71 set. Multiplying the rate by `firm.labour`
/// was only ever right while `labour` meant headcount; now that it carries
/// `H_f` in output units (A.65) that product overstates the bill by a factor of
/// `h_f` -- roughly 34x here, which emptied firm deposits in one quarter and
/// left them unable to buy inputs.
/// Interest due, instalment due, and the firm's short-term loan rate, read off
/// the loan book. A.80 needs all three, and the planning system does not hold
/// the bank rows.
fn firm_loan_obligations(state: &MacroEnvironment, firm_id: u32) -> (f64, f64, f64) {
    let mut interest = 0.0;
    let mut instalment = 0.0;
    let mut short_rate = 0.0;
    for loan in state
        .loan_book
        .loans
        .iter()
        .filter(|loan| loan.borrower_kind == BUYER_FIRM && loan.borrower_id == firm_id)
    {
        interest += loan.outstanding * loan.rate;
        instalment += loan.outstanding / loan.maturity_remaining_quarters.max(1) as f64;
        if loan.loan_class == LOAN_FIRM_SHORT {
            short_rate = loan.rate;
        }
    }
    (interest, instalment, short_rate)
}

/// Log growth of the last two recorded observations of a series.
fn last_two_growth(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }
    log_growth(series[series.len() - 1], series[series.len() - 2])
}

fn firm_wage_bill(firm: &Firm) -> f64 {
    firm.wage * firm.employees as f64
}

/// The wage bill implied by the firm's target labour input.
///
/// `target_labour` is `Ĥ_f = Ŷ_f` in output units (A.68), so the headcount it
/// implies is `Ĥ_f / h_f`.
fn target_wage_bill(firm: &Firm) -> f64 {
    let headcount = firm.target_labour / firm.work_effort.max(1e-9);
    firm.wage * headcount
}

fn seller_priority_weights(firms: &[Firm], seller_indices: &[usize], phi_gm: f64) -> Vec<f64> {
    let price_sum = seller_indices
        .iter()
        .map(|idx| (-phi_gm * firms[*idx].price).exp())
        .sum::<f64>()
        .max(1e-9);
    let production_sum = seller_indices
        .iter()
        .map(|idx| firms[*idx].production.max(0.0))
        .sum::<f64>()
        .max(1e-9);
    // A.140 ranks sellers by the *product* of relative price and relative
    // production, not their average. Averaging lets a large expensive firm rank
    // alongside a small cheap one, because a high score on either term alone
    // carries half the weight; the product requires both.
    let weights: Vec<f64> = seller_indices
        .iter()
        .map(|idx| {
            let relative_price = (-phi_gm * firms[*idx].price).exp() / price_sum;
            let relative_production = firms[*idx].production.max(0.0) / production_sum;
            (relative_price * relative_production).max(0.0)
        })
        .collect();
    // A firm with zero output would otherwise be unreachable even when it holds
    // sellable inventory, since the product term vanishes. Fall back to the
    // price term alone when every product is degenerate.
    if weights.iter().sum::<f64>() <= 1e-12 {
        return seller_indices
            .iter()
            .map(|idx| ((-phi_gm * firms[*idx].price).exp() / price_sum).max(0.0))
            .collect();
    }
    weights
}

fn weighted_choice(rng: &mut MacroRng, weights: &[f64]) -> usize {
    let total = weights.iter().sum::<f64>();
    if weights.is_empty() || total <= 1e-12 || !total.is_finite() {
        return 0;
    }
    let mut draw = rng.unit_f64() * total;
    for (idx, weight) in weights.iter().enumerate() {
        draw -= weight.max(0.0);
        if draw <= 0.0 {
            return idx;
        }
    }
    weights.len() - 1
}

fn distribute_excess_demand(firms: &mut [Firm], sector: usize, excess: f64, phi_gm: f64) {
    let seller_indices: Vec<usize> = firms
        .iter()
        .enumerate()
        .filter(|(_, firm)| firm.sector as usize == sector)
        .map(|(idx, _)| idx)
        .collect();
    let weights = seller_priority_weights(firms, &seller_indices, phi_gm);
    let total = weights.iter().sum::<f64>();
    if total <= 1e-12 {
        if let Some(firm) = firms.iter_mut().find(|firm| firm.sector as usize == sector) {
            firm.excess_demand += excess;
        }
        return;
    }
    for (idx, weight) in seller_indices.iter().zip(weights.iter()) {
        firms[*idx].excess_demand += excess * weight / total;
    }
}

fn previous_sector_prices(firms: &[Firm]) -> [f64; SECTORS] {
    let mut prices = [1.0; SECTORS];
    for (sector, slot) in prices.iter_mut().enumerate() {
        let mut value = 0.0;
        let mut weight = 0.0;
        for firm in firms.iter().filter(|firm| firm.sector as usize == sector) {
            let output = (firm.previous_production + firm.previous_inventory).max(0.0);
            value += firm.previous_price * output;
            weight += output;
        }
        if weight > 1e-9 {
            *slot = value / weight;
        }
    }
    prices
}

fn loan_maturity_quarters(state: &MacroEnvironment, loan_class: u8) -> u32 {
    match loan_class {
        LOAN_FIRM_SHORT => state.params.firm_short_maturity_quarters,
        LOAN_FIRM_LONG => state.params.firm_long_maturity_quarters,
        LOAN_HOUSEHOLD_CONSUMPTION => state.params.consumption_loan_maturity_quarters,
        LOAN_MORTGAGE => state.params.mortgage_maturity_quarters,
        _ => 1,
    }
}

fn account_production_tax(accounts: &[GovernmentAccount], sector: usize) -> f64 {
    accounts
        .first()
        .map(|account| account.production_tax_by_sector[sector])
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn update_government_accounts(
    accounts: &mut [GovernmentAccount],
    individuals: &[Individual],
    firms: &[Firm],
    banks: &[Bank],
    properties: &[Property],
    rows: &[RestOfWorld],
    receipts: &[GoodsReceipt],
    total_wages: f64,
    total_profits: f64,
    total_consumption: f64,
    total_gov: f64,
    policy_rate: f64,
    cpi: f64,
) {
    for account in accounts {
        // A.98's first term, printed as
        //   P^CPI(t) * (tau^SIW + tau^INC(1 - tau^SIW)) * sum_i w_i(t)
        //
        // ERRATUM. `tau^SIF` belongs in that bracket. The prose directly above
        // the equation -- identical in the paper (A.98) and the thesis (6.99) --
        // reads "The government collects social contributions (tau^SIF
        // employers', tau^SIW employees')"; Table A.12 / 6.10 lists tau^SIF
        // under Revenue; and footnote 66 / 63 sources it from OECD code D611,
        // "employers' contribution to social insurance". Only the equation
        // omits it. Poledna A.59 collects both legs.
        //
        // A.50 / 6.51 corroborates the cost side: `w_f(0)` is set from "the
        // initial total labour compensation of sector s obtained from
        // socio-economic accounts", and labour compensation (ESA D.1) is wages
        // and salaries plus employers' social contributions. So the firm is
        // already paying it inside `w_f`; without this term it is paid by
        // nobody and received by nobody.
        let employer_social_insurance = cpi * total_wages * account.social_insurance_firm_rate;
        let worker_social_insurance = cpi * total_wages * account.social_insurance_worker_rate;
        let labour_income_tax = cpi
            * total_wages
            * (1.0 - account.social_insurance_worker_rate)
            * account.income_tax_rate;
        let rental_income_tax = properties
            .iter()
            .filter(|property| property.owner_household_id != property.occupant_household_id)
            .map(|property| property.rent / 4.0)
            .sum::<f64>()
            * account.income_tax_rate;
        let vat = total_consumption * account.vat_rate;
        let household_capital_formation = receipts
            .iter()
            .filter(|receipt| {
                receipt.buyer_kind == BUYER_HOUSEHOLD && receipt.purpose == GOODS_CAPITAL
            })
            .map(|receipt| receipt.payment)
            .sum::<f64>()
            * account.capital_tax_rate;
        let production_tax = firms
            .iter()
            .map(|firm| {
                account.production_tax_by_sector[firm.sector as usize]
                    * firm.price
                    * firm.production
            })
            .sum::<f64>();
        let bank_positive_profits = banks
            .iter()
            .map(|bank| positive_part(bank.profit))
            .sum::<f64>();
        let corporate_tax =
            (total_profits.max(0.0) + bank_positive_profits) * account.corporate_tax_rate;
        let export_tax = rows.iter().map(|row| row.exports).sum::<f64>() * account.export_tax_rate;
        account.revenue = employer_social_insurance
            + worker_social_insurance
            + labour_income_tax
            + rental_income_tax
            + vat
            + household_capital_formation
            + production_tax
            + corporate_tax
            + export_tax;
        let unemployed = individuals
            .iter()
            .filter(|individual| individual.labour_status == LABOUR_UNEMPLOYED)
            .count() as f64;
        account.deficit = cpi
            * (account.other_benefits + unemployed * account.unemployment_benefit)
            + total_gov
            + policy_rate * account.debt
            - account.revenue;
        account.debt += account.deficit;
    }
}

#[derive(Clone, Debug, Default)]
struct LoanSettlementSummary {
    bank_interest: Vec<(u32, f64)>,
    firm_interest: Vec<(u32, f64)>,
    household_interest: Vec<(u32, f64)>,
}

impl LoanSettlementSummary {
    fn add_bank_interest(&mut self, bank_id: u32, amount: f64) {
        add_amount(&mut self.bank_interest, bank_id, amount);
    }

    fn add_firm_interest(&mut self, firm_id: u32, amount: f64) {
        add_amount(&mut self.firm_interest, firm_id, amount);
    }

    fn add_household_interest(&mut self, household_id: u32, amount: f64) {
        add_amount(&mut self.household_interest, household_id, amount);
    }

    fn bank_interest(&self, bank_id: u32) -> f64 {
        lookup_amount(&self.bank_interest, bank_id)
    }

    fn firm_interest(&self, firm_id: u32) -> f64 {
        lookup_amount(&self.firm_interest, firm_id)
    }

    #[allow(dead_code)]
    fn household_interest(&self, household_id: u32) -> f64 {
        lookup_amount(&self.household_interest, household_id)
    }
}

fn settle_loan_book(
    quarter: u64,
    state: &mut MacroEnvironment,
    banks: &mut [Bank],
    firms: &mut [Firm],
    households: &mut [Household],
) -> LoanSettlementSummary {
    let mut settlement = LoanSettlementSummary::default();
    for loan in &mut state.loan_book.loans {
        if loan.origin_quarter >= quarter || loan.outstanding <= 1e-9 {
            continue;
        }
        let remaining = loan.maturity_remaining_quarters.max(1) as f64;
        let principal_due = (loan.outstanding / remaining).min(loan.outstanding);
        let interest_due = loan.outstanding * loan.rate;
        match loan.borrower_kind {
            BUYER_FIRM => {
                if let Some(firm) = firms.iter_mut().find(|firm| firm.id == loan.borrower_id) {
                    firm.deposits -= principal_due + interest_due;
                    match loan.loan_class {
                        LOAN_FIRM_SHORT => {
                            firm.short_debt = positive_part(firm.short_debt - principal_due)
                        }
                        LOAN_FIRM_LONG => {
                            firm.long_debt = positive_part(firm.long_debt - principal_due)
                        }
                        _ => {}
                    }
                    settlement.add_firm_interest(firm.id, interest_due);
                }
            }
            BUYER_HOUSEHOLD => {
                if let Some(household) = households
                    .iter_mut()
                    .find(|household| household.id == loan.borrower_id)
                {
                    household.deposits -= principal_due + interest_due;
                    match loan.loan_class {
                        LOAN_HOUSEHOLD_CONSUMPTION => {
                            household.consumption_debt =
                                positive_part(household.consumption_debt - principal_due)
                        }
                        LOAN_MORTGAGE => {
                            household.mortgage_debt =
                                positive_part(household.mortgage_debt - principal_due)
                        }
                        _ => {}
                    }
                    settlement.add_household_interest(household.id, interest_due);
                }
            }
            _ => {}
        }
        if let Some(bank) = banks.iter_mut().find(|bank| bank.id == loan.bank_id) {
            match loan.loan_class {
                LOAN_FIRM_SHORT | LOAN_FIRM_LONG => {
                    bank.firm_loan_volume_by_sector[loan.sector as usize] = positive_part(
                        bank.firm_loan_volume_by_sector[loan.sector as usize] - principal_due,
                    )
                }
                LOAN_HOUSEHOLD_CONSUMPTION => {
                    bank.consumption_loan_volume =
                        positive_part(bank.consumption_loan_volume - principal_due)
                }
                LOAN_MORTGAGE => {
                    bank.mortgage_volume = positive_part(bank.mortgage_volume - principal_due)
                }
                _ => {}
            }
            settlement.add_bank_interest(bank.id, interest_due);
        }
        loan.outstanding = positive_part(loan.outstanding - principal_due);
        loan.maturity_remaining_quarters = loan.maturity_remaining_quarters.saturating_sub(1);
    }
    state
        .loan_book
        .loans
        .retain(|loan| loan.outstanding > 1e-9 && loan.maturity_remaining_quarters > 0);
    settlement
}

fn add_amount(items: &mut Vec<(u32, f64)>, id: u32, amount: f64) {
    if amount == 0.0 {
        return;
    }
    if let Some((_, value)) = items.iter_mut().find(|(item_id, _)| *item_id == id) {
        *value += amount;
    } else {
        items.push((id, amount));
    }
}

fn lookup_amount(items: &[(u32, f64)], id: u32) -> f64 {
    items
        .iter()
        .find(|(item_id, _)| *item_id == id)
        .map(|(_, value)| *value)
        .unwrap_or_default()
}

fn offered_rate(bank: &Bank, loan_class: u8) -> f64 {
    match loan_class {
        LOAN_FIRM_SHORT => bank.short_firm_rate,
        LOAN_FIRM_LONG => bank.long_firm_rate,
        LOAN_HOUSEHOLD_CONSUMPTION => bank.household_rate,
        LOAN_MORTGAGE => bank.mortgage_rate,
        _ => bank.household_rate,
    }
}

/// A.32: the bank's total credit envelope, `E_b(t-1)/rho^CAR - sum_l V_l`.
///
/// A.33-A.36 then *allocate* this envelope across firm, consumption and
/// mortgage lending by lagged non-performing-loan ratios; A.36 does not replace
/// A.32, it distributes it. The `credit_supply_max` clamp that used to sit here
/// is not in the paper and is now left at infinity by the generator.
fn bank_credit_supply(bank: &Bank, state: &MacroEnvironment) -> f64 {
    positive_part(
        bank.equity / state.params.car.max(1e-9)
            - bank.firm_loan_volume_by_sector.iter().sum::<f64>()
            - bank.consumption_loan_volume
            - bank.mortgage_volume,
    )
    .min(bank.credit_supply_max.max(0.0))
}

/// A.33-A.35 weights, normalised to A.32's envelope by A.36.
///
/// `V_hat ~ V(0) * exp(-phi^CS * nu(t-1))` for each class, so a class whose
/// loans are going bad is allocated a smaller share of the same total. `phi^CS`
/// is 2.0 (A.3.2) and was previously read by nothing at all -- the NPL ratios
/// were computed every quarter and discarded.
fn bank_class_credit_supply(
    bank: &Bank,
    state: &MacroEnvironment,
    loan_class: u8,
    sector: u8,
) -> f64 {
    let phi = state.params.credit_supply_phi;
    let firm_weight: f64 = (0..SECTORS)
        .map(|s| {
            bank.firm_loan_volume_by_sector[s].max(0.0) * (-phi * bank.npl_firm_by_sector[s]).exp()
        })
        .sum();
    let consumption_weight = bank.consumption_loan_volume.max(0.0) * (-phi * bank.npl_consumption).exp();
    let mortgage_weight = bank.mortgage_volume.max(0.0) * (-phi * bank.npl_mortgage).exp();
    let total = firm_weight + consumption_weight + mortgage_weight;
    let envelope = bank_credit_supply(bank, state);
    if total <= 1e-12 {
        // No loan book to weight against yet: the whole envelope is available.
        return envelope;
    }
    let share = match loan_class {
        LOAN_FIRM_SHORT | LOAN_FIRM_LONG => {
            let s = sector as usize;
            bank.firm_loan_volume_by_sector[s].max(0.0)
                * (-phi * bank.npl_firm_by_sector[s]).exp()
        }
        LOAN_HOUSEHOLD_CONSUMPTION => consumption_weight,
        LOAN_MORTGAGE => mortgage_weight,
        _ => 0.0,
    };
    envelope * share / total
}

fn borrower_credit_cap(
    state: &MacroEnvironment,
    firms: &[Firm],
    households: &[Household],
    app: CreditApplication,
    rate: f64,
) -> f64 {
    match app.loan_class {
        LOAN_FIRM_SHORT | LOAN_FIRM_LONG => firms
            .iter()
            .find(|firm| firm.id == app.borrower_id)
            .map(|firm| {
                // `sum_s P_s(t) K_fs(t)` -- the *value* of the capital stock.
                // Summing the bare quantities understated it by the whole price
                // level, and at the corrected `k_{s's}` magnitudes that is the
                // difference between a firm being bankable and not.
                let sector_prices = previous_sector_prices(firms);
                let capital_value: f64 = (0..SECTORS)
                    .map(|s| sector_prices[s] * firm.capital_stock[s])
                    .sum();
                let debt = firm.short_debt + firm.long_debt + firm.overdraft;
                let overdraft = negative_abs(firm.deposits);
                // A.3.3 sets the firm overdraft rate equal to the short-term
                // firm loan rate, which is the rate on offer here. Interest due
                // is approximated as `rate * debt` rather than summed per loan,
                // because the per-loan rates live in the loan book and this
                // screen runs before it is consulted.
                let interest = rate * debt;
                // A.25, debt-to-equity:
                //   V_l <= rho^DtE * sum_s P_s K_fs - L_f(t-1) + [D_f(t-1)]^-
                //          + r^F-O [D_f(t-1)]^- - sum_l r_l V_l
                // The `+ firm.equity` term the old code added is not in A.25 --
                // it double-counted the capital stock, since A.93 equity
                // already contains it.
                let dte_cap = positive_part(
                    state.params.debt_to_equity * capital_value - debt + overdraft
                        + rate * overdraft
                        - interest,
                );
                // A.26, return-on-equity:
                //   V_l <= sum_s P_s K_fs + D_f(t-1) - L_f(t-1) - Pi_f/rho^RoE
                // The old form was `Pi_f/rho^RoE - debt`: the profit term's sign
                // was inverted and the capital stock and deposits were missing
                // entirely, so a *more* profitable firm was granted *less*
                // credit and a firm with no predicted profit got the largest
                // cap of all.
                let roe_cap = positive_part(
                    capital_value + firm.deposits
                        - debt
                        - firm.predicted_profits / state.params.return_on_equity.max(1e-9),
                );
                // A.27.
                let roa_ok = ratio(firm.predicted_profits, debt + firm.equity)
                    >= state.params.return_on_assets;
                if roa_ok {
                    dte_cap.min(roe_cap)
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0),
        LOAN_HOUSEHOLD_CONSUMPTION => households
            .iter()
            .find(|household| household.id == app.borrower_id)
            .map(|household| {
                let six_month_income = household.income_history.iter().sum::<f64>()
                    / household.income_history.len() as f64;
                positive_part(
                    state.params.consumption_lti * six_month_income
                        - household.consumption_debt
                        - household.mortgage_debt,
                )
            })
            .unwrap_or(0.0),
        LOAN_MORTGAGE => households
            .iter()
            .find(|household| household.id == app.borrower_id)
            .map(|household| {
                let six_month_income = household.income_history.iter().sum::<f64>()
                    / household.income_history.len() as f64;
                let ltv = state.params.mortgage_ltv / (1.0 - state.params.mortgage_ltv).max(1e-9)
                    * (household.deposits + household.other_financial_assets).max(0.0);
                // A.30's income base is *annual*. Baptista et al. (2016) Eq.
                // (13) -- the source Wiese cites for the housing block -- reads
                // `q <= Phi_i y` with y the household's gross **annual**
                // income, and rho^LTI-M = 4.5 is the ESRB annual multiple.
                // A.30 as printed averages two quarterly incomes, which is a
                // quarterly rate: 4x too small. Measured at the quarterly
                // reading, A.30 bound 1431 of 1449 applications and the housing
                // market completed 3 sales in 40 quarters.
                let lti = state.params.mortgage_lti * QUARTERS_PER_YEAR * six_month_income
                    - household.consumption_debt
                    - household.mortgage_debt;
                // A.31's annuity is in the loan's own periods, and `r_l` here is
                // a quarterly rate, so the exponent must be the maturity in
                // *quarters*. Dividing by 4 mixed a quarterly rate with a
                // 25-unit exponent and understated the cap by an order of
                // magnitude.
                let maturity = state.params.mortgage_maturity_quarters as f64;
                // A.31 uses the same two-quarter average income as A.28 and
                // A.30, not the current quarter's prediction.
                let dsti_payment = state.params.mortgage_dsti * six_month_income;
                let dsti = if rate <= 1e-9 {
                    dsti_payment * maturity
                } else {
                    dsti_payment * (1.0 - (1.0 + rate).powf(-maturity)) / rate
                };
                (ltv, lti, dsti)
            })
            .map(|(ltv, lti, dsti)| positive_part(ltv.min(lti).min(dsti)))
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Diagnostic twin of the A.29/A.30/A.31 caps: returns the three bounds
/// separately so the binding one can be attributed.
fn mortgage_caps(
    household: &Household,
    params: &super::state::CountryParameters,
    rate: f64,
) -> (f64, f64, f64) {
    let six_month_income =
        household.income_history.iter().sum::<f64>() / household.income_history.len() as f64;
    let ltv = params.mortgage_ltv / (1.0f64 - params.mortgage_ltv).max(1e-9)
        * (household.deposits + household.other_financial_assets).max(0.0);
    let lti = params.mortgage_lti * QUARTERS_PER_YEAR * six_month_income
        - household.consumption_debt
        - household.mortgage_debt;
    let maturity = params.mortgage_maturity_quarters as f64;
    let dsti_payment = params.mortgage_dsti * six_month_income;
    let dsti = if rate <= 1e-9 {
        dsti_payment * maturity
    } else {
        dsti_payment * (1.0 - (1.0 + rate).powf(-maturity)) / rate
    };
    (ltv, lti, dsti)
}

fn apply_loan(
    bank: &mut Bank,
    firms: &mut [Firm],
    households: &mut [Household],
    app: CreditApplication,
    amount: f64,
) {
    bank.reserves -= amount;
    bank.deposits += amount;
    match app.loan_class {
        LOAN_FIRM_SHORT => {
            bank.firm_loan_volume_by_sector[app.sector as usize] += amount;
            if let Some(firm) = firms.iter_mut().find(|firm| firm.id == app.borrower_id) {
                firm.short_debt += amount;
                firm.deposits += amount;
                firm.granted_short_loan += amount;
            }
        }
        LOAN_FIRM_LONG => {
            bank.firm_loan_volume_by_sector[app.sector as usize] += amount;
            if let Some(firm) = firms.iter_mut().find(|firm| firm.id == app.borrower_id) {
                firm.long_debt += amount;
                firm.deposits += amount;
                firm.granted_long_loan += amount;
            }
        }
        LOAN_HOUSEHOLD_CONSUMPTION => {
            bank.consumption_loan_volume += amount;
            if let Some(household) = households
                .iter_mut()
                .find(|household| household.id == app.borrower_id)
            {
                household.consumption_debt += amount;
                household.deposits += amount;
                household.granted_consumption_loan += amount;
            }
        }
        LOAN_MORTGAGE => {
            bank.mortgage_volume += amount;
            if let Some(household) = households
                .iter_mut()
                .find(|household| household.id == app.borrower_id)
            {
                household.mortgage_debt += amount;
                household.deposits += amount;
                household.granted_mortgage += amount;
            }
        }
        _ => {}
    }
}

fn apply_buyer_goods(
    firms: &mut [Firm],
    households: &mut [Household],
    governments: &mut [GovernmentEntity],
    rows: &mut [RestOfWorld],
    demand: GoodsDemand,
    quantity: f64,
    payment: f64,
) {
    match demand.buyer_kind {
        // Purchases move real goods here but not cash. A.91 and A.123 settle
        // deposits once per quarter against the full flow -- revenue less costs
        // less taxes for firms, income less consumption for households -- so
        // debiting per transaction as well double-counted every purchase.
        BUYER_FIRM => {
            if let Some(firm) = firms.iter_mut().find(|firm| firm.id == demand.buyer_id) {
                match demand.purpose {
                    GOODS_CAPITAL => firm.realised_capital[demand.sector as usize] += quantity,
                    _ => firm.realised_intermediate[demand.sector as usize] += quantity,
                }
            }
        }
        BUYER_HOUSEHOLD => {
            let _ = &mut *households;
        }
        BUYER_GOVERNMENT => {
            if let Some(government) = governments
                .iter_mut()
                .find(|government| government.id == demand.buyer_id)
            {
                government.realised_consumption += payment;
            }
        }
        BUYER_ROW => {
            if let Some(row) = rows.iter_mut().find(|row| row.id == demand.buyer_id) {
                row.exports += payment;
            }
        }
        _ => {}
    }
}
