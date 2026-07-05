use std::sync::{Arc, Mutex};

use abm_framework::environment::{EnvKey, EnvironmentBoundary};
use abm_framework::messaging::{Capacity, MessageBufferSet, MessageHandle};
use abm_framework::model::ModelBuilder;
use abm_framework::{AccessSets, ComponentID, ECSReference, ECSResult, FnSystem, Read, Write};

use super::accounting::{negative_abs, positive_part, AccountingReport, GdpIdentity};
use super::components::*;
use super::equations::{
    bank_liabilities_a42, bank_reserves_a43, buy_probability_a110,
    constrained_goods_target_a83_a84, firm_predicted_profit_a61, firm_target_demand_a60,
    firm_target_production_a62, idiosyncratic_growth_a59,
    literal_price_or_rent_reduction_a113_a115, log_growth, min_input_constraint_a63_a64, ppi_a3,
    price_a73, purchase_cost_a109_literal_pdf, ratio, rent_cost_a108, sector_price_a5,
    target_capital_a79, target_intermediate_a78, work_effort_a66_a67,
};
use super::forecasting::fit_ar1_log_level_forecast;
use super::messages::*;
use super::state::{HousingReductionPolicy, MacroAggregates, MacroEnvironment, MACRO_ENV_KEY};

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

        for firm in &mut firms {
            let sector = firm.sector as usize;
            let gamma_applies = firm.excess_demand.abs() > 1e-9;
            let gamma_f = idiosyncratic_growth_a59(
                firm.previous_demand.max(1e-9),
                firm.previous_production.max(1e-9),
                firm.inventory_two_periods_ago.max(0.0),
                gamma_applies,
            );
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
            let labour_constraint = firm.labour.max(0.0);
            let intermediate_constraint = min_input_constraint_a63_a64(
                &firm.intermediate_stock,
                &state.params.io_matrix[sector],
            );
            let capital_constraint = min_input_constraint_a63_a64(
                &firm.capital_stock,
                &state.params.net_fixed_assets_matrix[sector],
            );
            firm.work_effort = work_effort_a66_a67(
                state.params.work_effort_max,
                firm.initial_work_effort,
                firm.labour.max(0.0),
                intermediate_constraint,
                capital_constraint,
            );
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
            firm.wage *= (1.0 + state.forecast.predicted_ppi_inflation).max(0.0);
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
            state.shuffle(&mut employees);
            while firm.labour > firm.target_labour && !employees.is_empty() {
                let idx = employees.pop().unwrap();
                if firm.labour - individuals[idx].labour_input < firm.target_labour {
                    break;
                }
                firm.labour -= individuals[idx].labour_input;
                firm.employees = firm.employees.saturating_sub(1);
                individuals[idx].labour_status = LABOUR_UNEMPLOYED;
                individuals[idx].employer_firm_id = NOT_LINKED;
                individuals[idx].wage = 0.0;
            }
        }

        let mut firm_order: Vec<usize> = (0..firms.len()).collect();
        state.shuffle(&mut firm_order);
        let mut job_seekers: Vec<usize> = individuals
            .iter()
            .enumerate()
            .filter(|(_, worker)| worker.labour_status == LABOUR_UNEMPLOYED)
            .map(|(idx, _)| idx)
            .collect();
        state.shuffle(&mut job_seekers);

        for firm_idx in firm_order {
            let firm = &mut firms[firm_idx];
            let needed = (firm.target_labour - firm.labour).ceil().max(0.0) as u32;
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
                    firm.labour += individuals[worker_idx].labour_input;
                    firm.employees += 1;
                    hired += 1;
                } else {
                    remaining_seekers.push(worker_idx);
                }
            }
            job_seekers = remaining_seekers;
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
            central_bank.predicted_policy_rate = positive_part(
                central_bank.rho * central_bank.policy_rate
                    + (1.0 - central_bank.rho)
                        * (central_bank.natural_rate
                            + central_bank.inflation_target
                            + central_bank.xi_pi
                                * (state.forecast.predicted_cpi_inflation
                                    - central_bank.inflation_target)
                            + central_bank.xi_gamma * state.forecast.predicted_growth),
            );
            central_bank.policy_rate = central_bank.predicted_policy_rate;
        }

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

        for firm in &mut firms {
            let sector = firm.sector as usize;
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
            let demand_pull = idiosyncratic_growth_a59(
                firm.previous_demand,
                firm.previous_production,
                firm.inventory_two_periods_ago,
                firm.excess_demand.abs() > 1e-9,
            )
            .max(0.0);
            let cost_push = (ratio(firm.unit_cost, firm.previous_price) - 1.0).max(0.0);
            firm.price = price_a73(
                firm.previous_price,
                state.forecast.predicted_ppi_inflation,
                state.calibration.phi_dp,
                demand_pull,
                state.calibration.phi_cp,
                cost_push,
            )
            .max(0.01);
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
            let short_need =
                firm.target_intermediate.iter().sum::<f64>() + firm.wage * firm.target_labour;
            let available_after_short = positive_part(firm.deposits - short_need);
            let long_need = firm.target_capital.iter().sum::<f64>();
            firm.target_short_loan = positive_part(short_need - firm.deposits);
            firm.target_long_loan = positive_part(long_need - available_after_short);
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

        for government in &mut governments {
            government.target_consumption = state.params.government_consumption_weights
                [government.sector as usize]
                * (1.0 + state.forecast.predicted_ppi_inflation)
                * state.aggregates.production.max(1.0)
                * state.params.government_consumption_share;
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
            row.target_exports = positive_part(
                (1.0 + row.adjustment_speed * (production_index - 1.0)) * row.exports.max(1.0),
            );
            row.target_imports = positive_part(
                (1.0 + row.adjustment_speed * (price_index - 1.0))
                    * (1.0 + row.adjustment_speed * (production_index - 1.0))
                    * row.imports.max(1.0),
            );
            row.imports = row.target_imports;
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
        for household in &mut households {
            let labour_income = individuals
                .iter()
                .filter(|individual| individual.household_id == household.id)
                .map(|individual| individual.predicted_income)
                .sum::<f64>();
            let rent_income = properties
                .iter()
                .filter(|property| {
                    property.owner_household_id == household.id
                        && property.occupant_household_id != household.id
                })
                .map(|property| property.rent)
                .sum::<f64>();
            household.predicted_income = labour_income
                + state.forecast.predicted_cpi * household.social_benefits_other
                + rent_income
                + state.params.financial_asset_income_phi * household.other_financial_assets;
            let financial_asset_epsilon =
                state.normal_f64(0.0, state.params.financial_asset_income_sigma);
            household.income = individuals
                .iter()
                .filter(|individual| individual.household_id == household.id)
                .map(|individual| individual.income)
                .sum::<f64>()
                + state.aggregates.cpi * household.social_benefits_other
                + rent_income
                + (1.0 + financial_asset_epsilon)
                    * state.params.financial_asset_income_phi
                    * household.other_financial_assets;
            household.income += wage_income_by_household
                .get(household.id as usize)
                .copied()
                .unwrap_or(0.0);
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
                .map(|property| property.rent / 4.0)
                .sum::<f64>();
            household.disposable_income_after_rent =
                positive_part(household.income - quarterly_rent);
            let consumption_gap =
                positive_part(desired_consumption - household.disposable_income_after_rent);
            let financial_assets_used = consumption_gap.min(household.other_financial_assets);
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
            account.unemployment_benefit *=
                (1.0 / (1.0 + state.forecast.predicted_growth)).max(1.0);
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
        let mut households = collect_rows::<Household>(ecs)?;
        let mut properties = collect_rows::<Property>(ecs)?;
        let banks = collect_rows::<Bank>(ecs)?;

        let cpi_lag = lagged_cpi_inflation(&state);
        for property in &mut properties {
            property.previous_price = property.price;
            property.previous_rent = property.rent;
            if property.market_status == PROPERTY_FOR_SALE {
                property.quarters_on_sale += 1;
                if state.unit_f64() < state.params.sale_price_reduction_probability {
                    let epsilon = state.normal_f64(
                        state.params.sale_price_reduction_mu,
                        state.params.sale_price_reduction_sigma,
                    );
                    property.price = housing_reduced_value(
                        state.policy.housing_reduction_policy,
                        property.previous_price,
                        epsilon,
                    );
                }
            }
            if property.market_status == PROPERTY_FOR_RENT {
                property.quarters_on_rent_market += 1;
                if state.unit_f64() < state.params.rent_reduction_probability {
                    let epsilon = state.normal_f64(
                        state.params.rent_reduction_mu,
                        state.params.rent_reduction_sigma,
                    );
                    property.rent = housing_reduced_value(
                        state.policy.housing_reduction_policy,
                        property.previous_rent,
                        epsilon,
                    );
                }
            }
            if property.market_status == PROPERTY_RENTAL {
                property.rent *= 1.0 + state.params.rent_partial_indexation_phi * cpi_lag;
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
        state.shuffle(&mut household_order);
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
            let considers_move = needs_home || state.unit_f64() >= stay_probability;
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
                }
            }
            let epsilon =
                state.normal_f64(state.params.housing_mu_hp, state.params.housing_sigma_hp);
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
                let rent_cost = rent_cost_a108(state.params.housing_mu_ps, property.rent);
                let purchase_cost = purchase_cost_a109_literal_pdf(
                    property.price,
                    household.deposits + household.other_financial_assets,
                    bank_rate,
                    state.params.mortgage_maturity_quarters,
                    state.forecast.predicted_hpi_inflation,
                    property.value,
                );
                buy_probability_a110(state.params.housing_phi_b, rent_cost, purchase_cost)
            } else {
                0.0
            };
            if nearest_sale.is_some() && state.unit_f64() < buy_probability {
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
            state.shuffle(&mut class_apps);
            for app in class_apps {
                let visits = if app.borrower_kind == BUYER_FIRM {
                    state.policy.firm_bank_visits
                } else {
                    state.policy.household_bank_visits
                }
                .max(1) as usize;
                let mut bank_order: Vec<usize> = (0..banks.len()).collect();
                state.shuffle(&mut bank_order);
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
                    let supply = bank_credit_supply(&banks[bank_idx], &state);
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
        let mut demands: Vec<GoodsDemand> = buffers.brute_force(messages.goods_demand)?.collect();
        let mut state = macro_state(ecs, env_boundary)?;
        state.shuffle(&mut demands);
        let mut firms = collect_rows::<Firm>(ecs)?;
        let mut households = collect_rows::<Household>(ecs)?;
        let mut governments = collect_rows::<GovernmentEntity>(ecs)?;
        let mut rows = collect_rows::<RestOfWorld>(ecs)?;

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
            while remaining > 1e-9 && remaining_budget > 1e-9 {
                let available_sellers: Vec<usize> = sellers
                    .iter()
                    .copied()
                    .filter(|idx| {
                        positive_part(
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
                let firm_idx = available_sellers[weighted_choice(&mut state, &weights)];
                let available = positive_part(
                    firms[firm_idx].production + firms[firm_idx].inventory
                        - firms[firm_idx].sales_quantity,
                );
                let affordable = remaining_budget / firms[firm_idx].price.max(1e-9);
                let quantity = remaining.min(available).min(affordable);
                if quantity <= 1e-9 {
                    continue;
                }
                let payment = quantity * firms[firm_idx].price;
                firms[firm_idx].deposits += payment;
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
        let excess: Vec<ExcessDemand> = buffers.brute_force(messages.excess_demand)?.collect();
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
        let loan_settlement = settle_loan_book(
            state.quarter,
            &mut state,
            &mut banks,
            &mut firms,
            &mut households,
        );

        for firm in &mut firms {
            let sector = firm.sector as usize;
            let previous_inventory = firm.inventory;
            let previous_production = firm.production;
            firm.inventory = positive_part(firm.inventory + firm.production - firm.sales_quantity);
            for s in 0..SECTORS {
                firm.intermediate_stock[s] = positive_part(
                    firm.intermediate_stock[s]
                        - state.params.io_matrix[sector][s] * firm.sales_quantity
                        + firm.realised_intermediate[s],
                );
                let installed_capital = firm.capital_to_install[s];
                firm.capital_stock[s] = positive_part(
                    firm.capital_stock[s]
                        - state.params.capital_depreciation_rate_by_sector[s]
                            * firm.capital_stock[s]
                        + installed_capital,
                );
                firm.capital_to_install[s] = firm.realised_capital[s];
                firm.realised_intermediate[s] = 0.0;
                firm.realised_capital[s] = 0.0;
            }
            firm.demand = firm.sales_quantity
                + excess
                    .iter()
                    .filter(|msg| msg.sector == firm.sector)
                    .map(|msg| msg.quantity)
                    .sum::<f64>();
            let intermediate_purchases = receipts
                .iter()
                .filter(|receipt| {
                    receipt.buyer_kind == BUYER_FIRM
                        && receipt.buyer_id == firm.id
                        && receipt.purpose == GOODS_INTERMEDIATE
                })
                .map(|receipt| receipt.payment)
                .sum::<f64>();
            let production_tax =
                account_production_tax(&accounts, sector) * firm.price * firm.production;
            let capital_depreciation_cost: f64 = state
                .params
                .capital_depreciation_rate_by_sector
                .iter()
                .zip(firm.capital_stock.iter())
                .map(|(rate, stock)| rate * stock)
                .sum();
            let loan_interest_cost = loan_settlement.firm_interest(firm.id);
            firm.costs = firm.wage * firm.labour
                + intermediate_purchases
                + production_tax
                + capital_depreciation_cost
                + loan_interest_cost;
            firm.unit_cost = ratio(firm.costs, firm.production.max(1e-9));
            let delta_inventory = firm.inventory - previous_inventory;
            firm.profits =
                firm.price * firm.sales_quantity + firm.price * delta_inventory - firm.costs;
            firm.deposits += firm.profits;
            firm.equity = firm.deposits + firm.inventory + firm.capital_stock.iter().sum::<f64>()
                - firm.short_debt
                - firm.long_debt
                - firm.overdraft;
            if firm.deposits < 0.0 && firm.equity < 0.0 {
                firm.bankrupt = true;
                firm.deposits = 0.0;
                firm.short_debt = 0.0;
                firm.long_debt = 0.0;
                firm.overdraft = 0.0;
                firm.equity = firm.inventory + firm.capital_stock.iter().sum::<f64>();
            }
            firm.inventory_two_periods_ago = firm.previous_inventory;
            firm.previous_inventory = firm.inventory;
            firm.previous_demand = firm.demand;
            firm.previous_production = previous_production;
            firm.previous_price = firm.price;
        }

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
            household.property_wealth = properties
                .iter()
                .filter(|property| property.owner_household_id == household.id)
                .map(|property| property.value)
                .sum();
            household.net_wealth = household.property_wealth
                + household.other_real_assets
                + household.deposits
                + household.other_financial_assets
                - household.consumption_debt
                - household.mortgage_debt;
            if household.net_wealth < 0.0 && household.deposits < 0.0 {
                household.bankrupt = true;
                household.consumption_debt = 0.0;
                household.mortgage_debt = 0.0;
                household.deposits = 0.0;
            }
            household.income_history.rotate_left(1);
            household.income_history[1] = household.income;
            household.previous_income = household.income;
        }

        let total_wages = firms
            .iter()
            .map(|firm| firm.wage * firm.labour)
            .sum::<f64>();
        let total_profits = firms.iter().map(|firm| firm.profits).sum::<f64>();
        let total_consumption = receipts
            .iter()
            .filter(|receipt| receipt.buyer_kind == BUYER_HOUSEHOLD)
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
            bank.deposit_rate = policy_rate;
            bank.household_overdraft_rate = bank.household_rate;
            bank.firm_overdraft_rate = bank.short_firm_rate;
            let positive_reserve_income = positive_part(bank.reserves) * policy_rate;
            let negative_reserve_cost = negative_abs(bank.reserves) * bank.short_firm_rate;
            bank.profit = loan_settlement.bank_interest(bank.id) + positive_reserve_income
                - negative_reserve_cost
                - bank.deposit_rate * bank.deposits;
            bank.equity += bank.profit;
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
            bank.insolvent = bank.equity
                / (bank.liabilities + positive_part(bank.reserves)).max(1e-9)
                < state.params.solvency_ratio;
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
        state.aggregates.gdp = GdpIdentity {
            output: state.aggregates.production,
            expenditure: state.aggregates.household_consumption
                + state.aggregates.government_consumption
                + state.aggregates.investment
                + state.aggregates.exports
                - state.aggregates.imports_nominal,
            income: state.aggregates.wage_income + state.aggregates.profit_income,
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
            failed_gdp_identity: !state.aggregates.gdp.holds(1e-6),
        };
        state
            .history
            .production
            .push(state.aggregates.production.max(1e-9));
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
    aggregate.wage_income = firms.iter().map(|firm| firm.wage * firm.labour).sum();
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

fn housing_reduced_value(policy: HousingReductionPolicy, previous: f64, epsilon: f64) -> f64 {
    match policy {
        HousingReductionPolicy::LiteralPaperFormula => {
            literal_price_or_rent_reduction_a113_a115(previous, epsilon)
        }
        HousingReductionPolicy::GuardedFractionalReduction => {
            positive_part((1.0 - (-epsilon).exp()) * previous)
        }
    }
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
    seller_indices
        .iter()
        .map(|idx| {
            let relative_price = (-phi_gm * firms[*idx].price).exp() / price_sum;
            let relative_production = firms[*idx].production.max(0.0) / production_sum;
            ((relative_price + relative_production) / 2.0).max(0.0)
        })
        .collect()
}

fn weighted_choice(state: &mut MacroEnvironment, weights: &[f64]) -> usize {
    let total = weights.iter().sum::<f64>();
    if weights.is_empty() || total <= 1e-12 || !total.is_finite() {
        return 0;
    }
    let mut draw = state.unit_f64() * total;
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
        let worker_social_insurance = total_wages * account.social_insurance_worker_rate;
        let firm_social_insurance = total_wages * account.social_insurance_firm_rate;
        let labour_income_tax =
            total_wages * (1.0 - account.social_insurance_worker_rate) * account.income_tax_rate;
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
        account.revenue = worker_social_insurance
            + firm_social_insurance
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

fn bank_credit_supply(bank: &Bank, state: &MacroEnvironment) -> f64 {
    positive_part(
        bank.equity / state.params.car
            - bank.firm_loan_volume_by_sector.iter().sum::<f64>()
            - bank.consumption_loan_volume
            - bank.mortgage_volume,
    )
    .min(bank.credit_supply_max.max(0.0))
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
                let capital_value = firm.capital_stock.iter().sum::<f64>();
                let debt = firm.short_debt + firm.long_debt + firm.overdraft;
                let dte_cap = positive_part(
                    state.params.debt_to_equity * (capital_value + firm.equity) - debt,
                );
                let roe_cap =
                    positive_part((firm.predicted_profits / state.params.return_on_equity) - debt);
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
                let lti = state.params.mortgage_lti * six_month_income
                    - household.consumption_debt
                    - household.mortgage_debt;
                let maturity = state.params.mortgage_maturity_quarters as f64 / 4.0;
                let dsti_payment = state.params.mortgage_dsti * household.predicted_income;
                let dsti = if rate <= 1e-9 {
                    dsti_payment * maturity
                } else {
                    dsti_payment * (1.0 - (1.0 + rate).powf(-maturity)) / rate
                };
                positive_part(ltv.min(lti).min(dsti))
            })
            .unwrap_or(0.0),
        _ => 0.0,
    }
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
        BUYER_FIRM => {
            if let Some(firm) = firms.iter_mut().find(|firm| firm.id == demand.buyer_id) {
                firm.deposits -= payment;
                match demand.purpose {
                    GOODS_CAPITAL => firm.realised_capital[demand.sector as usize] += quantity,
                    _ => firm.realised_intermediate[demand.sector as usize] += quantity,
                }
            }
        }
        BUYER_HOUSEHOLD => {
            if let Some(household) = households
                .iter_mut()
                .find(|household| household.id == demand.buyer_id)
            {
                household.deposits -= payment;
            }
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
                row.net_exports += payment;
            }
        }
        _ => {}
    }
}
