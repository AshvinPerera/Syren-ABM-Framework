#![cfg(all(feature = "model", feature = "messaging"))]

#[path = "lib.rs"]
mod macroeconomy;

use std::path::PathBuf;

use macroeconomy::*;

fn fixture() -> MacroeconomyExample {
    build_macroeconomy_model(MacroeconomyConfig::default(), FixtureDataProvider).unwrap()
}

#[test]
fn fixture_runs_quarterly_scheduler_in_paper_order() {
    let mut example = fixture();
    example.model.tick().unwrap();
    let state = macro_state(&example.model).unwrap();
    assert_eq!(state.quarter, 1);
    assert_eq!(state.scale_factor, SIM_SCALE_FACTOR);
    assert_eq!(
        state.audit.phase_log,
        vec![
            "aggregate_previous_state",
            "refit_expectations",
            "target_setting",
            "labour_market",
            "planning_and_production",
            "housing_preclear",
            "credit_market",
            "housing_completion",
            "goods_market",
            "realised_accounting",
        ]
    );
}

#[test]
fn market_audits_record_required_ordering_and_priority_rules() {
    let mut example = fixture();
    example.model.tick().unwrap();
    let state = macro_state(&example.model).unwrap();
    assert!(state.audit.labour_fired_before_hiring);
    assert_eq!(
        state.audit.credit_clearing_order,
        vec![
            LOAN_FIRM_SHORT,
            LOAN_FIRM_LONG,
            LOAN_HOUSEHOLD_CONSUMPTION,
            LOAN_MORTGAGE,
        ]
    );
    assert!(state.audit.lower_price_seller_priority_seen);
    assert!(state.audit.credit_visits_ordered_by_rate);
}

#[test]
fn accounting_harness_produces_finite_aggregates() {
    let mut example = fixture();
    example.model.tick().unwrap();
    let state = macro_state(&example.model).unwrap();
    assert!(state.aggregates.production >= 0.0);
    assert!(state.aggregates.ppi > 0.0);
    assert!(state.aggregates.cpi > 0.0);
    // The synthetic generator emits 8 quarters of history, not 4, and with
    // genuine variation in it. A.2 refits the AR(1) each quarter on real data
    // plus simulation output; a synthetic run has no real-data prefix, so a
    // flat 4-quarter history made every forecast predict zero change forever --
    // which is what pinned `hpi` at exactly 1.000000, since property value is
    // marked to market by *predicted* HPI inflation.
    // The generator emits 48 quarters of history. A.2 refits the AR(1) each
    // quarter on real data from 2000-Q1 *plus* simulation output -- roughly 50
    // real observations before a run even starts. With only 8 synthetic
    // quarters the fit had no anchor: it extrapolated a 239% growth forecast
    // from an oscillating series, which drove A.60 to target 62% above feasible
    // output and the Taylor rule to a 24% policy rate. Length is what stabilises
    // it.
    assert!(state.audit.last_expectation_fit_observations <= 53);

    // The GDP triple identity does not close yet: `expenditure` counts payments
    // for capital goods while `output` counts real production, and `income`
    // omits taxes, bank profits, and rental income. Once those are reconciled
    // this becomes `assert!(gdp.holds(relative_tolerance))`.
    // Until then, assert only that the residual is finite -- that catches NaN
    // propagation through the accounting pass, which is the failure mode a
    // tautological `failed == !holds` assertion could never catch.
    let gdp = state.accounting.gdp;
    assert!(gdp.output.is_finite());
    assert!(gdp.expenditure.is_finite());
    assert!(gdp.income.is_finite());
    assert!(gdp.max_gap().is_finite());
}

#[test]
fn table4_defaults_and_binary_search_grid_match_spec() {
    let params = CalibrationParameters::austria_npe_table4();
    assert_eq!(params.phi_f_q, 0.0);
    assert_eq!(params.phi_dp, 0.0);
    assert_eq!(params.phi_cp, 0.0);
    assert_eq!(params.phi_st_y, 0.10);
    assert_eq!(params.chi_h, 0.53);
    assert_eq!(params.chi_m, 0.03);
    assert_eq!(params.chi_k, 0.18);
    assert_eq!(CalibrationParameters::binary_search_combinations().len(), 8);
}

#[test]
fn real_data_mode_fails_fast_with_named_missing_assets() {
    let config = MacroeconomyConfig {
        mode: RunMode::RealData,
        data_dir: Some(PathBuf::from("missing")),
        ..MacroeconomyConfig::default()
    };
    let err = match build_macroeconomy_model(
        config,
        RealDataProvider {
            data_dir: PathBuf::from("missing"),
        },
    ) {
        Ok(_) => panic!("real-data provider should fail without external assets"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(message.contains("OECD ICIO"));
    assert!(message.contains("ECB HFCS"));
    assert!(message.contains("Compustat"));
}

#[test]
fn ar1_taylor_and_ardl_helpers_follow_spec_shapes() {
    let fit = fit_ar1_log_level_forecast(&[1.0, 2.0, 4.0, 8.0], 8.0);
    assert_eq!(fit.observations, 4);
    assert!((fit.forecast_level - 16.0).abs() < 1e-9);
    let sparse = fit_ar1_log_level_forecast(&[0.0, f64::NAN, 2.0], 7.0);
    assert_eq!(sparse.forecast_level, 7.0);
    let taylor = transform_taylor_rule(0.01, 0.8, 0.3, 0.1, 0.02);
    assert!((taylor.xi_pi - 1.5).abs() < 1e-9);
    assert!((taylor.xi_gamma - 0.5).abs() < 1e-9);
    let best = select_ardl_lag_by_aic(&[
        ArdlCandidate {
            p: 1,
            q: 1,
            r: 1,
            s: 1,
            aic: 10.0,
        },
        ArdlCandidate {
            p: 2,
            q: 1,
            r: 1,
            s: 1,
            aic: 9.0,
        },
    ])
    .unwrap();
    assert_eq!(best.p, 2);
    let delta = ardl_error_correction_delta_rate(&ArdlErrorCorrectionInput {
        previous_loan_rate: 0.04,
        current_policy_rate: 0.03,
        error_correction_phi: -0.2,
        long_run_pass_through_phi: 1.1,
        lagged_loan_rate_deltas: &[0.01],
        alpha: &[0.5],
        lagged_policy_rate_deltas: &[0.02],
        beta: &[0.25],
        lagged_ppi_inflation_deltas: &[0.03],
        gamma: &[0.1],
        lagged_npl_ratio_deltas: &[0.04],
        delta: &[0.05],
        mu: 0.001,
    });
    let expected =
        -0.2 * (0.04 - 1.1 * 0.03) + 0.5 * 0.01 + 0.25 * 0.02 + 0.1 * 0.03 + 0.05 * 0.04 + 0.001;
    assert!((delta - expected).abs() < 1e-12);
}

#[test]
fn config_overrides_parameters_without_changing_framework_api() {
    let mut environment = MacroEnvironment::new(7);
    apply_config_str(
        "
defaults:
  car: 0.12
  mortgage_ltv: 0.7
  housing_phi_hp: 41.5
  calibration.chi_h: 0.44
  policy.firm_bank_visits: 3
",
        None,
        &mut environment,
    )
    .unwrap();
    assert_eq!(environment.params.car, 0.12);
    assert_eq!(environment.params.mortgage_ltv, 0.7);
    assert_eq!(environment.params.housing_phi_hp, 41.5);
    assert_eq!(environment.calibration.chi_h, 0.44);
    assert_eq!(environment.policy.firm_bank_visits, 3);
}

#[test]
fn thesis_defaults_and_initialisation_recipe_are_exposed() {
    let environment = MacroEnvironment::new(42);
    assert_eq!(environment.history.first_real_data_quarter, "2000-Q1");
    assert_eq!(environment.params.goods_market_phi, 2.0);
    assert_eq!(environment.params.credit_supply_phi, 2.0);
    assert_eq!(environment.params.car, 0.08);
    assert_eq!(environment.params.solvency_ratio, 0.10);
    assert_eq!(environment.params.debt_to_equity, 1.0);
    assert_eq!(environment.params.return_on_equity, 0.15);
    assert_eq!(environment.params.return_on_assets, 0.05);
    assert_eq!(environment.params.consumption_lti, 0.36);
    assert_eq!(environment.params.work_effort_max, 1.5);
    assert_eq!(
        environment.params.capital_installation_delay_quarters,
        [1; SECTORS]
    );
    assert_eq!(environment.params.housing_phi_hp, 42.90);
    assert_eq!(environment.params.housing_beta_hp, 0.79);
    assert_eq!(environment.params.housing_mu_hp, -0.018);
    // Carro et al. (2023) Table 10 gives `eps_sigma = 0.4104` for A.107's noise,
    // and `normal_f64` takes a standard deviation. Wiese quotes 0.1684, which is
    // 0.4104 squared -- the variance. The old 0.17 here locked in that slip.
    assert_eq!(environment.params.housing_sigma_hp, 0.4104);
    assert_eq!(environment.params.housing_phi_hr, 17.22);
    assert_eq!(environment.params.housing_beta_hr, 0.35);
    assert_eq!(
        environment.policy.goods_clearing_policy,
        GoodsClearingPolicy::PolednaSearchAndMatching
    );

    let recipe = thesis_initialisation_recipe();
    assert!(recipe
        .iter()
        .any(|step| step.id == "firm-compustat-sampling"));
    assert!(recipe
        .iter()
        .any(|step| step.id == "household-hfcs-sampling"));
    assert!(recipe
        .iter()
        .any(|step| step.id == "linear-sum-assignments"));
    assert!(recipe.iter().any(|step| step.id == "scale-factor"));
}

#[test]
fn shard_count_is_derived_from_population() {
    // Shard count is fixed at construction and each shard addresses 2^22 - 1
    // entities, so under-provisioning is an unrecoverable mid-run failure.
    let tiny = FixtureDataProvider
        .load(&MacroeconomyConfig::default())
        .unwrap();
    assert_eq!(shards_for_population(&tiny), 1);

    let mut wide = tiny.clone();
    // 3M households at 2x headroom needs two shards.
    wide.households = vec![wide.households[0]; 3_000_000];
    assert!(
        shards_for_population(&wide) >= 2,
        "3M agents must not be squeezed into one {ENTITIES_PER_SHARD}-entity shard"
    );
}

/// Runs the fixture for `ticks` quarters inside a pool of exactly `threads`
/// workers, returning the raw bits of each tick's headline aggregates.
///
/// Raw bits rather than `f64` so the comparison is exact and a NaN cannot
/// silently compare unequal to itself and be mistaken for a thread-count bug.
fn trajectory_bits(seed: u64, ticks: u64, threads: usize) -> Vec<[u64; 7]> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let config = MacroeconomyConfig {
            seed,
            ticks,
            ..MacroeconomyConfig::default()
        };
        let mut example = build_macroeconomy_model(config, FixtureDataProvider).unwrap();
        let mut out = Vec::with_capacity(ticks as usize);
        for _ in 0..ticks {
            example.model.tick().unwrap();
            let state = macro_state(&example.model).unwrap();
            out.push([
                state.aggregates.production.to_bits(),
                state.aggregates.ppi.to_bits(),
                state.aggregates.cpi.to_bits(),
                state.aggregates.hpi.to_bits(),
                state.aggregates.rpi.to_bits(),
                state.aggregates.total_loans.to_bits(),
                state.accounting.gdp.max_gap().to_bits(),
            ]);
        }
        out
    })
}

/// The load-bearing determinism guarantee.
///
/// Rayon decides which worker processes which chunk by work stealing, so any
/// draw or accumulation whose result depends on visitation order will diverge
/// here. Every market restructuring must keep this green; it is the reason
/// draws are keyed on `(seed, tick, system_id, salt)` rather than taken from a
/// single shared sequential stream.
#[test]
fn same_seed_same_trajectory_at_1_and_8_threads() {
    let single = trajectory_bits(42, 6, 1);
    let multi = trajectory_bits(42, 6, 8);
    assert_eq!(single.len(), 6);
    assert_eq!(
        single, multi,
        "trajectory depends on thread count: some draw or accumulation is order-sensitive"
    );
}

#[test]
fn same_seed_reproduces_and_distinct_seeds_diverge() {
    let a = trajectory_bits(42, 4, 4);
    let b = trajectory_bits(42, 4, 4);
    assert_eq!(a, b, "identical seeds must reproduce exactly");

    let c = trajectory_bits(99, 4, 4);
    assert_ne!(
        a, c,
        "distinct seeds must diverge; if they do not, the model seed is not \
         reaching the draw sites through RunContext::simulation_seed"
    );
}

#[test]
fn csv_headers_are_single_line_with_unique_columns() {
    for columns in [HEADLINE_COLUMNS, AGGREGATE_COLUMNS, FIRM_COLUMNS] {
        let header = csv_header(columns);
        assert!(
            !header.contains('\n'),
            "a CSV header must be one physical line: {header}"
        );
        assert_eq!(
            header.split(',').count(),
            columns.len(),
            "joined header field count must equal the column list length"
        );
        let mut seen = std::collections::HashSet::new();
        for name in columns {
            assert!(seen.insert(name), "duplicate column name: {name}");
        }
    }
}

#[test]
fn csv_rows_match_their_headers_column_for_column() {
    let mut example = build_macroeconomy_model(
        MacroeconomyConfig {
            ticks: 1,
            ..MacroeconomyConfig::default()
        },
        FixtureDataProvider,
    )
    .unwrap();
    example.model.tick().unwrap();
    let state = macro_state(&example.model).unwrap();

    // Numeric fields never contain a comma, so the field count is the number of
    // comma-separated pieces.
    assert_eq!(
        headline_row(&state).split(',').count(),
        HEADLINE_COLUMNS.len(),
        "headline row width differs from HEADLINE_COLUMNS"
    );
    assert_eq!(
        aggregate_row(&state).split(',').count(),
        AGGREGATE_COLUMNS.len(),
        "aggregate row width differs from AGGREGATE_COLUMNS"
    );
    assert_eq!(
        firm_row(state.quarter, &FirmProbe::default())
            .split(',')
            .count(),
        FIRM_COLUMNS.len(),
        "firm row width differs from FIRM_COLUMNS"
    );
}

#[test]
fn housing_reduction_is_a_percentage_haircut() {
    // mu/sigma are log-normal parameters of a percentage reduction: the sale
    // median is exp(1.4531) = 4.28%, the rent median exp(1.6559) = 5.24%.
    let sale = price_or_rent_reduction_a113_a115(200.0, 1.4531);
    assert!((sale - 200.0 * (1.0 - 0.0428)).abs() < 0.2, "got {sale}");
    let rent = price_or_rent_reduction_a113_a115(100.0, 1.6559);
    assert!((rent - 100.0 * (1.0 - 0.0524)).abs() < 0.1, "got {rent}");

    // A reduction must reduce, never invert. `exp(eps)` is a percentage, so
    // reading it as a fraction would give `1 - 4.28 = -3.28` -- a negative
    // price -- for the median draw.
    for eps in [-3.0, 0.0, 1.4531, 3.0, 6.0, 12.0] {
        let out = price_or_rent_reduction_a113_a115(100.0, eps);
        assert!(out > 0.0 && out <= 100.0, "eps={eps} gave {out}");
    }
}

/// Aggregate output and the price level stay alive over a short run.
///
/// The two are coupled: A.73's cost-push term divides costs by production, so
/// output reaching zero sends `unit_cost` and every price index to infinity in
/// the same quarter. Checking both each tick pins the pair.
#[test]
fn production_does_not_collapse_over_a_short_run() {
    let mut example = fixture();
    for quarter in 1..=8 {
        example.model.tick().unwrap();
        let state = macro_state(&example.model).unwrap();
        assert!(
            state.aggregates.production > 0.0,
            "production collapsed to {} at quarter {quarter}",
            state.aggregates.production
        );
        assert!(
            state.aggregates.ppi.is_finite() && state.aggregates.ppi > 0.0,
            "ppi is {} at quarter {quarter}",
            state.aggregates.ppi
        );
    }
}

#[test]
fn named_equation_helpers_match_paper_arithmetic() {
    assert!((firm_target_demand_a60(0.1, 0.5, 0.2, 100.0) - 121.0).abs() < 1e-9);
    let target =
        firm_target_production_a62(100.0, 0.1, 80.0, 5.0, 0.5, 90.0, 0.25, 120.0, 0.75, 110.0);
    assert!((target - 95.0).abs() < 1e-9);
    assert_eq!(bank_liabilities_a42(10.0, 25.0, 20.0, -3.0), 52.0);
    assert_eq!(bank_reserves_a43(5.0, -2.0, 10.0, 4.0), 9.0);
}

#[test]
fn component_sizes_are_recorded() {
    // Row-struct sizes matter: `collect_rows_by` materialises whole rows, so a
    // system that touches a handful of fields still pulls every byte through
    // cache. This test documents the sizes rather than constraining them.
    eprintln!("Firm        {:>6} bytes", std::mem::size_of::<Firm>());
    eprintln!("Household   {:>6} bytes", std::mem::size_of::<Household>());
    eprintln!("Individual  {:>6} bytes", std::mem::size_of::<Individual>());
    eprintln!("Property    {:>6} bytes", std::mem::size_of::<Property>());
}
