#![cfg(all(feature = "model", feature = "messaging"))]

#[path = "../examples/macroeconomy/lib.rs"]
mod macroeconomy;

use std::path::PathBuf;

use macroeconomy::*;

fn fixture() -> MacroeconomyExample {
    build_macroeconomy_model(MacroeconomyConfig::default(), FixtureDataProvider).unwrap()
}

#[test]
fn paper_alignment_covers_every_appendix_equation() {
    let coverage = EquationCoverage::paper_aligned();
    assert_eq!(coverage.entries.len(), 142);
    assert!(coverage.missing_equations().is_empty());
    let appendix_coverage = EquationCoverage::appendix_coverage();
    for appendix in EquationCoverage::appendix_labels() {
        assert!(
            appendix_coverage
                .iter()
                .any(|entry| entry.appendix == appendix),
            "missing appendix coverage for {appendix}"
        );
    }
    assert!(coverage
        .entries
        .iter()
        .any(|entry| entry.equation == 15 && entry.status == CoverageStatus::ValidationOnly));
    assert!(coverage
        .unresolved_replication_blockers()
        .iter()
        .any(|entry| entry.equation == 24));
    assert!(coverage
        .entries
        .iter()
        .all(|entry| entry.source.contains("Paper Appendix A")));
    let gaps = EquationCoverage::blocker_log();
    assert_eq!(gaps.len(), 4);
    assert!(gaps.iter().any(|gap| gap.id == "ar1-missing-data-policy"));
    assert!(!gaps
        .iter()
        .any(|gap| gap.id == "goods-flow-preservation-pseudocode"));
    assert!(!gaps
        .iter()
        .any(|gap| gap.id == "trajectory-exact-randomness"));
    assert_eq!(
        coverage
            .entries
            .iter()
            .find(|entry| entry.equation == 140)
            .unwrap()
            .status,
        CoverageStatus::Implemented
    );
    assert!(EquationCoverage::gap_report_text().contains("remaining_exact_replication_gaps"));
    assert!(EquationCoverage::gap_report_json().contains("\"gap_id\""));
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
fn accounting_and_forecasting_harnesses_are_wired() {
    let mut example = fixture();
    example.model.tick().unwrap();
    let state = macro_state(&example.model).unwrap();
    assert!(state.aggregates.production >= 0.0);
    assert!(state.aggregates.ppi > 0.0);
    assert!(state.aggregates.cpi > 0.0);
    assert!(state.audit.last_expectation_fit_observations <= 4);
    assert_eq!(
        state.accounting.failed_gdp_identity,
        !state.accounting.gdp.holds(1e-6)
    );

    let forecast = run_forecast_batch(ForecastExperimentConfig::default()).unwrap();
    assert_eq!(forecast.countries, 38);
    assert_eq!(forecast.initialisation_quarters, 20);
    assert_eq!(forecast.horizon_quarters, 12);
    assert_eq!(forecast.trajectories, 1_000);
    assert_eq!(forecast.npe.stacks_or_layers, 5);
    assert_eq!(forecast.nre.stacks_or_layers, 2);
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
fn strict_replication_refuses_unresolved_blockers() {
    let config = MacroeconomyConfig {
        replication_policy: ReplicationPolicy::strict(),
        ..MacroeconomyConfig::default()
    };
    let err = match build_macroeconomy_model(config, FixtureDataProvider) {
        Ok(_) => panic!("strict replication should fail while blocker log is non-empty"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(message.contains("strict replication requested"));
    assert!(message.contains("credit-visit-limits"));
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
        car=0.12
        mortgage_ltv=0.7
        housing_phi_hp=41.5
        calibration.chi_h=0.44
        policy.firm_bank_visits=3
        ",
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
    assert_eq!(environment.params.housing_sigma_hp, 0.17);
    assert_eq!(environment.params.housing_phi_hr, 17.22);
    assert_eq!(environment.params.housing_beta_hr, 0.35);
    assert_eq!(
        MacroeconomyConfig::default().gap_report,
        GapReportMode::Text
    );
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
fn python_like_rng_is_seeded_and_reproducible() {
    let mut first = MacroEnvironment::new(42);
    let mut second = MacroEnvironment::new(42);
    let draws_a = [
        first.next_u32(),
        first.next_u32(),
        first.next_u32(),
        first.next_u32(),
    ];
    let draws_b = [
        second.next_u32(),
        second.next_u32(),
        second.next_u32(),
        second.next_u32(),
    ];
    assert_eq!(draws_a, draws_b);

    let mut values_a = [1, 2, 3, 4, 5, 6];
    let mut values_b = [1, 2, 3, 4, 5, 6];
    first.shuffle(&mut values_a);
    second.shuffle(&mut values_b);
    assert_eq!(values_a, values_b);
    assert!(first.bernoulli(1.0));
    assert!(!first.bernoulli(0.0));
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
