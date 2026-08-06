#[cfg(all(feature = "model", feature = "messaging"))]
#[path = "lib.rs"]
mod macroeconomy;

#[cfg(all(feature = "model", feature = "messaging"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    use macroeconomy::{
        build_macroeconomy_model, macro_state, FixtureDataProvider, RealDataProvider, RunMode,
    };

    let config = parse_args(env::args().skip(1))?;
    // Chrome Trace output, viewable in chrome://tracing or Perfetto. The spans
    // compile to nothing unless the `profiling` feature is on, so an ordinary
    // run pays nothing for them.
    if let Some(path) = &config.profile_path {
        abm_framework::init(path);
    }
    let mut example = match config.mode {
        RunMode::TinyFixture => build_macroeconomy_model(config.clone(), FixtureDataProvider)?,
        RunMode::RealData => {
            let data_dir = config
                .data_dir
                .clone()
                .ok_or("--data-dir is required in real-data mode")?;
            build_macroeconomy_model(config.clone(), RealDataProvider { data_dir })?
        }
    };

    let tracing = config.policy.trace;
    let mut agg_rows: Vec<String> = Vec::new();
    let mut firm_rows: Vec<String> = Vec::new();
    if tracing {
        agg_rows.push(
            "tick,production,ppi,cpi,hpi,rpi,employed,individuals,goods_demand,goods_supply,excess_demand,sales_rev,inv_chg,costs,c_wage,c_interm,c_cap,c_tax,c_int,firm_deposits,firm_debt,firm_equity,firms_bankrupt,hh_deposits,hh_ofa,hh_income,hh_consumption,hh_net_wealth,hh_bankrupt,bank_equity,bank_reserves,bank_deposits,gov_revenue,gov_deficit,gov_debt,unemp_benefit,avg_wage,total_loans,credit_req,credit_granted,roa_apps,roa_fails,roa_max,bail_ins,gdp_out,gdp_exp,b_loanint,b_resinc,b_rescost,b_depint,b_tax,b_seized,b_lost,policy_rate,cpi_infl,growth,blk_roa,blk_cap,blk_supply,envelope,cap_total,cap_dte,cap_roe,dte_zero,roe_zero,d_interm,d_fcap,d_cons,d_hcap,d_gov,d_exp"
                .to_owned(),
        );
        firm_rows.push(
            "tick,firm,employees,work_effort,h_f0,H,M,K,target,production,price,unit_cost,demand,excess_demand,wage,deposits,debt,equity,profits,sales,inventory,target_short_loan,granted_short_loan"
                .to_owned(),
        );
    }

    println!(
        "tick,production,ppi,cpi,hpi,rpi,total_loans,gdp_gap,blocked_mortgages,excess_demand,\
         prod_over_labour,labour_over_materials,employed,individuals,goods_demand,goods_supply,roa_apps,roa_fails,roa_max,credit_req,credit_granted,\n         sales_rev,inv_chg,costs,c_wage,c_interm,c_cap,c_tax,c_int"
    );
    for _ in 0..config.ticks {
        example.model.tick()?;
        let state = macro_state(&example.model)?;
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{},{},{:.6},{:.6},{},{},{:.5},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            state.quarter,
            state.aggregates.production,
            state.aggregates.ppi,
            state.aggregates.cpi,
            state.aggregates.hpi,
            state.aggregates.rpi,
            state.aggregates.total_loans,
            state.accounting.gdp.max_gap(),
            state.audit.mortgage_blocked_purchases,
            state.audit.goods_excess_demand,
            state.audit.max_production_over_labour,
            state.audit.max_labour_over_materials,
            state.audit.employed_headcount,
            state.audit.individual_headcount,
            state.audit.goods_demand_quantity,
            state.audit.goods_supply_quantity,
            state.audit.firm_credit_applications,
            state.audit.firm_credit_roa_failures,
            state.audit.firm_roa_max,
            state.audit.firm_credit_requested,
            state.audit.firm_credit_granted,
            state.audit.profit_sales_revenue,
            state.audit.profit_inventory_change,
            state.audit.profit_costs,
            state.audit.cost_wages,
            state.audit.cost_intermediate,
            state.audit.cost_capital,
            state.audit.cost_production_tax,
            state.audit.cost_interest
        );
        if tracing {
            let a = &state.audit;
            agg_rows.push(format!(
                "{},{:.4},{:.6},{:.6},{:.6},{:.6},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4},{},{},{:.5},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                state.quarter,
                state.aggregates.production, state.aggregates.ppi, state.aggregates.cpi,
                state.aggregates.hpi, state.aggregates.rpi,
                a.employed_headcount, a.individual_headcount,
                a.goods_demand_quantity, a.goods_supply_quantity, a.goods_excess_demand,
                a.profit_sales_revenue, a.profit_inventory_change, a.profit_costs,
                a.cost_wages, a.cost_intermediate, a.cost_capital, a.cost_production_tax,
                a.cost_interest,
                a.firm_deposits_total, a.firm_debt_total, a.firm_equity_total, a.firms_bankrupt,
                a.household_deposits_total, a.household_ofa_total, a.household_income_total,
                a.household_consumption_total, a.household_net_wealth_total, a.households_bankrupt,
                a.bank_equity_total, a.bank_reserves_total, a.bank_deposits_total,
                a.government_revenue, a.government_deficit, a.government_debt,
                a.unemployment_benefit, a.average_wage,
                state.aggregates.total_loans, a.firm_credit_requested, a.firm_credit_granted,
                a.firm_credit_applications, a.firm_credit_roa_failures, a.firm_roa_max,
                a.bank_bail_ins,
                state.aggregates.gdp.output, state.aggregates.gdp.expenditure,
                a.bank_loan_interest, a.bank_reserve_income, a.bank_reserve_cost,
                a.bank_deposit_interest, a.bank_corporate_tax, a.bank_writeoff_seized,
                a.bank_writeoff_lost, a.policy_rate, a.taylor_cpi_inflation, a.taylor_growth,
                a.credit_blocked_by_roa, a.credit_blocked_by_cap, a.credit_blocked_by_supply,
                a.credit_envelope_total, a.credit_cap_total,
                a.cap_dte_total, a.cap_roe_total, a.cap_dte_zero, a.cap_roe_zero,
                a.demand_firm_intermediate, a.demand_firm_capital,
                a.demand_household_consumption, a.demand_household_capital,
                a.demand_government, a.demand_export
            ));
            for f in &a.firm_trace {
                firm_rows.push(format!(
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    state.quarter, f.id, f.employees, f.work_effort, f.initial_work_effort,
                    f.labour, f.intermediate_constraint, f.capital_constraint,
                    f.target_production, f.production, f.price, f.unit_cost, f.demand,
                    f.excess_demand, f.wage, f.deposits, f.debt, f.equity, f.profits,
                    f.sales_quantity, f.inventory, f.target_short_loan, f.granted_short_loan
                ));
            }
        }
        if let Some(probe) = state.audit.firm_probe {
            eprintln!(
                "  firm {} q{}: employees={} h_f={:.4} (h_f0={:.4}) H={:.4} M={:.4} K={:.4} \
                 target={:.4} Y={:.4} P={:.4} U={:.4} Q={:.4} excess={:.4}",
                probe.id,
                state.quarter,
                probe.employees,
                probe.work_effort,
                probe.initial_work_effort,
                probe.labour,
                probe.intermediate_constraint,
                probe.capital_constraint,
                probe.target_production,
                probe.production,
                probe.price,
                probe.unit_cost,
                probe.demand,
                probe.excess_demand
            );
        }
    }
    if tracing {
        std::fs::write("trace_aggregates.csv", agg_rows.join("
"))?;
        std::fs::write("trace_firms.csv", firm_rows.join("
"))?;
        eprintln!("wrote trace_aggregates.csv and trace_firms.csv");
    }
    let state = macro_state(&example.model)?;
    if config.profile_path.is_some() {
        abm_framework::shutdown();
    }
    eprintln!("completed {} quarters", state.quarter);
    eprintln!(
        "housing (cumulative): listings={} sales={} blocked_mortgages={} transfer_value={:.1}",
        state.audit.housing_listings,
        state.audit.housing_sales,
        state.audit.mortgage_blocked_purchases,
        state.audit.housing_transfer_value
    );
    eprintln!(
        "mortgage caps binding: ltv={} lti={} dsti={} | mean cap={:.1} mean requested={:.1}",
        state.audit.mortgage_bind_ltv,
        state.audit.mortgage_bind_lti,
        state.audit.mortgage_bind_dsti,
        state.audit.mortgage_cap_sum
            / (state.audit.mortgage_bind_ltv
                + state.audit.mortgage_bind_lti
                + state.audit.mortgage_bind_dsti)
                .max(1) as f64,
        state.audit.mortgage_req_sum
            / (state.audit.mortgage_bind_ltv
                + state.audit.mortgage_bind_lti
                + state.audit.mortgage_bind_dsti)
                .max(1) as f64
    );
    Ok(())
}

#[cfg(all(feature = "model", feature = "messaging"))]
fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<macroeconomy::MacroeconomyConfig, Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    let mut config = macroeconomy::MacroeconomyConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixture" => {
                let name = args.next().ok_or("--fixture requires a value")?;
                if name != "tiny" {
                    return Err(format!("unknown fixture: {name}").into());
                }
                config.mode = macroeconomy::RunMode::TinyFixture;
            }
            "--ticks" => {
                config.ticks = args.next().ok_or("--ticks requires a value")?.parse()?;
            }
            "--seed" => {
                config.seed = args.next().ok_or("--seed requires a value")?.parse()?;
            }
            "--data-dir" => {
                config.mode = macroeconomy::RunMode::RealData;
                config.data_dir = Some(PathBuf::from(
                    args.next().ok_or("--data-dir requires a path")?,
                ));
            }
            "--scenario" => {
                config.scenario = Some(args.next().ok_or("--scenario requires a name")?);
            }
            "--config" => {
                config.config_path = Some(PathBuf::from(
                    args.next().ok_or("--config requires a path")?,
                ));
            }
            "--country" => {
                config.country = args.next().ok_or("--country requires a code")?;
            }
            "--initialisation" => {
                config.initialisation = args
                    .next()
                    .ok_or("--initialisation requires a yyyy-Qn value")?;
            }
            "--firms-per-sector" => {
                config.firms_per_sector =
                    args.next().ok_or("--firms-per-sector requires a value")?.parse()?;
            }
            "--profile" => {
                config.profile_path = Some(PathBuf::from(
                    args.next().ok_or("--profile requires a path")?,
                ));
            }
            "--trace" => {
                config.policy.trace = true;
            }
            "--debug-firm" => {
                config.policy.debug_firm_id =
                    Some(args.next().ok_or("--debug-firm requires a firm id")?.parse()?);
            }
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run --release --features \"model messaging\" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 [--config path] [--debug-firm id]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

#[cfg(not(all(feature = "model", feature = "messaging")))]
fn main() {
    eprintln!("the macroeconomy example requires --features \"model messaging\"");
}
