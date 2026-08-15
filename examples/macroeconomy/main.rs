#[cfg(all(feature = "model", feature = "messaging"))]
#[path = "lib.rs"]
mod macroeconomy;

#[cfg(all(feature = "model", feature = "messaging"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    use macroeconomy::{
        aggregate_row, build_macroeconomy_model, csv_header, firm_row, headline_row, macro_state,
        FixtureDataProvider, RealDataProvider, RunMode, AGGREGATE_COLUMNS, FIRM_COLUMNS,
        HEADLINE_COLUMNS,
    };

    let config = parse_args(env::args().skip(1))?;
    // Chrome Trace output, viewable in chrome://tracing or Perfetto. The spans
    // compile to nothing unless the `profiling` feature is on, so an ordinary
    // run pays nothing for them.
    if let Some(path) = &config.profile_path {
        syren::init(path);
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
        agg_rows.push(csv_header(AGGREGATE_COLUMNS));
        firm_rows.push(csv_header(FIRM_COLUMNS));
    }

    println!("{}", csv_header(HEADLINE_COLUMNS));
    for _ in 0..config.ticks {
        example.model.tick()?;
        let state = macro_state(&example.model)?;
        println!("{}", headline_row(&state));
        if tracing {
            agg_rows.push(aggregate_row(&state));
            for f in &state.audit.firm_trace {
                firm_rows.push(firm_row(state.quarter, f));
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
        let dir = config
            .trace_dir
            .clone()
            .ok_or("--trace requires an output directory")?;
        std::fs::create_dir_all(&dir)?;
        let aggregates = dir.join("trace_aggregates.csv");
        let firms = dir.join("trace_firms.csv");
        std::fs::write(&aggregates, agg_rows.join("\n"))?;
        std::fs::write(&firms, firm_rows.join("\n"))?;
        eprintln!("wrote {} and {}", aggregates.display(), firms.display());
    }
    let state = macro_state(&example.model)?;
    if config.profile_path.is_some() {
        syren::shutdown();
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
                config.firms_per_sector = args
                    .next()
                    .ok_or("--firms-per-sector requires a value")?
                    .parse()?;
            }
            "--profile" => {
                config.profile_path = Some(PathBuf::from(
                    args.next().ok_or("--profile requires a path")?,
                ));
            }
            "--trace" => {
                let dir = args.next().ok_or("--trace requires an output directory")?;
                config.trace_dir = Some(PathBuf::from(dir));
                config.policy.trace = true;
            }
            "--debug-firm" => {
                config.policy.debug_firm_id = Some(
                    args.next()
                        .ok_or("--debug-firm requires a firm id")?
                        .parse()?,
                );
            }
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run --release --features \"model messaging\" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 [--config path] [--scenario name] [--firms-per-sector n] [--trace out-dir] [--profile out.json] [--debug-firm id]"
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
