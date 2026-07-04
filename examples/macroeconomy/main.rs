#[cfg(all(feature = "model", feature = "messaging"))]
#[path = "lib.rs"]
mod macroeconomy;

#[cfg(all(feature = "model", feature = "messaging"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    use macroeconomy::{
        build_macroeconomy_model, macro_state, EquationCoverage, FixtureDataProvider,
        GapReportMode, RealDataProvider, RunMode,
    };

    let config = parse_args(env::args().skip(1))?;
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

    println!("tick,production,ppi,cpi,hpi,rpi,total_loans,gdp_gap,blocked_mortgages,excess_demand");
    for _ in 0..config.ticks {
        example.model.tick()?;
        let state = macro_state(&example.model)?;
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            state.quarter,
            state.aggregates.production,
            state.aggregates.ppi,
            state.aggregates.cpi,
            state.aggregates.hpi,
            state.aggregates.rpi,
            state.aggregates.total_loans,
            state.accounting.gdp.max_gap(),
            state.audit.mortgage_blocked_purchases,
            state.audit.goods_excess_demand
        );
    }
    let state = macro_state(&example.model)?;
    eprintln!(
        "completed {} quarters; paper coverage entries={} replication_blockers={} exact_replication_gaps={}",
        state.quarter,
        state.coverage.entries.len(),
        state.coverage.unresolved_replication_blockers().len(),
        EquationCoverage::blocker_log().len()
    );
    match config.gap_report {
        GapReportMode::Text => eprintln!("{}", EquationCoverage::gap_report_text()),
        GapReportMode::Json => eprintln!("{}", EquationCoverage::gap_report_json()),
        GapReportMode::None => {}
    }
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
            "--gap-report" => {
                let value = args
                    .next()
                    .ok_or("--gap-report requires text, json, or none")?;
                config.gap_report = match value.as_str() {
                    "text" => macroeconomy::GapReportMode::Text,
                    "json" => macroeconomy::GapReportMode::Json,
                    "none" => macroeconomy::GapReportMode::None,
                    _ => return Err("--gap-report requires text, json, or none".into()),
                };
            }
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run --release --features \"model messaging\" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 [--config path] [--gap-report text|json|none]"
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
