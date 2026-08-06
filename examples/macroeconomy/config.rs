use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use super::components::SECTORS;
use super::state::{GoodsClearingPolicy, MacroEnvironment};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    Io { path: PathBuf, message: String },
    Parse { line: usize, message: String },
    UnknownKey { line: usize, key: String },
    UnknownScenario { name: String, available: Vec<String> },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "failed to read config {}: {message}", path.display())
            }
            Self::Parse { line, message } => write!(f, "invalid config at line {line}: {message}"),
            Self::UnknownKey { line, key } => {
                write!(f, "unknown config key `{key}` at line {line}")
            }
            Self::UnknownScenario { name, available } => {
                write!(
                    f,
                    "unknown scenario `{name}`; available: {}",
                    available.join(", ")
                )
            }
        }
    }
}

impl Error for ConfigError {}

/// Loads `config.yaml`, applying the `defaults` block and then, if named, one
/// block from `scenarios`.
///
/// Scenario settings are applied *after* the defaults, so a scenario only needs
/// to list the keys it changes.
pub fn apply_config_file(
    path: impl AsRef<Path>,
    scenario: Option<&str>,
    environment: &mut MacroEnvironment,
) -> Result<(), ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|err| ConfigError::Io {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    apply_config_str(&text, scenario, environment)
}

/// One `key: value` pair with the source line it came from.
type Setting = (usize, String, String);

/// Splits the document into top-level blocks.
///
/// This is a deliberately small YAML subset: two levels, scalar leaves,
/// `#` comments. It exists so the example carries a single readable config file
/// without adding a YAML crate (and with it serde) to a framework whose whole
/// dependency list is six entries. Anything outside the subset is rejected
/// rather than silently misread.
fn parse_blocks(text: &str) -> Result<Vec<(String, Vec<Setting>)>, ConfigError> {
    let mut blocks: Vec<(String, Vec<Setting>)> = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let without_comment = raw_line.split('#').next().unwrap_or("");
        if without_comment.trim().is_empty() {
            continue;
        }
        if without_comment.contains('\t') {
            return Err(ConfigError::Parse {
                line: line_number,
                message: "tabs are not valid YAML indentation; use spaces".to_owned(),
            });
        }
        let indent = without_comment.len() - without_comment.trim_start().len();
        if indent % 2 != 0 {
            return Err(ConfigError::Parse {
                line: line_number,
                message: format!("indent of {indent} spaces is not a multiple of 2"),
            });
        }
        let depth = indent / 2;
        if depth > stack.len() {
            return Err(ConfigError::Parse {
                line: line_number,
                message: "unexpected indentation".to_owned(),
            });
        }
        stack.truncate(depth);
        let trimmed = without_comment.trim();
        let (key, value) = trimmed.split_once(':').ok_or_else(|| ConfigError::Parse {
            line: line_number,
            message: "expected `key: value` or `key:`".to_owned(),
        })?;
        let key = key.trim().to_owned();
        let value = value.trim();
        if value.is_empty() {
            if depth > 1 {
                return Err(ConfigError::Parse {
                    line: line_number,
                    message: "nesting deeper than two levels is not supported".to_owned(),
                });
            }
            stack.push(key.clone());
            let path = stack.join(".");
            if !blocks.iter().any(|(name, _)| *name == path) {
                blocks.push((path, Vec::new()));
            }
            continue;
        }
        let path = stack.join(".");
        let value = value.trim_matches('"').trim_matches('\'').to_owned();
        match blocks.iter_mut().find(|(name, _)| *name == path) {
            Some((_, settings)) => settings.push((line_number, key, value)),
            None => blocks.push((path, vec![(line_number, key, value)])),
        }
    }
    Ok(blocks)
}

pub fn apply_config_str(
    text: &str,
    scenario: Option<&str>,
    environment: &mut MacroEnvironment,
) -> Result<(), ConfigError> {
    let blocks = parse_blocks(text)?;
    let apply_block = |name: &str, environment: &mut MacroEnvironment| -> Result<bool, ConfigError> {
        let Some((_, settings)) = blocks.iter().find(|(block, _)| block == name) else {
            return Ok(false);
        };
        for (line, key, value) in settings {
            apply_setting(key, value, *line, environment)?;
        }
        Ok(true)
    };
    apply_block("defaults", environment)?;
    if let Some(scenario) = scenario {
        let path = format!("scenarios.{scenario}");
        if !apply_block(&path, environment)? {
            return Err(ConfigError::UnknownScenario {
                name: scenario.to_owned(),
                available: blocks
                    .iter()
                    .filter_map(|(block, _)| block.strip_prefix("scenarios.").map(str::to_owned))
                    .collect(),
            });
        }
    }
    Ok(())
}

fn apply_setting(
    key: &str,
    value: &str,
    line: usize,
    environment: &mut MacroEnvironment,
) -> Result<(), ConfigError> {
    match key {
        // Draws are keyed on `seed` at the point of use (see `MacroRng`), so
        // there is no generator state to re-seed here.
        "seed" => environment.seed = parse_u64(value, line)?,
        "quarter" => environment.quarter = parse_u64(value, line)?,
        "scale_factor" => environment.scale_factor = parse_f64(value, line)?,
        "car" => environment.params.car = parse_f64(value, line)?,
        "solvency_ratio" => environment.params.solvency_ratio = parse_f64(value, line)?,
        "debt_to_equity" => environment.params.debt_to_equity = parse_f64(value, line)?,
        "return_on_equity" => environment.params.return_on_equity = parse_f64(value, line)?,
        "return_on_assets" => environment.params.return_on_assets = parse_f64(value, line)?,
        "wage_effort_on_base" => {
            environment.params.wage_effort_on_base = matches!(value, "1" | "true" | "yes")
        }
        "consumption_lti" => environment.params.consumption_lti = parse_f64(value, line)?,
        "mortgage_ltv" => environment.params.mortgage_ltv = parse_f64(value, line)?,
        "mortgage_lti" => environment.params.mortgage_lti = parse_f64(value, line)?,
        "mortgage_dsti" => environment.params.mortgage_dsti = parse_f64(value, line)?,
        "firm_input_adjustment" => {
            environment.params.firm_input_adjustment = parse_f64(value, line)?
        }
        "firm_capital_adjustment" => {
            environment.params.firm_capital_adjustment = parse_f64(value, line)?
        }
        "firm_credit_shortfall_intermediate_sensitivity" => {
            environment
                .params
                .firm_credit_shortfall_intermediate_sensitivity = parse_f64(value, line)?
        }
        "firm_credit_shortfall_capital_sensitivity" => {
            environment.params.firm_credit_shortfall_capital_sensitivity = parse_f64(value, line)?
        }
        "theta_dividend" => environment.params.theta_dividend = parse_f64(value, line)?,
        "work_effort_max" => environment.params.work_effort_max = parse_f64(value, line)?,
        "wage_tightness_sensitivity" => {
            environment.params.wage_tightness_sensitivity = parse_f64(value, line)?
        }
        "wage_tightness_window" => {
            environment.params.wage_tightness_window = parse_usize(value, line)?
        }
        "phi_consumption_history" => {
            environment.params.phi_consumption_history = parse_f64(value, line)?
        }
        "other_real_asset_depreciation_rate" => {
            environment.params.other_real_asset_depreciation_rate = parse_f64(value, line)?
        }
        "financial_asset_income_phi" => {
            environment.params.financial_asset_income_phi = parse_f64(value, line)?
        }
        "financial_asset_income_sigma" => {
            environment.params.financial_asset_income_sigma = parse_f64(value, line)?
        }
        "government_consumption_share" => {
            environment.params.government_consumption_share = parse_f64(value, line)?
        }
        "goods_market_phi" => environment.params.goods_market_phi = parse_f64(value, line)?,
        "credit_supply_phi" => environment.params.credit_supply_phi = parse_f64(value, line)?,
        "firm_short_maturity_quarters" => {
            environment.params.firm_short_maturity_quarters = parse_u32(value, line)?
        }
        "firm_long_maturity_quarters" => {
            environment.params.firm_long_maturity_quarters = parse_u32(value, line)?
        }
        "consumption_loan_maturity_quarters" => {
            environment.params.consumption_loan_maturity_quarters = parse_u32(value, line)?
        }
        "mortgage_maturity_quarters" => {
            environment.params.mortgage_maturity_quarters = parse_u32(value, line)?
        }
        "unemployment_growth_h" => {
            environment.params.unemployment_growth_h = parse_f64(value, line)?
        }
        "employed_growth_h" => environment.params.employed_growth_h = parse_f64(value, line)?,
        "housing_phi_hp" => environment.params.housing_phi_hp = parse_f64(value, line)?,
        "housing_beta_hp" => environment.params.housing_beta_hp = parse_f64(value, line)?,
        "housing_mu_hp" => environment.params.housing_mu_hp = parse_f64(value, line)?,
        "housing_sigma_hp" => environment.params.housing_sigma_hp = parse_f64(value, line)?,
        "housing_mu_ps" => environment.params.housing_mu_ps = parse_f64(value, line)?,
        "housing_phi_b" => environment.params.housing_phi_b = parse_f64(value, line)?,
        "housing_phi_hr" => environment.params.housing_phi_hr = parse_f64(value, line)?,
        "housing_beta_hr" => environment.params.housing_beta_hr = parse_f64(value, line)?,
        "renter_stay_probability" => {
            environment.params.renter_stay_probability = parse_f64(value, line)?
        }
        "owner_stay_probability" => {
            environment.params.owner_stay_probability = parse_f64(value, line)?
        }
        "sale_price_reduction_probability" => {
            environment.params.sale_price_reduction_probability = parse_f64(value, line)?
        }
        "sale_price_reduction_mu" => {
            environment.params.sale_price_reduction_mu = parse_f64(value, line)?
        }
        "sale_price_reduction_sigma" => {
            environment.params.sale_price_reduction_sigma = parse_f64(value, line)?
        }
        "rent_reduction_probability" => {
            environment.params.rent_reduction_probability = parse_f64(value, line)?
        }
        "rent_reduction_mu" => environment.params.rent_reduction_mu = parse_f64(value, line)?,
        "rent_reduction_sigma" => environment.params.rent_reduction_sigma = parse_f64(value, line)?,
        "rent_partial_indexation_phi" => {
            environment.params.rent_partial_indexation_phi = parse_f64(value, line)?
        }
        "rent_partial_indexation_lag" => {
            environment.params.rent_partial_indexation_lag = parse_usize(value, line)?
        }
        "cpi_weights" => environment.params.cpi_weights = parse_array(value, line)?,
        "government_consumption_weights" => {
            environment.params.government_consumption_weights = parse_array(value, line)?
        }
        "household_investment_weights" => {
            environment.params.household_investment_weights = parse_array(value, line)?
        }
        "row_export_weights" => environment.params.row_export_weights = parse_array(value, line)?,
        "row_import_weights" => environment.params.row_import_weights = parse_array(value, line)?,
        "capital_depreciation_rate_by_sector" => {
            environment.params.capital_depreciation_rate_by_sector = parse_array(value, line)?
        }
        "capital_installation_delay_quarters" => {
            environment.params.capital_installation_delay_quarters = parse_u32_array(value, line)?
        }
        "io_matrix" => environment.params.io_matrix = parse_matrix(value, line)?,
        "net_fixed_assets_matrix" => {
            environment.params.net_fixed_assets_matrix = parse_matrix(value, line)?
        }
        "capital_compensation_matrix" => {
            environment.params.capital_compensation_matrix = parse_matrix(value, line)?
        }
        "calibration.phi_f_q" => environment.calibration.phi_f_q = parse_f64(value, line)?,
        "calibration.phi_dp" => environment.calibration.phi_dp = parse_f64(value, line)?,
        "calibration.phi_cp" => environment.calibration.phi_cp = parse_f64(value, line)?,
        "calibration.phi_st_y" => environment.calibration.phi_st_y = parse_f64(value, line)?,
        "calibration.chi_h" => environment.calibration.chi_h = parse_f64(value, line)?,
        "calibration.chi_m" => environment.calibration.chi_m = parse_f64(value, line)?,
        "calibration.chi_k" => environment.calibration.chi_k = parse_f64(value, line)?,
        "policy.firm_bank_visits" => environment.policy.firm_bank_visits = parse_u32(value, line)?,
        "policy.household_bank_visits" => {
            environment.policy.household_bank_visits = parse_u32(value, line)?
        }
        "policy.goods_clearing_policy" => {
            environment.policy.goods_clearing_policy = match value {
                "poledna_search"
                | "PolednaSearchAndMatching"
                | "proportional_fixture"
                | "ProportionalFixtureFlows" => GoodsClearingPolicy::PolednaSearchAndMatching,
                _ => {
                    return Err(ConfigError::Parse {
                        line,
                        message: "expected poledna_search".to_owned(),
                    });
                }
            }
        }
        _ => {
            return Err(ConfigError::UnknownKey {
                line,
                key: key.to_owned(),
            });
        }
    }
    Ok(())
}

fn parse_f64(value: &str, line: usize) -> Result<f64, ConfigError> {
    value.parse::<f64>().map_err(|err| ConfigError::Parse {
        line,
        message: err.to_string(),
    })
}

fn parse_u32(value: &str, line: usize) -> Result<u32, ConfigError> {
    value.parse::<u32>().map_err(|err| ConfigError::Parse {
        line,
        message: err.to_string(),
    })
}

fn parse_u64(value: &str, line: usize) -> Result<u64, ConfigError> {
    value.parse::<u64>().map_err(|err| ConfigError::Parse {
        line,
        message: err.to_string(),
    })
}

fn parse_usize(value: &str, line: usize) -> Result<usize, ConfigError> {
    value.parse::<usize>().map_err(|err| ConfigError::Parse {
        line,
        message: err.to_string(),
    })
}

fn parse_bool(value: &str, line: usize) -> Result<bool, ConfigError> {
    value.parse::<bool>().map_err(|err| ConfigError::Parse {
        line,
        message: err.to_string(),
    })
}

fn parse_array(value: &str, line: usize) -> Result<[f64; SECTORS], ConfigError> {
    let values = parse_f64_list(value, line)?;
    values
        .try_into()
        .map_err(|values: Vec<f64>| ConfigError::Parse {
            line,
            message: format!("expected {SECTORS} values, got {}", values.len()),
        })
}

fn parse_u32_array(value: &str, line: usize) -> Result<[u32; SECTORS], ConfigError> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>().map_err(|err| ConfigError::Parse {
                line,
                message: err.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|values: Vec<u32>| ConfigError::Parse {
            line,
            message: format!("expected {SECTORS} values, got {}", values.len()),
        })
}

fn parse_matrix(value: &str, line: usize) -> Result<[[f64; SECTORS]; SECTORS], ConfigError> {
    let values = parse_f64_list(value, line)?;
    if values.len() != SECTORS * SECTORS {
        return Err(ConfigError::Parse {
            line,
            message: format!(
                "expected {} values, got {}",
                SECTORS * SECTORS,
                values.len()
            ),
        });
    }
    let mut matrix = [[0.0; SECTORS]; SECTORS];
    for row in 0..SECTORS {
        for col in 0..SECTORS {
            matrix[row][col] = values[row * SECTORS + col];
        }
    }
    Ok(matrix)
}

fn parse_f64_list(value: &str, line: usize) -> Result<Vec<f64>, ConfigError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>().map_err(|err| ConfigError::Parse {
                line,
                message: err.to_string(),
            })
        })
        .collect()
}
