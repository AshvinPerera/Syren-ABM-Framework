use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use super::components::SECTORS;
use super::state::{GoodsClearingPolicy, HousingReductionPolicy, MacroEnvironment, PythonLikeRng};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    Io { path: PathBuf, message: String },
    Parse { line: usize, message: String },
    UnknownKey { line: usize, key: String },
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
        }
    }
}

impl Error for ConfigError {}

pub fn apply_config_file(
    path: impl AsRef<Path>,
    environment: &mut MacroEnvironment,
) -> Result<(), ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|err| ConfigError::Io {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    apply_config_str(&text, environment)
}

pub fn apply_config_str(text: &str, environment: &mut MacroEnvironment) -> Result<(), ConfigError> {
    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| ConfigError::Parse {
            line: line_number,
            message: "expected key=value".to_owned(),
        })?;
        apply_setting(key.trim(), value.trim(), line_number, environment)?;
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
        "seed" => {
            let seed = parse_u64(value, line)?;
            environment.seed = seed;
            environment.rng = PythonLikeRng::new(seed);
        }
        "quarter" => environment.quarter = parse_u64(value, line)?,
        "scale_factor" => environment.scale_factor = parse_f64(value, line)?,
        "car" => environment.params.car = parse_f64(value, line)?,
        "solvency_ratio" => environment.params.solvency_ratio = parse_f64(value, line)?,
        "debt_to_equity" => environment.params.debt_to_equity = parse_f64(value, line)?,
        "return_on_equity" => environment.params.return_on_equity = parse_f64(value, line)?,
        "return_on_assets" => environment.params.return_on_assets = parse_f64(value, line)?,
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
        "policy.allow_unresolved_blockers" => {
            environment.policy.allow_unresolved_blockers = parse_bool(value, line)?
        }
        "policy.housing_reduction_policy" => {
            environment.policy.housing_reduction_policy = match value {
                "literal" | "LiteralPaperFormula" => HousingReductionPolicy::LiteralPaperFormula,
                "guarded" | "GuardedFractionalReduction" => {
                    HousingReductionPolicy::GuardedFractionalReduction
                }
                _ => {
                    return Err(ConfigError::Parse {
                        line,
                        message: "expected literal or guarded".to_owned(),
                    });
                }
            }
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
