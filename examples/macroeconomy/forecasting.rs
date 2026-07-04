#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ar1Fit {
    pub intercept: f64,
    pub rho: f64,
    pub observations: usize,
    pub forecast_level: f64,
}

pub fn fit_ar1_log_level_forecast(history: &[f64], fallback: f64) -> Ar1Fit {
    let clean_logs: Vec<f64> = history
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::ln)
        .collect();
    if clean_logs.len() < 3 {
        return Ar1Fit {
            intercept: 0.0,
            rho: 1.0,
            observations: clean_logs.len(),
            forecast_level: fallback.max(0.0),
        };
    }

    let xs = &clean_logs[..clean_logs.len() - 1];
    let ys = &clean_logs[1..];
    let (intercept, rho) = ols_ar1(xs, ys);
    let last = *clean_logs.last().unwrap_or(&fallback.max(1e-9).ln());
    let forecast_log = intercept + rho * last;
    Ar1Fit {
        intercept,
        rho,
        observations: clean_logs.len(),
        forecast_level: forecast_log.exp().max(0.0),
    }
}

pub fn fit_ar1_level_forecast(history: &[f64], fallback: f64) -> Ar1Fit {
    let clean: Vec<f64> = history
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if clean.len() < 3 {
        return Ar1Fit {
            intercept: 0.0,
            rho: 1.0,
            observations: clean.len(),
            forecast_level: fallback.max(0.0),
        };
    }

    let (intercept, rho) = ols_ar1(&clean[..clean.len() - 1], &clean[1..]);
    let last = *clean.last().unwrap_or(&fallback);
    Ar1Fit {
        intercept,
        rho,
        observations: clean.len(),
        forecast_level: (intercept + rho * last).max(0.0),
    }
}

fn ols_ar1(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        sxx += (x - mean_x) * (x - mean_x);
        sxy += (x - mean_x) * (y - mean_y);
    }
    let rho = if sxx.abs() < f64::EPSILON {
        1.0
    } else {
        sxy / sxx
    };
    let intercept = mean_y - rho * mean_x;
    (intercept, rho)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TaylorRuleEstimate {
    pub alpha: f64,
    pub rho: f64,
    pub beta_pi: f64,
    pub beta_gamma: f64,
    pub pi_star: f64,
    pub natural_rate: f64,
    pub xi_pi: f64,
    pub xi_gamma: f64,
}

pub fn transform_taylor_rule(
    alpha: f64,
    rho: f64,
    beta_pi: f64,
    beta_gamma: f64,
    pi_star: f64,
) -> TaylorRuleEstimate {
    let denominator = (1.0 - rho).max(1e-9);
    let natural_rate = alpha / denominator - pi_star;
    TaylorRuleEstimate {
        alpha,
        rho,
        beta_pi,
        beta_gamma,
        pi_star,
        natural_rate,
        xi_pi: beta_pi / denominator,
        xi_gamma: beta_gamma / denominator,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArdlCandidate {
    pub p: usize,
    pub q: usize,
    pub r: usize,
    pub s: usize,
    pub aic: f64,
}

pub fn select_ardl_lag_by_aic(candidates: &[ArdlCandidate]) -> Option<ArdlCandidate> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.aic.is_finite())
        .min_by(|a, b| a.aic.total_cmp(&b.aic))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArdlErrorCorrectionInput<'a> {
    pub previous_loan_rate: f64,
    pub current_policy_rate: f64,
    pub error_correction_phi: f64,
    pub long_run_pass_through_phi: f64,
    pub lagged_loan_rate_deltas: &'a [f64],
    pub alpha: &'a [f64],
    pub lagged_policy_rate_deltas: &'a [f64],
    pub beta: &'a [f64],
    pub lagged_ppi_inflation_deltas: &'a [f64],
    pub gamma: &'a [f64],
    pub lagged_npl_ratio_deltas: &'a [f64],
    pub delta: &'a [f64],
    pub mu: f64,
}

pub fn ardl_error_correction_delta_rate(input: &ArdlErrorCorrectionInput<'_>) -> f64 {
    input.error_correction_phi
        * (input.previous_loan_rate - input.long_run_pass_through_phi * input.current_policy_rate)
        + weighted_sum(input.alpha, input.lagged_loan_rate_deltas)
        + weighted_sum(input.beta, input.lagged_policy_rate_deltas)
        + weighted_sum(input.gamma, input.lagged_ppi_inflation_deltas)
        + weighted_sum(input.delta, input.lagged_npl_ratio_deltas)
        + input.mu
}

fn weighted_sum(coefficients: &[f64], values: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(values.iter())
        .map(|(coefficient, value)| coefficient * value)
        .sum()
}
