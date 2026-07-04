#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationParameters {
    pub phi_f_q: f64,
    pub phi_dp: f64,
    pub phi_cp: f64,
    pub phi_st_y: f64,
    pub chi_h: f64,
    pub chi_m: f64,
    pub chi_k: f64,
}

impl CalibrationParameters {
    pub fn austria_npe_table4() -> Self {
        Self {
            phi_f_q: 0.0,
            phi_dp: 0.0,
            phi_cp: 0.0,
            phi_st_y: 0.10,
            chi_h: 0.53,
            chi_m: 0.03,
            chi_k: 0.18,
        }
    }

    pub fn binary_search_combinations() -> Vec<(f64, f64, f64)> {
        let mut rows = Vec::with_capacity(8);
        for phi_f_q in [0.0, 1.0] {
            for phi_dp in [0.0, 1.0] {
                for phi_cp in [0.0, 1.0] {
                    rows.push((phi_f_q, phi_dp, phi_cp));
                }
            }
        }
        rows
    }
}

impl Default for CalibrationParameters {
    fn default() -> Self {
        Self::austria_npe_table4()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuralPosteriorConfig {
    pub method: &'static str,
    pub flow_or_classifier: &'static str,
    pub stacks_or_layers: usize,
    pub blocks: usize,
    pub hidden_features: usize,
    pub learning_rate: f64,
    pub validation_fraction: f64,
    pub early_stop_patience: usize,
}

impl NeuralPosteriorConfig {
    pub fn npe() -> Self {
        Self {
            method: "NPE",
            flow_or_classifier: "Masked Autoregressive Flow with MADE",
            stacks_or_layers: 5,
            blocks: 2,
            hidden_features: 50,
            learning_rate: 5e-4,
            validation_fraction: 0.10,
            early_stop_patience: 20,
        }
    }

    pub fn nre() -> Self {
        Self {
            method: "NRE",
            flow_or_classifier: "ResNet classifier",
            stacks_or_layers: 2,
            blocks: 0,
            hidden_features: 50,
            learning_rate: 5e-4,
            validation_fraction: 0.10,
            early_stop_patience: 20,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BayesFactorConfig {
    pub hidden_sizes: [usize; 4],
    pub activation: &'static str,
    pub output: &'static str,
    pub input_features: usize,
    pub learning_rate: f64,
    pub validation_fraction: f64,
    pub max_epochs: usize,
    pub early_stop_patience: usize,
}

impl Default for BayesFactorConfig {
    fn default() -> Self {
        Self {
            hidden_sizes: [32, 32, 32, 16],
            activation: "ReLU",
            output: "scalar log Bayes factor",
            input_features: 5,
            learning_rate: 1e-3,
            validation_fraction: 0.20,
            max_epochs: 500,
            early_stop_patience: 50,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForecastExperimentConfig {
    pub countries: usize,
    pub initialisation_quarters: usize,
    pub horizon_quarters: usize,
    pub trajectories: usize,
}

impl Default for ForecastExperimentConfig {
    fn default() -> Self {
        Self {
            countries: 38,
            initialisation_quarters: 20,
            horizon_quarters: 12,
            trajectories: 1_000,
        }
    }
}
