/// The model's seven free parameters, set to the paper's Austria values.
///
/// These are the only parameters the paper calibrates; everything else in
/// `CountryParameters` is pinned by the source (Appendix A.3.2, A.5.1, A.9.1,
/// A.10.1, A.11.1). Setting these manually is therefore a complete substitute
/// for running the calibration pipeline -- no NPE/NRE machinery is needed.
///
/// Values are Table 4 (paper §4.3), NPE posterior for Austria, 1990-Q1 to
/// 2013-Q1. Override any of them through the `calibration.*` config keys.
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
            // The first three are *binary* (prior U({0,1})) and are not
            // estimated by NPE/NRE. The authors ran all eight combinations of
            // (phi_f_q, phi_dp, phi_cp) = (+/-1, +/-1, +/-1) and selected the
            // all-zero configuration as the one minimising forecast error --
            // and it is zero for all 38 countries under both NPE and NRE, not
            // just Austria (paper Sections 4.3 and 5).
            //
            // Zero here is the paper's answer, not a missing value. It switches
            // off the firm-specific terms: A.60 reduces to
            // `Qbar_f = (1 + gamma_s) * Q_f(t-1)` and A.73 to
            // `P_f = (1 + pi^PPI) * P_f(t-1)`, so growth and inflation are
            // driven by the sectoral and PPI forecasts alone. As a consequence
            // A.59, A.74 and A.76 are all multiplied out of the trajectory, and
            // unit cost A.77 becomes an accounting quantity rather than a
            // driver of prices.
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
