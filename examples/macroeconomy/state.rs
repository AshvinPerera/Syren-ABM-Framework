use std::path::PathBuf;

use syren::{DetRng, RunContext};

use super::accounting::{AccountingReport, GdpIdentity};
use super::calibration::CalibrationParameters;
use super::components::{SECTORS, SIM_SCALE_FACTOR};

pub const MACRO_ENV_KEY: &str = "macro_state";
pub const PHASE_AGGREGATE_DONE: &str = "phase_aggregate_done";
pub const PHASE_EXPECTATIONS_DONE: &str = "phase_expectations_done";
pub const PHASE_TARGETS_DONE: &str = "phase_targets_done";
pub const PHASE_LABOUR_DONE: &str = "phase_labour_done";
pub const PHASE_PLANNING_DONE: &str = "phase_planning_done";
pub const PHASE_HOUSING_PRECLEAR_DONE: &str = "phase_housing_preclear_done";
pub const PHASE_CREDIT_DONE: &str = "phase_credit_done";
pub const PHASE_HOUSING_COMPLETION_DONE: &str = "phase_housing_completion_done";
pub const PHASE_GOODS_DONE: &str = "phase_goods_done";
pub const PHASE_ACCOUNTING_DONE: &str = "phase_accounting_done";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunMode {
    TinyFixture,
    RealData,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroeconomyConfig {
    pub mode: RunMode,
    pub ticks: u64,
    pub seed: u64,
    pub country: String,
    pub initialisation: String,
    pub data_dir: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    /// Named block under `scenarios:` in the config file, applied after
    /// `defaults:`.
    pub scenario: Option<String>,
    /// Where to write a Chrome Trace profile, if `--profile` was given.
    pub profile_path: Option<PathBuf>,
    /// Directory for the `--trace` CSVs. The example never writes them to the
    /// working directory: a diagnostic dump should not land in whatever
    /// repository the run happened to start from.
    pub trace_dir: Option<PathBuf>,
    pub policy: ModelPolicy,
    /// Firms per sector. The paper runs Austria at 1:1000, giving ~600 firms
    /// over 18 sectors -- about 33 each. The default of 1 is a test fixture.
    pub firms_per_sector: u32,
}

impl Default for MacroeconomyConfig {
    fn default() -> Self {
        Self {
            mode: RunMode::TinyFixture,
            ticks: 8,
            seed: 42,
            country: "TST".to_owned(),
            initialisation: "2013-Q1".to_owned(),
            data_dir: None,
            config_path: None,
            scenario: None,
            profile_path: None,
            trace_dir: None,
            policy: ModelPolicy::default(),
            firms_per_sector: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoodsClearingPolicy {
    PolednaSearchAndMatching,
}

/// Salts distinguishing the draw sites of different systems within one tick.
///
/// `DetRng` keys on `(simulation_seed, tick, system_id, salt)`. The system id
/// already separates systems, so these exist to keep independent draw sites
/// *within* a system on separate streams.
pub mod rng_salt {
    pub const LABOUR_MARKET: u64 = 0x1A_B0_75;
    pub const PLANNING: u64 = 0x21_A4_44;
    pub const HOUSING_PRECLEAR: u64 = 0x40_57_11;
    pub const CREDIT_MARKET: u64 = 0xC2_ED_17;
    pub const GOODS_MARKET: u64 = 0x60_0D_50;
    /// Per-property draws in A.113/A.115, keyed on the property.
    pub const PROPERTY_REPRICE: u64 = 0x9E_11_C3;
    /// Per-household draws for A.104's financial-asset income noise.
    pub const HOUSEHOLD_ASSET_INCOME: u64 = 0xFA_1C_09;
}

/// Model-facing wrapper over the framework's [`DetRng`].
///
/// `DetRng` supplies uniform draws; the distributions the model actually needs
/// (Bernoulli, Gaussian, unbiased shuffle) live here.
///
/// Constructed per system per tick rather than carried in [`MacroEnvironment`].
/// The previous MT19937 stream lived inside the environment blob, so it was
/// deep-cloned on every system entry and written back wholesale — which meant
/// two systems could not draw concurrently without losing draws, forcing every
/// market loop to be single-threaded by construction.
#[derive(Clone, Copy, Debug)]
pub struct MacroRng {
    inner: DetRng,
}

impl MacroRng {
    /// Keys a stream on the run context, the model's own seed, and a salt.
    ///
    /// The model seed is folded in explicitly because `ModelBuilder` exposes no
    /// `with_seed`, leaving `RunContext::simulation_seed` permanently zero
    /// (`src/engine/activation.rs:103`). Without this fold every seed would
    /// produce an identical trajectory.
    pub fn new(context: RunContext, model_seed: u64, salt: u64) -> Self {
        Self {
            inner: DetRng::from_context(context, salt ^ model_seed.rotate_left(17)),
        }
    }

    /// Keys a stream on an agent id, so each agent draws independently of the
    /// order in which agents happen to be visited.
    pub fn for_agent(context: RunContext, model_seed: u64, salt: u64, agent_id: u64) -> Self {
        Self::new(
            context,
            model_seed,
            salt ^ agent_id.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        )
    }

    pub fn unit_f64(&mut self) -> f64 {
        self.inner.next_f64()
    }

    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    pub fn below(&mut self, upper: usize) -> usize {
        self.inner.next_index(upper)
    }

    pub fn bernoulli(&mut self, probability: f64) -> bool {
        self.unit_f64() < probability.clamp(0.0, 1.0)
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i + 1);
            values.swap(i, j);
        }
    }

    /// Box-Muller normal draw.
    pub fn normal_f64(&mut self, mean: f64, sigma: f64) -> f64 {
        if sigma == 0.0 {
            return mean;
        }
        let u1 = self.unit_f64().clamp(f64::MIN_POSITIVE, 1.0);
        let u2 = self.unit_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + sigma * z0
    }
}

/// Behavioural switches that change how markets clear.
///
/// This is not a replication policy -- the example reproduces the model's
/// mechanics, not the authors' trajectories. Each variant is a modelling choice
/// with different cost/fidelity trade-offs at scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPolicy {
    pub goods_clearing_policy: GoodsClearingPolicy,
    pub firm_bank_visits: u32,
    pub household_bank_visits: u32,
    /// When set, the production system records this firm's internals into
    /// [`MarketAudit::firm_probe`] each quarter. Diagnostic only; it does not
    /// affect the trajectory.
    pub debug_firm_id: Option<u32>,
    /// Capture every firm's state each quarter into `MarketAudit::firm_trace`.
    pub trace: bool,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            goods_clearing_policy: GoodsClearingPolicy::PolednaSearchAndMatching,
            firm_bank_visits: 2,
            household_bank_visits: 2,
            debug_firm_id: None,
            trace: false,
        }
    }
}

/// One firm's state at the moment A.72 is evaluated. The CSV aggregates are too
/// coarse to localise a feedback loop; this is the per-firm view.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FirmProbe {
    pub id: u32,
    pub employees: u32,
    pub work_effort: f64,
    pub initial_work_effort: f64,
    /// `H_f` from A.65.
    pub labour: f64,
    /// `M_f` from A.63.
    pub intermediate_constraint: f64,
    /// `K_f` from A.64.
    pub capital_constraint: f64,
    /// `Y_hat_f` from A.62.
    pub target_production: f64,
    /// `Y_f` from A.72.
    pub production: f64,
    pub price: f64,
    pub unit_cost: f64,
    pub demand: f64,
    pub excess_demand: f64,
    pub wage: f64,
    pub deposits: f64,
    pub debt: f64,
    pub equity: f64,
    pub profits: f64,
    pub sales_quantity: f64,
    pub inventory: f64,
    pub target_short_loan: f64,
    pub granted_short_loan: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CountryParameters {
    pub io_matrix: [[f64; SECTORS]; SECTORS],
    pub net_fixed_assets_matrix: [[f64; SECTORS]; SECTORS],
    pub capital_compensation_matrix: [[f64; SECTORS]; SECTORS],
    pub capital_depreciation_rate_by_sector: [f64; SECTORS],
    pub capital_installation_delay_quarters: [u32; SECTORS],
    pub cpi_weights: [f64; SECTORS],
    pub government_consumption_weights: [f64; SECTORS],
    pub household_investment_weights: [f64; SECTORS],
    pub row_export_weights: [f64; SECTORS],
    pub row_import_weights: [f64; SECTORS],
    /// A.24 ARDL error-correction coefficients for bank interest rates.
    pub ardl_error_correction_phi: f64,
    pub ardl_long_run_pass_through: f64,
    pub ardl_policy_beta: f64,
    pub ardl_inflation_gamma: f64,
    pub ardl_npl_delta: f64,
    /// `theta^DIV`: the dividend payout ratio (Poledna A.27/A.33/A.53).
    ///
    /// **Deviation from Wiese, adopted deliberately.** Wiese's A.80 drops the
    /// dividend term Poledna carries, so firm profits accumulate in deposits
    /// and never return to households as income. Restored at Poledna's Austrian
    /// value (Table 1: 0.7953).
    ///
    /// Set `theta_dividend=0.0` in a config file to recover Wiese exactly.
    /// See `docs/deviations.md`.
    pub theta_dividend: f64,
    pub firm_input_adjustment: f64,
    pub firm_capital_adjustment: f64,
    pub firm_credit_shortfall_intermediate_sensitivity: f64,
    pub firm_credit_shortfall_capital_sensitivity: f64,
    pub work_effort_max: f64,
    pub wage_tightness_sensitivity: f64,
    pub wage_tightness_window: usize,
    pub phi_consumption_history: f64,
    pub other_real_asset_depreciation_rate: f64,
    pub financial_asset_income_phi: f64,
    pub financial_asset_income_sigma: f64,
    pub government_consumption_share: f64,
    pub goods_market_phi: f64,
    pub credit_supply_phi: f64,
    pub car: f64,
    pub solvency_ratio: f64,
    pub debt_to_equity: f64,
    pub return_on_equity: f64,
    pub return_on_assets: f64,
    /// Apply the A.66 work-effort factor to a base wage (Poledna A.26) instead
    /// of to the previous wage (Wiese A.69).
    ///
    /// **Deviation from Wiese, adopted deliberately.** `phi^WE` is a level --
    /// input capacity over labour, capped at `h^max` -- but A.69 applies it as
    /// a growth factor to `w_i(t-1)`, so sustained overtime compounds the wage
    /// geometrically (measured: 10-24% a quarter). Poledna A.26, which Wiese
    /// cites as the source, applies the same factor to a base wage.
    ///
    /// Set `wage_effort_on_base=false` in a config file to recover Wiese
    /// exactly. See `docs/deviations.md`.
    pub wage_effort_on_base: bool,
    pub consumption_lti: f64,
    pub mortgage_ltv: f64,
    pub mortgage_lti: f64,
    pub mortgage_dsti: f64,
    pub firm_short_maturity_quarters: u32,
    pub firm_long_maturity_quarters: u32,
    pub consumption_loan_maturity_quarters: u32,
    pub mortgage_maturity_quarters: u32,
    pub unemployment_growth_h: f64,
    pub employed_growth_h: f64,
    pub housing_phi_hp: f64,
    pub housing_beta_hp: f64,
    pub housing_mu_hp: f64,
    pub housing_sigma_hp: f64,
    pub housing_mu_ps: f64,
    pub housing_phi_b: f64,
    pub housing_phi_hr: f64,
    pub housing_beta_hr: f64,
    pub renter_stay_probability: f64,
    pub owner_stay_probability: f64,
    pub sale_price_reduction_probability: f64,
    pub sale_price_reduction_mu: f64,
    pub sale_price_reduction_sigma: f64,
    pub rent_reduction_probability: f64,
    pub rent_reduction_mu: f64,
    pub rent_reduction_sigma: f64,
    pub rent_partial_indexation_phi: f64,
    pub rent_partial_indexation_lag: usize,
}

/// Builds a banded sector-linkage matrix with a prescribed row sum.
///
/// Row `s` puts `diagonal` on its own sector and spreads `off_diagonal_total`
/// over the others with weight decaying in the circular distance between
/// sectors, so nearby sectors trade more than distant ones.
///
/// These matrices come from OECD ICIO tables when real data is available. With
/// none, the previous default set only the diagonal, which made the 18 sectors
/// 18 mutually independent economies: no intermediate-input spillover, no
/// cross-sector investment demand, and therefore no propagation of a shock in
/// one sector to any other.
///
/// **The row sum is held at the previous diagonal-only value on purpose.** The
/// row sum is the aggregate input intensity of a sector, and the firm balance
/// sheets in the fixture were built around it -- raising it (0.10 to 0.45 in a
/// first attempt) pushes `unit_cost` above `price`, and since eq 6.73 feeds
/// `cost_push = unit_cost / price - 1` back into price, that closes a positive
/// loop with gain above one and the price level diverges within four quarters.
/// Only the *distribution* across sectors changes here; setting intensity and
/// balance sheets consistently is the population generator's job.
fn banded_sector_matrix(diagonal: f64, off_diagonal_total: f64) -> [[f64; SECTORS]; SECTORS] {
    let mut matrix = [[0.0; SECTORS]; SECTORS];
    // `s` selects the matrix row and drives the circular band arithmetic below,
    // so a range loop is clearer than indexing a single iterator.
    #[allow(clippy::needless_range_loop)]
    for s in 0..SECTORS {
        let mut weights = [0.0; SECTORS];
        let mut weight_sum = 0.0;
        for (t, weight) in weights.iter_mut().enumerate() {
            if t == s {
                continue;
            }
            let forward = (t + SECTORS - s) % SECTORS;
            let circular_distance = forward.min(SECTORS - forward) as f64;
            *weight = 1.0 / (1.0 + circular_distance);
            weight_sum += *weight;
        }
        matrix[s][s] = diagonal;
        if weight_sum > 0.0 {
            for t in 0..SECTORS {
                if t != s {
                    matrix[s][t] = off_diagonal_total * weights[t] / weight_sum;
                }
            }
        }
    }
    matrix
}

impl Default for CountryParameters {
    fn default() -> Self {
        // Row sums preserved from the diagonal-only defaults: 0.10 and 0.03.
        let io_matrix = banded_sector_matrix(0.06, 0.04);
        // `k_{s's}` and `d_{s's}` are different quantities and must not share a
        // magnitude. `k` is the capital *stock* needed per unit of output
        // (A.56, A.64); `d` is the capital *consumed* per unit of output
        // (A.79, A.87, A.89). A capital-output ratio of about one year of
        // production gives `k` a row sum near 4.0 at quarterly output, and
        // depreciation of roughly 6% of the stock per year gives `d` a row sum
        // near 0.06.
        //
        // The two matrices are distinct for that reason. Giving them the same
        // small row sum would leave every firm holding barely one quarter of
        // its own depreciation, and since A.87 consumes capital in proportion
        // to production the stock would drain inside a quarter and A.64 would
        // take output to zero.
        let net_fixed_assets_matrix = banded_sector_matrix(2.4, 1.6);
        let capital_compensation_matrix = banded_sector_matrix(0.036, 0.024);
        Self {
            io_matrix,
            net_fixed_assets_matrix,
            capital_compensation_matrix,
            // ~10%/yr. At 0.0 the capital evolution equation reduced to pure
            // accumulation and capital never wore out, so replacement
            // investment demand did not exist.
            capital_depreciation_rate_by_sector: [0.025; SECTORS],
            capital_installation_delay_quarters: [1; SECTORS],
            cpi_weights: [1.0 / SECTORS as f64; SECTORS],
            government_consumption_weights: [1.0 / SECTORS as f64; SECTORS],
            household_investment_weights: [1.0 / SECTORS as f64; SECTORS],
            row_export_weights: [1.0 / SECTORS as f64; SECTORS],
            row_import_weights: [1.0 / SECTORS as f64; SECTORS],
            // Error correction pulls the loan rate toward `phi^LR * r(t)`; the
            // long-run pass-through exceeds one because lending rates sit above
            // the policy rate, as Leroy & Lucotte (2016) find for the euro area.
            // Signs follow A.24.
            ardl_error_correction_phi: -0.25,
            ardl_long_run_pass_through: 1.4,
            ardl_policy_beta: 0.6,
            ardl_inflation_gamma: 0.2,
            ardl_npl_delta: 0.5,
            theta_dividend: 0.7953,
            firm_input_adjustment: 0.5,
            firm_capital_adjustment: 0.25,
            firm_credit_shortfall_intermediate_sensitivity: 0.0,
            firm_credit_shortfall_capital_sensitivity: 0.0,
            work_effort_max: 1.5,
            // phi_W. At 0.0 the wage rule degenerated to pure price indexation,
            // removing the labour-market tightness channel entirely -- no
            // Phillips-curve relationship could emerge.
            // phi^WN. A.5.1 is explicit: "we assume no wage mark-ups when firms
            // fail to meet the labour targets, phi^WN = 0". A.70 is therefore
            // inert by the paper's own choice, and the previous default of 0.5
            // described a Phillips-curve channel Table 4 was never calibrated
            // against.
            wage_tightness_sensitivity: 0.0,
            wage_tightness_window: 8,
            phi_consumption_history: 1.0,
            other_real_asset_depreciation_rate: 0.05,
            financial_asset_income_phi: 0.01,
            financial_asset_income_sigma: 0.0,
            government_consumption_share: 0.08,
            goods_market_phi: 2.0,
            credit_supply_phi: 2.0,
            car: 0.08,
            solvency_ratio: 0.10,
            debt_to_equity: 1.0,
            return_on_equity: 0.15,
            return_on_assets: 0.05,
            wage_effort_on_base: true,
            consumption_lti: 0.36,
            mortgage_ltv: 0.80,
            mortgage_lti: 4.5,
            mortgage_dsti: 0.35,
            firm_short_maturity_quarters: 1,
            firm_long_maturity_quarters: 8,
            consumption_loan_maturity_quarters: 1,
            mortgage_maturity_quarters: 100,
            unemployment_growth_h: 0.0,
            employed_growth_h: 0.0,
            housing_phi_hp: 42.90,
            housing_beta_hp: 0.79,
            housing_mu_hp: -0.018,
            // A.107's noise. `normal_f64` takes a *standard deviation*, and
            // Carro et al. (2023) Table 10 gives `eps_sigma = 0.4104` for this
            // draw. Wiese quotes `sigma^HP = 0.1684`, which is 0.4104 squared
            // -- the variance. Used as an s.d. it understated the dispersion of
            // desired purchase prices by a factor of about 2.4. Every other
            // Carro parameter transfers exactly (phi^HP 42.9036, beta^HP 0.7892,
            // mu^HP -0.0177), so this is a transcription slip, not a
            // re-estimation.
            housing_sigma_hp: 0.4104,
            housing_mu_ps: 0.4,
            housing_phi_b: 0.001,
            housing_phi_hr: 17.22,
            housing_beta_hr: 0.35,
            renter_stay_probability: 7.0 / 8.0,
            owner_stay_probability: 79.0 / 80.0,
            sale_price_reduction_probability: 0.1964,
            sale_price_reduction_mu: 1.4531,
            // Carro Table 11 gives 0.7070 for the sale-price reduction s.d.;
            // Wiese quotes 0.4889 and the two do not reconcile (0.7070^2 =
            // 0.4998, so it is not the same variance/s.d. confusion as
            // sigma^HP). Every other value in that chain matches exactly,
            // including both reduction probabilities once converted from Carro's
            // monthly to Wiese's quarterly frequency. Carro is the primary
            // source, so we take 0.7070 and record the deviation.
            sale_price_reduction_sigma: 0.7070,
            rent_reduction_probability: 0.2848,
            rent_reduction_mu: 1.6559,
            rent_reduction_sigma: 0.7855,
            rent_partial_indexation_phi: 1.0,
            rent_partial_indexation_lag: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroAggregates {
    pub production: f64,
    pub sector_production: [f64; SECTORS],
    pub ppi: f64,
    pub cpi: f64,
    pub hpi: f64,
    pub rpi: f64,
    pub total_loans: f64,
    pub firm_loans_by_sector: [f64; SECTORS],
    pub consumption_loans: f64,
    pub mortgages: f64,
    pub firm_npl_by_sector: [f64; SECTORS],
    pub consumption_npl: f64,
    pub mortgage_npl: f64,
    pub imports_nominal: f64,
    pub imports_real: f64,
    pub exports: f64,
    pub household_consumption: f64,
    pub government_consumption: f64,
    pub investment: f64,
    pub wage_income: f64,
    pub profit_income: f64,
    pub gdp: GdpIdentity,
}

impl Default for MacroAggregates {
    fn default() -> Self {
        Self {
            production: 0.0,
            sector_production: [0.0; SECTORS],
            ppi: 1.0,
            cpi: 1.0,
            hpi: 1.0,
            rpi: 1.0,
            total_loans: 0.0,
            firm_loans_by_sector: [0.0; SECTORS],
            consumption_loans: 0.0,
            mortgages: 0.0,
            firm_npl_by_sector: [0.0; SECTORS],
            consumption_npl: 0.0,
            mortgage_npl: 0.0,
            imports_nominal: 0.0,
            imports_real: 0.0,
            exports: 0.0,
            household_consumption: 0.0,
            government_consumption: 0.0,
            investment: 0.0,
            wage_income: 0.0,
            profit_income: 0.0,
            gdp: GdpIdentity::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastState {
    pub predicted_growth: f64,
    pub predicted_sector_growth: [f64; SECTORS],
    pub predicted_ppi_inflation: f64,
    pub predicted_cpi_inflation: f64,
    pub predicted_hpi_inflation: f64,
    pub predicted_rpi_inflation: f64,
    pub predicted_ppi: f64,
    pub predicted_cpi: f64,
    pub predicted_hpi: f64,
    pub predicted_rpi: f64,
    /// `C_hat^CG(t)` from A.95's AR(1) on realised government consumption.
    pub predicted_government_consumption: f64,
    pub ar1_observations: usize,
}

impl Default for ForecastState {
    fn default() -> Self {
        Self {
            predicted_growth: 0.0,
            predicted_sector_growth: [0.0; SECTORS],
            predicted_ppi_inflation: 0.0,
            predicted_cpi_inflation: 0.0,
            predicted_hpi_inflation: 0.0,
            predicted_rpi_inflation: 0.0,
            predicted_ppi: 1.0,
            predicted_cpi: 1.0,
            predicted_hpi: 1.0,
            predicted_rpi: 1.0,
            predicted_government_consumption: 0.0,
            ar1_observations: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroHistory {
    pub first_real_data_quarter: &'static str,
    pub production: Vec<f64>,
    pub ppi: Vec<f64>,
    pub cpi: Vec<f64>,
    pub hpi: Vec<f64>,
    pub rpi: Vec<f64>,
    pub sector_production: Vec<[f64; SECTORS]>,
    /// Realised total government consumption, for A.95's AR(1).
    ///
    /// The model produces this series itself, so the AR(1) the paper specifies
    /// needs no external input -- which is what made a fixed share of output a
    /// defect rather than a replication blocker.
    pub government_consumption: Vec<f64>,
}

impl Default for MacroHistory {
    fn default() -> Self {
        Self {
            first_real_data_quarter: "2000-Q1",
            production: Vec::new(),
            ppi: Vec::new(),
            cpi: Vec::new(),
            hpi: Vec::new(),
            rpi: Vec::new(),
            sector_production: Vec::new(),
            government_consumption: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MarketAudit {
    pub labour_fired_before_hiring: bool,
    pub credit_clearing_order: Vec<u8>,
    pub credit_visits_ordered_by_rate: bool,
    pub mortgage_blocked_purchases: u32,
    pub lower_price_seller_priority_seen: bool,
    /// Quantity of demand left unserved after clearing, summed over the
    /// buyers of **this quarter**. Reset by the goods market each tick.
    pub goods_excess_demand: f64,
    pub last_expectation_fit_observations: usize,
    pub phase_log: Vec<String>,
    /// Largest amount by which any firm's production exceeded its labour inputs
    /// `H_f` this quarter. A.72 is `min(Y_hat, H_f, M_f, K_f)`, so any positive
    /// value means the labour leg is not binding and the economy is producing
    /// above the ceiling `h^max * h_f(0) * sum_i H_i`.
    pub max_production_over_labour: f64,
    /// Largest amount by which any firm's `H_f` exceeded `min(M_f, K_f)`.
    /// Substituting A.66 into A.65 gives
    /// `H_f = min(h^max * h_f(0) * sum_i H_i, min(M_f, K_f))`, so labour weakly
    /// dominates the material constraints and this must stay at zero.
    pub max_labour_over_materials: f64,
    /// Employees booked by firms, against the individual population. The labour
    /// market draws hires from an unemployed pool, so the first must never
    /// exceed the second.
    pub employed_headcount: u64,
    pub individual_headcount: u64,
    /// Populated when [`ModelPolicy::debug_firm_id`] is set.
    pub firm_probe: Option<FirmProbe>,
    /// Total real quantity buyers asked for on the goods market this quarter,
    /// against the total available for sale. A synthetic initial state that is
    /// stock-flow consistent should open with these roughly equal; a large
    /// standing gap means the expenditure side of the SAM does not add up to
    /// the output side, and firms will accumulate inventory and cut output
    /// through A.62's first term regardless of how the rest of the model
    /// behaves.
    pub goods_demand_quantity: f64,
    /// A.60 propagates `Q_f(t-1)` multiplicatively, so any gap between realised
    /// demand and production at t=1 compounds forever. These break the demand
    /// side down by buyer so the gap can be attributed to a component.
    pub demand_firm_intermediate: f64,
    pub demand_firm_capital: f64,
    pub demand_household_consumption: f64,
    pub demand_household_capital: f64,
    pub demand_government: f64,
    pub demand_export: f64,
    pub goods_supply_quantity: f64,
    /// A.27 return-on-assets screen. It gates every firm loan, so if
    /// `firm_credit_roa_failures` tracks `firm_credit_applications` the firm
    /// sector cannot borrow at all, whatever the A.25/A.26 caps say.
    /// `firm_roa_max` is the best ratio any applicant achieved: compare it
    /// against `rho^RoA` to see whether the threshold is reachable.
    pub firm_credit_applications: u32,
    pub firm_credit_roa_failures: u32,
    pub firm_credit_requested: f64,
    pub firm_credit_granted: f64,
    pub firm_roa_max: f64,
    /// A.90's three terms, summed over firms:
    /// `Pi_f = P_f*Q~_f + P_f*dS_f - C_f`. The equation is implemented
    /// correctly; these exist to show which of its *inputs* is moving when
    /// profit falls, since "not selling" and "costs too high" need opposite
    /// fixes. The cost fields break A.89 down the same way.
    pub profit_sales_revenue: f64,
    pub profit_inventory_change: f64,
    pub profit_costs: f64,
    pub cost_wages: f64,
    pub cost_intermediate: f64,
    pub cost_capital: f64,
    pub cost_production_tax: f64,
    pub cost_interest: f64,
    /// A.44 bail-ins executed and total equity moved between banks.
    pub bank_bail_ins: u32,
    pub bank_bail_in_amount: f64,
    /// Value of completed housing transfers reaching the accounting pass.
    pub housing_transfer_value: f64,
    /// Housing-market activity. `property.value` is only written on a completed
    /// sale, so a zero sale count pins HPI at 1.000 by construction.
    pub housing_listings: u32,
    pub housing_bids: u32,
    pub housing_sales: u32,
    /// Which of A.29/A.30/A.31 is the binding minimum on a mortgage application.
    pub mortgage_bind_ltv: u32,
    pub mortgage_bind_lti: u32,
    pub mortgage_bind_dsti: u32,
    pub mortgage_cap_sum: f64,
    pub mortgage_req_sum: f64,
    /// Every firm's state at the end of the quarter, for offline diagnosis.
    /// Populated only when [`ModelPolicy::trace`] is set -- it is O(firms) per
    /// quarter and has no place in a production run.
    pub firm_trace: Vec<FirmProbe>,
    /// Sector balances, so a collapse can be attributed to whichever sector is
    /// losing the money the others gain.
    pub firm_deposits_total: f64,
    pub firm_debt_total: f64,
    pub firm_equity_total: f64,
    pub household_deposits_total: f64,
    pub household_ofa_total: f64,
    pub household_income_total: f64,
    pub household_consumption_total: f64,
    pub household_net_wealth_total: f64,
    pub bank_equity_total: f64,
    pub bank_reserves_total: f64,
    pub bank_deposits_total: f64,
    pub government_revenue: f64,
    pub government_deficit: f64,
    pub government_debt: f64,
    pub unemployment_benefit: f64,
    pub average_wage: f64,
    pub firms_bankrupt: u32,
    pub households_bankrupt: u32,
    /// A.40 bank profit, term by term, plus the A.41 adjustments that follow
    /// it. Equity falling while nothing is being lent has to show up in one of
    /// these.
    pub bank_loan_interest: f64,
    pub bank_reserve_income: f64,
    pub bank_reserve_cost: f64,
    pub bank_deposit_interest: f64,
    pub bank_corporate_tax: f64,
    pub bank_writeoff_seized: f64,
    pub bank_writeoff_lost: f64,
    /// A.45 Taylor-rule inputs, so a runaway policy rate can be attributed to
    /// the inflation term or the growth term rather than guessed at.
    pub policy_rate: f64,
    pub taylor_cpi_inflation: f64,
    pub taylor_growth: f64,
    /// Why a credit application failed: the A.27 screen (demand side), the
    /// A.25/A.26 caps, or A.32's envelope (supply side). Without this split a
    /// zero grant is unattributable.
    pub credit_blocked_by_roa: u32,
    pub credit_blocked_by_cap: u32,
    pub credit_blocked_by_supply: u32,
    pub credit_envelope_total: f64,
    pub credit_cap_total: f64,
    /// A.25 and A.26 separately, so a zero joint cap is attributable.
    pub cap_dte_total: f64,
    pub cap_roe_total: f64,
    pub cap_dte_zero: u32,
    pub cap_roe_zero: u32,
}

impl MarketAudit {
    /// True when every quantity invariant the paper guarantees held this
    /// quarter. Reported rather than asserted so a violated invariant shows up
    /// as a measurable number in the CLI output instead of a panic.
    pub fn quantity_invariants_hold(&self, tolerance: f64) -> bool {
        self.max_production_over_labour <= tolerance
            && self.max_labour_over_materials <= tolerance
            && self.employed_headcount <= self.individual_headcount
    }
}

impl Default for MarketAudit {
    fn default() -> Self {
        Self {
            labour_fired_before_hiring: false,
            credit_clearing_order: Vec::new(),
            credit_visits_ordered_by_rate: true,
            mortgage_blocked_purchases: 0,
            lower_price_seller_priority_seen: false,
            goods_excess_demand: 0.0,
            last_expectation_fit_observations: 0,
            phase_log: Vec::new(),
            max_production_over_labour: 0.0,
            max_labour_over_materials: 0.0,
            employed_headcount: 0,
            individual_headcount: 0,
            firm_probe: None,
            goods_demand_quantity: 0.0,
            demand_firm_intermediate: 0.0,
            demand_firm_capital: 0.0,
            demand_household_consumption: 0.0,
            demand_household_capital: 0.0,
            demand_government: 0.0,
            demand_export: 0.0,
            goods_supply_quantity: 0.0,
            firm_credit_applications: 0,
            firm_credit_roa_failures: 0,
            firm_credit_requested: 0.0,
            firm_credit_granted: 0.0,
            firm_roa_max: 0.0,
            profit_sales_revenue: 0.0,
            profit_inventory_change: 0.0,
            profit_costs: 0.0,
            cost_wages: 0.0,
            cost_intermediate: 0.0,
            cost_capital: 0.0,
            cost_production_tax: 0.0,
            cost_interest: 0.0,
            bank_bail_ins: 0,
            bank_bail_in_amount: 0.0,
            housing_transfer_value: 0.0,
            housing_listings: 0,
            housing_bids: 0,
            housing_sales: 0,
            mortgage_bind_ltv: 0,
            mortgage_bind_lti: 0,
            mortgage_bind_dsti: 0,
            mortgage_cap_sum: 0.0,
            mortgage_req_sum: 0.0,
            firm_trace: Vec::new(),
            firm_deposits_total: 0.0,
            firm_debt_total: 0.0,
            firm_equity_total: 0.0,
            household_deposits_total: 0.0,
            household_ofa_total: 0.0,
            household_income_total: 0.0,
            household_consumption_total: 0.0,
            household_net_wealth_total: 0.0,
            bank_equity_total: 0.0,
            bank_reserves_total: 0.0,
            bank_deposits_total: 0.0,
            government_revenue: 0.0,
            government_deficit: 0.0,
            government_debt: 0.0,
            unemployment_benefit: 0.0,
            average_wage: 0.0,
            firms_bankrupt: 0,
            households_bankrupt: 0,
            bank_loan_interest: 0.0,
            bank_reserve_income: 0.0,
            bank_reserve_cost: 0.0,
            bank_deposit_interest: 0.0,
            bank_corporate_tax: 0.0,
            bank_writeoff_seized: 0.0,
            bank_writeoff_lost: 0.0,
            policy_rate: 0.0,
            taylor_cpi_inflation: 0.0,
            taylor_growth: 0.0,
            credit_blocked_by_roa: 0,
            credit_blocked_by_cap: 0,
            credit_blocked_by_supply: 0,
            credit_envelope_total: 0.0,
            credit_cap_total: 0.0,
            cap_dte_total: 0.0,
            cap_roe_total: 0.0,
            cap_dte_zero: 0,
            cap_roe_zero: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroEnvironment {
    pub quarter: u64,
    pub seed: u64,
    pub scale_factor: f64,
    pub policy: ModelPolicy,
    pub params: CountryParameters,
    pub calibration: CalibrationParameters,
    pub aggregates: MacroAggregates,
    pub previous_aggregates: MacroAggregates,
    pub forecast: ForecastState,
    pub history: MacroHistory,
    pub loan_book: LoanBook,
    pub accounting: AccountingReport,
    pub audit: MarketAudit,
}

impl MacroEnvironment {
    pub fn new(seed: u64) -> Self {
        Self {
            quarter: 0,
            seed,
            scale_factor: SIM_SCALE_FACTOR,
            policy: ModelPolicy::default(),
            params: CountryParameters::default(),
            calibration: CalibrationParameters::default(),
            aggregates: MacroAggregates::default(),
            previous_aggregates: MacroAggregates::default(),
            forecast: ForecastState::default(),
            history: MacroHistory::default(),
            loan_book: LoanBook::default(),
            accounting: AccountingReport::default(),
            audit: MarketAudit::default(),
        }
    }

    pub fn push_phase(&mut self, phase: &str) {
        self.audit.phase_log.push(phase.to_owned());
    }

    /// Opens a deterministic stream for the calling system.
    ///
    /// `context` comes from `ECSReference::run_context()`.
    pub fn rng(&self, context: RunContext, salt: u64) -> MacroRng {
        MacroRng::new(context, self.seed, salt)
    }

    /// Opens a per-agent deterministic stream.
    ///
    /// Prefer this inside loops over agents: the draw then depends on the
    /// agent, not on the position at which the agent happens to be visited,
    /// which is what makes the loop safe to parallelise.
    pub fn rng_for_agent(&self, context: RunContext, salt: u64, agent_id: u64) -> MacroRng {
        MacroRng::for_agent(context, self.seed, salt, agent_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Loan {
    pub id: u64,
    pub bank_id: u32,
    pub borrower_kind: u8,
    pub borrower_id: u32,
    pub sector: u8,
    pub loan_class: u8,
    pub principal: f64,
    pub outstanding: f64,
    pub rate: f64,
    pub maturity_remaining_quarters: u32,
    pub origin_quarter: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoanBook {
    pub next_id: u64,
    pub loans: Vec<Loan>,
}

impl LoanBook {
    // A loan record carries every one of these fields; a parameter object would
    // just be the `Loan` struct this method already constructs.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        bank_id: u32,
        borrower_kind: u8,
        borrower_id: u32,
        sector: u8,
        loan_class: u8,
        principal: f64,
        rate: f64,
        maturity_quarters: u32,
        origin_quarter: u64,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        self.loans.push(Loan {
            id,
            bank_id,
            borrower_kind,
            borrower_id,
            sector,
            loan_class,
            principal,
            outstanding: principal,
            rate,
            maturity_remaining_quarters: maturity_quarters.max(1),
            origin_quarter,
        });
    }
}
