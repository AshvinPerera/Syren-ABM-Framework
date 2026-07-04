use std::path::PathBuf;

use super::accounting::{AccountingReport, GdpIdentity};
use super::calibration::CalibrationParameters;
use super::components::{SECTORS, SIM_SCALE_FACTOR};
use super::coverage::EquationCoverage;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapReportMode {
    Text,
    Json,
    None,
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
    pub replication_policy: ReplicationPolicy,
    pub gap_report: GapReportMode,
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
            replication_policy: ReplicationPolicy::default(),
            gap_report: GapReportMode::Text,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankSignPolicy {
    PositiveAndNegativePartHelpers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HousingReductionPolicy {
    GuardedFractionalReduction,
    LiteralPaperFormula,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoodsClearingPolicy {
    PolednaSearchAndMatching,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonLikeRng {
    state: [u32; 624],
    index: usize,
}

impl PythonLikeRng {
    pub fn new(seed: u64) -> Self {
        let mut state = [0_u32; 624];
        state[0] = (seed as u32).max(1);
        for idx in 1..624 {
            let previous = state[idx - 1];
            state[idx] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(idx as u32);
        }
        Self { state, index: 624 }
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;

        for idx in 0..624 {
            let x = (self.state[idx] & UPPER_MASK) | (self.state[(idx + 1) % 624] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.state[idx] = self.state[(idx + 397) % 624] ^ x_a;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn unit_f64(&mut self) -> f64 {
        let high = (self.next_u32() >> 5) as u64;
        let low = (self.next_u32() >> 6) as u64;
        ((high << 26) + low) as f64 / ((1_u64 << 53) as f64)
    }

    pub fn below(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be positive");
        let upper = upper as u64;
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let draw = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
            if draw < zone {
                return (draw % upper) as usize;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicationPolicy {
    pub allow_unresolved_blockers: bool,
    pub bank_sign_policy: BankSignPolicy,
    pub housing_reduction_policy: HousingReductionPolicy,
    pub goods_clearing_policy: GoodsClearingPolicy,
    pub firm_bank_visits: u32,
    pub household_bank_visits: u32,
    pub assignment_tie_breaker: &'static str,
    pub missing_data_policy: &'static str,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self {
            allow_unresolved_blockers: true,
            bank_sign_policy: BankSignPolicy::PositiveAndNegativePartHelpers,
            housing_reduction_policy: HousingReductionPolicy::LiteralPaperFormula,
            goods_clearing_policy: GoodsClearingPolicy::PolednaSearchAndMatching,
            firm_bank_visits: 2,
            household_bank_visits: 2,
            assignment_tie_breaker: "seeded Python-like MT19937 shuffle; Bernoulli(0.5) random tie policy where explicit ties remain",
            missing_data_policy: "real-data mode fails fast with named missing assets",
        }
    }
}

impl ReplicationPolicy {
    pub fn strict() -> Self {
        Self {
            allow_unresolved_blockers: false,
            ..Self::default()
        }
    }
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

impl Default for CountryParameters {
    fn default() -> Self {
        let mut io_matrix = [[0.0; SECTORS]; SECTORS];
        let mut net_fixed_assets_matrix = [[0.0; SECTORS]; SECTORS];
        let mut capital_compensation_matrix = [[0.0; SECTORS]; SECTORS];
        for s in 0..SECTORS {
            io_matrix[s][s] = 0.10;
            net_fixed_assets_matrix[s][s] = 0.03;
            capital_compensation_matrix[s][s] = 0.03;
        }
        Self {
            io_matrix,
            net_fixed_assets_matrix,
            capital_compensation_matrix,
            capital_depreciation_rate_by_sector: [0.0; SECTORS],
            capital_installation_delay_quarters: [1; SECTORS],
            cpi_weights: [1.0 / SECTORS as f64; SECTORS],
            government_consumption_weights: [1.0 / SECTORS as f64; SECTORS],
            household_investment_weights: [1.0 / SECTORS as f64; SECTORS],
            row_export_weights: [1.0 / SECTORS as f64; SECTORS],
            row_import_weights: [1.0 / SECTORS as f64; SECTORS],
            firm_input_adjustment: 0.5,
            firm_capital_adjustment: 0.25,
            firm_credit_shortfall_intermediate_sensitivity: 0.0,
            firm_credit_shortfall_capital_sensitivity: 0.0,
            work_effort_max: 1.5,
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
            housing_sigma_hp: 0.17,
            housing_mu_ps: 0.4,
            housing_phi_b: 0.001,
            housing_phi_hr: 17.22,
            housing_beta_hr: 0.35,
            renter_stay_probability: 7.0 / 8.0,
            owner_stay_probability: 79.0 / 80.0,
            sale_price_reduction_probability: 0.1964,
            sale_price_reduction_mu: 1.4531,
            sale_price_reduction_sigma: 0.4889,
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
    pub goods_excess_demand: f64,
    pub last_expectation_fit_observations: usize,
    pub phase_log: Vec<String>,
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
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroEnvironment {
    pub quarter: u64,
    pub seed: u64,
    pub rng: PythonLikeRng,
    pub scale_factor: f64,
    pub policy: ReplicationPolicy,
    pub params: CountryParameters,
    pub calibration: CalibrationParameters,
    pub aggregates: MacroAggregates,
    pub previous_aggregates: MacroAggregates,
    pub forecast: ForecastState,
    pub history: MacroHistory,
    pub loan_book: LoanBook,
    pub accounting: AccountingReport,
    pub coverage: EquationCoverage,
    pub audit: MarketAudit,
}

impl MacroEnvironment {
    pub fn new(seed: u64) -> Self {
        Self {
            quarter: 0,
            seed,
            rng: PythonLikeRng::new(seed),
            scale_factor: SIM_SCALE_FACTOR,
            policy: ReplicationPolicy::default(),
            params: CountryParameters::default(),
            calibration: CalibrationParameters::default(),
            aggregates: MacroAggregates::default(),
            previous_aggregates: MacroAggregates::default(),
            forecast: ForecastState::default(),
            history: MacroHistory::default(),
            loan_book: LoanBook::default(),
            accounting: AccountingReport::default(),
            coverage: EquationCoverage::default(),
            audit: MarketAudit::default(),
        }
    }

    pub fn push_phase(&mut self, phase: &str) {
        self.audit.phase_log.push(phase.to_owned());
    }

    pub fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    pub fn unit_f64(&mut self) -> f64 {
        self.rng.unit_f64()
    }

    pub fn bernoulli(&mut self, probability: f64) -> bool {
        self.unit_f64() < probability.clamp(0.0, 1.0)
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.rng.below(i + 1);
            values.swap(i, j);
        }
    }

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
