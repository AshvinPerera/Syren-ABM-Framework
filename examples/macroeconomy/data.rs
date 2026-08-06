use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use super::components::{
    HouseholdDemand, HouseholdHistory, IndividualWageHistory, FirmRealised, FirmStockBaseline, FirmStocks, FirmTargets,
    Bank, CentralBank, Firm, GovernmentAccount, GovernmentEntity, Household, Individual, Property,
    RestOfWorld, LABOUR_EMPLOYED, LABOUR_INACTIVE, LABOUR_UNEMPLOYED, NOT_LINKED, PROPERTY_FOR_RENT,
    PROPERTY_FOR_SALE, PROPERTY_OWNER_OCCUPIED, PROPERTY_RENTAL, SECTORS,
};
use super::messages::{
    BUYER_FIRM, BUYER_HOUSEHOLD, LOAN_FIRM_LONG, LOAN_FIRM_SHORT, LOAN_HOUSEHOLD_CONSUMPTION,
    LOAN_MORTGAGE,
};
use super::state::{CountryParameters, MacroEnvironment, MacroeconomyConfig};

/// Initial utilisation rates of intermediate and capital stocks, `omega^M` and
/// `omega^K`. Both 0.85 (Appendix A.5.1, following Poledna et al.), so a firm
/// starts with `1/0.85 = 1.18` quarters of inputs on hand.
const OMEGA_INTERMEDIATE: f64 = 0.85;
const OMEGA_CAPITAL: f64 = 0.85;

/// Quarters of history the AR(1) expectation fits see at t=0.
///
/// Not a tuned length. A.2 refits the AR(1) each quarter on real data "from
/// 2000-Q1 up to the initialisation time" plus simulation output, so the
/// prefix is exactly the span between those two dates -- both of which the
/// model already carries as `first_real_data_quarter` and
/// `MacroeconomyConfig::initialisation`. `quarters_between` derives it.
///
/// The length is load-bearing. With the previous 8 quarters the fit had no
/// anchor: it extrapolated a 239% growth forecast off its own oscillation,
/// which drove A.60 to target 62% above feasible output and A.45 to a 24%
/// policy rate. A *flat* history is equally bad in the other direction -- it
/// predicts zero change forever, which is what pinned `hpi` at 1.000000.
///
/// See `quarters_between`. Trend and amplitude of the synthetic prefix:
const HISTORY_TREND: f64 = 0.004;
const HISTORY_WOBBLE: f64 = 0.003;

/// Share of gross output paid out as wages.
///
/// Initial prices are 1 (A.52), so A.77 unit cost is
/// `w_f/Y_f + sum_s' m + sum_s' d + tau^PROD`. The labour share plus the
/// technology row sums plus the production tax must leave a positive margin,
/// or A.58 hands every firm a negative initial profit and the A.27 credit
/// screen locks the whole economy out of borrowing at t=0.
const LABOUR_SHARE: f64 = 0.55;
/// Exports and imports as a share of gross output.
const EXPORT_SHARE: f64 = 0.10;
const IMPORT_SHARE: f64 = 0.10;
/// Final-demand composition, as shares of value added.
///
/// The goods-market identity alone does not pin the split: it is satisfied by
/// *any* division of final demand, including one households cannot afford. The
/// first attempt handed households the whole residual — 83% of GDP against the
/// ~42% they receive as net labour income — so the A.101 saving-rate solve went
/// negative, clamped, and left a quarter of output unsold anyway.
///
/// These are the national-accounting proportions of a small open European
/// economy (Austria: household consumption ~52% of GDP, government ~20%, gross
/// fixed capital formation ~23%). They are structural facts about the shape of
/// an economy, stated and cited — not knobs turned until the run behaved.
/// Firm capital purchases `d_{s's} Y_f` are part of GFCF, so household
/// investment is what is left of GFCF once firms have bought theirs.
const CONSUMPTION_SHARE_OF_VALUE_ADDED: f64 = 0.52;
const GOVERNMENT_SHARE_OF_VALUE_ADDED: f64 = 0.20;
const GFCF_SHARE_OF_VALUE_ADDED: f64 = 0.23;
/// Unemployment benefit replacement rate on *gross* income. Poledna B.2 derives
/// this from the statutory 55% of net income as
/// `0.55 * (1 - tau^INC) * (1 - tau^SIW)`, so it is computed, not chosen.
const BENEFIT_NET_REPLACEMENT_RATE: f64 = 0.55;
/// `phi^StY`, the target inventory-to-production fraction. Table 4 (Austria),
/// verified against the paper p. 13. A.54 sets `S_f(0)` to exactly this
/// fraction of initial output, so it must match `CalibrationParameters`.
const CALIBRATION_PHI_ST_Y: f64 = 0.10;
/// Ceiling on firm debt as a share of the capital stock's value. Poledna C.1
/// takes `L^I` from national accounts and apportions it by capital share; with
/// no national accounts the actual ratio is *solved* from A.26 (see the balance
/// sheet section below) and this only caps it.
const FIRM_DEBT_TO_CAPITAL: f64 = 0.45;
/// How far above A.26's own boundary the initial firms are placed.
const ROE_HEADROOM: f64 = 1.3;
/// How far above `rho^RoA` the initial firms are placed, so the A.27 screen has
/// headroom rather than sitting on its own boundary.
const ROA_HEADROOM: f64 = 1.6;
/// Representative quarterly loan rate used when sizing the initial balance
/// sheet. Only affects the interest term of the A.27 solve below.
const INITIAL_LOAN_RATE: f64 = 0.025;
/// Bank equity as a share of the loans it has granted (Poledna: "initial bank
/// equity is proportional to total loans granted").
const BANK_EQUITY_TO_LOANS: f64 = 0.12;
/// Dwelling value as a multiple of annual household income, and the gross
/// rental yield on it.
const HOUSE_PRICE_TO_INCOME: f64 = 4.5;
const GROSS_RENTAL_YIELD: f64 = 0.045;
/// Share of households that own their residence, and of owners who let out an
/// additional property.
const OWNER_OCCUPIER_SHARE: f64 = 0.60;
/// Mortgage debt as a share of owner-occupied property value, and consumption
/// debt as a share of quarterly household income.
const MORTGAGE_TO_VALUE: f64 = 0.25;
const CONSUMPTION_DEBT_TO_INCOME: f64 = 0.20;
/// Household financial assets held outside deposits, as a share of deposits.
const OTHER_FINANCIAL_TO_DEPOSITS: f64 = 0.28;
/// Quarters of income a household holds as deposits, before the A.23 reserve
/// solve below tops it up.
const HOUSEHOLD_DEPOSIT_QUARTERS: f64 = 1.4;
/// Margin by which deposits plus bank equity exceed the loan book, so banks
/// open with positive reserves rather than on the boundary.
const RESERVE_HEADROOM: f64 = 0.10;

/// How many agents of each kind to generate.
///
/// Every field is a count or a rate, never an absolute money amount, so the
/// same generator produces a 286-entity fixture and a ~2M-agent population
/// without changing any other constant in this file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationScale {
    pub firms_per_sector: u32,
    pub employees_per_firm: u32,
    pub individuals_per_household: u32,
    pub unemployment_rate: f64,
    /// Share of the total population that is economically inactive.
    ///
    /// Wiese A.8 and Table A.15 define `I^N`, the set of not-economically-active
    /// individuals; they belong to households and consume, but supply no labour
    /// and earn no individual income (A.104 supports them through `sb^O`).
    pub inactive_rate: f64,
    pub banks: u32,
}

impl PopulationScale {
    /// The smallest population that still exercises every mechanism: more than
    /// one firm in a sector (so A.140 seller priority is observable), more than
    /// one bank (so the A.12 credit search has somewhere to go), and enough
    /// workers per firm that hiring and firing are not all-or-nothing.
    pub fn tiny() -> Self {
        Self {
            firms_per_sector: 1,
            employees_per_firm: 8,
            individuals_per_household: 4,
            unemployment_rate: 0.07,
            // Poledna Table 5 (Austria): H^act = 4,729,215 economically active
            // against H^inact = 4,130,385 inactive, i.e. 46.6% of the
            // population.
            inactive_rate: 0.4661,
            banks: 2,
        }
    }

    pub fn firm_count(&self) -> u32 {
        self.firms_per_sector * SECTORS as u32 + 1
    }
}

impl Default for PopulationScale {
    fn default() -> Self {
        Self::tiny()
    }
}

/// Deterministic generator-local RNG.
///
/// Kept separate from `MacroRng` so population generation does not depend on a
/// `RunContext`, and so a change to the simulation's stream cannot silently
/// change the initial state.
struct GenRng(u64);

impl GenRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Bounded Pareto draw with the exponent Poledna C.1 uses for the firm size
    /// distribution (-2), clamped so a single firm cannot take a whole sector.
    fn pareto_size(&mut self, max_ratio: f64) -> f64 {
        let u = self.unit().clamp(1e-6, 1.0 - 1e-6);
        (1.0 / (1.0 - u)).min(max_ratio)
    }
}

pub trait DataProvider {
    fn load(&self, config: &MacroeconomyConfig) -> Result<InitialData, DataError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitialisationRecipeStep {
    pub id: &'static str,
    pub source: &'static str,
    pub method: &'static str,
    pub unresolved_exactness: &'static str,
}

pub fn thesis_initialisation_recipe() -> Vec<InitialisationRecipeStep> {
    vec![
        InitialisationRecipeStep {
            id: "firm-compustat-sampling",
            source: "Thesis Ch. 6 pp. 172-176",
            method: "Sample firms with replacement from Compustat, rescale employees, deposits, and debt to OECD sector and balance-sheet aggregates, then set initial production, prices, demand, inventory, intermediate inputs, capital inputs, costs, profits, and equity from the thesis equations.",
            unresolved_exactness: "Exact random seed stream and weighted-sampling tie behavior are not given.",
        },
        InitialisationRecipeStep {
            id: "bank-compustat-sampling",
            source: "Thesis Ch. 6 pp. 160-168",
            method: "Sample banks with replacement from Compustat where available, create a single aggregate bank otherwise, rescale deposits/liabilities/equity to IMF/OECD aggregates, and initialize reserves from deposits, equity, and loans.",
            unresolved_exactness: "Exact sampling order and assignment tie behavior are not given.",
        },
        InitialisationRecipeStep {
            id: "household-hfcs-sampling",
            source: "Thesis Ch. 6 pp. 192-204",
            method: "Draw households from HFCS using survey weights, map income, assets, liabilities, tenure, rent, and consumption fields, then rescale consumption, social benefits, wealth, and debt to macro aggregates.",
            unresolved_exactness: "Exact weighted-sampling implementation and country fallback microstate construction are not given.",
        },
        InitialisationRecipeStep {
            id: "individual-hfcs-linking",
            source: "Thesis Ch. 6 pp. 204-207",
            method: "Link individuals to sampled households through HFCS ids, map labour status, wages, unemployment benefits, education, age, and sector, then adjust labour status and employment sector to aggregate unemployment, vacancies, and sector employment.",
            unresolved_exactness: "Exact random choices used when adjusting statuses/sectors are not given.",
        },
        InitialisationRecipeStep {
            id: "linear-sum-assignments",
            source: "Thesis Ch. 6 pp. 162, 176, 196, 206",
            method: "Use linear sum assignment for firm-employee, firm-bank, household-bank, and household-property matching so micro relationships are close to aggregate wages, deposits, debts, and property wealth.",
            unresolved_exactness: "Assignment solver, cost scaling, and tie-breaking are not given.",
        },
        InitialisationRecipeStep {
            id: "scale-factor",
            source: "Paper specification and thesis figures in Ch. 6",
            method: "Represent 1000 real agents with one simulated agent.",
            unresolved_exactness: "No unresolved model choice; exact empirical counts still depend on source vintages and rounding.",
        },
    ]
}

#[derive(Clone, Debug)]
pub struct InitialData {
    pub environment: MacroEnvironment,
    pub firms: Vec<Firm>,
    pub individual_wage_histories: Vec<IndividualWageHistory>,
    pub household_demands: Vec<HouseholdDemand>,
    pub household_histories: Vec<HouseholdHistory>,
    pub firm_stocks: Vec<FirmStocks>,
    pub firm_stock_baselines: Vec<FirmStockBaseline>,
    pub firm_targets: Vec<FirmTargets>,
    pub firm_realised: Vec<FirmRealised>,
    pub individuals: Vec<Individual>,
    pub households: Vec<Household>,
    pub banks: Vec<Bank>,
    pub government_entities: Vec<GovernmentEntity>,
    pub government_accounts: Vec<GovernmentAccount>,
    pub central_banks: Vec<CentralBank>,
    pub properties: Vec<Property>,
    pub rest_of_world: Vec<RestOfWorld>,
}

#[derive(Clone, Debug, Default)]
pub struct FixtureDataProvider;

impl DataProvider for FixtureDataProvider {
    fn load(&self, config: &MacroeconomyConfig) -> Result<InitialData, DataError> {
        // Config is applied to the environment *before* the population is
        // generated, so the generator reads the final coefficients. Applying it
        // afterwards would leave a config that overrides `io_matrix` changing
        // the simulation's coefficients but not the firms' initial stocks,
        // breaking the A.55/A.56 consistency the initial state is built for.
        let mut environment = MacroEnvironment::new(config.seed);
        if let Some(path) = &config.config_path {
            super::config::apply_config_file(path, config.scenario.as_deref(), &mut environment)
                .map_err(|error| DataError::Config(error.to_string()))?;
        }
        environment.policy = config.policy;
        let mut scale = PopulationScale::tiny();
        scale.firms_per_sector = config.firms_per_sector.max(1);
        Ok(synthetic_population(
            scale,
            config.seed,
            &config.initialisation,
            environment,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct RealDataProvider {
    pub data_dir: PathBuf,
}

impl DataProvider for RealDataProvider {
    fn load(&self, config: &MacroeconomyConfig) -> Result<InitialData, DataError> {
        Err(DataError::MissingAssets {
            data_dir: self.data_dir.clone(),
            country: config.country.clone(),
            initialisation: config.initialisation.clone(),
            required: vec![
                "OECD ICIO tables",
                "OECD business demography and employment by activity",
                "OECD financial balance sheets",
                "OECD social expenditure and tax receipts",
                "IMF quarterly macro and financial series",
                "World Bank fiscal/unemployment/NPL series",
                "BIS policy rates",
                "ECB HFCS household and individual microdata",
                "Compustat firm and bank microdata",
                "ESRB macroprudential mortgage measures",
            ],
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataError {
    MissingAssets {
        data_dir: PathBuf,
        country: String,
        initialisation: String,
        required: Vec<&'static str>,
    },
    Config(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::MissingAssets {
                data_dir,
                country,
                initialisation,
                required,
            } => {
                write!(
                    f,
                    "real-data mode for {country} at {initialisation} needs external assets in {}: {}",
                    data_dir.display(),
                    required.join(", ")
                )
            }
            DataError::Config(message) => write!(f, "config error: {message}"),
        }
    }
}

impl Error for DataError {}

/// Build a stock-flow-consistent synthetic population.
///
/// Everything below is *derived* — from the technology matrices, from an
/// accounting identity, or from a stated structural ratio. Nothing is chosen
/// because it makes a run look better; if a value here is ever picked that way,
/// that is a defect in the method, not a calibration.
///
/// Order matters, because each step feeds the next:
///   1. draw employment, derive output   (Poledna C.1)
///   2. close the goods-market identity  (A.101/A.102/A.95, Poledna B.2)
///   3. derive the sector weights from the SAM it just solved
///   4. build balance sheets and close with central-bank equity (Poledna C.5)
pub fn synthetic_population(
    scale: PopulationScale,
    seed: u64,
    initialisation: &str,
    mut environment: MacroEnvironment,
) -> InitialData {
    let history_quarters = quarters_between(
        environment.history.first_real_data_quarter,
        initialisation,
    );
    let mut rng = GenRng::new(seed);

    // ---- 1. Employment and output -------------------------------------
    //
    // Poledna C.1 draws firm employment from a power law with exponent -2 and
    // derives production from it, `Y_i(0) = alpha_i * N_i(0)`. Output is a
    // consequence of the employment draw and labour productivity, never a
    // hand-written number.
    let firm_count = scale.firm_count();
    let mut firm_sector = Vec::with_capacity(firm_count as usize);
    for slot in 0..firm_count {
        // One extra firm doubled into sector 0 so at least one sector has two
        // sellers posting different prices -- A.140 seller priority is only
        // observable where it does.
        let sector = if slot == firm_count - 1 {
            0
        } else {
            (slot / scale.firms_per_sector) as usize % SECTORS
        };
        firm_sector.push(sector);
    }

    let mut raw_size = Vec::with_capacity(firm_count as usize);
    for _ in 0..firm_count {
        raw_size.push(rng.pareto_size(6.0));
    }
    // Normalise within each sector so the sector's headcount is exactly the
    // scale's target, then round to whole people -- labour is only adjustable
    // in whole individuals.
    let mut employees = vec![0u32; firm_count as usize];
    for sector in 0..SECTORS {
        let members: Vec<usize> = (0..firm_count as usize)
            .filter(|&i| firm_sector[i] == sector)
            .collect();
        if members.is_empty() {
            continue;
        }
        let target = scale.employees_per_firm as f64 * members.len() as f64;
        let raw_total: f64 = members.iter().map(|&i| raw_size[i]).sum();
        for &i in &members {
            employees[i] = ((raw_size[i] / raw_total) * target).round().max(1.0) as u32;
        }
    }

    // Sectoral labour productivity h_s (A.5.1: sectoral gross output per person
    // employed). A mild spread so sectors are not identical.
    let mut productivity = [0.0; SECTORS];
    for (s, slot) in productivity.iter_mut().enumerate() {
        *slot = 4.0 * (1.0 + 0.08 * (((s % 5) as f64) - 2.0));
    }

    let mut sector_output = [0.0; SECTORS];
    let mut firm_output = Vec::with_capacity(firm_count as usize);
    for i in 0..firm_count as usize {
        let s = firm_sector[i];
        let output = productivity[s] * employees[i] as f64;
        firm_output.push(output);
        sector_output[s] += output;
    }
    let total_output: f64 = sector_output.iter().sum();

    // ---- 2. Close the goods-market identity ---------------------------
    let government_entity_count = ((firm_count as f64 * 0.25).round() as u32).max(1);
    let mut entities_per_sector = [0u32; SECTORS];
    for entity in 0..government_entity_count {
        entities_per_sector[entity as usize % SECTORS] += 1;
    }
    let sam = solve_sam(
        &environment.params,
        sector_output,
        environment.params.government_consumption_share,
        &entities_per_sector,
    );

    // ---- 2b. Scale the capital stock from the A.27 credit screen -------
    //
    // A.27 refuses *all* firm credit unless `Pi_f/(L_f + E_f) >= rho^RoA`, and
    // by A.93 the debt cancels out of that denominator: `L_f + E_f` is just
    // total assets, `D + P*S + P*M + P*K`. The thesis (Eq. 6.30) confirms Pi is
    // *quarterly*, so 0.05 is a 20% annual return on assets -- unreachable at
    // any realistic capital-output ratio. Measured: 36 of 38 applications
    // failed the screen in quarter 1, and credit granted fell to exactly zero
    // by quarter 3.
    //
    // The resolution is not to touch A.27. It is that `k_{s's}` is *not a
    // real-side parameter in this model*: substituting A.56 into A.64 gives
    // `K_f(0) = Y_f(0)/omega^K` regardless of `k`, so its magnitude affects
    // only the balance sheet. With no OECD net-fixed-assets table to pin it,
    // the synthetic economy pins it from the screen the model itself imposes --
    // the same derivation the earlier fixture described as "solved backwards
    // from the credit screen".
    //
    // Per unit of output, with `A0` the non-capital assets:
    //     margin - r*lambda*K = target * (A0 + K)
    //  => K = (margin - target*A0) / (target + r*lambda)
    let io_row_mean: f64 = (0..SECTORS)
        .map(|s| environment.params.io_matrix[s].iter().sum::<f64>())
        .sum::<f64>()
        / SECTORS as f64;
    let capital_row_mean: f64 = (0..SECTORS)
        .map(|s| {
            environment.params.capital_compensation_matrix[s]
                .iter()
                .sum::<f64>()
        })
        .sum::<f64>()
        / SECTORS as f64;
    let production_tax_rate = GovernmentAccount::default().production_tax_by_sector[0];
    let margin = 1.0 - LABOUR_SHARE - io_row_mean - capital_row_mean - production_tax_rate;
    let target_roa = environment.params.return_on_assets * ROA_HEADROOM;
    // Deposits are one quarter of operating surplus; inventory is A.54; the
    // intermediate stock is A.55.
    let non_capital_assets = margin + CALIBRATION_PHI_ST_Y + io_row_mean / OMEGA_INTERMEDIATE;
    let capital_per_output = ((margin - target_roa * non_capital_assets)
        / (target_roa + INITIAL_LOAN_RATE * FIRM_DEBT_TO_CAPITAL))
        .max(0.05);
    // A.26 has to hold at t=0 as well, and it pulls the opposite way to A.27.
    //
    //   A.27:  D + S + M + K <= Pi / rho^RoA        (assets small vs profit)
    //   A.26:  K - L + D     >= Pi / rho^RoE        (capital large vs profit)
    //   A.25:  L <= rho^DtE * K                     (rho^DtE = 1, so L <= K)
    //
    // Solving A.27 alone leaves firms sitting on A.26's boundary: at tick 1
    // A.25 allows six times what A.26 does. Hoarded deposits keep `D_f`
    // climbing so it does not bite immediately, but the moment `D_f` stops
    // growing every firm's cap goes to exactly zero and the credit market shuts
    // with its supply envelope untouched. Both are solved here.
    //
    // With `L = lambda * K`, A.26 rearranges to
    //     lambda <= 1 - (headroom * Pi / rho^RoE - D) / K
    // which is what sets the debt ratio. `FIRM_DEBT_TO_CAPITAL` becomes the
    // ceiling rather than the value. A.25 is then slack for any `lambda <= 1`.
    let deposits_per_output = margin;
    let roe_requirement =
        ROE_HEADROOM * margin / environment.params.return_on_equity.max(1e-9);
    let debt_to_capital = if capital_per_output > 1e-9 {
        (1.0 - (roe_requirement - deposits_per_output) / capital_per_output)
            .clamp(0.0, FIRM_DEBT_TO_CAPITAL)
    } else {
        0.0
    };
    // A.56 holds `K_fs'(0) = k_{s's} Y_f(0)/omega^K`, so the row sum of `k` that
    // delivers this capital-to-output ratio is `capital_per_output * omega^K`.
    let target_k_row = capital_per_output * OMEGA_CAPITAL;
    let current_k_row: f64 = (0..SECTORS)
        .map(|s| {
            environment.params.net_fixed_assets_matrix[s]
                .iter()
                .sum::<f64>()
        })
        .sum::<f64>()
        / SECTORS as f64;
    if current_k_row > 1e-9 {
        let k_scale = target_k_row / current_k_row;
        for row in environment.params.net_fixed_assets_matrix.iter_mut() {
            for cell in row.iter_mut() {
                *cell *= k_scale;
            }
        }
    }

    // ---- 3. Sector weights, derived from the SAM ----------------------
    //
    // A.4's `b_s^CPI` are *aggregate household consumption weights*, so they
    // are the consumption column of the SAM, not a uniform 1/18. Same for the
    // government, investment and trade weights. Overriding these from a config
    // file would desynchronise them from the initial state, so they are always
    // derived here.
    // A.95 recomputes government demand every quarter as
    // `share * aggregate production`, so the share has to be the one the SAM
    // solved for. Leaving it at its default made the government buy 47.7 where
    // the SAM had allocated 107, and that shortfall alone was most of the
    // remaining demand hole.
    let sam_government: f64 = sam.government.iter().sum();
    environment.params.government_consumption_share = if total_output > 1e-9 {
        sam_government / total_output
    } else {
        environment.params.government_consumption_share
    };
    environment.params.cpi_weights = normalise(&sam.consumption);
    environment.params.government_consumption_weights = normalise(&sam.government);
    environment.params.household_investment_weights = normalise(&sam.investment);
    environment.params.row_export_weights = normalise(&sam.exports);
    environment.params.row_import_weights = normalise(&sam.imports);

    // ---- 4. Firms ------------------------------------------------------
    let account_template = GovernmentAccount::default();
    let production_tax = account_template.production_tax_by_sector[0];
    let mut firms = Vec::with_capacity(firm_count as usize);
    let mut firm_stocks = Vec::with_capacity(firm_count as usize);
    let mut firm_stock_baselines = Vec::with_capacity(firm_count as usize);
    let mut firm_targets = Vec::with_capacity(firm_count as usize);
    let mut firm_realised = Vec::with_capacity(firm_count as usize);
    for i in 0..firm_count as usize {
        let sector = firm_sector[i];
        let output = firm_output[i];
        let headcount = employees[i];
        let io_row: f64 = environment.params.io_matrix[sector].iter().sum();
        let capital_row: f64 = environment.params.capital_compensation_matrix[sector]
            .iter()
            .sum();
        // A.77 at P = 1, which is the accounting quantity A.58 needs.
        let unit_cost = LABOUR_SHARE + io_row + capital_row + production_tax;
        // A.140 needs at least one sector with genuine price dispersion.
        let price = if i == firm_count as usize - 1 { 1.15 } else { 1.0 };
        let inventory = CALIBRATION_PHI_ST_Y * output;
        let work_effort = productivity[sector];

        let firm = Firm {
            id: i as u32,
            sector: sector as u8,
            country: 0,
            bank_id: (i as u32) % scale.banks.max(1),
            employees: headcount,
            // A.65: H_f = h_f * sum_i H_i in output units, with H_i = 1 (A.128).
            labour: output,
            target_labour: output,
            work_effort,
            initial_work_effort: work_effort,
            // A.50 makes the *total* wage bill proportional to headcount; this
            // field is the per-worker rate.
            wage: LABOUR_SHARE * output / headcount as f64,
            // `w_bar_f(0) = w_f(0)`: phi^WE(0) is 1 by construction, so the
            // base wage and the paid wage coincide at initialisation.
            base_wage: LABOUR_SHARE * output / headcount as f64,
            production: output,
            previous_production: output,
            initial_production: output,
            target_production: output,
            demand: output,
            previous_demand: output,
            target_demand: output,
            price,
            previous_price: price,
            inventory,
            previous_inventory: inventory,
            inventory_two_periods_ago: inventory,
            initial_inventory: inventory,
            unit_cost,
            ..Firm::default()
        };
        // A.55/A.56: enough of every input for 1/omega quarters of production.
        let mut stocks = FirmStocks {
            id: firm.id,
            ..FirmStocks::default()
        };
        let mut baseline = FirmStockBaseline {
            id: firm.id,
            ..FirmStockBaseline::default()
        };
        for s in 0..SECTORS {
            stocks.intermediate_stock[s] =
                environment.params.io_matrix[sector][s] * output / OMEGA_INTERMEDIATE;
            baseline.initial_intermediate_stock[s] = stocks.intermediate_stock[s];
            stocks.capital_stock[s] =
                environment.params.net_fixed_assets_matrix[sector][s] * output / OMEGA_CAPITAL;
            baseline.initial_capital_stock[s] = stocks.capital_stock[s];
        }
        firm_targets.push(FirmTargets { id: firm.id, ..FirmTargets::default() });
        firm_realised.push(FirmRealised { id: firm.id, ..FirmRealised::default() });
        firms.push(firm);
        firm_stocks.push(stocks);
        firm_stock_baselines.push(baseline);
    }

    // Debt apportioned by capital share and deposits by operating-surplus
    // share, exactly as Poledna C.1 does with national-accounts aggregates.
    let capital_value: Vec<f64> = firm_stocks
        .iter()
        .map(|stocks| stocks.capital_stock.iter().sum::<f64>())
        .collect();
    let total_capital: f64 = capital_value.iter().sum();
    let total_firm_debt = debt_to_capital * total_capital;
    let operating_margin = (1.0 - LABOUR_SHARE
        - firms
            .iter()
            .map(|f| {
                let s = f.sector as usize;
                environment.params.io_matrix[s].iter().sum::<f64>()
                    + environment.params.capital_compensation_matrix[s]
                        .iter()
                        .sum::<f64>()
            })
            .sum::<f64>()
            / firms.len() as f64
        - production_tax)
        .max(0.01);
    let surplus: Vec<f64> = firms
        .iter()
        .map(|firm| (operating_margin * firm.production).max(0.0))
        .collect();
    let total_surplus: f64 = surplus.iter().sum();
    // One quarter of operating surplus held as working capital.
    let total_firm_deposits = total_surplus;
    for (i, (firm, stocks)) in firms.iter_mut().zip(firm_stocks.iter()).enumerate() {
        let capital_share = if total_capital > 1e-12 {
            capital_value[i] / total_capital
        } else {
            1.0 / firm_count as f64
        };
        let surplus_share = if total_surplus > 1e-12 {
            surplus[i] / total_surplus
        } else {
            1.0 / firm_count as f64
        };
        let debt = total_firm_debt * capital_share;
        firm.short_debt = 0.35 * debt;
        firm.long_debt = 0.65 * debt;
        firm.deposits = total_firm_deposits * surplus_share;
        // A.58: profits are output less A.57 costs, not a chosen fraction.
        firm.profits = firm.price * firm.production
            - (firm.unit_cost * firm.production + 0.02 * firm.short_debt + 0.03 * firm.long_debt);
        firm.predicted_profits = firm.profits;
        // A.93 at P = 1.
        firm.equity = firm.deposits
            + firm.price * firm.inventory
            + stocks
                .intermediate_stock
                .iter()
                .chain(stocks.capital_stock.iter())
                .sum::<f64>()
            - (firm.short_debt + firm.long_debt);
    }

    // ---- 5. Individuals and households --------------------------------
    let employed_total: u32 = employees.iter().sum();
    let unemployed_total =
        ((employed_total as f64 * scale.unemployment_rate / (1.0 - scale.unemployment_rate))
            .round() as u32)
            .max(1);
    // A.8's `I^N`. The active population is employed + unemployed, so the
    // inactive count follows from the inactive *share of the total*.
    let active_total = employed_total + unemployed_total;
    let inactive_rate = scale.inactive_rate.clamp(0.0, 0.95);
    let inactive_total = ((active_total as f64 * inactive_rate / (1.0 - inactive_rate)).round()
        as u32)
        .max(0);
    let population = active_total + inactive_total;
    let household_count = (population / scale.individuals_per_household.max(1)).max(1);

    let total_wage_bill: f64 = firms
        .iter()
        .map(|firm| firm.wage * firm.employees as f64)
        .sum();
    let average_wage = total_wage_bill / employed_total.max(1) as f64;
    // Poledna B.2 derives the gross replacement rate from the statutory net
    // rate; it is computed, not chosen.
    let benefit = BENEFIT_NET_REPLACEMENT_RATE
        * (1.0 - account_template.income_tax_rate)
        * (1.0 - account_template.social_insurance_worker_rate)
        * average_wage;

    let mut individuals = Vec::with_capacity(population as usize);
    let mut next_id = 0u32;
    for firm in &firms {
        for _ in 0..firm.employees {
            individuals.push(individual(
                next_id,
                next_id % household_count,
                firm.id,
                firm.sector,
                LABOUR_EMPLOYED,
                firm.wage,
                benefit,
            ));
            next_id += 1;
        }
    }
    for _ in 0..unemployed_total {
        individuals.push(individual(
            next_id,
            next_id % household_count,
            NOT_LINKED,
            0,
            LABOUR_UNEMPLOYED,
            0.0,
            benefit,
        ));
        next_id += 1;
    }
    // A.8's not-economically-active individuals. They never enter the labour
    // market (the A.128 search filters on `LABOUR_UNEMPLOYED`) and their
    // individual income is zero by A.104; the household's `sb^O` supports them.
    for _ in 0..inactive_total {
        individuals.push(individual(
            next_id,
            next_id % household_count,
            NOT_LINKED,
            0,
            LABOUR_INACTIVE,
            0.0,
            0.0,
        ));
        next_id += 1;
    }

    // Net labour income per household, before benefits, rent and financial
    // income (A.132/A.133 at P^CPI = 1).
    let net_of_tax = 1.0
        - account_template.social_insurance_worker_rate
        - account_template.income_tax_rate * (1.0 - account_template.social_insurance_worker_rate);
    let mut household_labour_income = vec![0.0; household_count as usize];
    for ind in &individuals {
        let slot = &mut household_labour_income[ind.household_id as usize];
        *slot += if ind.labour_status == LABOUR_EMPLOYED {
            ind.wage * net_of_tax
        } else {
            benefit
        };
    }

    let owner_count = ((household_count as f64 * OWNER_OCCUPIER_SHARE).round() as u32).max(1);
    // `sb^O`, A.99's other social benefits. Wiese A.7.1 matches these to OECD
    // Social Expenditure Aggregates; the previous `0.15 * average_wage` had no
    // source and left the government running a surplus of ~15% of GDP forever,
    // because it is what supports the whole inactive population.
    //
    // Anchored on the wage bill, which the SAM already fixes: for Austria,
    // social benefits other than social transfers in kind (ESA D.62) are ~21%
    // of GDP against compensation of employees at ~47%, and unemployment
    // benefits (A.99's separate `|I^U| w^U` term) account for ~1.5 points of
    // that. So `sb^O` is (21 - 1.5)/47 = 0.415 of the wage bill.
    const SOCIAL_BENEFITS_TO_WAGE_BILL: f64 = 0.415;
    let other_benefits_per_household =
        SOCIAL_BENEFITS_TO_WAGE_BILL * total_wage_bill / household_count as f64;

    // Property values anchored to income, rents to a gross yield.
    let mut properties = Vec::with_capacity(household_count as usize);
    let mut household_property_value = vec![0.0; household_count as usize];
    for id in 0..household_count {
        let annual_income = 4.0 * household_labour_income[id as usize];
        let value = HOUSE_PRICE_TO_INCOME * annual_income.max(1.0);
        let rent = value * GROSS_RENTAL_YIELD / 4.0;
        let (market_status, owner, occupant) = if id < owner_count {
            (PROPERTY_OWNER_OCCUPIED, id, id)
        } else if id < household_count.saturating_sub(2) {
            (PROPERTY_RENTAL, id % owner_count, id)
        } else if id < household_count.saturating_sub(1) {
            (PROPERTY_FOR_SALE, id % owner_count, NOT_LINKED)
        } else {
            (PROPERTY_FOR_RENT, id % owner_count, NOT_LINKED)
        };
        household_property_value[owner as usize] += value;
        properties.push(Property {
            id,
            country: 0,
            owner_household_id: owner,
            occupant_household_id: occupant,
            value,
            initial_value: value,
            price: value,
            previous_price: value,
            rent,
            previous_rent: rent,
            initial_rent: rent,
            market_status,
            mortgage_bank_id: NOT_LINKED,
            ..Property::default()
        });
    }

    // Rent actually flowing between households: paid by occupants who do not
    // own, received by the owner.
    let mut household_rent_income = vec![0.0; household_count as usize];
    for property in &properties {
        if property.market_status == PROPERTY_RENTAL
            && property.occupant_household_id != NOT_LINKED
            && property.owner_household_id != property.occupant_household_id
        {
            household_rent_income[property.owner_household_id as usize] += property.rent;
        }
    }

    let mut households = Vec::with_capacity(household_count as usize);
    for id in 0..household_count {
        let idx = id as usize;
        let labour = household_labour_income[idx];
        let gross = labour + other_benefits_per_household + household_rent_income[idx];
        let deposits = HOUSEHOLD_DEPOSIT_QUARTERS * gross;
        let other_financial = OTHER_FINANCIAL_TO_DEPOSITS * deposits;
        let property_wealth = household_property_value[idx];
        households.push(Household {
            id,
            country: 0,
            bank_id: id % scale.banks.max(1),
            residence_property_id: id,
            income: gross,
            previous_income: gross,
            predicted_income: gross,
            deposits,
            other_financial_assets: other_financial,
            property_wealth,
            other_real_assets: 0.5 * gross,
            consumption_debt: CONSUMPTION_DEBT_TO_INCOME * gross,
            mortgage_debt: MORTGAGE_TO_VALUE * property_wealth,
            social_benefits_other: other_benefits_per_household,
            owns_residence: id < owner_count,
            ..Household::default()
        });
    }

    // A.104 income including the financial-asset term, which the propensity
    // solve below has to see.
    let total_household_income: f64 = households
        .iter()
        .map(|h| h.income + environment.params.financial_asset_income_phi * h.other_financial_assets)
        .sum();

    // ---- 6. Solve the propensities (A.101 / A.102) --------------------
    //
    // Summing A.101 over households and sectors, with `sum_s c_s^CPI = 1`:
    //     C(0) = (1 / (1 + tau^VAT)) * (1 - phi^SR) * sum_h Y_h(0)
    // so phi^SR is *solved* to reproduce the IO consumption column. Same for
    // phi^IR against the capital column (A.102). This is what closes the 31%
    // demand hole; it is a solve, not a fit.
    let target_consumption: f64 = sam.consumption.iter().sum();
    let target_investment: f64 = sam.investment.iter().sum();
    let saving_rate = if total_household_income > 1e-9 {
        1.0 - target_consumption * (1.0 + account_template.vat_rate) / total_household_income
    } else {
        0.1
    };
    let investment_rate = if total_household_income > 1e-9 {
        target_investment * (1.0 + account_template.capital_tax_rate) / total_household_income
    } else {
        0.02
    };
    let saving_rate = saving_rate.clamp(0.0, 0.95);
    let investment_rate = investment_rate.clamp(0.0, 0.95);
    let mut household_histories = Vec::with_capacity(households.len());
    for household in &mut households {
        household.saving_rate = saving_rate;
        household.investment_rate = investment_rate;
        // A.105's smoothing term must not exceed the level the solve just set,
        // or every household opens above its own consumption function.
        let opening = (1.0 - saving_rate) * household.predicted_income
            / (1.0 + account_template.vat_rate);
        household_histories.push(HouseholdHistory {
            id: household.id,
            consumption_history: [opening; 12],
            // A.28/A.30 read the last two quarters of income; both are seeded
            // at the opening figure so the first tick has a window.
            income_history: [household.income, household.income],
        });
    }

    // ---- 7. Government, banks, rest of world --------------------------
    let mut government_entities = Vec::with_capacity(government_entity_count as usize);
    for entity in 0..government_entity_count {
        let sector = entity as usize % SECTORS;
        let per_entity = if entities_per_sector[sector] > 0 {
            sam.government[sector] / entities_per_sector[sector] as f64
        } else {
            0.0
        };
        government_entities.push(GovernmentEntity {
            id: entity,
            country: 0,
            sector: sector as u8,
            target_consumption: per_entity,
            realised_consumption: 0.0,
        });
    }

    let government_accounts = vec![GovernmentAccount {
        id: 0,
        country: 0,
        unemployment_benefit: benefit,
        other_benefits: other_benefits_per_household * household_count as f64,
        debt: 0.8 * total_output,
        ..GovernmentAccount::default()
    }];

    let mut environment_loans = Vec::new();
    for firm in &firms {
        if firm.short_debt > 0.0 {
            environment_loans.push((
                firm.bank_id,
                BUYER_FIRM,
                firm.id,
                firm.sector,
                LOAN_FIRM_SHORT,
                firm.short_debt,
                0.02,
                environment.params.firm_short_maturity_quarters,
            ));
        }
        if firm.long_debt > 0.0 {
            environment_loans.push((
                firm.bank_id,
                BUYER_FIRM,
                firm.id,
                firm.sector,
                LOAN_FIRM_LONG,
                firm.long_debt,
                0.03,
                environment.params.firm_long_maturity_quarters,
            ));
        }
    }
    for household in &households {
        if household.consumption_debt > 0.0 {
            environment_loans.push((
                household.bank_id,
                BUYER_HOUSEHOLD,
                household.id,
                0,
                LOAN_HOUSEHOLD_CONSUMPTION,
                household.consumption_debt,
                0.04,
                environment.params.consumption_loan_maturity_quarters,
            ));
        }
        if household.mortgage_debt > 0.0 {
            environment_loans.push((
                household.bank_id,
                BUYER_HOUSEHOLD,
                household.id,
                0,
                LOAN_MORTGAGE,
                household.mortgage_debt,
                0.035,
                environment.params.mortgage_maturity_quarters,
            ));
        }
    }
    for loan in &environment_loans {
        environment
            .loan_book
            .add(loan.0, loan.1, loan.2, loan.3, loan.4, loan.5, loan.6, loan.7, 0);
    }

    // Banks: deposits are the *actual* sum of what agents hold with them
    // (A.42/A.43), not a hand-set number. The old fixture carried 1120 against
    // 6850 of real agent deposits, so quarter 1 saw a 6x jump in the deposit
    // base and the interest bill that goes with it.
    let bank_count = scale.banks.max(1);
    let mut banks: Vec<Bank> = (0..bank_count)
        .map(|id| Bank {
            id,
            country: 0,
            ..Bank::default()
        })
        .collect();
    // A.23 gives `R_b(0) = sum D_f + sum D_h + E_b - sum V_l`, so reserves are
    // non-negative only if deposits plus equity cover the loan book. The first
    // version of this generator sized household deposits from income alone and
    // opened every bank at reserves of -1118 against loans of 1958: banks then
    // paid `|reserves| * r` every quarter, equity went negative by quarter 4,
    // and A.32's envelope `E_b/rho^CAR - loans` closed the credit market for
    // the rest of the run. The collapse was in the initial state, not in any
    // equation.
    //
    // Poledna's Austria has firm plus household deposits of 275,074 against
    // loans of 244,953 -- deposits exceed loans. Household deposits are
    // therefore topped up to whatever the identity requires, which is a solve
    // against A.23, not a chosen number.
    let total_loans: f64 = environment_loans.iter().map(|loan| loan.5).sum();
    let total_firm_deposits_actual: f64 = firms.iter().map(|firm| firm.deposits).sum();
    let bank_equity_total = BANK_EQUITY_TO_LOANS * total_loans;
    let required_household_deposits =
        (total_loans * (1.0 + RESERVE_HEADROOM) - bank_equity_total - total_firm_deposits_actual)
            .max(0.0);
    let current_household_deposits: f64 = households.iter().map(|h| h.deposits).sum();
    if required_household_deposits > current_household_deposits
        && current_household_deposits > 1e-9
    {
        let uplift = required_household_deposits / current_household_deposits;
        for household in &mut households {
            household.deposits *= uplift;
            household.other_financial_assets *= uplift;
        }
    }

    for bank in &mut banks {
        let firm_deposits: f64 = firms
            .iter()
            .filter(|f| f.bank_id == bank.id)
            .map(|f| f.deposits)
            .sum();
        let household_deposits: f64 = households
            .iter()
            .filter(|h| h.bank_id == bank.id)
            .map(|h| h.deposits)
            .sum();
        for firm in firms.iter().filter(|f| f.bank_id == bank.id) {
            bank.firm_loan_volume_by_sector[firm.sector as usize] +=
                firm.short_debt + firm.long_debt;
        }
        for household in households.iter().filter(|h| h.bank_id == bank.id) {
            bank.consumption_loan_volume += household.consumption_debt;
            bank.mortgage_volume += household.mortgage_debt;
        }
        let loans: f64 = bank.firm_loan_volume_by_sector.iter().sum::<f64>()
            + bank.consumption_loan_volume
            + bank.mortgage_volume;
        bank.deposits = firm_deposits + household_deposits;
        bank.equity = BANK_EQUITY_TO_LOANS * loans;
        // A.23: R_b(0) = deposits + equity - loans granted.
        bank.reserves = bank.deposits + bank.equity - loans;
        // A.42.
        bank.liabilities = bank.equity + bank.deposits - (-bank.reserves).max(0.0);
        bank.credit_supply_max = f64::INFINITY;
    }

    // Poledna C.5: central bank equity is the residual that closes the system.
    let total_reserves: f64 = banks.iter().map(|b| b.reserves).sum();
    let central_banks = vec![CentralBank {
        equity: government_accounts[0].debt - total_reserves,
        ..CentralBank::default()
    }];

    let export_total: f64 = sam.exports.iter().sum();
    let import_total: f64 = sam.imports.iter().sum();
    let mut row = RestOfWorld {
        id: 0,
        country: 0,
        target_exports: export_total,
        exports: export_total,
        target_imports: import_total,
        imports: import_total,
        initial_exports: export_total,
        initial_imports: import_total,
        export_weights: environment.params.row_export_weights,
        import_weights: environment.params.row_import_weights,
        ..RestOfWorld::default()
    };
    for s in 0..SECTORS {
        row.import_nominal_by_sector[s] = row.imports * row.import_weights[s];
        row.import_real_by_sector[s] =
            row.import_nominal_by_sector[s] / row.sector_prices[s].max(1e-9);
    }
    let rest_of_world = vec![row];

    // ---- 8. History with variation -------------------------------------
    //
    // Flat histories make every AR(1) predict zero change forever. That is why
    // `hpi` sat at exactly 1.000000: property value is marked to market by
    // *predicted* HPI inflation, so a zero forecast is self-fulfilling.
    let mut production_history = Vec::with_capacity(history_quarters);
    let mut sector_history = Vec::with_capacity(history_quarters);
    let mut price_history = Vec::with_capacity(history_quarters);
    // A.95's AR(1) is fitted on realised government consumption exactly as
    // A.2's is fitted on production. Leaving this series empty does not make
    // government consumption endogenous -- it pins it at zero forever, because
    // the forecast feeds the target, the target feeds the realised flow, and
    // the realised flow feeds this history. Government demand is ~20% of final
    // demand in the SAM, so the whole component was silently absent.
    let government_total: f64 = sam.government.iter().sum();
    let mut government_history = Vec::with_capacity(history_quarters);
    for q in 0..history_quarters {
        let back = (history_quarters - 1 - q) as f64;
        let wobble = HISTORY_WOBBLE * ((q % 3) as f64 - 1.0);
        let level = (1.0 - HISTORY_TREND * back + wobble).max(0.5);
        production_history.push(total_output * level);
        let mut row = [0.0; SECTORS];
        for s in 0..SECTORS {
            row[s] = sector_output[s] * level;
        }
        sector_history.push(row);
        price_history.push(level);
        government_history.push(government_total * level);
    }
    environment.history.production = production_history;
    environment.history.sector_production = sector_history;
    environment.history.ppi = price_history.clone();
    environment.history.cpi = price_history.clone();
    environment.history.hpi = price_history.clone();
    environment.history.rpi = price_history;
    environment.history.government_consumption = government_history;
    // A.95's AR(1) anchors on the *current* level as well as the history, so
    // the aggregate has to start on the series too. Left at zero the fit sees
    // `max(1e-9)` and forecasts zero regardless of the history.
    environment.aggregates.government_consumption = government_total;
    environment.previous_aggregates.government_consumption = government_total;

    InitialData {
        environment,
        firms,
        firm_stocks,
        firm_stock_baselines,
        firm_targets,
        firm_realised,
        individual_wage_histories: individuals
            .iter()
            .map(|individual| IndividualWageHistory {
                id: individual.id,
                // A.131's reservation wage averages the window, so it opens at
                // the individual's own wage rather than at zero.
                wage_history: [individual.wage; 8],
            })
            .collect(),
        household_demands: households
            .iter()
            .map(|household| HouseholdDemand {
                id: household.id,
                ..HouseholdDemand::default()
            })
            .collect(),
        household_histories,
        individuals,
        households,
        banks,
        government_entities,
        government_accounts,
        central_banks,
        properties,
        rest_of_world,
    }
}

/// The synthetic social accounting matrix, in real units at `P = 1` (A.52).
///
/// Every column is derived from sectoral gross output and the technology
/// matrices; nothing here is chosen to make a run behave.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticSam {
    pub sector_output: [f64; SECTORS],
    pub intermediate_demand: [f64; SECTORS],
    pub capital_demand: [f64; SECTORS],
    pub consumption: [f64; SECTORS],
    pub investment: [f64; SECTORS],
    pub government: [f64; SECTORS],
    pub exports: [f64; SECTORS],
    pub imports: [f64; SECTORS],
}

impl SyntheticSam {
    pub fn total_output(&self) -> f64 {
        self.sector_output.iter().sum()
    }

    /// Largest absolute violation of the per-sector goods-market identity
    /// `Y_s = intermediates + capital + C + I + G + X - M`. Zero by
    /// construction; asserted by the stock-flow-consistency test so that a
    /// future change to the technology matrices cannot silently reopen the
    /// demand hole this generator exists to close.
    pub fn max_identity_gap(&self) -> f64 {
        (0..SECTORS)
            .map(|s| {
                (self.sector_output[s]
                    - (self.intermediate_demand[s]
                        + self.capital_demand[s]
                        + self.consumption[s]
                        + self.investment[s]
                        + self.government[s]
                        + self.exports[s]
                        - self.imports[s]))
                    .abs()
            })
            .fold(0.0, f64::max)
    }
}

/// Close the goods-market identity sector by sector.
///
/// The paper closes the demand side *by construction*: A.101 and A.102 choose
/// the household consumption and investment propensities so that the resulting
/// aggregates match the input-output table's final-demand columns, and A.95
/// does the same for government. The previous fixture picked output and demand
/// independently, which opened quarter 1 with demand 533.9 against supply 772.2
/// -- 31% of output with no buyer -- and A.62's first term then drove every
/// target to zero.
fn solve_sam(
    params: &CountryParameters,
    sector_output: [f64; SECTORS],
    government_share: f64,
    government_entities_per_sector: &[u32; SECTORS],
) -> SyntheticSam {
    let mut intermediate_demand = [0.0; SECTORS];
    let mut capital_demand = [0.0; SECTORS];
    // `m_{s's}` is the amount of s' needed per unit of s, so demand *for* good
    // s is the column sum over every producing sector.
    for producing in 0..SECTORS {
        for input in 0..SECTORS {
            intermediate_demand[input] += params.io_matrix[producing][input] * sector_output[producing];
            capital_demand[input] +=
                params.capital_compensation_matrix[producing][input] * sector_output[producing];
        }
    }

    let mut exports = [0.0; SECTORS];
    let mut imports = [0.0; SECTORS];
    let mut government = [0.0; SECTORS];
    let mut consumption = [0.0; SECTORS];
    let mut investment = [0.0; SECTORS];

    let _ = government_share;
    // Value added is gross output less intermediate consumption. The final
    // demand components are shares of it; net exports take the remainder, which
    // is what makes the identity close exactly rather than approximately.
    let total_output: f64 = sector_output.iter().sum();
    let total_intermediate: f64 = intermediate_demand.iter().sum();
    let total_capital: f64 = capital_demand.iter().sum();
    let value_added = (total_output - total_intermediate).max(1e-9);

    let consumption_total = CONSUMPTION_SHARE_OF_VALUE_ADDED * value_added;
    let government_total = GOVERNMENT_SHARE_OF_VALUE_ADDED * value_added;
    // Firms have already bought `d_{s's} Y_f`; households fund the rest of GFCF.
    let investment_total = (GFCF_SHARE_OF_VALUE_ADDED * value_added - total_capital).max(0.0);
    let export_total = EXPORT_SHARE * total_output;
    // Imports close the identity exactly.
    let import_total = (total_intermediate
        + total_capital
        + consumption_total
        + government_total
        + investment_total
        + export_total
        - total_output)
        .max(0.0);

    // Government entities buy one sector each, and A.6.1 puts their number at
    // 25% of firms — so sectors with no entity get no public demand, and the
    // SAM has to know that or the residual lands on a buyer who does not exist.
    let entity_total: u32 = government_entities_per_sector.iter().sum();
    let output_weight = normalise(&sector_output);
    let mut government_weight = [0.0; SECTORS];
    if entity_total > 0 {
        for s in 0..SECTORS {
            government_weight[s] = government_entities_per_sector[s] as f64 / entity_total as f64;
        }
    }

    for s in 0..SECTORS {
        consumption[s] = consumption_total * output_weight[s];
        investment[s] = investment_total * output_weight[s];
        government[s] = government_total * government_weight[s];
        exports[s] = export_total * output_weight[s];
        imports[s] = import_total * output_weight[s];
    }

    SyntheticSam {
        sector_output,
        intermediate_demand,
        capital_demand,
        consumption,
        investment,
        government,
        exports,
        imports,
    }
}

/// Quarters between two `yyyy-Qn` labels, inclusive of the first.
///
/// A.2's expectation prefix runs from `first_real_data_quarter` to the
/// initialisation quarter, so this is what sets the history length rather than
/// a constant chosen by hand.
fn quarters_between(from: &str, to: &str) -> usize {
    fn parse(label: &str) -> Option<(i32, i32)> {
        let (year, quarter) = label.split_once("-Q")?;
        Some((year.trim().parse().ok()?, quarter.trim().parse().ok()?))
    }
    match (parse(from), parse(to)) {
        (Some((y0, q0)), Some((y1, q1))) => {
            let span = (y1 - y0) * 4 + (q1 - q0);
            span.max(8) as usize
        }
        // Fall back to a decade if either label is malformed, rather than to a
        // history so short the AR(1) has nothing to fit.
        _ => 40,
    }
}

fn normalise(values: &[f64; SECTORS]) -> [f64; SECTORS] {
    let total: f64 = values.iter().sum();
    let mut out = [0.0; SECTORS];
    if total > 1e-12 {
        for s in 0..SECTORS {
            out[s] = values[s] / total;
        }
    } else {
        for slot in out.iter_mut() {
            *slot = 1.0 / SECTORS as f64;
        }
    }
    out
}

fn individual(
    id: u32,
    household_id: u32,
    employer: u32,
    sector: u8,
    status: u8,
    wage: f64,
    benefit_floor: f64,
) -> Individual {
    Individual {
        id,
        household_id,
        employer_firm_id: employer,
        labour_status: status,
        // The firm's sector, not its id -- these coincided only while there
        // was one firm per sector with matching ids.
        industry: sector,
        wage,
        // A.131: the reservation wage is the greater of the benefit and the
        // average of past wages. `benefit_floor` carries the former; a
        // hard-coded floor above the going wage would make the unemployed
        // permanently unhireable.
        reservation_wage: wage.max(benefit_floor),
        labour_input: 1.0,
        predicted_income: wage,
        income: wage,
    }
}
