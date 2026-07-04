use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use super::components::{
    Bank, CentralBank, Firm, GovernmentAccount, GovernmentEntity, Household, Individual, Property,
    RestOfWorld, LABOUR_EMPLOYED, LABOUR_UNEMPLOYED, NOT_LINKED, PROPERTY_FOR_RENT,
    PROPERTY_FOR_SALE, PROPERTY_OWNER_OCCUPIED, PROPERTY_RENTAL, SECTORS,
};
use super::messages::{
    BUYER_FIRM, BUYER_HOUSEHOLD, LOAN_FIRM_LONG, LOAN_FIRM_SHORT, LOAN_HOUSEHOLD_CONSUMPTION,
    LOAN_MORTGAGE,
};
use super::state::{MacroEnvironment, MacroeconomyConfig};

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
        let mut data = tiny_fixture(config.seed);
        data.environment.policy = config.replication_policy.clone();
        Ok(data)
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
        }
    }
}

impl Error for DataError {}

fn tiny_fixture(seed: u64) -> InitialData {
    let mut environment = MacroEnvironment::new(seed);
    environment.history.production = vec![90.0, 94.0, 99.0, 105.0];
    environment.history.ppi = vec![0.98, 1.0, 1.01, 1.02];
    environment.history.cpi = vec![0.99, 1.0, 1.02, 1.03];
    environment.history.hpi = vec![0.95, 1.0, 1.04, 1.05];
    environment.history.rpi = vec![0.97, 1.0, 1.01, 1.02];
    environment.history.sector_production = vec![
        sector_row(4.8),
        sector_row(5.0),
        sector_row(5.2),
        sector_row(5.5),
    ];

    let mut firms = Vec::new();
    for (id, sector, price, production, deposits) in [
        (0, 0, 0.92, 34.0, 180.0),
        (1, 1, 1.06, 38.0, 160.0),
        (2, 2, 1.00, 42.0, 210.0),
        (3, 0, 1.20, 18.0, 120.0),
    ] {
        let mut firm = Firm {
            id,
            sector,
            country: 0,
            bank_id: id % 2,
            employees: 1,
            labour: 1.0,
            target_labour: 1.0,
            wage: 18.0 + id as f64,
            reservation_wage_anchor: 15.0,
            production,
            previous_production: production,
            initial_production: production,
            target_production: production,
            demand: production,
            previous_demand: production,
            target_demand: production,
            price,
            previous_price: price,
            inventory: production * 0.20,
            previous_inventory: production * 0.20,
            inventory_two_periods_ago: production * 0.20,
            initial_inventory: production * 0.20,
            deposits,
            short_debt: 20.0 + id as f64 * 5.0,
            long_debt: 40.0 + id as f64 * 5.0,
            equity: 120.0,
            profits: 8.0,
            predicted_profits: 8.0,
            unit_cost: 0.7,
            ..Firm::default()
        };
        for s in 0..SECTORS {
            firm.intermediate_stock[s] = if s == sector as usize {
                production * 0.2
            } else {
                1.0
            };
            firm.initial_intermediate_stock[s] = firm.intermediate_stock[s];
            firm.capital_stock[s] = if s == sector as usize {
                production * 0.6
            } else {
                1.0
            };
            firm.initial_capital_stock[s] = firm.capital_stock[s];
        }
        firms.push(firm);
    }

    let households = vec![
        Household {
            id: 0,
            country: 0,
            bank_id: 0,
            residence_property_id: 0,
            deposits: 140.0,
            other_financial_assets: 50.0,
            property_wealth: 260.0,
            other_real_assets: 20.0,
            consumption_debt: 5.0,
            mortgage_debt: 60.0,
            saving_rate: 0.15,
            investment_rate: 0.03,
            social_benefits_other: 2.0,
            previous_income: 30.0,
            income_history: [30.0, 30.0],
            owns_residence: true,
            ..Household::default()
        },
        Household {
            id: 1,
            country: 0,
            bank_id: 1,
            residence_property_id: 1,
            deposits: 90.0,
            other_financial_assets: 25.0,
            property_wealth: 0.0,
            other_real_assets: 10.0,
            consumption_debt: 8.0,
            mortgage_debt: 0.0,
            saving_rate: 0.05,
            investment_rate: 0.02,
            social_benefits_other: 1.0,
            previous_income: 24.0,
            income_history: [24.0, 24.0],
            owns_residence: false,
            ..Household::default()
        },
        Household {
            id: 2,
            country: 0,
            bank_id: 0,
            residence_property_id: NOT_LINKED,
            deposits: 70.0,
            other_financial_assets: 10.0,
            property_wealth: 0.0,
            other_real_assets: 8.0,
            consumption_debt: 4.0,
            mortgage_debt: 0.0,
            saving_rate: 0.02,
            investment_rate: 0.01,
            social_benefits_other: 1.0,
            previous_income: 14.0,
            income_history: [14.0, 14.0],
            owns_residence: false,
            ..Household::default()
        },
    ];

    let individuals = vec![
        individual(0, 0, 0, LABOUR_EMPLOYED, 18.0),
        individual(1, 1, 1, LABOUR_EMPLOYED, 19.0),
        individual(2, 2, 2, LABOUR_EMPLOYED, 20.0),
        individual(3, 2, NOT_LINKED, LABOUR_UNEMPLOYED, 0.0),
    ];

    let mut banks = vec![
        Bank {
            id: 0,
            country: 0,
            reserves: 120.0,
            equity: 180.0,
            liabilities: 900.0,
            deposits: 600.0,
            credit_supply_max: 1_000.0,
            ..Bank::default()
        },
        Bank {
            id: 1,
            country: 0,
            reserves: 100.0,
            equity: 160.0,
            liabilities: 800.0,
            deposits: 520.0,
            credit_supply_max: 900.0,
            ..Bank::default()
        },
    ];

    let government_entity_count = ((firms.len() as f64) * 0.25).round().max(1.0) as u32;
    let government_entities = (0..government_entity_count)
        .map(|id| GovernmentEntity {
            id,
            country: 0,
            sector: (id as usize % SECTORS) as u8,
            target_consumption: 8.0,
            realised_consumption: 0.0,
        })
        .collect();

    let government_accounts = vec![GovernmentAccount {
        id: 0,
        country: 0,
        unemployment_benefit: 6.0,
        other_benefits: 4.0,
        debt: 500.0,
        ..GovernmentAccount::default()
    }];

    let central_banks = vec![CentralBank::default()];

    let properties = vec![
        Property {
            id: 0,
            country: 0,
            owner_household_id: 0,
            occupant_household_id: 0,
            value: 260.0,
            initial_value: 250.0,
            price: 265.0,
            previous_price: 265.0,
            rent: 10.0,
            previous_rent: 10.0,
            initial_rent: 10.0,
            predicted_annual_rent_price: 40.0,
            predicted_annual_buy_price: 265.0,
            predicted_rental_yield: 40.0 / 265.0,
            market_status: PROPERTY_OWNER_OCCUPIED,
            mortgage_bank_id: 0,
            ..Property::default()
        },
        Property {
            id: 1,
            country: 0,
            owner_household_id: 0,
            occupant_household_id: 1,
            value: 160.0,
            initial_value: 150.0,
            price: 166.0,
            previous_price: 166.0,
            rent: 8.0,
            previous_rent: 8.0,
            initial_rent: 8.0,
            predicted_annual_rent_price: 32.0,
            predicted_annual_buy_price: 166.0,
            predicted_rental_yield: 32.0 / 166.0,
            market_status: PROPERTY_RENTAL,
            mortgage_bank_id: NOT_LINKED,
            ..Property::default()
        },
        Property {
            id: 2,
            country: 0,
            owner_household_id: 0,
            occupant_household_id: NOT_LINKED,
            value: 190.0,
            initial_value: 190.0,
            price: 190.0,
            previous_price: 190.0,
            rent: 9.0,
            previous_rent: 9.0,
            initial_rent: 9.0,
            predicted_annual_rent_price: 36.0,
            predicted_annual_buy_price: 190.0,
            predicted_rental_yield: 36.0 / 190.0,
            market_status: PROPERTY_FOR_SALE,
            mortgage_bank_id: NOT_LINKED,
            ..Property::default()
        },
        Property {
            id: 3,
            country: 0,
            owner_household_id: 0,
            occupant_household_id: NOT_LINKED,
            value: 130.0,
            initial_value: 130.0,
            price: 132.0,
            previous_price: 132.0,
            rent: 7.0,
            previous_rent: 7.0,
            initial_rent: 7.0,
            predicted_annual_rent_price: 28.0,
            predicted_annual_buy_price: 132.0,
            predicted_rental_yield: 28.0 / 132.0,
            market_status: PROPERTY_FOR_RENT,
            mortgage_bank_id: NOT_LINKED,
            ..Property::default()
        },
    ];

    let mut rest_of_world_agent = RestOfWorld {
        id: 0,
        country: 0,
        target_exports: 20.0,
        exports: 20.0,
        target_imports: 15.0,
        imports: 15.0,
        ..RestOfWorld::default()
    };
    for s in 0..SECTORS {
        rest_of_world_agent.import_nominal_by_sector[s] =
            rest_of_world_agent.imports * rest_of_world_agent.import_weights[s];
        rest_of_world_agent.import_real_by_sector[s] = rest_of_world_agent.import_nominal_by_sector
            [s]
            / rest_of_world_agent.sector_prices[s].max(1e-9);
    }
    let rest_of_world = vec![rest_of_world_agent];

    for firm in &firms {
        if firm.short_debt > 0.0 {
            environment.loan_book.add(
                firm.bank_id,
                BUYER_FIRM,
                firm.id,
                firm.sector,
                LOAN_FIRM_SHORT,
                firm.short_debt,
                0.02,
                environment.params.firm_short_maturity_quarters,
                0,
            );
        }
        if firm.long_debt > 0.0 {
            environment.loan_book.add(
                firm.bank_id,
                BUYER_FIRM,
                firm.id,
                firm.sector,
                LOAN_FIRM_LONG,
                firm.long_debt,
                0.03,
                environment.params.firm_long_maturity_quarters,
                0,
            );
        }
    }
    for household in &households {
        if household.consumption_debt > 0.0 {
            environment.loan_book.add(
                household.bank_id,
                BUYER_HOUSEHOLD,
                household.id,
                0,
                LOAN_HOUSEHOLD_CONSUMPTION,
                household.consumption_debt,
                0.04,
                environment.params.consumption_loan_maturity_quarters,
                0,
            );
        }
        if household.mortgage_debt > 0.0 {
            environment.loan_book.add(
                household.bank_id,
                BUYER_HOUSEHOLD,
                household.id,
                0,
                LOAN_MORTGAGE,
                household.mortgage_debt,
                0.035,
                environment.params.mortgage_maturity_quarters,
                0,
            );
        }
    }
    for bank in &mut banks {
        for firm in firms.iter().filter(|firm| firm.bank_id == bank.id) {
            bank.firm_loan_volume_by_sector[firm.sector as usize] +=
                firm.short_debt + firm.long_debt;
        }
        for household in households
            .iter()
            .filter(|household| household.bank_id == bank.id)
        {
            bank.consumption_loan_volume += household.consumption_debt;
            bank.mortgage_volume += household.mortgage_debt;
        }
    }

    InitialData {
        environment,
        firms,
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

fn sector_row(value: f64) -> [f64; SECTORS] {
    let mut row = [0.0; SECTORS];
    for item in row.iter_mut().take(3) {
        *item = value;
    }
    row
}

fn individual(id: u32, household_id: u32, employer: u32, status: u8, wage: f64) -> Individual {
    Individual {
        id,
        household_id,
        employer_firm_id: employer,
        labour_status: status,
        industry: if employer == NOT_LINKED {
            0
        } else {
            employer as u8
        },
        wage,
        reservation_wage: wage.max(6.0),
        labour_input: 1.0,
        predicted_income: wage,
        income: wage,
        wage_history: [wage; 8],
    }
}
