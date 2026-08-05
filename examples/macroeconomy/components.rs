pub const SECTORS: usize = 18;
pub const SIM_SCALE_FACTOR: f64 = 1_000.0;
pub const NOT_LINKED: u32 = u32::MAX;

pub const LABOUR_INACTIVE: u8 = 0;
pub const LABOUR_UNEMPLOYED: u8 = 1;
pub const LABOUR_EMPLOYED: u8 = 2;

pub const PROPERTY_OWNER_OCCUPIED: u8 = 0;
pub const PROPERTY_RENTAL: u8 = 1;
pub const PROPERTY_FOR_SALE: u8 = 2;
pub const PROPERTY_FOR_RENT: u8 = 3;
pub const PROPERTY_TENTATIVE_SALE: u8 = 4;
pub const PROPERTY_TENTATIVE_RENT: u8 = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Firm {
    pub id: u32,
    pub sector: u8,
    pub country: u16,
    pub bank_id: u32,
    pub employees: u32,
    pub labour: f64,
    pub target_labour: f64,
    pub wage: f64,
    /// `w_bar_f`: the wage before the A.66 work-effort factor is applied.
    ///
    /// Only read when `CountryParameters::wage_effort_on_base` selects the
    /// Poledna A.26 form, where work effort scales a base wage instead of
    /// compounding onto the previous one.
    pub base_wage: f64,
    pub reservation_wage_anchor: f64,
    pub production: f64,
    pub previous_production: f64,
    pub initial_production: f64,
    pub target_production: f64,
    pub demand: f64,
    pub previous_demand: f64,
    pub target_demand: f64,
    pub price: f64,
    pub previous_price: f64,
    pub inventory: f64,
    pub previous_inventory: f64,
    pub inventory_two_periods_ago: f64,
    pub initial_inventory: f64,
    pub intermediate_stock: [f64; SECTORS],
    pub initial_intermediate_stock: [f64; SECTORS],
    pub capital_stock: [f64; SECTORS],
    pub initial_capital_stock: [f64; SECTORS],
    pub target_intermediate: [f64; SECTORS],
    pub target_capital: [f64; SECTORS],
    pub realised_intermediate: [f64; SECTORS],
    pub realised_capital: [f64; SECTORS],
    pub capital_to_install: [f64; SECTORS],
    pub target_short_loan: f64,
    pub target_long_loan: f64,
    pub granted_short_loan: f64,
    pub granted_long_loan: f64,
    pub deposits: f64,
    pub short_debt: f64,
    pub long_debt: f64,
    pub overdraft: f64,
    pub equity: f64,
    pub profits: f64,
    pub predicted_profits: f64,
    pub costs: f64,
    pub unit_cost: f64,
    pub work_effort: f64,
    pub initial_work_effort: f64,
    pub target_labour_gap_history: [f64; 8],
    pub sales_quantity: f64,
    pub sales_revenue: f64,
    pub excess_demand: f64,
    pub bankrupt: bool,
}

impl Default for Firm {
    fn default() -> Self {
        Self {
            id: 0,
            sector: 0,
            country: 0,
            bank_id: NOT_LINKED,
            employees: 0,
            labour: 0.0,
            target_labour: 0.0,
            wage: 0.0,
            base_wage: 0.0,
            reservation_wage_anchor: 0.0,
            production: 0.0,
            previous_production: 0.0,
            initial_production: 1.0,
            target_production: 0.0,
            demand: 0.0,
            previous_demand: 0.0,
            target_demand: 0.0,
            price: 1.0,
            previous_price: 1.0,
            inventory: 0.0,
            previous_inventory: 0.0,
            inventory_two_periods_ago: 0.0,
            initial_inventory: 0.0,
            intermediate_stock: [0.0; SECTORS],
            initial_intermediate_stock: [0.0; SECTORS],
            capital_stock: [0.0; SECTORS],
            initial_capital_stock: [0.0; SECTORS],
            target_intermediate: [0.0; SECTORS],
            target_capital: [0.0; SECTORS],
            realised_intermediate: [0.0; SECTORS],
            realised_capital: [0.0; SECTORS],
            capital_to_install: [0.0; SECTORS],
            target_short_loan: 0.0,
            target_long_loan: 0.0,
            granted_short_loan: 0.0,
            granted_long_loan: 0.0,
            deposits: 0.0,
            short_debt: 0.0,
            long_debt: 0.0,
            overdraft: 0.0,
            equity: 0.0,
            profits: 0.0,
            predicted_profits: 0.0,
            costs: 0.0,
            unit_cost: 1.0,
            work_effort: 1.0,
            initial_work_effort: 1.0,
            target_labour_gap_history: [0.0; 8],
            sales_quantity: 0.0,
            sales_revenue: 0.0,
            excess_demand: 0.0,
            bankrupt: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Individual {
    pub id: u32,
    pub household_id: u32,
    pub employer_firm_id: u32,
    pub labour_status: u8,
    pub industry: u8,
    pub wage: f64,
    pub reservation_wage: f64,
    pub labour_input: f64,
    pub predicted_income: f64,
    pub income: f64,
    pub wage_history: [f64; 8],
}

impl Default for Individual {
    fn default() -> Self {
        Self {
            id: 0,
            household_id: NOT_LINKED,
            employer_firm_id: NOT_LINKED,
            labour_status: LABOUR_INACTIVE,
            industry: 0,
            wage: 0.0,
            reservation_wage: 0.0,
            labour_input: 1.0,
            predicted_income: 0.0,
            income: 0.0,
            wage_history: [0.0; 8],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Household {
    pub id: u32,
    pub country: u16,
    pub bank_id: u32,
    pub residence_property_id: u32,
    pub desired_property_id: u32,
    pub income: f64,
    pub previous_income: f64,
    pub income_history: [f64; 2],
    pub predicted_income: f64,
    pub consumption_target: [f64; SECTORS],
    pub investment_target: [f64; SECTORS],
    pub consumption_history: [f64; 12],
    pub saving_rate: f64,
    pub investment_rate: f64,
    pub deposits: f64,
    pub other_financial_assets: f64,
    pub property_wealth: f64,
    pub other_real_assets: f64,
    pub consumption_debt: f64,
    pub mortgage_debt: f64,
    pub net_wealth: f64,
    pub desired_consumption_loan: f64,
    pub granted_consumption_loan: f64,
    pub desired_mortgage: f64,
    pub granted_mortgage: f64,
    pub consumption_gap_after_financial_assets: f64,
    pub disposable_income_after_rent: f64,
    pub desired_house_price: f64,
    pub desired_rent: f64,
    /// Dividends received last quarter (see `CountryParameters::theta_dividend`).
    pub dividend_income: f64,
    pub social_benefits_other: f64,
    pub owns_residence: bool,
    pub bankrupt: bool,
}

impl Default for Household {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            bank_id: NOT_LINKED,
            residence_property_id: NOT_LINKED,
            desired_property_id: NOT_LINKED,
            income: 0.0,
            previous_income: 0.0,
            income_history: [0.0; 2],
            predicted_income: 0.0,
            consumption_target: [0.0; SECTORS],
            investment_target: [0.0; SECTORS],
            consumption_history: [0.0; 12],
            saving_rate: 0.1,
            investment_rate: 0.05,
            deposits: 0.0,
            other_financial_assets: 0.0,
            property_wealth: 0.0,
            other_real_assets: 0.0,
            consumption_debt: 0.0,
            mortgage_debt: 0.0,
            net_wealth: 0.0,
            desired_consumption_loan: 0.0,
            granted_consumption_loan: 0.0,
            desired_mortgage: 0.0,
            granted_mortgage: 0.0,
            consumption_gap_after_financial_assets: 0.0,
            disposable_income_after_rent: 0.0,
            desired_house_price: 0.0,
            desired_rent: 0.0,
            dividend_income: 0.0,
            social_benefits_other: 0.0,
            owns_residence: false,
            bankrupt: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bank {
    pub id: u32,
    pub country: u16,
    pub reserves: f64,
    pub equity: f64,
    pub liabilities: f64,
    pub profit: f64,
    pub deposits: f64,
    pub firm_loan_volume_by_sector: [f64; SECTORS],
    pub consumption_loan_volume: f64,
    pub mortgage_volume: f64,
    pub credit_supply_max: f64,
    pub short_firm_rate: f64,
    pub long_firm_rate: f64,
    pub household_rate: f64,
    pub mortgage_rate: f64,
    pub deposit_rate: f64,
    pub household_overdraft_rate: f64,
    pub firm_overdraft_rate: f64,
    pub npl_firm_by_sector: [f64; SECTORS],
    pub npl_consumption: f64,
    pub npl_mortgage: f64,
    pub insolvent: bool,
}

impl Default for Bank {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            reserves: 0.0,
            equity: 0.0,
            liabilities: 0.0,
            profit: 0.0,
            deposits: 0.0,
            firm_loan_volume_by_sector: [0.0; SECTORS],
            consumption_loan_volume: 0.0,
            mortgage_volume: 0.0,
            // No cap beyond the regulatory one. `bank_credit_supply` already
            // binds lending to `equity / CAR` less outstanding loans; this
            // field is an *additional* institution-specific ceiling. Defaulting
            // it to 0.0 meant any bank not explicitly given a value lent
            // nothing at all, silently.
            credit_supply_max: f64::INFINITY,
            short_firm_rate: 0.02,
            long_firm_rate: 0.03,
            household_rate: 0.04,
            mortgage_rate: 0.035,
            deposit_rate: 0.01,
            household_overdraft_rate: 0.04,
            firm_overdraft_rate: 0.02,
            npl_firm_by_sector: [0.0; SECTORS],
            npl_consumption: 0.0,
            npl_mortgage: 0.0,
            insolvent: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GovernmentEntity {
    pub id: u32,
    pub country: u16,
    pub sector: u8,
    pub target_consumption: f64,
    pub realised_consumption: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GovernmentAccount {
    pub id: u32,
    pub country: u16,
    pub unemployment_benefit: f64,
    pub other_benefits: f64,
    pub revenue: f64,
    pub deficit: f64,
    pub debt: f64,
    pub vat_rate: f64,
    pub income_tax_rate: f64,
    pub social_insurance_worker_rate: f64,
    pub social_insurance_firm_rate: f64,
    pub capital_tax_rate: f64,
    pub corporate_tax_rate: f64,
    pub export_tax_rate: f64,
    pub production_tax_rate: f64,
    pub production_tax_by_sector: [f64; SECTORS],
}

impl Default for GovernmentAccount {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            unemployment_benefit: 1.0,
            other_benefits: 0.0,
            revenue: 0.0,
            deficit: 0.0,
            debt: 0.0,
            // Wiese A.6.2: "The income tax rate, corporate tax rate, export
            // taxes, value-added tax rate, and social insurance rates are taken
            // directly from the OECD database." These were round-number
            // placeholders; the values are Poledna Table 1 for Austria.
            vat_rate: 0.1529,
            income_tax_rate: 0.2134,
            social_insurance_worker_rate: 0.1711,
            social_insurance_firm_rate: 0.2122,
            capital_tax_rate: 0.2521,
            corporate_tax_rate: 0.0779,
            export_tax_rate: 0.003,
            production_tax_rate: 0.02,
            production_tax_by_sector: [0.02; SECTORS],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentralBank {
    pub id: u32,
    pub country: u16,
    /// `E^CB(0)`. The residual that closes the initial balance sheet: central
    /// bank assets (government debt) less reserves held for banks and the net
    /// position with the rest of the world (Poledna Appendix C.5). Without it
    /// there is nowhere for the stock-flow closure to land, which is why the
    /// hand-written fixture could carry bank deposits of 1120 against 6850 of
    /// actual agent deposits and nothing complained.
    pub equity: f64,
    pub policy_rate: f64,
    pub predicted_policy_rate: f64,
    pub inflation_target: f64,
    pub natural_rate: f64,
    pub rho: f64,
    pub xi_pi: f64,
    pub xi_gamma: f64,
}

impl Default for CentralBank {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            equity: 0.0,
            policy_rate: 0.01,
            predicted_policy_rate: 0.01,
            inflation_target: 0.02,
            natural_rate: 0.01,
            rho: 0.8,
            xi_pi: 1.5,
            xi_gamma: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Property {
    pub id: u32,
    pub country: u16,
    pub owner_household_id: u32,
    pub occupant_household_id: u32,
    pub value: f64,
    pub initial_value: f64,
    pub price: f64,
    pub previous_price: f64,
    pub rent: f64,
    pub previous_rent: f64,
    pub initial_rent: f64,
    pub predicted_annual_rent_price: f64,
    pub predicted_annual_buy_price: f64,
    pub predicted_rental_yield: f64,
    pub market_status: u8,
    pub quarters_on_sale: u32,
    pub quarters_on_rent_market: u32,
    pub mortgage_bank_id: u32,
}

impl Default for Property {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            owner_household_id: NOT_LINKED,
            occupant_household_id: NOT_LINKED,
            value: 0.0,
            initial_value: 1.0,
            price: 0.0,
            previous_price: 0.0,
            rent: 0.0,
            previous_rent: 0.0,
            initial_rent: 1.0,
            predicted_annual_rent_price: 0.0,
            predicted_annual_buy_price: 0.0,
            predicted_rental_yield: 0.0,
            market_status: PROPERTY_FOR_SALE,
            quarters_on_sale: 0,
            quarters_on_rent_market: 0,
            mortgage_bank_id: NOT_LINKED,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestOfWorld {
    pub id: u32,
    pub country: u16,
    pub target_exports: f64,
    pub exports: f64,
    pub target_imports: f64,
    pub imports: f64,
    /// `Y^ROW(0)` and `C^ROW(0)`. A.137 and A.138 index target trade to these
    /// initial levels, not to the previous quarter's realised flow.
    pub initial_exports: f64,
    pub initial_imports: f64,
    pub import_nominal_by_sector: [f64; SECTORS],
    pub import_real_by_sector: [f64; SECTORS],
    pub export_weights: [f64; SECTORS],
    pub import_weights: [f64; SECTORS],
    pub sector_prices: [f64; SECTORS],
    pub net_exports: f64,
    pub adjustment_speed: f64,
}

impl Default for RestOfWorld {
    fn default() -> Self {
        Self {
            id: 0,
            country: 0,
            target_exports: 0.0,
            exports: 0.0,
            target_imports: 0.0,
            imports: 0.0,
            initial_exports: 0.0,
            initial_imports: 0.0,
            import_nominal_by_sector: [0.0; SECTORS],
            import_real_by_sector: [0.0; SECTORS],
            export_weights: [1.0 / SECTORS as f64; SECTORS],
            import_weights: [1.0 / SECTORS as f64; SECTORS],
            sector_prices: [1.0; SECTORS],
            net_exports: 0.0,
            adjustment_speed: 1.0,
        }
    }
}
