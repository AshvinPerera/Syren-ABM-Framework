use super::accounting::{negative_abs, positive_part};
use super::components::SECTORS;

pub fn ratio(num: f64, den: f64) -> f64 {
    if den.abs() <= 1e-12 {
        0.0
    } else {
        num / den
    }
}

pub fn log_growth(new_value: f64, old_value: f64) -> f64 {
    if new_value > 0.0 && old_value > 0.0 {
        (new_value / old_value).ln()
    } else {
        0.0
    }
}

// A.3
pub fn ppi_a3(firm_value: f64, imports_nominal: f64, firm_quantity: f64, imports_real: f64) -> f64 {
    ratio(firm_value + imports_nominal, firm_quantity + imports_real).max(1e-9)
}

// A.5
pub fn sector_price_a5(
    firm_value: f64,
    imports_nominal: f64,
    firm_quantity: f64,
    imports_real: f64,
) -> f64 {
    ratio(firm_value + imports_nominal, firm_quantity + imports_real).max(1e-9)
}

// A.59 and A.74 share the same ratio. The caller supplies the paper's case condition.
pub fn idiosyncratic_growth_a59(
    previous_demand: f64,
    previous_production: f64,
    inventory_two_periods_ago: f64,
    applies: bool,
) -> f64 {
    if applies {
        ratio(
            previous_demand,
            previous_production + inventory_two_periods_ago,
        ) - 1.0
    } else {
        0.0
    }
}

// A.60
pub fn firm_target_demand_a60(
    predicted_sector_growth: f64,
    phi_f_q: f64,
    gamma_f: f64,
    previous_demand: f64,
) -> f64 {
    ((1.0 + predicted_sector_growth) * (1.0 + phi_f_q * gamma_f) * previous_demand).max(0.0)
}

// A.61
pub fn firm_predicted_profit_a61(
    predicted_ppi_inflation: f64,
    gamma_f: f64,
    previous_profit: f64,
) -> f64 {
    (1.0 + predicted_ppi_inflation) * (1.0 + gamma_f) * previous_profit
}

// A.62
pub fn firm_target_production_a62(
    predicted_demand: f64,
    phi_st_y: f64,
    previous_production: f64,
    previous_inventory: f64,
    chi_h: f64,
    labour_constraint: f64,
    chi_m: f64,
    intermediate_constraint: f64,
    chi_k: f64,
    capital_constraint: f64,
) -> f64 {
    [
        predicted_demand + phi_st_y * previous_production - previous_inventory,
        predicted_demand + chi_h * (labour_constraint - predicted_demand),
        predicted_demand + chi_m * (intermediate_constraint - predicted_demand),
        predicted_demand + chi_k * (capital_constraint - predicted_demand),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
    .max(0.0)
}

// A.63-A.64
pub fn min_input_constraint_a63_a64(stocks: &[f64; SECTORS], coeffs: &[f64; SECTORS]) -> f64 {
    let value = stocks
        .iter()
        .zip(coeffs.iter())
        .filter_map(|(stock, coeff)| (*coeff > 0.0).then_some(stock / coeff))
        .fold(f64::INFINITY, f64::min);
    if value.is_finite() {
        value.max(0.0)
    } else {
        f64::MAX / 4.0
    }
}

// A.66-A.67
pub fn work_effort_a66_a67(
    h_max: f64,
    initial_work_effort: f64,
    labour_input_sum: f64,
    intermediate_constraint: f64,
    capital_constraint: f64,
) -> f64 {
    if initial_work_effort <= 0.0 || labour_input_sum <= 0.0 {
        return initial_work_effort.max(0.0);
    }
    let multiplier = (intermediate_constraint.min(capital_constraint))
        / (initial_work_effort * labour_input_sum);
    multiplier.min(h_max).max(0.0) * initial_work_effort
}

// A.73
pub fn price_a73(
    previous_price: f64,
    predicted_ppi_inflation: f64,
    phi_dp: f64,
    demand_pull: f64,
    phi_cp: f64,
    cost_push: f64,
) -> f64 {
    ((1.0 + predicted_ppi_inflation)
        * (1.0 + phi_dp * demand_pull)
        * (1.0 + phi_cp * cost_push)
        * previous_price)
        .max(0.0)
}

// A.78
pub fn target_intermediate_a78(
    io_coeff: f64,
    target_production: f64,
    phi_m: f64,
    previous_stock: f64,
    initial_stock: f64,
    current_production: f64,
    initial_production: f64,
) -> f64 {
    positive_part(
        io_coeff * target_production
            - phi_m
                * (previous_stock - initial_stock * ratio(current_production, initial_production)),
    )
}

// A.79
pub fn target_capital_a79(
    capital_compensation_coeff: f64,
    target_production: f64,
    phi_k: f64,
    previous_stock: f64,
    initial_stock: f64,
    current_production: f64,
    initial_production: f64,
) -> f64 {
    positive_part(
        capital_compensation_coeff * target_production
            - phi_k
                * (previous_stock - initial_stock * ratio(current_production, initial_production)),
    )
}

// A.83-A.84
pub fn constrained_goods_target_a83_a84(
    financial_friction_free_target: f64,
    phi_credit_shortfall: f64,
    target_loan: f64,
    granted_loan: f64,
    predicted_ppi_inflation: f64,
    previous_sector_price: f64,
) -> f64 {
    positive_part(
        financial_friction_free_target
            - phi_credit_shortfall * (target_loan - granted_loan)
                / ((1.0 + predicted_ppi_inflation) * previous_sector_price).max(1e-9),
    )
}

// A.108
pub fn rent_cost_a108(mu_ps: f64, annual_rent: f64) -> f64 {
    4.0 * (1.0 + mu_ps) * annual_rent
}

// A.109, transcribed literally from the visually checked PDF/OCR snippet.
pub fn purchase_cost_a109_literal_pdf(
    property_price: f64,
    financial_assets: f64,
    mortgage_rate: f64,
    mortgage_maturity_quarters: u32,
    predicted_hpi_inflation: f64,
    property_value: f64,
) -> f64 {
    let principal = positive_part(property_price - financial_assets);
    let maturity = mortgage_maturity_quarters.max(1) as f64;
    let principal_repayment = 4.0 * principal / maturity;
    let interest = if mortgage_rate.abs() <= 1e-12 {
        0.0
    } else {
        4.0 * mortgage_rate * principal / (1.0 - (1.0 + mortgage_rate).powf(maturity))
    };
    let expected_revaluation = ((1.0 + predicted_hpi_inflation).powi(4) - 1.0) * property_value;
    principal_repayment + interest - expected_revaluation
}

// A.110
pub fn buy_probability_a110(phi_b: f64, rent_cost: f64, purchase_cost: f64) -> f64 {
    1.0 / (1.0 + (phi_b * (rent_cost - purchase_cost)).exp())
}

// A.113 and A.115 as printed.
pub fn literal_price_or_rent_reduction_a113_a115(previous: f64, epsilon: f64) -> f64 {
    (1.0 - epsilon.exp()) * previous
}

// A.42
pub fn bank_liabilities_a42(
    equity: f64,
    positive_firm_deposits: f64,
    positive_household_deposits: f64,
    previous_reserves: f64,
) -> f64 {
    equity + positive_firm_deposits + positive_household_deposits - negative_abs(previous_reserves)
}

// A.43
pub fn bank_reserves_a43(
    firm_deposits: f64,
    household_deposits: f64,
    equity: f64,
    outstanding_loans: f64,
) -> f64 {
    firm_deposits + household_deposits + equity - outstanding_loans
}
