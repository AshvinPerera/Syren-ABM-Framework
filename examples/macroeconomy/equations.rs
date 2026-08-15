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

// A.62. The argument list mirrors the paper's equation inputs one-to-one;
// bundling them into a struct would obscure that mapping.
#[allow(clippy::too_many_arguments)]
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
    predicted_demand: f64,
    intermediate_constraint: f64,
    capital_constraint: f64,
) -> f64 {
    if initial_work_effort <= 0.0 || labour_input_sum <= 0.0 {
        return initial_work_effort.max(0.0);
    }
    // PAPER ERRATUM (docs/errata.md (A.66)): `predicted_demand`
    // is inside this min.
    //
    // A.66 as printed reads `min(h^max, min(M_f, K_f) / (h_f(0) sum_i H_i))`.
    // Poledna's online appendix A.25, which A.66 cites as its source, is
    //     alpha_i(t) = alpha_bar_i * min(1.5,
    //         min(Q_i^s(t), beta_i M_i(t-1), kappa_i K_i(t-1)) / (N_i alpha_bar_i))
    // -- the inner min includes `Q_i^s`, the firm's *desired supply*. Wiese's
    // A.72 also carries the demand term (`min(Y_hat_f, H_f, M_f, K_f)`); only
    // A.66 drops it.
    //
    // The term is what makes the rule mean anything: a firm works overtime
    // because it wants to produce more, not because its warehouse is full.
    // Without it, A.55 opens every firm at `M_f(0) = Y_f(0)/omega^M` and A.78
    // with `phi^M = 1` holds that ratio, so `phi^WE` is pinned at
    // `1/0.85 = 1.176` forever and A.69 compounds wages 17.6% a quarter until
    // the wage bill exceeds revenue and the economy fails.
    //
    // `Q_bar_f` (A.60 predicted demand) is used rather than `Y_hat_f` (A.62
    // target production): target production depends on `H_f`, which depends on
    // this factor, so using it would be circular. Predicted demand is Poledna's
    // `Q^s` in any case -- a supply choice from expected growth and previous
    // demand, formed before any constraint is applied.
    let multiplier = predicted_demand
        .min(intermediate_constraint)
        .min(capital_constraint)
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

/// A.77: unit costs.
///
/// `U_f(t) = w_f(t)/Y_f(t) + sum_s' m_{s's} P_s'(t-1) + sum_s' d_{s's} P_s'(t-1)
///           + tau^PROD P_f(t-1)`
///
/// Only the wage term is divided by output. The intermediate and depreciation
/// terms are *per unit of output already*, being technology coefficients times
/// prices.
///
/// Computing this as `total_costs / production` instead — as the code did — is
/// wrong in two ways that compound. Total costs include **restocking**
/// purchases (A.89 buys `M(t) - M(t-1) + m*Y`, not just the `m*Y` consumed)
/// and loan interest, neither of which is a unit cost of production. And when
/// output falls the whole numerator is divided by a collapsing denominator, so
/// `U_f` explodes; since A.76 feeds `U_f/P_f - 1` straight back into price,
/// that is a divergent loop.
pub fn unit_cost_a77(
    total_wages: f64,
    production: f64,
    io_coeffs: &[f64; SECTORS],
    depreciation_coeffs: &[f64; SECTORS],
    sector_prices: &[f64; SECTORS],
    production_tax_rate: f64,
    previous_price: f64,
) -> f64 {
    let labour_cost = ratio(total_wages, production);
    let intermediate_cost: f64 = io_coeffs
        .iter()
        .zip(sector_prices.iter())
        .map(|(coeff, price)| coeff * price)
        .sum();
    let depreciation_cost: f64 = depreciation_coeffs
        .iter()
        .zip(sector_prices.iter())
        .map(|(coeff, price)| coeff * price)
        .sum();
    labour_cost + intermediate_cost + depreciation_cost + production_tax_rate * previous_price
}

/// A.76: cost-push inflation, `U_f(t-1)/P_f(t-1) - 1`.
///
/// Deliberately unfloored. The code clamped this at zero, which turned the
/// pricing rule into a ratchet: prices could rise when costs rose but never
/// fall back when they eased, so the price level could only drift upward.
pub fn cost_push_inflation_a76(previous_unit_cost: f64, previous_price: f64) -> f64 {
    ratio(previous_unit_cost, previous_price) - 1.0
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
pub fn rent_cost_a108(mu_ps: f64, quarterly_rent: f64) -> f64 {
    4.0 * (1.0 + mu_ps) * quarterly_rent
}

/// A.109: the annual cost of buying a property, against which A.108's annual
/// cost of renting is compared.
///
/// The annuity denominator is `1 - (1 + r*)^{-m_l}`. The paper prints a
/// positive exponent, which with `r* > 0` over 100 quarters makes the
/// denominator large and negative and the interest term come out negative --
/// buying would look cheaper the higher mortgage rates went. See
/// `docs/deviations.md`.
pub fn purchase_cost_a109(
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
        4.0 * mortgage_rate * principal / (1.0 - (1.0 + mortgage_rate).powf(-maturity))
    };
    let expected_revaluation = ((1.0 + predicted_hpi_inflation).powi(4) - 1.0) * property_value;
    principal_repayment + interest - expected_revaluation
}

// A.110
pub fn buy_probability_a110(phi_b: f64, rent_cost: f64, purchase_cost: f64) -> f64 {
    1.0 / (1.0 + (phi_b * (rent_cost - purchase_cost)).exp())
}

// A.113 and A.115 as printed.
/// A.113 / A.115: an unsold or unlet property cuts its asking price or rent.
///
/// `epsilon` is a normal draw whose exponential is the reduction **as a
/// percentage**, not as a fraction. The thesis reports `(mu, sigma)` of a
/// log-normal fitted to observed reductions: `(1.4531, 0.4889)` for sale
/// prices and `(1.6559, 0.7855)` for rents, giving median haircuts of
/// `exp(1.4531) = 4.28%` and `exp(1.6559) = 5.24%` respectively.
///
/// Reading `exp(epsilon)` as a fraction instead — as `(1 - exp(eps)) * prev`
/// did — yields a multiplier of `1 - 4.28 = -3.28`, i.e. a negative price.
///
/// The clamp guards the far tail: `exp(epsilon) >= 100` needs `epsilon >= 4.6`,
/// about 6.4 sigma out, but a single such draw would otherwise flip the sign
/// of a price and poison every downstream index.
pub fn price_or_rent_reduction_a113_a115(previous: f64, epsilon: f64) -> f64 {
    let reduction_fraction = (epsilon.exp() / 100.0).clamp(0.0, 0.95);
    previous * (1.0 - reduction_fraction)
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
