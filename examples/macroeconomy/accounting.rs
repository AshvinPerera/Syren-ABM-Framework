#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GdpIdentity {
    pub output: f64,
    pub expenditure: f64,
    pub income: f64,
}

impl GdpIdentity {
    pub fn max_gap(self) -> f64 {
        (self.output - self.expenditure)
            .abs()
            .max((self.output - self.income).abs())
            .max((self.expenditure - self.income).abs())
    }

    pub fn holds(self, tolerance: f64) -> bool {
        self.max_gap() <= tolerance
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AccountingReport {
    pub gdp: GdpIdentity,
    pub bank_equity: f64,
    pub firm_equity: f64,
    pub household_net_wealth: f64,
    pub government_debt: f64,
    pub failed_gdp_identity: bool,
}

pub fn positive_part(value: f64) -> f64 {
    value.max(0.0)
}

pub fn negative_abs(value: f64) -> f64 {
    (-value).max(0.0)
}
