use abm_framework::messaging::{BruteForceMessage, Message};

pub const BUYER_FIRM: u8 = 1;
pub const BUYER_HOUSEHOLD: u8 = 2;
pub const BUYER_GOVERNMENT: u8 = 3;
pub const BUYER_ROW: u8 = 4;

pub const LOAN_FIRM_SHORT: u8 = 1;
pub const LOAN_FIRM_LONG: u8 = 2;
pub const LOAN_HOUSEHOLD_CONSUMPTION: u8 = 3;
pub const LOAN_MORTGAGE: u8 = 4;

pub const GOODS_INTERMEDIATE: u8 = 1;
pub const GOODS_CAPITAL: u8 = 2;
pub const GOODS_CONSUMPTION: u8 = 3;
pub const GOODS_GOVERNMENT: u8 = 4;
pub const GOODS_EXPORT: u8 = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LabourOffer {
    pub firm_id: u32,
    pub wage: f64,
    pub slots: u32,
}
impl Message for LabourOffer {}
impl BruteForceMessage for LabourOffer {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WagePayment {
    pub firm_id: u32,
    pub household_id: u32,
    pub individual_id: u32,
    pub amount: f64,
}
impl Message for WagePayment {}
impl BruteForceMessage for WagePayment {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GoodsDemand {
    pub buyer_kind: u8,
    pub buyer_id: u32,
    pub purpose: u8,
    pub sector: u8,
    pub quantity: f64,
    pub max_spend: f64,
}
impl Message for GoodsDemand {}
impl BruteForceMessage for GoodsDemand {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GoodsReceipt {
    pub buyer_kind: u8,
    pub buyer_id: u32,
    pub seller_kind: u8,
    pub seller_id: u32,
    pub purpose: u8,
    pub sector: u8,
    pub quantity: f64,
    pub payment: f64,
}
impl Message for GoodsReceipt {}
impl BruteForceMessage for GoodsReceipt {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExcessDemand {
    pub buyer_kind: u8,
    pub buyer_id: u32,
    pub sector: u8,
    pub quantity: f64,
}
impl Message for ExcessDemand {}
impl BruteForceMessage for ExcessDemand {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CreditApplication {
    pub borrower_kind: u8,
    pub borrower_id: u32,
    pub loan_class: u8,
    pub sector: u8,
    pub amount: f64,
    pub collateral: f64,
    pub income: f64,
}
impl Message for CreditApplication {}
impl BruteForceMessage for CreditApplication {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CreditGrant {
    pub borrower_kind: u8,
    pub borrower_id: u32,
    pub loan_class: u8,
    pub bank_id: u32,
    pub amount: f64,
    pub rate: f64,
}
impl Message for CreditGrant {}
impl BruteForceMessage for CreditGrant {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CreditFailure {
    pub borrower_kind: u8,
    pub borrower_id: u32,
    pub loan_class: u8,
    pub requested: f64,
    pub reason_code: u8,
}
impl Message for CreditFailure {}
impl BruteForceMessage for CreditFailure {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MortgageNeed {
    pub household_id: u32,
    pub property_id: u32,
    pub desired_price: f64,
    pub amount: f64,
}
impl Message for MortgageNeed {}
impl BruteForceMessage for MortgageNeed {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TentativePurchase {
    pub household_id: u32,
    pub seller_household_id: u32,
    pub property_id: u32,
    pub price: f64,
    pub mortgage_required: f64,
}
impl Message for TentativePurchase {}
impl BruteForceMessage for TentativePurchase {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TentativeRental {
    pub household_id: u32,
    pub owner_household_id: u32,
    pub property_id: u32,
    pub annual_rent: f64,
}
impl Message for TentativeRental {}
impl BruteForceMessage for TentativeRental {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PropertyTransfer {
    pub household_id: u32,
    pub seller_household_id: u32,
    pub property_id: u32,
    pub price: f64,
    pub mortgage_amount: f64,
}
impl Message for PropertyTransfer {}
impl BruteForceMessage for PropertyTransfer {}
