#![cfg(all(feature = "model", feature = "messaging"))]

use std::sync::{Arc, Mutex, RwLock};

use abm_framework::advanced::EntityShards;
use abm_framework::agents::AgentTemplate;
use abm_framework::environment::EnvironmentBoundary;
use abm_framework::messaging::{
    BruteForceMessage, BucketMessage, Capacity, Message, MessageBufferSet,
};
use abm_framework::model::ModelBuilder;
use abm_framework::{AccessSets, ComponentRegistry, Count, ECSResult, FnSystem, Read, Sum, Write};

const HOUSEHOLDS: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
struct Household {
    id: u32,
    cash: u32,
    goods: u32,
    labor: u32,
    desired_goods: u32,
    consume_need: u32,
    consumed: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Firm {
    cash: u32,
    inventory: u32,
    wage: u32,
    price: u32,
    productivity: u32,
    produced: u32,
    wages_paid: u32,
    sales_revenue: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LaborOffer {
    household_id: u32,
    hours: u32,
}
impl Message for LaborOffer {}
impl BruteForceMessage for LaborOffer {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GoodsOrder {
    household_id: u32,
    quantity: u32,
    max_spend: u32,
}
impl Message for GoodsOrder {}
impl BruteForceMessage for GoodsOrder {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WagePayment {
    household_id: u32,
    amount: u32,
}
impl Message for WagePayment {}
impl BucketMessage for WagePayment {
    fn bucket_key(&self) -> u32 {
        self.household_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TradeReceipt {
    household_id: u32,
    quantity: u32,
    payment: u32,
}
impl Message for TradeReceipt {}
impl BucketMessage for TradeReceipt {
    fn bucket_key(&self) -> u32 {
        self.household_id
    }
}

#[derive(Clone, Copy)]
struct EconomyIds {
    household: abm_framework::ComponentID,
    firm: abm_framework::ComponentID,
}

fn register_components() -> (Arc<RwLock<ComponentRegistry>>, EconomyIds) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let ids = {
        let mut reg = registry.write().unwrap();
        let household = reg.register::<Household>().unwrap();
        let firm = reg.register::<Firm>().unwrap();
        (household, firm)
    };
    (
        registry,
        EconomyIds {
            household: ids.0,
            firm: ids.1,
        },
    )
}

fn build_closed_market_model() -> (abm_framework::model::Model, EconomyIds) {
    let (registry, ids) = register_components();
    let mut builder = ModelBuilder::new()
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(2).unwrap());

    let message_boundary = builder.message_boundary_id();
    let production_key = builder
        .register_environment::<u32>("last_production", 0)
        .unwrap();
    let wages_key = builder
        .register_environment::<u32>("last_wages", 0)
        .unwrap();
    let sales_key = builder
        .register_environment::<u32>("last_sales", 0)
        .unwrap();
    let settled_key = builder
        .register_environment::<u32>("last_settled_goods", 0)
        .unwrap();

    let labor = builder
        .register_brute_force_message::<LaborOffer>(Capacity::unbounded(64))
        .unwrap();
    let orders = builder
        .register_brute_force_message::<GoodsOrder>(Capacity::unbounded(64))
        .unwrap();
    let wages = builder
        .register_bucket_message::<WagePayment>(HOUSEHOLDS, Capacity::unbounded(64))
        .unwrap();
    let receipts = builder
        .register_bucket_message::<TradeReceipt>(HOUSEHOLDS, Capacity::unbounded(64))
        .unwrap();

    let mut emit_access = AccessSets::default();
    emit_access.read.set(ids.household);
    emit_access.produces.insert(labor.channel_id());
    emit_access.produces.insert(orders.channel_id());

    let mut market_access = AccessSets::default();
    market_access.write.set(ids.firm);
    market_access.consumes.insert(labor.channel_id());
    market_access.consumes.insert(orders.channel_id());
    market_access.produces.insert(wages.channel_id());
    market_access.produces.insert(receipts.channel_id());
    market_access.produces.insert(production_key.channel_id());
    market_access.produces.insert(wages_key.channel_id());
    market_access.produces.insert(sales_key.channel_id());

    let mut settle_access = AccessSets::default();
    settle_access.write.set(ids.household);
    settle_access.consumes.insert(wages.channel_id());
    settle_access.consumes.insert(receipts.channel_id());
    settle_access.produces.insert(settled_key.channel_id());

    let mut consume_access = AccessSets::default();
    consume_access.write.set(ids.household);
    consume_access.consumes.insert(settled_key.channel_id());

    let model = builder
        .with_agent_template(
            AgentTemplate::builder("household")
                .with_component::<Household>(ids.household)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_agent_template(
            AgentTemplate::builder("firm")
                .with_component::<Firm>(ids.firm)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_system(FnSystem::new(
            1,
            "household_emit_orders",
            emit_access,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
                let rows = Arc::new(Mutex::new(Vec::<Household>::new()));
                let rows_for_query = Arc::clone(&rows);
                let q = ecs.query()?.read::<Household>()?.build()?;
                ecs.for_each::<(Read<Household>,)>(q, &move |household| {
                    rows_for_query.lock().unwrap().push(*household.0);
                })?;

                for household in rows.lock().unwrap().iter().copied() {
                    if household.labor > 0 {
                        buffers.emit(
                            labor,
                            LaborOffer {
                                household_id: household.id,
                                hours: household.labor,
                            },
                        )?;
                    }
                    if household.desired_goods > 0 && household.cash > 0 {
                        buffers.emit(
                            orders,
                            GoodsOrder {
                                household_id: household.id,
                                quantity: household.desired_goods,
                                max_spend: household.cash,
                            },
                        )?;
                    }
                }
                Ok(())
            },
        ))
        .with_system(FnSystem::new(
            2,
            "firm_market_clearing",
            market_access,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
                let labor_offers: Vec<_> = buffers.brute_force(labor)?.collect();
                let goods_orders: Vec<_> = buffers.brute_force(orders)?.collect();
                let env = ecs.boundary::<EnvironmentBoundary>(0)?;

                let firm_state = Arc::new(Mutex::new(None::<Firm>));
                let firm_for_query = Arc::clone(&firm_state);
                let read_q = ecs.query()?.read::<Firm>()?.build()?;
                ecs.for_each::<(Read<Firm>,)>(read_q, &move |firm| {
                    *firm_for_query.lock().unwrap() = Some(*firm.0);
                })?;

                let mut firm = firm_state.lock().unwrap().expect("fixture has one firm");
                let total_labor: u32 = labor_offers.iter().map(|offer| offer.hours).sum();
                let total_wages = total_labor * firm.wage;
                let production = total_labor * firm.productivity;

                firm.cash -= total_wages;
                firm.inventory += production;
                firm.produced += production;
                firm.wages_paid += total_wages;

                for offer in &labor_offers {
                    buffers.emit(
                        wages,
                        WagePayment {
                            household_id: offer.household_id,
                            amount: offer.hours * firm.wage,
                        },
                    )?;
                }

                let mut sales = 0;
                for order in &goods_orders {
                    let affordable = order.max_spend / firm.price;
                    let quantity = order.quantity.min(affordable).min(firm.inventory);
                    if quantity == 0 {
                        continue;
                    }
                    let payment = quantity * firm.price;
                    firm.inventory -= quantity;
                    firm.cash += payment;
                    firm.sales_revenue += payment;
                    sales += payment;
                    buffers.emit(
                        receipts,
                        TradeReceipt {
                            household_id: order.household_id,
                            quantity,
                            payment,
                        },
                    )?;
                }

                let write_q = ecs.query()?.write::<Firm>()?.build()?;
                ecs.for_each::<(Write<Firm>,)>(write_q, &move |slot| {
                    *slot.0 = firm;
                })?;

                env.environment().set(production_key.name(), production)?;
                env.environment().set(wages_key.name(), total_wages)?;
                env.environment().set(sales_key.name(), sales)?;
                Ok(())
            },
        ))
        .with_system(FnSystem::new(
            3,
            "household_settlement",
            settle_access,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
                let env = ecs.boundary::<EnvironmentBoundary>(0)?;
                let mut wage_income = vec![0u32; HOUSEHOLDS as usize];
                let mut receipt_rows = vec![Vec::<TradeReceipt>::new(); HOUSEHOLDS as usize];
                for id in 0..HOUSEHOLDS {
                    wage_income[id as usize] = buffers
                        .bucket(wages, id)
                        .unwrap()
                        .map(|payment| payment.amount)
                        .sum();
                    receipt_rows[id as usize] = buffers.bucket(receipts, id).unwrap().collect();
                }
                let settled_goods: u32 = receipt_rows
                    .iter()
                    .flat_map(|rows| rows.iter())
                    .map(|receipt| receipt.quantity)
                    .sum();
                let q = ecs.query()?.write::<Household>()?.build()?;
                ecs.for_each::<(Write<Household>,)>(q, &move |state| {
                    state.0.cash += wage_income[state.0.id as usize];
                    for receipt in &receipt_rows[state.0.id as usize] {
                        assert!(state.0.cash >= receipt.payment);
                        state.0.cash -= receipt.payment;
                        state.0.goods += receipt.quantity;
                    }
                })?;
                env.environment().set(settled_key.name(), settled_goods)?;
                Ok(())
            },
        ))
        .with_system(FnSystem::new(
            4,
            "household_consumption",
            consume_access,
            move |ecs| {
                let q = ecs.query()?.write::<Household>()?.build()?;
                ecs.for_each::<(Write<Household>,)>(q, &|state| {
                    let consumed = state.0.goods.min(state.0.consume_need);
                    state.0.goods -= consumed;
                    state.0.consumed += consumed;
                })?;
                Ok(())
            },
        ))
        .build()
        .unwrap();

    (model, ids)
}

fn spawn_fixture(model: &mut abm_framework::model::Model, ids: EconomyIds) {
    let world = model.ecs().world_ref();
    let household = model.agents().get("household").unwrap();
    for id in 0..HOUSEHOLDS {
        household
            .spawner()
            .set(
                ids.household,
                Household {
                    id,
                    cash: 100,
                    goods: 0,
                    labor: 2,
                    desired_goods: 3,
                    consume_need: 1,
                    consumed: 0,
                },
            )
            .unwrap()
            .spawn(world)
            .unwrap();
    }

    let firm = model.agents().get("firm").unwrap();
    firm.spawner()
        .set(
            ids.firm,
            Firm {
                cash: 10_000,
                inventory: 20,
                wage: 5,
                price: 2,
                productivity: 3,
                produced: 0,
                wages_paid: 0,
                sales_revenue: 0,
            },
        )
        .unwrap()
        .spawn(world)
        .unwrap();

    model.ecs().apply_deferred_commands().unwrap();
}

#[derive(Debug)]
struct EconomySnapshot {
    household_cash: u32,
    household_goods: u32,
    household_consumed: u32,
    firm_cash: u32,
    firm_inventory: u32,
    produced: u32,
    wages_paid: u32,
    sales_revenue: u32,
    firm_count: usize,
    household_count: usize,
}

fn snapshot(model: &abm_framework::model::Model) -> ECSResult<EconomySnapshot> {
    let world = model.ecs().world_ref();
    let households = Arc::new(Mutex::new(Vec::<Household>::new()));
    let households_for_query = Arc::clone(&households);
    let q_households = world.query()?.read::<Household>()?.build()?;
    world.for_each::<(Read<Household>,)>(q_households.clone(), &move |state| {
        households_for_query.lock().unwrap().push(*state.0);
    })?;
    let household_count = world
        .reduce_read::<Household, Count>(
            q_households,
            Count::default,
            |acc, _| acc.0 += 1,
            |acc, other| acc.0 += other.0,
        )?
        .0 as usize;

    let firms = Arc::new(Mutex::new(Vec::<Firm>::new()));
    let firms_for_query = Arc::clone(&firms);
    let q_firms = world.query()?.read::<Firm>()?.build()?;
    world.for_each::<(Read<Firm>,)>(q_firms.clone(), &move |state| {
        firms_for_query.lock().unwrap().push(*state.0);
    })?;
    let firm_count = world
        .reduce_read::<Firm, Count>(
            q_firms,
            Count::default,
            |acc, _| acc.0 += 1,
            |acc, other| acc.0 += other.0,
        )?
        .0 as usize;

    let households = households.lock().unwrap();
    let firms = firms.lock().unwrap();
    let firm = firms.first().copied().unwrap_or_default();
    Ok(EconomySnapshot {
        household_cash: households.iter().map(|h| h.cash).sum(),
        household_goods: households.iter().map(|h| h.goods).sum(),
        household_consumed: households.iter().map(|h| h.consumed).sum(),
        firm_cash: firm.cash,
        firm_inventory: firm.inventory,
        produced: firm.produced,
        wages_paid: firm.wages_paid,
        sales_revenue: firm.sales_revenue,
        firm_count,
        household_count,
    })
}

#[test]
fn closed_market_accounting_holds() -> ECSResult<()> {
    let (mut model, ids) = build_closed_market_model();
    spawn_fixture(&mut model, ids);

    let before = snapshot(&model)?;
    model.tick()?;
    let after = snapshot(&model)?;

    assert_eq!(before.household_count, HOUSEHOLDS as usize);
    assert_eq!(before.firm_count, 1);
    assert_eq!(after.household_count, HOUSEHOLDS as usize);
    assert_eq!(after.firm_count, 1);

    assert_eq!(
        before.household_cash + before.firm_cash,
        after.household_cash + after.firm_cash,
        "cash is conserved inside the closed economy"
    );
    assert_eq!(
        before.household_goods + before.firm_inventory + after.produced,
        after.household_goods + after.household_consumed + after.firm_inventory,
        "goods either remain in inventories, are held by households, or are consumed"
    );
    assert_eq!(after.produced, HOUSEHOLDS * 2 * 3);
    assert_eq!(after.wages_paid, HOUSEHOLDS * 2 * 5);
    assert_eq!(after.sales_revenue, HOUSEHOLDS * 3 * 2);
    assert_eq!(
        model.environment().get::<u32>("last_production")?,
        after.produced
    );
    assert_eq!(
        model.environment().get::<u32>("last_wages")?,
        after.wages_paid
    );
    assert_eq!(
        model.environment().get::<u32>("last_sales")?,
        after.sales_revenue
    );
    Ok(())
}

#[test]
fn settlement_messages_reach_their_household_buckets() -> ECSResult<()> {
    let (mut model, ids) = build_closed_market_model();
    spawn_fixture(&mut model, ids);
    model.tick()?;

    let world = model.ecs().world_ref();
    let q = world.query()?.read::<Household>()?.build()?;
    let total_consumed = world.reduce_read::<Household, Sum>(
        q,
        Sum::default,
        |acc, state| acc.0 += state.consumed as f64,
        |acc, other| acc.0 += other.0,
    )?;

    assert_eq!(total_consumed.0 as u32, HOUSEHOLDS);
    Ok(())
}
