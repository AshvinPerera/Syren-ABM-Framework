use abm_framework::engine::component::{
    Bundle, register_component, freeze_components, component_id_of,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSReference, ECSData};
use abm_framework::engine::systems::{AccessSets, System};
use abm_framework::engine::scheduler::Scheduler;
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::{ECSError, ECSResult};
use abm_framework::engine::reduce::{Count, Sum};

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct AgentTag(pub u8);

#[derive(Clone, Copy)]
struct Cash(pub f32);

#[derive(Clone, Copy)]
struct Hunger(pub f32);

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct FirmTag(pub u8);

#[derive(Clone, Copy)]
struct Inventory(pub f32);

#[derive(Clone, Copy)]
struct Production(pub f32);

#[derive(Clone, Copy)]
struct Wage(pub f32);

#[derive(Clone, Copy)]
struct TargetInventory(pub f32);

#[derive(Clone, Copy)]
struct Price(pub f32);

struct ProductionSystem;

impl System for ProductionSystem {
    fn id(&self) -> u16 { 1 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Production>().unwrap());
        a.read.set(component_id_of::<TargetInventory>().unwrap());
        a.write.set(component_id_of::<Inventory>().unwrap());
        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> Result<(), ECSError> {
        let qb = ecs.query()?;
        let query = qb
            .read::<Production>()?
            .read::<TargetInventory>()?
            .write::<Inventory>()?
            .build()?;

        ecs.for_each_read2_write1::<Production, TargetInventory, Inventory>(
            query,
            |prod, target, inv| {
                if inv.0 < target.0 {
                    inv.0 += prod.0;
                }
            },
        )?;

        Ok(())
    }
}

struct WagePaymentSystem;

impl System for WagePaymentSystem {
    fn id(&self) -> u16 { 2 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Production>().unwrap());
        a.read.set(component_id_of::<Wage>().unwrap());
        a.write.set(component_id_of::<Cash>().unwrap());
        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> Result<(), ECSError> {
        let qb = ecs.query()?;
        let query = qb
            .read::<Production>()?
            .read::<Wage>()?
            .write::<Cash>()?
            .build()?;

        ecs.for_each_read2_write1::<Production, Wage, Cash>(
            query,
            |prod, wage, cash| {
                cash.0 -= prod.0 * wage.0;
            },
        )?;

        Ok(())
    }
}

struct SpendingSystem;

impl System for SpendingSystem {
    fn id(&self) -> u16 { 3 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<AgentTag>().unwrap());
        a.write.set(component_id_of::<Cash>().unwrap());
        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> Result<(), ECSError> {
        let qb = ecs.query()?;
        let query = qb
            .read::<AgentTag>()?
            .write::<Cash>()?
            .build()?;

        ecs.for_each_read_write::<AgentTag, Cash>(
            query,
            |_, cash| {
                if cash.0 >= 1.0 {
                    cash.0 -= 1.0;
                }
            },
        )?;

        Ok(())
    }
}

struct PriceSystem;

impl System for PriceSystem {
    fn id(&self) -> u16 { 4 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Inventory>().unwrap());
        a.read.set(component_id_of::<TargetInventory>().unwrap());
        a.write.set(component_id_of::<Price>().unwrap());
        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> Result<(), ECSError> {
        let qb = ecs.query()?;
        let query = qb
            .read::<Inventory>()?
            .read::<TargetInventory>()?
            .write::<Price>()?
            .build()?;

        ecs.for_each_read2_write1::<Inventory, TargetInventory, Price>(
            query,
            |inv, target, price| {
                if inv.0 < target.0 {
                    price.0 *= 1.01;
                } else {
                    price.0 *= 0.99;
                }
            },
        )?;

        Ok(())
    }
}

struct HungerSystem;

impl System for HungerSystem {
    fn id(&self) -> u16 { 5 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Cash>().unwrap());
        a.write.set(component_id_of::<Hunger>().unwrap());
        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> Result<(), ECSError> {
        let qb = ecs.query()?;
        let query = qb
            .read::<Cash>()?
            .write::<Hunger>()?
            .build()?;

        ecs.for_each_read_write::<Cash, Hunger>(
            query,
            |cash, hunger| {
                if cash.0 >= 0.0 {
                    hunger.0 = (hunger.0 - 1.0).max(0.0);
                } else {
                    hunger.0 += 1.0;
                }
            },
        )?;

        Ok(())
    }
}

#[test]
fn toy_economy_ecs_abm() -> ECSResult<()> {
    let _ = register_component::<Cash>();
    let _ = register_component::<Hunger>();
    let _ = register_component::<AgentTag>();
    let _ = register_component::<FirmTag>();
    let _ = register_component::<Inventory>();
    let _ = register_component::<Production>();
    let _ = register_component::<Wage>();
    let _ = register_component::<TargetInventory>();
    let _ = register_component::<Price>();
    let _ = freeze_components();

    let shards = EntityShards::new(4);
    let ecs = ECSManager::new(ECSData::new(shards));

    let world = ecs.world_ref();

    world.with_exclusive(|_| -> Result<(), ECSError> {
        for _ in 0..100 {
            let mut b = Bundle::new();
            b.insert(component_id_of::<FirmTag>()?, FirmTag(0));
            b.insert(component_id_of::<Cash>()?, Cash(10_000.0));
            b.insert(component_id_of::<Inventory>()?, Inventory(100.0));
            b.insert(component_id_of::<Production>()?, Production(5.0));
            b.insert(component_id_of::<Wage>()?, Wage(1.0));
            b.insert(component_id_of::<TargetInventory>()?, TargetInventory(200.0));
            b.insert(component_id_of::<Price>()?, Price(1.0));
            world.defer(Command::Spawn { bundle: b })?;
        }

        for _ in 0..1_000_000 {
            let mut b = Bundle::new();
            b.insert(component_id_of::<AgentTag>()?, AgentTag(0));
            b.insert(component_id_of::<Cash>()?, Cash(100.0));
            b.insert(component_id_of::<Hunger>()?, Hunger(0.0));
            world.defer(Command::Spawn { bundle: b })?;
        }

        Ok(())
    })?;

    ecs.apply_deferred_commands()?;

    let mut scheduler = Scheduler::new();
    scheduler.add_system(ProductionSystem);
    scheduler.add_system(WagePaymentSystem);
    scheduler.add_system(PriceSystem);
    scheduler.add_system(SpendingSystem);
    scheduler.add_system(HungerSystem);

    for step in 0..1000 {
        ecs.run(&mut scheduler)?;

        let world = ecs.world_ref();

        let qb = world.query()?;
        let query = qb
            .read::<Price>()?
            .build()?;

        // Sum of prices
        let sum = world.reduce_read::<Price, Sum>(
            query.clone(),
            Sum::default,
            |acc, price| acc.0 += price.0 as f64,
            |a, b| a.0 += b.0,
        )?;

        // Count of firms
        let count = world.reduce_read::<Price, Count>(
            query,
            Count::default,
            |acc, _| acc.0 += 1,
            |a, b| a.0 += b.0,
        )?;

        let avg = sum.0 / count.0 as f64;
        println!("{step},{avg}");
    }

    Ok(())
}
