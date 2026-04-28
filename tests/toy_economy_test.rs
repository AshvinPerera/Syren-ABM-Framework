use std::sync::{Arc, RwLock};

use abm_framework::{
    advanced::{ECSData, EntityShards},
    AccessSets, Bundle, Command, ComponentRegistry, Count, ECSError, ECSManager, ECSReference,
    ECSResult, Read, Scheduler, Sum, System, Write,
};

use abm_framework::{init, shutdown, span, thread_name, Arg};

// -----------------------------------------------------------------------------
// Components
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Systems
// -----------------------------------------------------------------------------

struct ProductionSystem {
    access: AccessSets,
}

impl ProductionSystem {
    fn new(registry: &ComponentRegistry) -> Self {
        let mut a = AccessSets::default();
        a.read.set(registry.id_of::<Production>().unwrap());
        a.read.set(registry.id_of::<TargetInventory>().unwrap());
        a.write.set(registry.id_of::<Inventory>().unwrap());
        Self { access: a }
    }
}

impl System for ProductionSystem {
    fn id(&self) -> u16 {
        1
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let query = ecs
            .query()?
            .read::<Production>()?
            .read::<TargetInventory>()?
            .write::<Inventory>()?
            .build()?;

        ecs.for_each::<(Read<Production>, Read<TargetInventory>, Write<Inventory>)>(
            query,
            &|(prod, target, inv)| {
                if inv.0 < target.0 {
                    inv.0 += prod.0;
                }
            },
        )?;

        Ok(())
    }
}

struct WagePaymentSystem {
    access: AccessSets,
}

impl WagePaymentSystem {
    fn new(registry: &ComponentRegistry) -> Self {
        let mut a = AccessSets::default();
        a.read.set(registry.id_of::<Production>().unwrap());
        a.read.set(registry.id_of::<Wage>().unwrap());
        a.write.set(registry.id_of::<Cash>().unwrap());
        Self { access: a }
    }
}

impl System for WagePaymentSystem {
    fn id(&self) -> u16 {
        2
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let query = ecs
            .query()?
            .read::<Production>()?
            .read::<Wage>()?
            .write::<Cash>()?
            .build()?;

        ecs.for_each::<(Read<Production>, Read<Wage>, Write<Cash>)>(query, &|(
            prod,
            wage,
            cash,
        )| {
            cash.0 -= prod.0 * wage.0;
        })?;

        Ok(())
    }
}

struct SpendingSystem {
    access: AccessSets,
}

impl SpendingSystem {
    fn new(registry: &ComponentRegistry) -> Self {
        let mut a = AccessSets::default();
        a.read.set(registry.id_of::<AgentTag>().unwrap());
        a.write.set(registry.id_of::<Cash>().unwrap());
        Self { access: a }
    }
}

impl System for SpendingSystem {
    fn id(&self) -> u16 {
        3
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let query = ecs.query()?.read::<AgentTag>()?.write::<Cash>()?.build()?;

        ecs.for_each::<(Read<AgentTag>, Write<Cash>)>(query, &|(_, cash)| {
            if cash.0 >= 1.0 {
                cash.0 -= 1.0;
            }
        })?;

        Ok(())
    }
}

struct PriceSystem {
    access: AccessSets,
}

impl PriceSystem {
    fn new(registry: &ComponentRegistry) -> Self {
        let mut a = AccessSets::default();
        a.read.set(registry.id_of::<Inventory>().unwrap());
        a.read.set(registry.id_of::<TargetInventory>().unwrap());
        a.write.set(registry.id_of::<Price>().unwrap());
        Self { access: a }
    }
}

impl System for PriceSystem {
    fn id(&self) -> u16 {
        4
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let query = ecs
            .query()?
            .read::<Inventory>()?
            .read::<TargetInventory>()?
            .write::<Price>()?
            .build()?;

        ecs.for_each::<(Read<Inventory>, Read<TargetInventory>, Write<Price>)>(
            query,
            &|(inv, target, price)| {
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

struct HungerSystem {
    access: AccessSets,
}

impl HungerSystem {
    fn new(registry: &ComponentRegistry) -> Self {
        let mut a = AccessSets::default();
        a.read.set(registry.id_of::<Cash>().unwrap());
        a.write.set(registry.id_of::<Hunger>().unwrap());
        Self { access: a }
    }
}

impl System for HungerSystem {
    fn id(&self) -> u16 {
        5
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let query = ecs.query()?.read::<Cash>()?.write::<Hunger>()?.build()?;

        ecs.for_each::<(Read<Cash>, Write<Hunger>)>(query, &|(cash, hunger)| {
            if cash.0 >= 0.0 {
                hunger.0 = (hunger.0 - 1.0).max(0.0);
            } else {
                hunger.0 += 1.0;
            }
        })?;

        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------

#[test]
fn toy_economy_ecs_abm() -> ECSResult<()> {
    // --- PROFILER SETUP -------------------------------------------------------
    init("profile/toy_economy_trace.json");
    thread_name("Main");

    // --- COMPONENT REGISTRATION ----------------------------------------------
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));

    {
        let mut reg = registry.write().unwrap();
        reg.register::<Cash>().map_err(ECSError::from)?;
        reg.register::<Hunger>().map_err(ECSError::from)?;
        reg.register::<AgentTag>().map_err(ECSError::from)?;
        reg.register::<FirmTag>().map_err(ECSError::from)?;
        reg.register::<Inventory>().map_err(ECSError::from)?;
        reg.register::<Production>().map_err(ECSError::from)?;
        reg.register::<Wage>().map_err(ECSError::from)?;
        reg.register::<TargetInventory>().map_err(ECSError::from)?;
        reg.register::<Price>().map_err(ECSError::from)?;
        reg.freeze();
    }

    // --- WORLD SETUP ----------------------------------------------------------
    let _setup = span("setup::spawn_entities");

    let shards = EntityShards::new(4)?;
    let ecs = ECSManager::new(ECSData::new(shards, registry.clone()));
    let world = ecs.world_ref();

    let reg = registry.read().unwrap();

    let firm_tag_id = reg.id_of::<FirmTag>().unwrap();
    let cash_id = reg.id_of::<Cash>().unwrap();
    let inventory_id = reg.id_of::<Inventory>().unwrap();
    let production_id = reg.id_of::<Production>().unwrap();
    let wage_id = reg.id_of::<Wage>().unwrap();
    let target_inventory_id = reg.id_of::<TargetInventory>().unwrap();
    let price_id = reg.id_of::<Price>().unwrap();
    let agent_tag_id = reg.id_of::<AgentTag>().unwrap();
    let hunger_id = reg.id_of::<Hunger>().unwrap();

    world.with_exclusive(|_| -> Result<(), ECSError> {
        // Firms
        for _ in 0..100 {
            let mut b = Bundle::new();
            b.insert(firm_tag_id, FirmTag(0));
            b.insert(cash_id, Cash(10_000.0));
            b.insert(inventory_id, Inventory(100.0));
            b.insert(production_id, Production(5.0));
            b.insert(wage_id, Wage(1.0));
            b.insert(target_inventory_id, TargetInventory(200.0));
            b.insert(price_id, Price(1.0));
            world.defer(Command::Spawn { bundle: b })?;
        }

        // Agents
        for _ in 0..1_000_000 {
            let mut b = Bundle::new();
            b.insert(agent_tag_id, AgentTag(0));
            b.insert(cash_id, Cash(100.0));
            b.insert(hunger_id, Hunger(0.0));
            world.defer(Command::Spawn { bundle: b })?;
        }

        Ok(())
    })?;

    ecs.apply_deferred_commands()?;
    drop(_setup);

    // --- SCHEDULER ------------------------------------------------------------
    let mut scheduler = Scheduler::new();
    scheduler.add_system(ProductionSystem::new(&reg));
    scheduler.add_system(WagePaymentSystem::new(&reg));
    scheduler.add_system(PriceSystem::new(&reg));
    scheduler.add_system(SpendingSystem::new(&reg));
    scheduler.add_system(HungerSystem::new(&reg));

    drop(reg);

    // --- SIMULATION LOOP ------------------------------------------------------
    for step in 0..1000 {
        let _tick = span("tick").arg("step", Arg::U64(step as u64));

        ecs.run(&mut scheduler)?;

        let world = ecs.world_ref();
        let query = world.query()?.read::<Price>()?.build()?;

        let sum = {
            let _g = span("reduce::sum_price");
            world.reduce_read::<Price, Sum>(
                query.clone(),
                Sum::default,
                |acc, price| acc.0 += price.0 as f64,
                |a, b| a.0 += b.0,
            )?
        };

        let count = {
            let _g = span("reduce::count_firms");
            world.reduce_read::<Price, Count>(
                query,
                Count::default,
                |acc, _| acc.0 += 1,
                |a, b| a.0 += b.0,
            )?
        };

        let avg = sum.0 / count.0 as f64;
        println!("{step},{avg}");
    }

    shutdown();
    Ok(())
}
