use std::sync::atomic::{AtomicU32, Ordering};

use abm_framework::engine::component::{register_component, freeze_components, component_id_of};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSReference};
use abm_framework::engine::systems::{System};
use abm_framework::engine::scheduler::{make_stages, run_schedule};
use abm_framework::engine::types::{Bundle, COMPONENT_CAP, AccessSets};
use abm_framework::engine::commands::Command;

#[allow(dead_code)]
#[derive(Clone, Copy)] struct AgentTag(pub u8);
#[derive(Clone, Copy)] struct Cash(pub f32);
#[derive(Clone, Copy)] struct Hunger(pub f32);

#[allow(dead_code)]
#[derive(Clone, Copy)] struct FirmTag(pub u8);
#[derive(Clone, Copy)] struct Inventory(pub f32);
#[derive(Clone, Copy)] struct Production(pub f32);
#[derive(Clone, Copy)] struct Wage(pub f32);
#[derive(Clone, Copy)] struct TargetInventory(pub f32);
#[derive(Clone, Copy)] struct Price(pub f32);

struct ProductionSystem;

impl System for ProductionSystem {
    fn id(&self) -> u16 { 1 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Production>());
        a.read.set(component_id_of::<TargetInventory>());
        a.write.set(component_id_of::<Inventory>());
        a
    }

    fn run(&self, ecs: ECSReference) {
        let data = ecs.data_mut();

        let query = data.query()
            .read::<Production>()
            .read::<TargetInventory>()
            .write::<Inventory>()
            .build();

        data.for_each_read2_write_1::<Production, TargetInventory, Inventory>(
            query,
            |prod, target, inv| {
                if inv.0 < target.0 {
                    inv.0 += prod.0;
                }
            },
        );
    }
}

struct WagePaymentSystem;

impl System for WagePaymentSystem {
    fn id(&self) -> u16 { 2 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Production>());
        a.read.set(component_id_of::<Wage>());
        a.write.set(component_id_of::<Cash>());
        a
    }

    fn run(&self, ecs: ECSReference) {
        let data = ecs.data_mut();

        let query = data.query()
            .read::<Production>()
            .read::<Wage>()
            .write::<Cash>()
            .build();

        data.for_each_read2_write_1::<Production, Wage, Cash>(
            query,
            |prod, wage, cash| {
                cash.0 -= prod.0 * wage.0;
            },
        );
    }
}

struct SpendingSystem;

impl System for SpendingSystem {
    fn id(&self) -> u16 { 3 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<AgentTag>());
        a.write.set(component_id_of::<Cash>());
        a
    }

    fn run(&self, ecs: ECSReference) {
        let data = ecs.data_mut();

        let query = data.query()
            .read::<AgentTag>()
            .write::<Cash>()
            .build();

        data.for_each_read_write::<AgentTag, Cash>(
            query,
            |_, cash| {
                if cash.0 >= 1.0 {
                    cash.0 -= 1.0;
                }
            },
        );
    }
}

struct PriceSystem;

impl System for PriceSystem {
    fn id(&self) -> u16 { 4 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();

        a.read.set(component_id_of::<Inventory>());
        a.read.set(component_id_of::<TargetInventory>());

        a.write.set(component_id_of::<Price>());

        a
    }

    fn run(&self, ecs: ECSReference) {
        let data = ecs.data_mut();

        let query = data.query()
            .read::<Inventory>()
            .read::<TargetInventory>()
            .write::<Price>()
            .build();

        data.for_each_read2_write_1::<Inventory, TargetInventory, Price>(
            query,
            |inv, target, price| {
                if inv.0 < target.0 {
                    price.0 *= 1.01;
                } else {
                    price.0 *= 0.99;
                }
            },
        );
    }
}

struct HungerSystem;

impl System for HungerSystem {
    fn id(&self) -> u16 { 5 }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Cash>());
        a.write.set(component_id_of::<Hunger>());
        a
    }

    fn run(&self, ecs: ECSReference) {
        let data = ecs.data_mut();

        let query = data.query()
            .read::<Cash>()
            .write::<Hunger>()
            .build();

        data.for_each_read_write::<Cash, Hunger>(
            query,
            |cash, hunger| {
                if cash.0 >= 0.0 {
                    hunger.0 = (hunger.0 - 1.0).max(0.0);
                } else {
                    hunger.0 += 1.0;
                }
            },
        );
    }
}


#[test]
fn toy_economy_ecs_abm() {
    register_component::<Cash>();
    register_component::<Hunger>();
    register_component::<AgentTag>();
    register_component::<FirmTag>();
    register_component::<Inventory>();
    register_component::<Production>();
    register_component::<Wage>();
    register_component::<TargetInventory>();
    register_component::<Price>();
    freeze_components();

    let shards = EntityShards::new(4);
    let mut ecs = ECSManager::new(shards);

    {
        let world = ecs.world_ref();
        let data = world.data_mut();

        for _ in 0..10 {
            let mut b = Bundle::with_len(COMPONENT_CAP);
            b.insert(component_id_of::<FirmTag>(), FirmTag(0));
            b.insert(component_id_of::<Cash>(), Cash(10_000.0));
            b.insert(component_id_of::<Inventory>(), Inventory(100.0));
            b.insert(component_id_of::<Production>(), Production(5.0));
            b.insert(component_id_of::<Wage>(), Wage(1.0));
            b.insert(component_id_of::<TargetInventory>(), TargetInventory(200.0));
            b.insert(component_id_of::<Price>(), Price(1.0));

            data.defer(Command::Spawn { bundle: b });
        }

        for _ in 0..10000 {
            let mut b = Bundle::with_len(COMPONENT_CAP);
            b.insert(component_id_of::<AgentTag>(), AgentTag(0));
            b.insert(component_id_of::<Cash>(), Cash(100.0));
            b.insert(component_id_of::<Hunger>(), Hunger(0.0));

            data.defer(Command::Spawn { bundle: b });
        }

    }
    ecs.apply_deferred_commands();

    let systems: Vec<Box<dyn System>> = vec![
        Box::new(ProductionSystem),
        Box::new(WagePaymentSystem),
        Box::new(PriceSystem),
        Box::new(SpendingSystem),
        Box::new(HungerSystem),
    ];

    let stages = make_stages(systems);

    for step in 0..1000 {
        run_schedule(&mut ecs, &stages);

        let world = ecs.world_ref();
        let data = world.data_mut();

        if step == 999 {
            let sum_bits = AtomicU32::new(0);
            let count = AtomicU32::new(0);

            let query = data.query()
                .read::<Price>()
                .read::<FirmTag>()
                .build();

            data.for_each_read2::<Price, FirmTag>(
                query,
                |price, _| {
                    sum_bits.fetch_add(price.0.to_bits(), Ordering::Relaxed);
                    count.fetch_add(1, Ordering::Relaxed);
                },
            );

            let total = f32::from_bits(sum_bits.load(Ordering::Relaxed));
            let n = count.load(Ordering::Relaxed) as f32;
            let avg_price = total / n;

        println!("{},{}", step, avg_price);
        }
    }
}