#![cfg(all(feature = "model", feature = "messaging_gpu"))]

use std::sync::{Arc, Mutex, RwLock};

use abm_framework::advanced::EntityShards;
use abm_framework::agents::AgentTemplate;
use abm_framework::messaging::{
    BruteForceMessage, Capacity, GpuMessage, Message, MessageBufferSet,
};
use abm_framework::model::ModelBuilder;
use abm_framework::{
    AccessSets, ComponentRegistry, ECSError, ECSReference, ECSResult, ExecutionError, GPUPod,
    GpuSystem, Read, System, SystemBackend, Write,
};

const HOUSEHOLDS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHousehold {
    id: u32,
    cash: u32,
    goods: u32,
    desired_goods: u32,
}

unsafe impl GPUPod for GpuHousehold {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GpuFirm {
    cash: u32,
    inventory: u32,
    price: u32,
    sales_revenue: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGoodsOrder {
    household_id: u32,
    quantity: u32,
    max_spend: u32,
    _pad: u32,
}

impl Message for GpuGoodsOrder {}
impl BruteForceMessage for GpuGoodsOrder {}
unsafe impl GpuMessage for GpuGoodsOrder {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTradeReceipt {
    household_id: u32,
    quantity: u32,
    payment: u32,
    _pad: u32,
}

impl Message for GpuTradeReceipt {}
impl BruteForceMessage for GpuTradeReceipt {}
unsafe impl GpuMessage for GpuTradeReceipt {}

#[derive(Clone, Copy)]
struct EconomyGpuIds {
    household: abm_framework::ComponentID,
    firm: abm_framework::ComponentID,
}

struct HouseholdOrderGpuSystem {
    access: AccessSets,
    resources: [abm_framework::GPUResourceID; 1],
    writes: [abm_framework::GPUResourceID; 1],
}

impl HouseholdOrderGpuSystem {
    fn new(
        household_id: abm_framework::ComponentID,
        orders: abm_framework::messaging::GpuMessageHandle<GpuGoodsOrder>,
    ) -> Self {
        let mut access = AccessSets::default();
        access.read.set(household_id);
        access.produces.insert(orders.channel_id());
        let resource = orders.resource_ids().resource;
        Self {
            access,
            resources: [resource],
            writes: [resource],
        }
    }
}

impl System for HouseholdOrderGpuSystem {
    fn id(&self) -> abm_framework::SystemID {
        1
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn backend(&self) -> SystemBackend {
        SystemBackend::GPU
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        Ok(())
    }

    fn gpu(&self) -> Option<&dyn GpuSystem> {
        Some(self)
    }
}

impl GpuSystem for HouseholdOrderGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Household {
    id: u32,
    cash: u32,
    goods: u32,
    desired_goods: u32,
};

struct Order {
    household_id: u32,
    quantity: u32,
    max_spend: u32,
    pad: u32,
};

struct Params {
    entity_len: u32,
    archetype_base: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> households: array<Household>;
@group(0) @binding(1) var<uniform> params: Params;

@group(1) @binding(1) var<storage, read_write> raw_gpu: array<Order>;
@group(1) @binding(2) var<storage, read_write> valid_flags: array<u32>;
@group(1) @binding(5) var<storage, read_write> control: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }

    let slot = params.archetype_base + i;
    if (slot >= arrayLength(&raw_gpu) || slot >= arrayLength(&valid_flags)) {
        atomicStore(&control[4], 1u);
        return;
    }

    let household = households[i];
    if (household.desired_goods == 0u || household.cash == 0u) { return; }

    raw_gpu[slot] = Order(household.id, household.desired_goods, household.cash, 0u);
    valid_flags[slot] = 1u;
}
"#
    }

    fn workgroup_size(&self) -> u32 {
        64
    }

    fn uses_resources(&self) -> &[abm_framework::GPUResourceID] {
        &self.resources
    }

    fn writes_resources(&self) -> &[abm_framework::GPUResourceID] {
        &self.writes
    }
}

struct HouseholdReceiptGpuSystem {
    access: AccessSets,
    resources: [abm_framework::GPUResourceID; 1],
}

impl HouseholdReceiptGpuSystem {
    fn new(
        household_id: abm_framework::ComponentID,
        receipts: abm_framework::messaging::GpuMessageHandle<GpuTradeReceipt>,
    ) -> Self {
        let mut access = AccessSets::default();
        access.write.set(household_id);
        access.consumes.insert(receipts.channel_id());
        Self {
            access,
            resources: [receipts.resource_ids().resource],
        }
    }
}

impl System for HouseholdReceiptGpuSystem {
    fn id(&self) -> abm_framework::SystemID {
        3
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn backend(&self) -> SystemBackend {
        SystemBackend::GPU
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        Ok(())
    }

    fn gpu(&self) -> Option<&dyn GpuSystem> {
        Some(self)
    }
}

impl GpuSystem for HouseholdReceiptGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Household {
    id: u32,
    cash: u32,
    goods: u32,
    desired_goods: u32,
};

struct Receipt {
    household_id: u32,
    quantity: u32,
    payment: u32,
    pad: u32,
};

struct Params {
    entity_len: u32,
    archetype_base: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> households: array<Household>;
@group(0) @binding(1) var<uniform> params: Params;

@group(1) @binding(3) var<storage, read> final_messages: array<Receipt>;
@group(1) @binding(5) var<storage, read_write> control: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }

    var household = households[i];
    let final_count = atomicLoad(&control[2]);
    var cursor = 0u;
    loop {
        if (cursor >= final_count) { break; }
        let receipt = final_messages[cursor];
        if (receipt.household_id == household.id) {
            household.cash = household.cash - receipt.payment;
            household.goods = household.goods + receipt.quantity;
        }
        cursor = cursor + 1u;
    }
    households[i] = household;
}
"#
    }

    fn workgroup_size(&self) -> u32 {
        64
    }

    fn uses_resources(&self) -> &[abm_framework::GPUResourceID] {
        &self.resources
    }
}

fn register_components() -> (Arc<RwLock<ComponentRegistry>>, EconomyGpuIds) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let ids = {
        let mut reg = registry.write().unwrap();
        let household = reg.register_gpu::<GpuHousehold>().unwrap();
        let firm = reg.register::<GpuFirm>().unwrap();
        (household, firm)
    };
    (
        registry,
        EconomyGpuIds {
            household: ids.0,
            firm: ids.1,
        },
    )
}

fn build_gpu_model() -> (abm_framework::model::Model, EconomyGpuIds) {
    let (registry, ids) = register_components();
    let mut builder = ModelBuilder::new()
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(2).unwrap());
    let message_boundary = builder.message_boundary_id();
    let orders = builder
        .register_gpu_brute_force_message::<GpuGoodsOrder>(Capacity::bounded(
            HOUSEHOLDS as usize,
            HOUSEHOLDS as usize,
        ))
        .unwrap();
    let receipts = builder
        .register_gpu_brute_force_message::<GpuTradeReceipt>(Capacity::bounded(
            HOUSEHOLDS as usize,
            HOUSEHOLDS as usize,
        ))
        .unwrap();

    let mut market_access = AccessSets::default();
    market_access.write.set(ids.firm);
    market_access.consumes.insert(orders.channel_id());
    market_access.produces.insert(receipts.channel_id());

    let model = builder
        .with_agent_template(
            AgentTemplate::builder("gpu_household")
                .with_component::<GpuHousehold>(ids.household)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_agent_template(
            AgentTemplate::builder("gpu_firm")
                .with_component::<GpuFirm>(ids.firm)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_system(HouseholdOrderGpuSystem::new(ids.household, orders))
        .with_system(abm_framework::FnSystem::new(
            2,
            "cpu_market_clear",
            market_access,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
                let orders_for_tick: Vec<_> = buffers.brute_force(orders.cpu())?.collect();

                let firm_state = Arc::new(Mutex::new(None::<GpuFirm>));
                let firm_for_query = Arc::clone(&firm_state);
                let read_q = ecs.query()?.read::<GpuFirm>()?.build()?;
                ecs.for_each::<(Read<GpuFirm>,)>(read_q, &move |firm| {
                    *firm_for_query.lock().unwrap() = Some(*firm.0);
                })?;

                let mut firm = firm_state.lock().unwrap().expect("fixture has one firm");
                let mut receipt_rows = Vec::new();
                for order in &orders_for_tick {
                    let affordable = order.max_spend / firm.price;
                    let quantity = order.quantity.min(affordable).min(firm.inventory);
                    if quantity == 0 {
                        continue;
                    }
                    let payment = quantity * firm.price;
                    firm.inventory -= quantity;
                    firm.cash += payment;
                    firm.sales_revenue += payment;
                    receipt_rows.push(GpuTradeReceipt {
                        household_id: order.household_id,
                        quantity,
                        payment,
                        _pad: 0,
                    });
                }

                let write_q = ecs.query()?.write::<GpuFirm>()?.build()?;
                ecs.for_each::<(Write<GpuFirm>,)>(write_q, &move |slot| {
                    *slot.0 = firm;
                })?;

                for receipt in receipt_rows {
                    buffers.emit(receipts.cpu(), receipt)?;
                }
                Ok(())
            },
        ))
        .with_system(HouseholdReceiptGpuSystem::new(ids.household, receipts))
        .build()
        .unwrap();
    (model, ids)
}

fn spawn_fixture(model: &mut abm_framework::model::Model, ids: EconomyGpuIds) {
    let world = model.ecs().world_ref();
    let household = model.agents().get("gpu_household").unwrap();
    for id in 0..HOUSEHOLDS {
        household
            .spawner()
            .set(
                ids.household,
                GpuHousehold {
                    id,
                    cash: 100,
                    goods: 0,
                    desired_goods: 2,
                },
            )
            .unwrap()
            .spawn(world)
            .unwrap();
    }

    let firm = model.agents().get("gpu_firm").unwrap();
    firm.spawner()
        .set(
            ids.firm,
            GpuFirm {
                cash: 1000,
                inventory: 16,
                price: 3,
                sales_revenue: 0,
            },
        )
        .unwrap()
        .spawn(world)
        .unwrap();

    model.ecs().apply_deferred_commands().unwrap();
}

fn households(model: &abm_framework::model::Model) -> ECSResult<Vec<GpuHousehold>> {
    let out = Arc::new(Mutex::new(Vec::new()));
    let out_for_query = Arc::clone(&out);
    let world = model.ecs().world_ref();
    let q = world.query()?.read::<GpuHousehold>()?.build()?;
    world.for_each::<(Read<GpuHousehold>,)>(q, &move |household| {
        out_for_query.lock().unwrap().push(*household.0);
    })?;
    let mut out = out.lock().unwrap().clone();
    out.sort_by_key(|household| household.id);
    Ok(out)
}

fn firm(model: &abm_framework::model::Model) -> ECSResult<GpuFirm> {
    let out = Arc::new(Mutex::new(Vec::new()));
    let out_for_query = Arc::clone(&out);
    let world = model.ecs().world_ref();
    let q = world.query()?.read::<GpuFirm>()?.build()?;
    world.for_each::<(Read<GpuFirm>,)>(q, &move |firm| {
        out_for_query.lock().unwrap().push(*firm.0);
    })?;
    let firm = out.lock().unwrap()[0];
    Ok(firm)
}

#[test]
fn gpu_order_and_receipt_messages_match_cpu_expectation() -> ECSResult<()> {
    let (mut model, ids) = build_gpu_model();
    spawn_fixture(&mut model, ids);

    match model.tick() {
        Ok(()) => {}
        Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) => {
            eprintln!("skipping GPU economy messaging test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err),
    }

    let households = households(&model)?;
    let firm = firm(&model)?;

    let expected_households: Vec<_> = (0..HOUSEHOLDS)
        .map(|id| GpuHousehold {
            id,
            cash: 94,
            goods: 2,
            desired_goods: 2,
        })
        .collect();

    assert_eq!(households, expected_households);
    assert_eq!(
        firm,
        GpuFirm {
            cash: 1024,
            inventory: 8,
            price: 3,
            sales_revenue: 24,
        }
    );
    Ok(())
}
