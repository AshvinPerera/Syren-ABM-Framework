// This module is compiled into two targets: the `macroeconomy` example binary
// and the `macroeconomy` integration test. Items the binary never reaches --
// the equation helpers and initialisation recipe the tests assert against --
// are dead code from the binary's point of view alone.
#![allow(unused_imports, dead_code)]

pub mod accounting;
pub mod calibration;
pub mod components;
pub mod config;
pub mod data;
pub mod equations;
pub mod forecasting;
pub mod messages;
pub mod output;
pub mod state;
pub mod systems;

use std::error::Error;
use std::sync::{Arc, RwLock};

use syren::advanced::EntityShards;
use syren::agents::AgentTemplate;
use syren::model::{Model, ModelBuilder};
use syren::ComponentRegistry;

pub use accounting::{AccountingReport, GdpIdentity};
pub use calibration::CalibrationParameters;
pub use components::*;
pub use config::{apply_config_file, apply_config_str, ConfigError};
pub use data::{
    thesis_initialisation_recipe, DataError, DataProvider, FixtureDataProvider, InitialData,
    InitialisationRecipeStep, RealDataProvider,
};
pub use equations::*;
pub use forecasting::{
    ardl_error_correction_delta_rate, fit_ar1_level_forecast, fit_ar1_log_level_forecast,
    select_ardl_lag_by_aic, transform_taylor_rule, Ar1Fit, ArdlCandidate, ArdlErrorCorrectionInput,
    TaylorRuleEstimate,
};
pub use messages::*;
pub use output::{
    aggregate_row, csv_header, firm_row, headline_row, AGGREGATE_COLUMNS, FIRM_COLUMNS,
    HEADLINE_COLUMNS,
};
pub use state::{
    rng_salt, FirmProbe, GoodsClearingPolicy, MacroEnvironment, MacroRng, MacroeconomyConfig,
    MarketAudit, ModelPolicy, RunMode, MACRO_ENV_KEY,
};
pub use systems::{MacroComponentIds, MacroMessageHandles, PhaseKeys};

use self::state::{
    PHASE_ACCOUNTING_DONE, PHASE_AGGREGATE_DONE, PHASE_CREDIT_DONE, PHASE_EXPECTATIONS_DONE,
    PHASE_GOODS_DONE, PHASE_HOUSING_COMPLETION_DONE, PHASE_HOUSING_PRECLEAR_DONE,
    PHASE_LABOUR_DONE, PHASE_PLANNING_DONE, PHASE_TARGETS_DONE,
};

pub struct MacroeconomyExample {
    pub model: Model,
    pub ids: MacroComponentIds,
    pub messages: MacroMessageHandles,
    pub phases: PhaseKeys,
}

pub fn build_macroeconomy_model<P>(
    config: MacroeconomyConfig,
    provider: P,
) -> Result<MacroeconomyExample, Box<dyn Error>>
where
    P: DataProvider,
{
    // The provider applies the config file itself, *before* generating the
    // population, so the initial state is built against the final parameters.
    // Re-applying it here would clobber the sector weights the generator
    // derives from the solved SAM.
    let data = provider.load(&config)?;
    let (registry, ids) = register_components()?;

    let mut builder = ModelBuilder::new()
        .with_seed(data.environment.seed)
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(shards_for_population(&data))?);

    let _macro_key = builder
        .register_environment::<MacroEnvironment>(state::MACRO_ENV_KEY, data.environment.clone())?;
    let phases = register_phase_keys(&mut builder)?;
    let messages = systems::register_message_handles(&mut builder)?;

    builder = builder
        .with_agent_template(
            AgentTemplate::builder("firm")
                .with_component::<Firm>(ids.firm)?
                .with_component::<FirmStocks>(ids.firm_stocks)?
                .with_component::<FirmStockBaseline>(ids.firm_stock_baseline)?
                .with_component::<FirmTargets>(ids.firm_targets)?
                .with_component::<FirmRealised>(ids.firm_realised)?
                .with_capacity(data.firms.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("individual")
                .with_component::<Individual>(ids.individual)?
                .with_component::<IndividualWageHistory>(ids.individual_wage_history)?
                .with_capacity(data.individuals.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("household")
                .with_component::<Household>(ids.household)?
                .with_component::<HouseholdDemand>(ids.household_demand)?
                .with_component::<HouseholdHistory>(ids.household_history)?
                .with_capacity(data.households.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("bank")
                .with_component::<Bank>(ids.bank)?
                .with_capacity(data.banks.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("government_entity")
                .with_component::<GovernmentEntity>(ids.government_entity)?
                .with_capacity(data.government_entities.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("government_account")
                .with_component::<GovernmentAccount>(ids.government_account)?
                .with_capacity(data.government_accounts.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("central_bank")
                .with_component::<CentralBank>(ids.central_bank)?
                .with_capacity(data.central_banks.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("property")
                .with_component::<Property>(ids.property)?
                .with_capacity(data.properties.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("rest_of_world")
                .with_component::<RestOfWorld>(ids.rest_of_world)?
                .with_capacity(data.rest_of_world.len())
                .build(),
        )?;

    builder = builder
        .with_agent_population("firm", ids.firm, data.firms)?
        .with_agent_population("firm", ids.firm_stocks, data.firm_stocks)?
        .with_agent_population("firm", ids.firm_stock_baseline, data.firm_stock_baselines)?
        .with_agent_population("firm", ids.firm_targets, data.firm_targets)?
        .with_agent_population("firm", ids.firm_realised, data.firm_realised)?
        .with_agent_population("individual", ids.individual, data.individuals)?
        .with_agent_population(
            "individual",
            ids.individual_wage_history,
            data.individual_wage_histories,
        )?
        .with_agent_population("household", ids.household, data.households)?
        .with_agent_population("household", ids.household_demand, data.household_demands)?
        .with_agent_population("household", ids.household_history, data.household_histories)?
        .with_agent_population("bank", ids.bank, data.banks)?
        .with_agent_population(
            "government_entity",
            ids.government_entity,
            data.government_entities,
        )?
        .with_agent_population(
            "government_account",
            ids.government_account,
            data.government_accounts,
        )?
        .with_agent_population("central_bank", ids.central_bank, data.central_banks)?
        .with_agent_population("property", ids.property, data.properties)?
        .with_agent_population("rest_of_world", ids.rest_of_world, data.rest_of_world)?;

    builder = systems::add_macroeconomy_systems(builder, ids, messages, phases);
    let model = builder.build()?;
    Ok(MacroeconomyExample {
        model,
        ids,
        messages,
        phases,
    })
}

pub fn macro_state(model: &Model) -> Result<MacroEnvironment, Box<dyn Error>> {
    Ok(model
        .environment()
        .get::<MacroEnvironment>(state::MACRO_ENV_KEY)?)
}

/// Entities addressable per shard.
///
/// `EntityID` packs `| version(32) | shard(10) | index(22) |`, so each shard
/// holds `2^22 - 1` entities (`INDEX_CAP`, `src/engine/types.rs:101`). The
/// constant is not re-exported from the crate root, so it is mirrored here.
pub const ENTITIES_PER_SHARD: usize = (1 << 22) - 1;

/// Spare capacity above the initial population.
///
/// Shard count is fixed at construction (`EntityShards::new`) and cannot grow,
/// so under-provisioning is an unrecoverable mid-run failure. Firm respawn
/// after bankruptcy and any future loan/property entities all spawn into the
/// same budget.
const SHARD_HEADROOM: usize = 2;

/// Derive the shard count from the initial population.
///
/// A fixed shard count caps the world silently, so it is sized from the
/// population that is actually being built.
pub fn shards_for_population(data: &InitialData) -> usize {
    let total = data.firms.len()
        + data.individuals.len()
        + data.households.len()
        + data.banks.len()
        + data.government_entities.len()
        + data.government_accounts.len()
        + data.central_banks.len()
        + data.properties.len()
        + data.rest_of_world.len();
    total
        .saturating_mul(SHARD_HEADROOM)
        .div_ceil(ENTITIES_PER_SHARD)
        .max(1)
}

/// Shared handle to the component registry, cloned into the builder and kept by
/// the caller so populations can be seeded after `build`.
type SharedRegistry = Arc<RwLock<ComponentRegistry>>;

fn register_components() -> Result<(SharedRegistry, MacroComponentIds), Box<dyn Error>> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let ids = {
        let mut reg = registry
            .write()
            .map_err(|_| "component registry lock poisoned")?;
        MacroComponentIds {
            firm: reg.register::<Firm>()?,
            firm_stocks: reg.register::<FirmStocks>()?,
            firm_stock_baseline: reg.register::<FirmStockBaseline>()?,
            firm_targets: reg.register::<FirmTargets>()?,
            firm_realised: reg.register::<FirmRealised>()?,
            individual: reg.register::<Individual>()?,
            individual_wage_history: reg.register::<IndividualWageHistory>()?,
            household: reg.register::<Household>()?,
            household_demand: reg.register::<HouseholdDemand>()?,
            household_history: reg.register::<HouseholdHistory>()?,
            bank: reg.register::<Bank>()?,
            government_entity: reg.register::<GovernmentEntity>()?,
            government_account: reg.register::<GovernmentAccount>()?,
            central_bank: reg.register::<CentralBank>()?,
            property: reg.register::<Property>()?,
            rest_of_world: reg.register::<RestOfWorld>()?,
        }
    };
    Ok((registry, ids))
}

fn register_phase_keys(builder: &mut ModelBuilder) -> Result<PhaseKeys, syren::model::ModelError> {
    Ok(PhaseKeys {
        aggregate_done: builder.register_environment::<u64>(PHASE_AGGREGATE_DONE, 0)?,
        expectations_done: builder.register_environment::<u64>(PHASE_EXPECTATIONS_DONE, 0)?,
        targets_done: builder.register_environment::<u64>(PHASE_TARGETS_DONE, 0)?,
        labour_done: builder.register_environment::<u64>(PHASE_LABOUR_DONE, 0)?,
        planning_done: builder.register_environment::<u64>(PHASE_PLANNING_DONE, 0)?,
        housing_preclear_done: builder
            .register_environment::<u64>(PHASE_HOUSING_PRECLEAR_DONE, 0)?,
        credit_done: builder.register_environment::<u64>(PHASE_CREDIT_DONE, 0)?,
        housing_completion_done: builder
            .register_environment::<u64>(PHASE_HOUSING_COMPLETION_DONE, 0)?,
        goods_done: builder.register_environment::<u64>(PHASE_GOODS_DONE, 0)?,
        accounting_done: builder.register_environment::<u64>(PHASE_ACCOUNTING_DONE, 0)?,
    })
}
