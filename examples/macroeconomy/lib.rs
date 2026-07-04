#![allow(dead_code, unused_imports)]

pub mod accounting;
pub mod calibration;
pub mod components;
pub mod config;
pub mod coverage;
pub mod data;
pub mod equations;
pub mod forecasting;
pub mod messages;
pub mod state;
pub mod systems;

use std::error::Error;
use std::sync::{Arc, RwLock};

use abm_framework::advanced::EntityShards;
use abm_framework::agents::AgentTemplate;
use abm_framework::model::{Model, ModelBuilder};
use abm_framework::ComponentRegistry;

pub use accounting::{AccountingReport, GdpIdentity};
pub use calibration::{
    BayesFactorConfig, CalibrationParameters, ForecastExperimentConfig, NeuralPosteriorConfig,
};
pub use components::*;
pub use config::{apply_config_file, apply_config_str, ConfigError};
pub use coverage::{CoverageStatus, EquationCoverage, EquationCoverageEntry, ReplicationBlocker};
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
pub use state::{
    GapReportMode, GoodsClearingPolicy, MacroEnvironment, MacroeconomyConfig, PythonLikeRng,
    ReplicationPolicy, RunMode, MACRO_ENV_KEY,
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
    let mut data = provider.load(&config)?;
    if let Some(config_path) = &config.config_path {
        apply_config_file(config_path, &mut data.environment)?;
    }
    validate_replication_policy(&data.environment)?;
    let (registry, ids) = register_components()?;

    let mut builder = ModelBuilder::new()
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(2)?);

    let _macro_key = builder
        .register_environment::<MacroEnvironment>(state::MACRO_ENV_KEY, data.environment.clone())?;
    let phases = register_phase_keys(&mut builder)?;
    let messages = systems::register_message_handles(&mut builder)?;

    builder = builder
        .with_agent_template(
            AgentTemplate::builder("firm")
                .with_component::<Firm>(ids.firm)?
                .with_capacity(data.firms.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("individual")
                .with_component::<Individual>(ids.individual)?
                .with_capacity(data.individuals.len())
                .build(),
        )?
        .with_agent_template(
            AgentTemplate::builder("household")
                .with_component::<Household>(ids.household)?
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
        .with_agent_population("individual", ids.individual, data.individuals)?
        .with_agent_population("household", ids.household, data.households)?
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

pub fn validate_replication_policy(state: &MacroEnvironment) -> Result<(), Box<dyn Error>> {
    if state.policy.allow_unresolved_blockers {
        return Ok(());
    }
    let blockers = EquationCoverage::blocker_log();
    if blockers.is_empty() {
        return Ok(());
    }
    let ids = blockers
        .iter()
        .map(|blocker| blocker.id)
        .collect::<Vec<_>>()
        .join(", ");
    Err(
        format!("strict replication requested, but unresolved replication blockers remain: {ids}")
            .into(),
    )
}

pub fn run_forecast_batch(
    config: ForecastExperimentConfig,
) -> Result<ForecastBatchSummary, Box<dyn Error>> {
    Ok(ForecastBatchSummary {
        countries: config.countries,
        initialisation_quarters: config.initialisation_quarters,
        horizon_quarters: config.horizon_quarters,
        trajectories: config.trajectories,
        npe: NeuralPosteriorConfig::npe(),
        nre: NeuralPosteriorConfig::nre(),
        bayes_factor: BayesFactorConfig::default(),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForecastBatchSummary {
    pub countries: usize,
    pub initialisation_quarters: usize,
    pub horizon_quarters: usize,
    pub trajectories: usize,
    pub npe: NeuralPosteriorConfig,
    pub nre: NeuralPosteriorConfig,
    pub bayes_factor: BayesFactorConfig,
}

pub fn macro_state(model: &Model) -> Result<MacroEnvironment, Box<dyn Error>> {
    Ok(model
        .environment()
        .get::<MacroEnvironment>(state::MACRO_ENV_KEY)?)
}

fn register_components(
) -> Result<(Arc<RwLock<ComponentRegistry>>, MacroComponentIds), Box<dyn Error>> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let ids = {
        let mut reg = registry
            .write()
            .map_err(|_| "component registry lock poisoned")?;
        MacroComponentIds {
            firm: reg.register::<Firm>()?,
            individual: reg.register::<Individual>()?,
            household: reg.register::<Household>()?,
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

fn register_phase_keys(
    builder: &mut ModelBuilder,
) -> Result<PhaseKeys, abm_framework::model::ModelError> {
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
