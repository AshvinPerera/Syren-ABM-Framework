//! Fluent builder for [`Model`](crate::model::Model).

use std::any::Any;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use crate::agents::{AgentRegistry, AgentTemplate};
use crate::engine::channel_allocator::ChannelAllocator;
use crate::engine::component::ComponentRegistry;
use crate::engine::entity::EntityShards;
use crate::engine::manager::ECSManager;
use crate::engine::scheduler::Scheduler;
use crate::engine::systems::System;
use crate::engine::types::BoundaryID;
use crate::engine::types::ChannelID;
use crate::engine::workers::max_workers;
use crate::environment::{EnvKey, EnvironmentBoundary, EnvironmentBuilder};

#[cfg(feature = "messaging_gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "messaging")]
use crate::messaging::{
    BruteForceMessage, BucketMessage, Capacity, MessageBufferSet, MessageHandle, MessageRegistry,
    SpatialConfig, SpatialMessage, TargetedMessage,
};

#[cfg(feature = "messaging_gpu")]
use crate::messaging::{
    GpuBucketMessage, GpuMessage, GpuMessageHandle, GpuMessageOptions, GpuMessageResource,
    GpuSpatialMessage, GpuTargetedMessage, Specialisation,
};

#[cfg(feature = "messaging_gpu")]
use crate::messaging::gpu::GpuKeyMetadata;

use super::error::ModelError;
use super::model::Model;
use super::nested::NestedModel;
use super::sub_scheduler::SubScheduler;

/// Fluent builder for top-level models.
///
/// Model-owned boundary resources are registered in a private deterministic
/// order. Public pre-build helpers such as
/// [`environment_boundary_id`](Self::environment_boundary_id) and
/// [`message_boundary_id`](Self::message_boundary_id) are derived from that
/// registry, and `build` asserts that the IDs returned by `ECSManager` match
/// the planned order.
pub struct ModelBuilder {
    channel_allocator: ChannelAllocator,
    model_boundaries: ModelBoundaryRegistry,
    component_registry: Option<Arc<RwLock<ComponentRegistry>>>,
    shards: Option<EntityShards>,
    scheduler: Scheduler,
    sub_schedulers: Vec<SubScheduler>,
    nested_models: Vec<NestedModel>,
    environment_builder: Option<EnvironmentBuilder>,
    agents: AgentRegistry,
    pending_agent_populations: Vec<PendingAgentPopulation>,
    #[cfg(feature = "messaging")]
    message_registry: MessageRegistry,
    #[cfg(feature = "messaging_gpu")]
    gpu_message_resources: Vec<PendingGpuMessageResource>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelBoundaryKind {
    Environment,
    #[cfg(feature = "messaging")]
    Messages,
}

#[derive(Clone, Debug)]
struct ModelBoundaryRegistry {
    order: Vec<ModelBoundaryKind>,
}

impl ModelBoundaryRegistry {
    fn new() -> Self {
        #[cfg(feature = "messaging")]
        let order = vec![ModelBoundaryKind::Environment, ModelBoundaryKind::Messages];
        #[cfg(not(feature = "messaging"))]
        let order = vec![ModelBoundaryKind::Environment];
        Self { order }
    }

    fn id_of(&self, kind: ModelBoundaryKind) -> BoundaryID {
        self.order
            .iter()
            .position(|candidate| *candidate == kind)
            .expect("model boundary kind must be registered") as BoundaryID
    }
}

#[cfg(feature = "messaging_gpu")]
struct PendingGpuMessageResource {
    resource_id: GPUResourceID,
    resource: GpuMessageResource,
}

struct PendingAgentPopulation {
    template_name: String,
    component_id: crate::ComponentID,
    values: Vec<Box<dyn Any + Send>>,
}

impl ModelBuilder {
    /// Creates a builder with empty defaults.
    pub fn new() -> Self {
        Self {
            channel_allocator: ChannelAllocator::new(),
            model_boundaries: ModelBoundaryRegistry::new(),
            component_registry: None,
            shards: None,
            scheduler: Scheduler::new(),
            sub_schedulers: Vec::new(),
            nested_models: Vec::new(),
            environment_builder: None,
            agents: AgentRegistry::new(),
            pending_agent_populations: Vec::new(),
            #[cfg(feature = "messaging")]
            message_registry: MessageRegistry::new(),
            #[cfg(feature = "messaging_gpu")]
            gpu_message_resources: Vec::new(),
        }
    }

    /// Overrides the component registry.
    pub fn with_component_registry(mut self, registry: Arc<RwLock<ComponentRegistry>>) -> Self {
        self.component_registry = Some(registry);
        self
    }

    /// Overrides entity shards.
    pub fn with_shards(mut self, shards: EntityShards) -> Self {
        self.shards = Some(shards);
        self
    }

    /// Supplies an environment builder.
    pub fn with_environment(mut self, environment_builder: EnvironmentBuilder) -> Self {
        self.environment_builder = Some(environment_builder);
        self
    }

    /// Boundary ID that will hold the model environment after build.
    pub fn environment_boundary_id(&self) -> BoundaryID {
        self.model_boundaries.id_of(ModelBoundaryKind::Environment)
    }

    /// Registers an environment key and returns its typed scheduler handle.
    pub fn register_environment<T>(
        &mut self,
        key: &'static str,
        default: T,
    ) -> Result<EnvKey<T>, ModelError>
    where
        T: std::any::Any + Clone + Send + Sync + 'static,
    {
        let channel_id = self.channel_allocator.alloc()?;
        let builder = self.environment_builder.take().unwrap_or_default();
        self.environment_builder =
            Some(builder.register_with_channel::<T>(key, default, channel_id)?);
        Ok(EnvKey::new(key, channel_id))
    }

    /// Adds a root scheduler system.
    pub fn with_system<S: System + 'static>(mut self, system: S) -> Self {
        self.scheduler.add_system(system);
        self
    }

    /// Adds an already-built sub-scheduler.
    pub fn with_sub_scheduler(mut self, sub_scheduler: SubScheduler) -> Self {
        self.sub_schedulers.push(sub_scheduler);
        self
    }

    /// Adds an isolated nested child model.
    pub fn with_nested_model(mut self, nested_model: NestedModel) -> Self {
        self.nested_models.push(nested_model);
        self
    }

    /// Registers an agent template.
    pub fn with_agent_template(mut self, template: AgentTemplate) -> Result<Self, ModelError> {
        self.agents.register(template)?;
        Ok(self)
    }

    /// Adds an initial single-column population to materialise during build.
    pub fn with_agent_population<T>(
        mut self,
        template_name: impl Into<String>,
        component_id: crate::ComponentID,
        values: Vec<T>,
    ) -> Result<Self, ModelError>
    where
        T: Any + Send + 'static,
    {
        let template_name = template_name.into();
        self.agents.get(&template_name)?;
        self.pending_agent_populations.push(PendingAgentPopulation {
            template_name,
            component_id,
            values: values
                .into_iter()
                .map(|value| Box::new(value) as Box<dyn Any + Send>)
                .collect(),
        });
        Ok(self)
    }

    /// Boundary ID that will hold model messages after build.
    #[cfg(feature = "messaging")]
    pub fn message_boundary_id(&self) -> BoundaryID {
        self.model_boundaries.id_of(ModelBoundaryKind::Messages)
    }

    /// Registers a brute-force message.
    #[cfg(feature = "messaging")]
    pub fn register_brute_force_message<M: BruteForceMessage>(
        &mut self,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, ModelError> {
        Ok(self
            .message_registry
            .register_brute_force::<M>(&mut self.channel_allocator, capacity)?)
    }

    /// Registers a bucketed message.
    #[cfg(feature = "messaging")]
    pub fn register_bucket_message<M: BucketMessage>(
        &mut self,
        max_buckets: u32,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, ModelError> {
        Ok(self.message_registry.register_bucket::<M>(
            &mut self.channel_allocator,
            max_buckets,
            capacity,
        )?)
    }

    /// Registers a spatial message.
    #[cfg(feature = "messaging")]
    pub fn register_spatial_message<M: SpatialMessage>(
        &mut self,
        config: SpatialConfig,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, ModelError> {
        Ok(self.message_registry.register_spatial::<M>(
            &mut self.channel_allocator,
            config,
            capacity,
        )?)
    }

    /// Registers a targeted message.
    #[cfg(feature = "messaging")]
    pub fn register_targeted_message<M: TargetedMessage>(
        &mut self,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, ModelError> {
        Ok(self
            .message_registry
            .register_targeted::<M>(&mut self.channel_allocator, capacity)?)
    }

    /// Registers a GPU-backed brute-force message.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_brute_force_message<M>(
        &mut self,
        capacity: Capacity,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + BruteForceMessage,
    {
        let cpu = self.register_brute_force_message::<M>(capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::BruteForce,
            capacity,
            GpuMessageOptions::deterministic(capacity.max.unwrap_or(capacity.initial).max(1)),
            GpuKeyMetadata::None,
        )
    }

    /// Registers a GPU-backed brute-force message with explicit GPU options.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_brute_force_message_with_options<M>(
        &mut self,
        capacity: Capacity,
        options: GpuMessageOptions,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + BruteForceMessage,
    {
        let cpu = self.register_brute_force_message::<M>(capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::BruteForce,
            capacity,
            options,
            GpuKeyMetadata::None,
        )
    }

    /// Registers a GPU-backed bucketed message.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_bucket_message<M>(
        &mut self,
        max_buckets: u32,
        capacity: Capacity,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + BucketMessage,
    {
        let cpu = self.register_bucket_message::<M>(max_buckets, capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Bucket { max_buckets },
            capacity,
            GpuMessageOptions::deterministic(capacity.max.unwrap_or(capacity.initial).max(1)),
            GpuKeyMetadata::None,
        )
    }

    /// Registers a GPU-backed bucketed message with explicit GPU options and
    /// GPU key metadata.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_bucket_message_with_options<M>(
        &mut self,
        max_buckets: u32,
        capacity: Capacity,
        options: GpuMessageOptions,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + BucketMessage + GpuBucketMessage,
    {
        let cpu = self.register_bucket_message::<M>(max_buckets, capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Bucket { max_buckets },
            capacity,
            options,
            GpuKeyMetadata::Bucket {
                key_word_offset: M::BUCKET_KEY_WORD_OFFSET,
            },
        )
    }

    /// Registers a GPU-backed spatial message.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_spatial_message<M>(
        &mut self,
        config: SpatialConfig,
        capacity: Capacity,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + SpatialMessage,
    {
        let cpu = self.register_spatial_message::<M>(config, capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Spatial(config),
            capacity,
            GpuMessageOptions::deterministic(capacity.max.unwrap_or(capacity.initial).max(1)),
            GpuKeyMetadata::None,
        )
    }

    /// Registers a GPU-backed spatial message with explicit GPU options and
    /// GPU key metadata.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_spatial_message_with_options<M>(
        &mut self,
        config: SpatialConfig,
        capacity: Capacity,
        options: GpuMessageOptions,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + SpatialMessage + GpuSpatialMessage,
    {
        let cpu = self.register_spatial_message::<M>(config, capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Spatial(config),
            capacity,
            options,
            GpuKeyMetadata::Spatial {
                x_word_offset: M::X_WORD_OFFSET,
                y_word_offset: M::Y_WORD_OFFSET,
            },
        )
    }

    /// Registers a GPU-backed targeted message.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_targeted_message<M>(
        &mut self,
        capacity: Capacity,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + TargetedMessage,
    {
        let cpu = self.register_targeted_message::<M>(capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Targeted,
            capacity,
            GpuMessageOptions::deterministic(capacity.max.unwrap_or(capacity.initial).max(1)),
            GpuKeyMetadata::None,
        )
    }

    /// Registers a GPU-backed targeted message with explicit GPU options and
    /// GPU key metadata.
    #[cfg(feature = "messaging_gpu")]
    pub fn register_gpu_targeted_message_with_options<M>(
        &mut self,
        capacity: Capacity,
        options: GpuMessageOptions,
    ) -> Result<GpuMessageHandle<M>, ModelError>
    where
        M: GpuMessage + TargetedMessage + GpuTargetedMessage,
    {
        let cpu = self.register_targeted_message::<M>(capacity)?;
        self.attach_gpu_message::<M>(
            cpu,
            Specialisation::Targeted,
            capacity,
            options,
            GpuKeyMetadata::Targeted {
                layout: M::TARGET_KEY_LAYOUT,
            },
        )
    }

    #[cfg(feature = "messaging_gpu")]
    fn attach_gpu_message<M: GpuMessage>(
        &mut self,
        cpu: MessageHandle<M>,
        specialisation: Specialisation,
        capacity: Capacity,
        options: GpuMessageOptions,
        key_metadata: GpuKeyMetadata,
    ) -> Result<GpuMessageHandle<M>, ModelError> {
        let resource_id = self.gpu_message_resources.len() as GPUResourceID;
        self.message_registry.attach_gpu_resource(cpu, resource_id);
        let resource = GpuMessageResource::new(
            std::any::type_name::<M>(),
            std::mem::size_of::<M>(),
            capacity,
            specialisation,
            options,
            key_metadata,
        )?;
        self.gpu_message_resources.push(PendingGpuMessageResource {
            resource_id,
            resource,
        });
        Ok(GpuMessageHandle::new(cpu, specialisation, resource_id))
    }

    /// Builds the model and validates all schedulers.
    pub fn build(mut self) -> Result<Model, ModelError> {
        let registry = self
            .component_registry
            .unwrap_or_else(|| Arc::new(RwLock::new(ComponentRegistry::new())));
        registry
            .write()
            .map_err(|_| ModelError::LockPoisoned("component registry"))?
            .freeze();

        let shards = match self.shards {
            Some(shards) => shards,
            None => EntityShards::new(max_workers() as usize)?,
        };

        let environment = self
            .environment_builder
            .unwrap_or_default()
            .build_with_allocator(&mut self.channel_allocator)?;
        let environment_channels: HashSet<ChannelID> =
            environment.all_channel_ids().into_iter().collect();

        #[cfg(feature = "messaging")]
        let message_channels: HashSet<ChannelID> = self
            .message_registry
            .descriptors()
            .iter()
            .map(|d| d.channel_id)
            .collect();
        #[cfg(not(feature = "messaging"))]
        let message_channels: HashSet<ChannelID> = HashSet::new();

        Self::validate_unique_sub_scheduler_names(&self.sub_schedulers)?;

        self.scheduler.try_rebuild()?;
        for sub in &mut self.sub_schedulers {
            sub.scheduler_mut().try_rebuild()?;
        }

        Self::validate_channel_scope(
            &self.scheduler,
            &self.sub_schedulers,
            &environment_channels,
            &message_channels,
        )?;

        let ecs = ECSManager::with_registry(shards, registry);

        #[cfg(feature = "messaging_gpu")]
        {
            for pending in self.gpu_message_resources {
                let assigned = ecs.world_ref().register_gpu_resource(pending.resource)?;
                debug_assert_eq!(assigned, pending.resource_id);
            }
        }

        let environment_boundary_id =
            ecs.register_boundary(EnvironmentBoundary::new(Arc::clone(&environment)))?;
        debug_assert_eq!(
            environment_boundary_id,
            self.model_boundaries.id_of(ModelBoundaryKind::Environment)
        );

        #[cfg(feature = "messaging")]
        let (message_boundary_id, message_registry) = {
            self.message_registry.freeze();
            let message_registry = Arc::new(self.message_registry);
            let message_boundary_id =
                ecs.register_boundary(MessageBufferSet::new(Arc::clone(&message_registry))?)?;
            debug_assert_eq!(
                message_boundary_id,
                self.model_boundaries.id_of(ModelBoundaryKind::Messages)
            );
            (message_boundary_id, message_registry)
        };

        self.agents.seal();

        let mut model = Model {
            ecs,
            environment,
            agents: self.agents,
            scheduler: self.scheduler,
            sub_schedulers: self.sub_schedulers,
            nested_models: self.nested_models,
            environment_boundary_id,
            #[cfg(feature = "messaging")]
            message_boundary_id,
            #[cfg(feature = "messaging")]
            message_registry,
            tick_count: 0,
        };

        for population in self.pending_agent_populations {
            model.spawn_agent_batch_boxed(
                &population.template_name,
                population.component_id,
                population.values,
            )?;
        }

        Ok(model)
    }

    fn validate_unique_sub_scheduler_names(subs: &[SubScheduler]) -> Result<(), ModelError> {
        let mut seen = HashSet::new();
        for sub in subs {
            let name = sub.name();
            if !seen.insert(name.to_owned()) {
                return Err(ModelError::DuplicateSubSchedulerName {
                    name: name.to_owned(),
                });
            }
        }
        Ok(())
    }

    fn validate_channel_scope(
        root: &Scheduler,
        subs: &[SubScheduler],
        environment_channels: &HashSet<ChannelID>,
        message_channels: &HashSet<ChannelID>,
    ) -> Result<(), ModelError> {
        struct Scope<'a> {
            name: String,
            order: usize,
            scheduler: &'a Scheduler,
        }

        let mut scopes = Vec::new();
        for (idx, sub) in subs.iter().enumerate() {
            scopes.push(Scope {
                name: sub.name().to_owned(),
                order: idx,
                scheduler: sub.scheduler(),
            });
        }
        scopes.push(Scope {
            name: "root".to_owned(),
            order: subs.len(),
            scheduler: root,
        });

        for producer in &scopes {
            let produces = producer.scheduler.aggregate_produces();
            for consumer in &scopes {
                if producer.order == consumer.order {
                    continue;
                }
                let consumes = consumer.scheduler.aggregate_consumes();

                for channel_id in produces.iter() {
                    if !consumes.contains(channel_id) {
                        continue;
                    }

                    if message_channels.contains(&channel_id) {
                        return Err(ModelError::CrossSchedulerMessageChannel {
                            channel_id,
                            producer_scope: producer.name.clone(),
                            consumer_scope: consumer.name.clone(),
                        });
                    }

                    if environment_channels.contains(&channel_id) && producer.order > consumer.order
                    {
                        return Err(ModelError::BackwardEnvironmentChannel {
                            channel_id,
                            producer_scope: producer.name.clone(),
                            consumer_scope: consumer.name.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::AgentTemplate;
    use crate::engine::component::ComponentRegistry;
    use crate::engine::systems::{AccessSets, FnSystem};
    use crate::environment::EnvironmentBoundary;
    use crate::model::ModelError;
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "messaging")]
    use crate::messaging::BruteForceMessage;
    #[cfg(feature = "messaging")]
    use crate::messaging::{Capacity, Message};

    #[cfg(feature = "messaging_gpu")]
    use bytemuck::{Pod, Zeroable};

    #[test]
    fn empty_model_builds_and_ticks() {
        let mut model = ModelBuilder::new().build().unwrap();
        model.tick().unwrap();
        assert_eq!(model.tick_count(), 1);
    }

    #[test]
    fn environment_boundary_id_is_registry_derived_and_matches_model() {
        let builder = ModelBuilder::new();
        let expected = builder.environment_boundary_id();
        let model = builder.build().unwrap();
        assert_eq!(expected, 0);
        assert_eq!(model.environment_boundary_id(), expected);
    }

    #[cfg(feature = "messaging")]
    #[test]
    fn message_boundary_id_is_registry_derived_and_matches_model() {
        let builder = ModelBuilder::new();
        let env_id = builder.environment_boundary_id();
        let message_id = builder.message_boundary_id();
        assert_ne!(env_id, message_id);
        let model = builder.build().unwrap();
        assert_eq!(model.environment_boundary_id(), env_id);
        assert_eq!(model.message_boundary_id(), message_id);
    }

    #[test]
    fn sub_scheduler_runs_before_root() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut sub = SubScheduler::new("sub");
        let sub_order = Arc::clone(&order);
        sub.scheduler_mut()
            .add_system(FnSystem::new(0, "sub", AccessSets::default(), move |_| {
                sub_order.lock().unwrap().push(1);
                Ok(())
            }));

        let root_order = Arc::clone(&order);
        let mut model = ModelBuilder::new()
            .with_sub_scheduler(sub)
            .with_system(FnSystem::new(1, "root", AccessSets::default(), move |_| {
                root_order.lock().unwrap().push(2);
                Ok(())
            }))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn agent_registry_is_sealed_after_build() {
        let model = ModelBuilder::new().build().unwrap();
        assert!(model.agents().is_sealed());
    }

    #[test]
    fn duplicate_sub_scheduler_names_are_rejected() {
        let result = ModelBuilder::new()
            .with_sub_scheduler(SubScheduler::new("dup"))
            .with_sub_scheduler(SubScheduler::new("dup"))
            .build();
        let Err(err) = result else {
            panic!("expected duplicate sub-scheduler name error");
        };
        assert!(matches!(
            err,
            ModelError::DuplicateSubSchedulerName { name } if name == "dup"
        ));
    }

    #[test]
    fn tagged_agent_spawn_hooks_fire_at_scheduler_boundary() {
        #[derive(Default)]
        struct Marker {
            _value: u8,
        }
        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let marker_id = registry.write().unwrap().register::<Marker>().unwrap();

        let calls = Arc::new(Mutex::new(0usize));
        let hook_calls = Arc::clone(&calls);
        let registered = AgentTemplate::builder("agent")
            .with_component::<Marker>(marker_id)
            .unwrap()
            .on_spawn(Box::new(move |_, _| {
                *hook_calls.lock().unwrap() += 1;
            }))
            .build();
        let spawn_template = AgentTemplate::builder("agent")
            .with_component::<Marker>(marker_id)
            .unwrap()
            .build();

        let mut model = ModelBuilder::new()
            .with_component_registry(registry)
            .with_agent_template(registered)
            .unwrap()
            .with_system(FnSystem::new(
                0,
                "spawn_agent",
                AccessSets::default(),
                move |ecs| spawn_template.spawner().spawn(ecs),
            ))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn tagged_agent_despawn_hooks_fire_at_scheduler_boundary() {
        #[derive(Default)]
        struct Marker {
            _value: u8,
        }
        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let marker_id = registry.write().unwrap().register::<Marker>().unwrap();

        let spawned_entity = Arc::new(Mutex::new(None));
        let spawned_for_hook = Arc::clone(&spawned_entity);
        let despawn_calls = Arc::new(Mutex::new(0usize));
        let despawn_calls_for_hook = Arc::clone(&despawn_calls);

        let registered = AgentTemplate::builder("agent")
            .with_component::<Marker>(marker_id)
            .unwrap()
            .on_spawn(Box::new(move |_, entity| {
                *spawned_for_hook.lock().unwrap() = Some(entity);
            }))
            .on_despawn(Box::new(move |_, _| {
                *despawn_calls_for_hook.lock().unwrap() += 1;
            }))
            .build();
        let spawn_template = AgentTemplate::builder("agent")
            .with_component::<Marker>(marker_id)
            .unwrap()
            .build();
        let despawn_template = AgentTemplate::builder("agent")
            .with_component::<Marker>(marker_id)
            .unwrap()
            .build();

        let spawned_once = Arc::new(Mutex::new(false));
        let spawned_once_for_system = Arc::clone(&spawned_once);
        let entity_for_system = Arc::clone(&spawned_entity);

        let mut model = ModelBuilder::new()
            .with_component_registry(registry)
            .with_agent_template(registered)
            .unwrap()
            .with_system(FnSystem::new(
                0,
                "spawn_or_despawn_agent",
                AccessSets::default(),
                move |ecs| {
                    let mut spawned = spawned_once_for_system.lock().unwrap();
                    if !*spawned {
                        *spawned = true;
                        spawn_template.spawner().spawn(ecs)?;
                    } else if let Some(entity) = *entity_for_system.lock().unwrap() {
                        despawn_template.despawn(ecs, entity)?;
                    }
                    Ok(())
                },
            ))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*despawn_calls.lock().unwrap(), 0);
        model.tick().unwrap();
        assert_eq!(*despawn_calls.lock().unwrap(), 1);
    }

    #[test]
    fn builder_agent_population_materialises_and_indexes_entities() {
        #[derive(Clone, Copy, Default, Debug, PartialEq)]
        struct Household {
            cash: u32,
        }

        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let household_id = registry.write().unwrap().register::<Household>().unwrap();
        let template = AgentTemplate::builder("household")
            .with_component::<Household>(household_id)
            .unwrap()
            .with_capacity(3)
            .build();

        let model = ModelBuilder::new()
            .with_component_registry(registry)
            .with_agent_template(template)
            .unwrap()
            .with_agent_population(
                "household",
                household_id,
                vec![
                    Household { cash: 10 },
                    Household { cash: 20 },
                    Household { cash: 30 },
                ],
            )
            .unwrap()
            .build()
            .unwrap();

        let template_id = model.agents().id("household").unwrap();
        let entities = model.agents().entities(template_id);
        assert_eq!(entities.len(), 3);
        assert_eq!(
            model.agents().entity_template(entities[0]),
            Some(template_id)
        );
        assert_eq!(
            model
                .ecs()
                .world_ref()
                .read_entity_component::<Household>(entities[1], household_id)
                .unwrap(),
            Household { cash: 20 }
        );
    }

    #[test]
    fn runtime_agent_batch_spawn_and_despawn_update_indexes_and_hooks() {
        #[derive(Clone, Copy, Default)]
        struct Firm {
            _inventory: u32,
        }

        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let firm_id = registry.write().unwrap().register::<Firm>().unwrap();
        let spawn_batches = Arc::new(Mutex::new(0usize));
        let despawn_batches = Arc::new(Mutex::new(0usize));
        let spawn_batches_hook = Arc::clone(&spawn_batches);
        let despawn_batches_hook = Arc::clone(&despawn_batches);

        let template = AgentTemplate::builder("firm")
            .with_component::<Firm>(firm_id)
            .unwrap()
            .on_spawn_batch(Box::new(move |_, entities| {
                *spawn_batches_hook.lock().unwrap() += entities.len();
            }))
            .on_despawn_batch(Box::new(move |_, entities| {
                *despawn_batches_hook.lock().unwrap() += entities.len();
            }))
            .build();

        let mut model = ModelBuilder::new()
            .with_component_registry(registry)
            .with_agent_template(template)
            .unwrap()
            .build()
            .unwrap();

        let entities = model
            .spawn_agent_batch(
                "firm",
                firm_id,
                vec![Firm { _inventory: 1 }, Firm { _inventory: 2 }],
            )
            .unwrap();
        let template_id = model.agents().id("firm").unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(model.agents().entities(template_id).len(), 2);
        assert_eq!(*spawn_batches.lock().unwrap(), 2);

        model.despawn_agent_batch("firm", entities).unwrap();
        assert!(model.agents().entities(template_id).is_empty());
        assert_eq!(*despawn_batches.lock().unwrap(), 2);
    }

    #[test]
    fn nested_model_bridge_runs_before_root_scheduler() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let child_order = Arc::clone(&order);
        let child = ModelBuilder::new()
            .with_system(FnSystem::new(
                0,
                "child",
                AccessSets::default(),
                move |_| {
                    child_order.lock().unwrap().push(1);
                    Ok(())
                },
            ))
            .build()
            .unwrap();

        let bridge_order = Arc::clone(&order);
        let nested = NestedModel::new("child", child).with_bridge(move |_, _| {
            bridge_order.lock().unwrap().push(2);
            Ok(())
        });

        let root_order = Arc::clone(&order);
        let mut model = ModelBuilder::new()
            .with_nested_model(nested)
            .with_system(FnSystem::new(1, "root", AccessSets::default(), move |_| {
                root_order.lock().unwrap().push(3);
                Ok(())
            }))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn sub_env_writer_is_visible_to_root_reader() {
        let mut builder = ModelBuilder::new();
        let env_id = builder.environment_boundary_id();
        let key = builder
            .register_environment::<u32>("counter", 0u32)
            .unwrap();

        let mut write_access = AccessSets::default();
        write_access.produces.insert(key.channel_id());
        let mut sub = SubScheduler::new("sub");
        sub.scheduler_mut()
            .add_system(FnSystem::new(0, "write_env", write_access, move |ecs| {
                let boundary = ecs.boundary::<EnvironmentBoundary>(env_id)?;
                boundary.environment().set(key.name(), 7u32)?;
                Ok(())
            }));

        let seen = Arc::new(Mutex::new(0u32));
        let seen_root = Arc::clone(&seen);
        let mut read_access = AccessSets::default();
        read_access.consumes.insert(key.channel_id());
        let mut model = builder
            .with_sub_scheduler(sub)
            .with_system(FnSystem::new(1, "read_env", read_access, move |ecs| {
                let boundary = ecs.boundary::<EnvironmentBoundary>(env_id)?;
                *seen_root.lock().unwrap() = boundary.environment().get::<u32>(key.name())?;
                Ok(())
            }))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*seen.lock().unwrap(), 7);
    }

    #[test]
    fn root_env_writer_to_sub_reader_is_rejected() {
        let mut builder = ModelBuilder::new();
        let key = builder
            .register_environment::<u32>("counter", 0u32)
            .unwrap();

        let mut root_access = AccessSets::default();
        root_access.produces.insert(key.channel_id());
        let mut sub_access = AccessSets::default();
        sub_access.consumes.insert(key.channel_id());

        let mut sub = SubScheduler::new("sub");
        sub.scheduler_mut()
            .add_system(FnSystem::new(0, "read", sub_access, |_| Ok(())));

        let result = builder
            .with_sub_scheduler(sub)
            .with_system(FnSystem::new(1, "write", root_access, |_| Ok(())))
            .build();
        let Err(err) = result else {
            panic!("expected backward environment channel error");
        };
        assert!(matches!(err, ModelError::BackwardEnvironmentChannel { .. }));
    }

    #[cfg(feature = "messaging")]
    #[derive(Clone, Copy)]
    struct TestMsg;
    #[cfg(feature = "messaging")]
    impl Message for TestMsg {}
    #[cfg(feature = "messaging")]
    impl BruteForceMessage for TestMsg {}

    #[cfg(feature = "messaging")]
    #[test]
    fn message_channels_do_not_overlap_environment_channels() {
        let mut builder = ModelBuilder::new();
        let env_key = builder
            .register_environment::<u32>("counter", 0u32)
            .unwrap();
        let msg = builder
            .register_brute_force_message::<TestMsg>(Capacity::unbounded(4))
            .unwrap();
        assert_ne!(env_key.channel_id(), msg.channel_id());
    }

    #[cfg(feature = "messaging")]
    #[test]
    fn cross_scheduler_message_channel_is_rejected() {
        let mut builder = ModelBuilder::new();
        let handle = builder
            .register_brute_force_message::<TestMsg>(Capacity::unbounded(4))
            .unwrap();

        let mut produce = AccessSets::default();
        produce.produces.insert(handle.channel_id());
        let mut consume = AccessSets::default();
        consume.consumes.insert(handle.channel_id());

        let mut sub = SubScheduler::new("sub");
        sub.scheduler_mut()
            .add_system(FnSystem::new(0, "produce", produce, |_| Ok(())));

        let result = builder
            .with_sub_scheduler(sub)
            .with_system(FnSystem::new(1, "consume", consume, |_| Ok(())))
            .build();
        let Err(err) = result else {
            panic!("expected cross scheduler message channel error");
        };

        assert!(matches!(
            err,
            ModelError::CrossSchedulerMessageChannel { .. }
        ));
    }

    #[cfg(feature = "messaging_gpu")]
    #[repr(C)]
    #[derive(Clone, Copy, Zeroable, Pod)]
    struct GpuMsgA {
        value: u32,
    }

    #[cfg(feature = "messaging_gpu")]
    impl Message for GpuMsgA {}
    #[cfg(feature = "messaging_gpu")]
    impl BruteForceMessage for GpuMsgA {}
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuMessage for GpuMsgA {}

    #[cfg(feature = "messaging_gpu")]
    #[repr(C)]
    #[derive(Clone, Copy, Zeroable, Pod)]
    struct GpuMsgB {
        value: u32,
    }

    #[cfg(feature = "messaging_gpu")]
    impl Message for GpuMsgB {}
    #[cfg(feature = "messaging_gpu")]
    impl BruteForceMessage for GpuMsgB {}
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuMessage for GpuMsgB {}

    #[cfg(feature = "messaging_gpu")]
    #[repr(C)]
    #[derive(Clone, Copy, Zeroable, Pod)]
    struct GpuBucketMsg {
        bucket: u32,
        value: u32,
    }

    #[cfg(feature = "messaging_gpu")]
    impl Message for GpuBucketMsg {}
    #[cfg(feature = "messaging_gpu")]
    impl BucketMessage for GpuBucketMsg {
        fn bucket_key(&self) -> u32 {
            self.bucket
        }
    }
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuMessage for GpuBucketMsg {}
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuBucketMessage for GpuBucketMsg {
        const BUCKET_KEY_WORD_OFFSET: u32 = 0;
    }

    #[cfg(feature = "messaging_gpu")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Zeroable, Pod)]
    struct BadGpuBucketMsg {
        bucket: u32,
    }

    #[cfg(feature = "messaging_gpu")]
    impl Message for BadGpuBucketMsg {}
    #[cfg(feature = "messaging_gpu")]
    impl BucketMessage for BadGpuBucketMsg {
        fn bucket_key(&self) -> u32 {
            self.bucket
        }
    }
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuMessage for BadGpuBucketMsg {}
    #[cfg(feature = "messaging_gpu")]
    unsafe impl crate::messaging::GpuBucketMessage for BadGpuBucketMsg {
        const BUCKET_KEY_WORD_OFFSET: u32 = 4;
    }

    #[cfg(feature = "messaging_gpu")]
    #[test]
    fn gpu_message_handles_expose_stable_resource_bindings() {
        let mut builder = ModelBuilder::new();
        let a = builder
            .register_gpu_brute_force_message::<GpuMsgA>(Capacity::bounded(4, 8))
            .unwrap();
        let b = builder
            .register_gpu_brute_force_message::<GpuMsgB>(Capacity::bounded(4, 8))
            .unwrap();

        assert_eq!(a.resource_ids().resource, 0);
        assert_eq!(b.resource_ids().resource, 1);
        assert_eq!(a.bindings().raw_cpu_stream, 0);
        assert_eq!(a.bindings().raw_gpu_stream, 1);
        assert_eq!(a.bindings().fixed_valid_flags, 2);
        assert_eq!(a.bindings().final_messages, 3);
        assert_eq!(a.bindings().index_metadata, 4);
        assert_eq!(a.bindings().control, 5);
        assert_eq!(a.bindings().params, 5);
        assert_eq!(a.bindings().append_counter, 5);
        assert_eq!(a.bindings().overflow_status, 5);

        let model = builder.build().unwrap();
        model
            .ecs()
            .world_ref()
            .with_exclusive(|data| {
                let ids = [b.resource_ids().resource, a.resource_ids().resource];
                let layout = data.gpu_resources().flattened_binding_descs(&ids);
                assert_eq!(layout.len(), 12);
                assert!(!layout[1].read_only);
                assert!(layout[6].read_only);
                assert!(!layout[11].read_only);
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "messaging_gpu")]
    #[test]
    fn gpu_message_with_options_validates_key_metadata() {
        let mut builder = ModelBuilder::new();
        let handle = builder
            .register_gpu_bucket_message_with_options::<GpuBucketMsg>(
                4,
                Capacity::bounded(4, 8),
                crate::messaging::GpuMessageOptions::deterministic(8),
            )
            .unwrap();

        assert_eq!(handle.resource_ids().resource, 0);

        let mut bad = ModelBuilder::new();
        let err = bad
            .register_gpu_bucket_message_with_options::<BadGpuBucketMsg>(
                4,
                Capacity::bounded(4, 8),
                crate::messaging::GpuMessageOptions::deterministic(8),
            )
            .unwrap_err();
        assert!(matches!(err, ModelError::Messaging(_)));
    }
}
