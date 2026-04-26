//! Fluent builder for [`Model`](crate::model::Model).

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use crate::agents::{AgentRegistry, AgentTemplate};
use crate::engine::channel_allocator::ChannelAllocator;
use crate::engine::component::ComponentRegistry;
use crate::engine::entity::EntityShards;
use crate::engine::manager::ECSManager;
use crate::engine::scheduler::Scheduler;
use crate::engine::systems::System;
use crate::engine::types::ChannelID;
use crate::engine::types::BoundaryID;
use crate::engine::workers::max_workers;
use crate::environment::{EnvKey, EnvironmentBoundary, EnvironmentBuilder};

#[cfg(feature = "messaging")]
use crate::messaging::{
    BruteForceMessage, BucketMessage, Capacity, MessageBufferSet, MessageHandle, MessageRegistry,
    SpatialConfig, SpatialMessage, TargetedMessage,
};

use super::error::ModelError;
use super::model::Model;
use super::sub_scheduler::SubScheduler;

/// Fluent builder for top-level models.
pub struct ModelBuilder {
    channel_allocator: ChannelAllocator,
    component_registry: Option<Arc<RwLock<ComponentRegistry>>>,
    shards: Option<EntityShards>,
    scheduler: Scheduler,
    sub_schedulers: Vec<SubScheduler>,
    environment_builder: Option<EnvironmentBuilder>,
    agents: AgentRegistry,
    #[cfg(feature = "messaging")]
    message_registry: MessageRegistry,
}

impl ModelBuilder {
    /// Creates a builder with empty defaults.
    pub fn new() -> Self {
        Self {
            channel_allocator: ChannelAllocator::new(),
            component_registry: None,
            shards: None,
            scheduler: Scheduler::new(),
            sub_schedulers: Vec::new(),
            environment_builder: None,
            agents: AgentRegistry::new(),
            #[cfg(feature = "messaging")]
            message_registry: MessageRegistry::new(),
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
        0
    }

    /// Registers an environment key and returns its typed scheduler handle.
    pub fn register_environment<T>(
        &mut self,
        key: &'static str,
        default: T,
    ) -> EnvKey<T>
    where
        T: std::any::Any + Clone + Send + Sync + 'static,
    {
        let channel_id = self.channel_allocator.alloc();
        let builder = self.environment_builder.take().unwrap_or_default();
        self.environment_builder =
            Some(builder.register_with_channel::<T>(key, default, channel_id));
        EnvKey::new(key, channel_id)
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

    /// Registers an agent template.
    pub fn with_agent_template(mut self, template: AgentTemplate) -> Result<Self, ModelError> {
        self.agents.register(template)?;
        Ok(self)
    }

    /// Boundary ID that will hold model messages after build.
    #[cfg(feature = "messaging")]
    pub fn message_boundary_id(&self) -> BoundaryID {
        1
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
            .build_with_allocator(&mut self.channel_allocator);
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
        let environment_boundary_id =
            ecs.register_boundary(EnvironmentBoundary::new(Arc::clone(&environment)))?;

        #[cfg(feature = "messaging")]
        let (message_boundary_id, message_registry) = {
            self.message_registry.freeze();
            let message_registry = Arc::new(self.message_registry);
            let message_boundary_id =
                ecs.register_boundary(MessageBufferSet::new(Arc::clone(&message_registry)))?;
            (message_boundary_id, message_registry)
        };

        Ok(Model {
            ecs,
            environment,
            agents: self.agents,
            scheduler: self.scheduler,
            sub_schedulers: self.sub_schedulers,
            environment_boundary_id,
            #[cfg(feature = "messaging")]
            message_boundary_id,
            #[cfg(feature = "messaging")]
            message_registry,
            tick_count: 0,
        })
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

                    if environment_channels.contains(&channel_id)
                        && producer.order > consumer.order
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
    use crate::engine::systems::{AccessSets, FnSystem};
    use crate::environment::EnvironmentBoundary;
    use crate::model::ModelError;
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "messaging")]
    use crate::messaging::{Capacity, Message};
    #[cfg(feature = "messaging")]
    use crate::messaging::BruteForceMessage;

    #[test]
    fn empty_model_builds_and_ticks() {
        let mut model = ModelBuilder::new().build().unwrap();
        model.tick().unwrap();
        assert_eq!(model.tick_count(), 1);
    }

    #[test]
    fn sub_scheduler_runs_before_root() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut sub = SubScheduler::new("sub");
        let sub_order = Arc::clone(&order);
        sub.scheduler_mut().add_system(FnSystem::new(
            0,
            "sub",
            AccessSets::default(),
            move |_| {
                sub_order.lock().unwrap().push(1);
                Ok(())
            },
        ));

        let root_order = Arc::clone(&order);
        let mut model = ModelBuilder::new()
            .with_sub_scheduler(sub)
            .with_system(FnSystem::new(
                1,
                "root",
                AccessSets::default(),
                move |_| {
                    root_order.lock().unwrap().push(2);
                    Ok(())
                },
            ))
            .build()
            .unwrap();

        model.tick().unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn sub_env_writer_is_visible_to_root_reader() {
        let mut builder = ModelBuilder::new();
        let env_id = builder.environment_boundary_id();
        let key = builder.register_environment::<u32>("counter", 0u32);

        let mut write_access = AccessSets::default();
        write_access.produces.insert(key.channel_id());
        let mut sub = SubScheduler::new("sub");
        sub.scheduler_mut().add_system(FnSystem::new(
            0,
            "write_env",
            write_access,
            move |ecs| {
                let boundary = ecs.boundary::<EnvironmentBoundary>(env_id)?;
                boundary.environment().set(key.name(), 7u32)?;
                Ok(())
            },
        ));

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
        let key = builder.register_environment::<u32>("counter", 0u32);

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
        let env_key = builder.register_environment::<u32>("counter", 0u32);
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
}
