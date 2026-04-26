//! Runtime storage and boundary lifecycle for typed messages.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::boundary::{BoundaryContext, BoundaryResource};
use crate::engine::entity::Entity;
use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;

use super::aligned_buffer::AlignedBuffer;
use super::error::MessagingError;
use super::message::{
    BruteForceMessage, BucketMessage, Message, MessageHandle, MessageTypeID, SpatialMessage,
    TargetedMessage,
};
use super::registry::{MessageRegistry, Specialisation};
use super::specialisations::brute_force::{BruteForceBuffer, BruteForceIter};
use super::specialisations::bucket::{BucketBuffer, BucketIter};
use super::specialisations::spatial::{SpatialBuffer, SpatialQueryIter};
use super::specialisations::targeted::{InboxIter, TargetedBuffer};
use super::thread_local_emit::{self, MessageRuntimeID};

#[cfg(feature = "messaging_gpu")]
use super::gpu::{targeted_index_words, GpuFinaliseOutput, GpuMessageResource};

#[cfg(feature = "messaging_gpu")]
enum GpuRoute {
    GpuFinalised(Option<GpuFinaliseOutput>),
    CpuFallback,
}

pub(crate) struct MessageBufferSetInner {
    brute_force: Vec<Option<BruteForceBuffer>>,
    bucket: Vec<Option<BucketBuffer>>,
    spatial: Vec<Option<SpatialBuffer>>,
    targeted: Vec<Option<TargetedBuffer>>,
    channel_to_mtid: HashMap<ChannelID, MessageTypeID>,
}

unsafe impl Sync for MessageBufferSetInner {}

impl MessageBufferSetInner {
    fn new(registry: &MessageRegistry) -> Self {
        let n = registry.descriptors().len();
        let mut inner = MessageBufferSetInner {
            brute_force: (0..n).map(|_| None).collect(),
            bucket: (0..n).map(|_| None).collect(),
            spatial: (0..n).map(|_| None).collect(),
            targeted: (0..n).map(|_| None).collect(),
            channel_to_mtid: HashMap::new(),
        };

        for desc in registry.descriptors() {
            let idx = desc.message_type_id.index();
            inner
                .channel_to_mtid
                .insert(desc.channel_id, desc.message_type_id);
            match desc.specialisation {
                Specialisation::BruteForce => {
                    inner.brute_force[idx] = Some(BruteForceBuffer::new(
                        desc.item_size,
                        desc.item_align,
                        desc.capacity.initial,
                    ));
                }
                Specialisation::Bucket { max_buckets } => {
                    inner.bucket[idx] = Some(BucketBuffer::new(
                        desc.item_size,
                        desc.item_align,
                        max_buckets,
                        desc.capacity.initial,
                    ));
                }
                Specialisation::Spatial(cfg) => {
                    inner.spatial[idx] = Some(SpatialBuffer::new(
                        desc.item_size,
                        desc.item_align,
                        cfg,
                        desc.capacity.initial,
                    ));
                }
                Specialisation::Targeted => {
                    inner.targeted[idx] = Some(TargetedBuffer::new(
                        desc.item_size,
                        desc.item_align,
                        desc.capacity.initial,
                    ));
                }
            }
        }

        inner
    }

    fn brute_force_iter<M: BruteForceMessage>(&self, mtid: MessageTypeID) -> BruteForceIter<'_, M> {
        match self.brute_force.get(mtid.index()).and_then(|s| s.as_ref()) {
            Some(buf) => BruteForceIter::new(buf),
            None => BruteForceIter::empty(),
        }
    }

    fn bucket_iter<M: BucketMessage>(&self, mtid: MessageTypeID, key: u32) -> BucketIter<'_, M> {
        match self.bucket.get(mtid.index()).and_then(|s| s.as_ref()) {
            Some(buf) => BucketIter::new(buf, key),
            None => BucketIter::empty(),
        }
    }

    fn spatial_query<M: SpatialMessage>(
        &self,
        mtid: MessageTypeID,
        cx: f32,
        cy: f32,
        r: f32,
    ) -> SpatialQueryIter<'_, M> {
        match self.spatial.get(mtid.index()).and_then(|s| s.as_ref()) {
            Some(buf) => SpatialQueryIter::new(buf, cx, cy, r),
            None => SpatialQueryIter::empty(),
        }
    }

    fn inbox_iter<M: TargetedMessage>(
        &self,
        mtid: MessageTypeID,
        recipient: Entity,
    ) -> InboxIter<'_, M> {
        match self.targeted.get(mtid.index()).and_then(|s| s.as_ref()) {
            Some(buf) => InboxIter::new(buf, recipient),
            None => InboxIter::empty(),
        }
    }
}

/// Runtime container for all message buffers in one model.
pub struct MessageBufferSet {
    registry: Arc<MessageRegistry>,
    inner: MessageBufferSetInner,
    runtime_id: MessageRuntimeID,
    channels: Vec<ChannelID>,
}

impl MessageBufferSet {
    /// Creates a new buffer set from a frozen registry.
    pub fn new(registry: Arc<MessageRegistry>) -> ECSResult<Self> {
        if !registry.is_frozen() {
            return Err(MessagingError::RegistryNotFrozen.into());
        }
        let channels = registry
            .descriptors()
            .iter()
            .map(|d| d.channel_id)
            .collect();
        Ok(Self {
            inner: MessageBufferSetInner::new(&registry),
            registry,
            runtime_id: thread_local_emit::alloc_runtime_id(),
            channels,
        })
    }

    /// Returns the message registry used by this runtime.
    pub fn registry(&self) -> &Arc<MessageRegistry> {
        &self.registry
    }

    /// Emits one message into this model's per-worker staging buffers.
    pub fn emit<M: Message>(&self, handle: MessageHandle<M>, msg: M) -> ECSResult<()> {
        self.validate_handle(handle)?;
        let desc = self.registry.descriptor(handle.message_type_id);
        thread_local_emit::emit(
            self.runtime_id,
            self.registry.descriptors().len(),
            handle.message_type_id,
            desc.item_size,
            desc.item_align,
            desc.capacity.initial,
            msg,
        )?;
        Ok(())
    }

    /// Iterates all messages for a brute-force message type.
    pub fn brute_force<M: BruteForceMessage>(
        &self,
        handle: MessageHandle<M>,
    ) -> ECSResult<BruteForceIter<'_, M>> {
        self.validate_handle(handle)?;
        Ok(self.inner.brute_force_iter(handle.message_type_id))
    }

    /// Iterates all messages in one bucket.
    pub fn bucket<M: BucketMessage>(
        &self,
        handle: MessageHandle<M>,
        key: u32,
    ) -> ECSResult<BucketIter<'_, M>> {
        self.validate_handle(handle)?;
        Ok(self.inner.bucket_iter(handle.message_type_id, key))
    }

    /// Queries spatial messages by radius.
    pub fn spatial<M: SpatialMessage>(
        &self,
        handle: MessageHandle<M>,
        cx: f32,
        cy: f32,
        r: f32,
    ) -> ECSResult<SpatialQueryIter<'_, M>> {
        self.validate_handle(handle)?;
        Ok(self.inner.spatial_query(handle.message_type_id, cx, cy, r))
    }

    /// Iterates all messages addressed to `recipient`.
    pub fn inbox<M: TargetedMessage>(
        &self,
        handle: MessageHandle<M>,
        recipient: Entity,
    ) -> ECSResult<InboxIter<'_, M>> {
        self.validate_handle(handle)?;
        Ok(self.inner.inbox_iter(handle.message_type_id, recipient))
    }

    fn validate_handle<M: Message>(&self, handle: MessageHandle<M>) -> ECSResult<()> {
        if handle.registry_id != self.registry.registry_id() {
            return Err(MessagingError::WrongRegistry.into());
        }
        if handle.message_type_id.index() >= self.registry.descriptors().len() {
            return Err(MessagingError::WrongRegistry.into());
        }
        Ok(())
    }

    fn tick_begin(&mut self, ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        #[cfg(not(feature = "messaging_gpu"))]
        let _ = ctx;

        for desc in self.registry.descriptors() {
            thread_local_emit::clear_for_tick(self.runtime_id, desc.message_type_id)?;
            #[cfg(feature = "messaging_gpu")]
            if let Some(resource_id) = desc.gpu_resource_id {
                if let Some(gpu_resources) = ctx.gpu_resources_mut() {
                    if let Some(resource) =
                        gpu_resources.get_mut_typed::<GpuMessageResource>(resource_id)
                    {
                        resource.begin_tick();
                        gpu_resources.mark_cpu_dirty(resource_id)?;
                    }
                }
            }
        }

        for buf in self.inner.brute_force.iter_mut().flatten() {
            buf.begin_tick();
        }
        for buf in self.inner.bucket.iter_mut().flatten() {
            buf.begin_tick();
        }
        for buf in self.inner.spatial.iter_mut().flatten() {
            buf.begin_tick();
        }
        for buf in self.inner.targeted.iter_mut().flatten() {
            buf.begin_tick();
        }
        Ok(())
    }

    fn tick_finalise(
        &mut self,
        ctx: &mut BoundaryContext<'_>,
        channels: &[ChannelID],
    ) -> ECSResult<()> {
        #[cfg(not(feature = "messaging_gpu"))]
        let _ = ctx;

        for &channel_id in channels {
            let Some(&mtid) = self.inner.channel_to_mtid.get(&channel_id) else {
                continue;
            };
            let idx = mtid.index();
            let desc = self.registry.descriptor(mtid);
            let mut raw = AlignedBuffer::with_capacity(
                desc.item_size,
                desc.item_align,
                desc.capacity.initial,
            );
            thread_local_emit::drain_into(self.runtime_id, mtid, &mut raw)?;
            #[cfg(feature = "messaging_gpu")]
            let cpu_raw_bytes_for_gpu = raw.as_bytes().to_vec();

            #[cfg(feature = "messaging_gpu")]
            if let Some(route) = Self::try_gpu_route(ctx, channel_id, desc, &mut raw)? {
                if let GpuRoute::GpuFinalised(output) = route {
                    if let Some(output) = output {
                        Self::apply_gpu_finalised(
                            &mut self.inner,
                            idx,
                            desc.specialisation,
                            output,
                        );
                    }
                    continue;
                }
            }

            #[cfg(feature = "messaging_gpu")]
            Self::publish_gpu_cpu_raw(ctx, desc.gpu_resource_id, &cpu_raw_bytes_for_gpu);

            if let Some(max) = desc.capacity.max {
                if raw.len() > max {
                    return Err(MessagingError::EmitCapacityExceeded {
                        channel_id,
                        len: raw.len(),
                        max,
                    }
                    .into());
                }
            }

            match desc.specialisation {
                Specialisation::BruteForce => {
                    if let Some(buf) = self.inner.brute_force[idx].as_mut() {
                        buf.data.extend_from(&raw);
                        buf.finalise();
                        #[cfg(feature = "messaging_gpu")]
                        Self::publish_gpu_finalised(
                            ctx,
                            desc.gpu_resource_id,
                            buf.data.as_bytes(),
                            &[0, buf.data.len() as u32],
                        );
                    }
                }
                Specialisation::Bucket { .. } => {
                    if let Some(buf) = self.inner.bucket[idx].as_mut() {
                        unsafe { buf.finalise(&raw, &desc.erased_fns)? };
                        #[cfg(feature = "messaging_gpu")]
                        Self::publish_gpu_finalised(
                            ctx,
                            desc.gpu_resource_id,
                            buf.data.as_bytes(),
                            &buf.bucket_starts,
                        );
                    }
                }
                Specialisation::Spatial(_) => {
                    if let Some(buf) = self.inner.spatial[idx].as_mut() {
                        unsafe { buf.finalise(&raw, &desc.erased_fns) };
                        #[cfg(feature = "messaging_gpu")]
                        Self::publish_gpu_finalised(
                            ctx,
                            desc.gpu_resource_id,
                            buf.data.as_bytes(),
                            &buf.cell_starts,
                        );
                    }
                }
                Specialisation::Targeted => {
                    if let Some(buf) = self.inner.targeted[idx].as_mut() {
                        unsafe { buf.finalise(&raw, &desc.erased_fns) };
                        #[cfg(feature = "messaging_gpu")]
                        {
                            let mut rows: Vec<_> = buf
                                .inbox_index
                                .iter()
                                .map(|(&entity, range)| (entity.to_raw(), range.start, range.end))
                                .collect();
                            rows.sort_unstable_by_key(|row| row.0);
                            let mut index_words = Vec::with_capacity(rows.len() * 4);
                            for (entity_raw, start, end) in rows {
                                index_words.extend_from_slice(&targeted_index_words(
                                    entity_raw, start, end,
                                ));
                            }
                            Self::publish_gpu_finalised(
                                ctx,
                                desc.gpu_resource_id,
                                buf.data.as_bytes(),
                                &index_words,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "messaging_gpu")]
    fn publish_gpu_finalised(
        ctx: &mut BoundaryContext<'_>,
        resource_id: Option<crate::engine::types::GPUResourceID>,
        bytes: &[u8],
        index_words: &[u32],
    ) {
        let Some(resource_id) = resource_id else {
            return;
        };
        let Some(gpu_resources) = ctx.gpu_resources_mut() else {
            return;
        };
        if let Some(resource) = gpu_resources.get_mut_typed::<GpuMessageResource>(resource_id) {
            resource.publish_finalised(bytes, index_words);
            let _ = gpu_resources.mark_cpu_dirty(resource_id);
        }
    }

    #[cfg(feature = "messaging_gpu")]
    fn publish_gpu_cpu_raw(
        ctx: &mut BoundaryContext<'_>,
        resource_id: Option<crate::engine::types::GPUResourceID>,
        bytes: &[u8],
    ) {
        let Some(resource_id) = resource_id else {
            return;
        };
        let Some(gpu_resources) = ctx.gpu_resources_mut() else {
            return;
        };
        if let Some(resource) = gpu_resources.get_mut_typed::<GpuMessageResource>(resource_id) {
            resource.publish_cpu_raw(bytes);
            let _ = gpu_resources.mark_cpu_dirty(resource_id);
        }
    }

    #[cfg(feature = "messaging_gpu")]
    fn try_gpu_route(
        ctx: &mut BoundaryContext<'_>,
        channel_id: ChannelID,
        desc: &super::registry::MessageDescriptor,
        raw: &mut AlignedBuffer,
    ) -> ECSResult<Option<GpuRoute>> {
        let Some(resource_id) = desc.gpu_resource_id else {
            return Ok(None);
        };

        let profile = ctx.channel_profile(channel_id).unwrap_or(
            crate::engine::boundary::BoundaryChannelProfile {
                channel_id,
                cpu_producer: true,
                gpu_producer: false,
                cpu_consumer: true,
                gpu_consumer: false,
            },
        );
        let needs_cpu_download = profile.cpu_consumer;

        let Some(prefer_gpu) = ctx.with_gpu_dispatch(|dispatch, gpu_resources| {
            let Some(resource) = gpu_resources.get_mut_typed::<GpuMessageResource>(resource_id)
            else {
                return Ok(None);
            };
            let prefer_gpu = resource.prefers_gpu_finalise(
                raw.len(),
                profile.gpu_producer,
                profile.gpu_consumer,
            );
            if prefer_gpu {
                resource.publish_cpu_raw(raw.as_bytes());
                let output = resource.finalise_gpu(dispatch, needs_cpu_download)?;
                return Ok(Some(GpuRoute::GpuFinalised(
                    needs_cpu_download.then_some(output),
                )));
            }
            if profile.gpu_producer {
                resource.download_gpu_raw_into(dispatch, raw)?;
            }
            Ok(Some(GpuRoute::CpuFallback))
        })?
        else {
            return Ok(None);
        };

        Ok(prefer_gpu)
    }

    #[cfg(feature = "messaging_gpu")]
    fn apply_gpu_finalised(
        inner: &mut MessageBufferSetInner,
        idx: usize,
        specialisation: Specialisation,
        output: GpuFinaliseOutput,
    ) {
        let final_bytes = output.final_bytes.unwrap_or_default();
        let index_words = output.index_words.unwrap_or_default();
        match specialisation {
            Specialisation::BruteForce => {
                if let Some(buf) = inner.brute_force[idx].as_mut() {
                    buf.data.clear();
                    buf.data.extend_from_bytes(&final_bytes);
                }
            }
            Specialisation::Bucket { max_buckets } => {
                if let Some(buf) = inner.bucket[idx].as_mut() {
                    buf.data.clear();
                    buf.data.extend_from_bytes(&final_bytes);
                    buf.bucket_starts.clear();
                    buf.bucket_starts.extend_from_slice(&index_words);
                    if buf.bucket_starts.len() != max_buckets as usize + 1 {
                        buf.bucket_starts.resize(max_buckets as usize + 1, 0);
                    }
                }
            }
            Specialisation::Spatial(cfg) => {
                if let Some(buf) = inner.spatial[idx].as_mut() {
                    buf.data.clear();
                    buf.data.extend_from_bytes(&final_bytes);
                    buf.cell_starts.clear();
                    buf.cell_starts.extend_from_slice(&index_words);
                    if buf.cell_starts.len() != cfg.total_cells() + 1 {
                        buf.cell_starts.resize(cfg.total_cells() + 1, 0);
                    }
                }
            }
            Specialisation::Targeted => {
                if let Some(buf) = inner.targeted[idx].as_mut() {
                    buf.data.clear();
                    buf.data.extend_from_bytes(&final_bytes);
                    buf.inbox_index.clear();
                    for row in index_words.chunks_exact(4) {
                        let raw = row[0] as u64 | ((row[1] as u64) << 32);
                        buf.inbox_index
                            .insert(Entity::from_raw(raw), row[2]..row[3]);
                    }
                }
            }
        }
    }

    fn tick_end(&self) -> ECSResult<()> {
        Ok(())
    }
}

impl BoundaryResource for MessageBufferSet {
    fn name(&self) -> &str {
        "MessageBufferSet"
    }

    fn channels(&self) -> &[ChannelID] {
        &self.channels
    }

    fn begin_tick(&mut self, ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        self.tick_begin(ctx)
    }

    fn end_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        self.tick_end()
    }

    fn finalise(&mut self, ctx: &mut BoundaryContext<'_>, channels: &[ChannelID]) -> ECSResult<()> {
        self.tick_finalise(ctx, channels)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::channel_allocator::ChannelAllocator;
    use crate::engine::error::ECSError;
    use crate::messaging::error::MessagingError;
    use crate::messaging::{BucketMessage, Capacity, SpatialConfig};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct GlobalMsg(u32);
    impl Message for GlobalMsg {}
    impl BruteForceMessage for GlobalMsg {}

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct BucketMsg {
        bucket: u32,
        value: u32,
    }
    impl Message for BucketMsg {}
    impl BucketMessage for BucketMsg {
        fn bucket_key(&self) -> u32 {
            self.bucket
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SpatialMsg {
        x: f32,
        y: f32,
        value: u32,
    }
    impl Message for SpatialMsg {}
    impl SpatialMessage for SpatialMsg {
        fn position(&self) -> (f32, f32) {
            (self.x, self.y)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct DirectMsg {
        to: Entity,
        value: u32,
    }
    impl Message for DirectMsg {}
    impl TargetedMessage for DirectMsg {
        fn recipient(&self) -> Entity {
            self.to
        }
    }

    #[test]
    fn brute_force_roundtrip_and_clear() {
        let mut alloc = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let handle = registry
            .register_brute_force::<GlobalMsg>(&mut alloc, Capacity::unbounded(4))
            .unwrap();
        registry.freeze();
        let mut buffers = MessageBufferSet::new(Arc::new(registry)).unwrap();

        buffers.emit(handle, GlobalMsg(7)).unwrap();
        let mut ctx = BoundaryContext::empty();
        buffers
            .tick_finalise(&mut ctx, &[handle.channel_id()])
            .unwrap();
        let got: Vec<_> = buffers.brute_force(handle).unwrap().collect();
        assert_eq!(got, vec![GlobalMsg(7)]);

        buffers.tick_begin(&mut ctx).unwrap();
        buffers
            .tick_finalise(&mut ctx, &[handle.channel_id()])
            .unwrap();
        assert_eq!(buffers.brute_force(handle).unwrap().count(), 0);
    }

    #[test]
    fn new_rejects_unfrozen_registry() {
        let registry = MessageRegistry::new();
        let err = match MessageBufferSet::new(Arc::new(registry)) {
            Ok(_) => panic!("expected unfrozen registry to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            ECSError::Messaging(MessagingError::RegistryNotFrozen)
        ));
    }

    #[test]
    fn bucket_spatial_and_targeted_roundtrip() {
        let mut alloc = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let bucket = registry
            .register_bucket::<BucketMsg>(&mut alloc, 3, Capacity::unbounded(4))
            .unwrap();
        let spatial = registry
            .register_spatial::<SpatialMsg>(
                &mut alloc,
                SpatialConfig {
                    width: 100.0,
                    height: 100.0,
                    cell_size: 10.0,
                },
                Capacity::unbounded(4),
            )
            .unwrap();
        let direct = registry
            .register_targeted::<DirectMsg>(&mut alloc, Capacity::unbounded(4))
            .unwrap();
        registry.freeze();
        let mut buffers = MessageBufferSet::new(Arc::new(registry)).unwrap();
        let recipient = Entity::from_raw(42);

        buffers
            .emit(
                bucket,
                BucketMsg {
                    bucket: 2,
                    value: 11,
                },
            )
            .unwrap();
        buffers
            .emit(
                spatial,
                SpatialMsg {
                    x: 15.0,
                    y: 15.0,
                    value: 22,
                },
            )
            .unwrap();
        buffers
            .emit(
                direct,
                DirectMsg {
                    to: recipient,
                    value: 33,
                },
            )
            .unwrap();

        buffers
            .tick_finalise(
                &mut BoundaryContext::empty(),
                &[
                    bucket.channel_id(),
                    spatial.channel_id(),
                    direct.channel_id(),
                ],
            )
            .unwrap();

        assert_eq!(
            buffers.bucket(bucket, 2).unwrap().collect::<Vec<_>>()[0].value,
            11
        );
        assert_eq!(
            buffers
                .spatial(spatial, 15.0, 15.0, 1.0)
                .unwrap()
                .collect::<Vec<_>>()[0]
                .value,
            22
        );
        assert_eq!(
            buffers
                .inbox(direct, recipient)
                .unwrap()
                .collect::<Vec<_>>()[0]
                .value,
            33
        );
    }

    #[test]
    fn bucket_out_of_range_fails_at_finalise() {
        let mut alloc = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let handle = registry
            .register_bucket::<BucketMsg>(&mut alloc, 2, Capacity::unbounded(4))
            .unwrap();
        registry.freeze();
        let mut buffers = MessageBufferSet::new(Arc::new(registry)).unwrap();
        buffers
            .emit(
                handle,
                BucketMsg {
                    bucket: 9,
                    value: 1,
                },
            )
            .unwrap();

        assert!(buffers
            .tick_finalise(&mut BoundaryContext::empty(), &[handle.channel_id()])
            .is_err());
    }

    #[test]
    fn independent_runtimes_do_not_cross_drain() {
        let mut alloc_a = ChannelAllocator::new();
        let mut reg_a = MessageRegistry::new();
        let handle_a = reg_a
            .register_brute_force::<GlobalMsg>(&mut alloc_a, Capacity::unbounded(4))
            .unwrap();
        reg_a.freeze();
        let mut a = MessageBufferSet::new(Arc::new(reg_a)).unwrap();

        let mut alloc_b = ChannelAllocator::new();
        let mut reg_b = MessageRegistry::new();
        let handle_b = reg_b
            .register_brute_force::<GlobalMsg>(&mut alloc_b, Capacity::unbounded(4))
            .unwrap();
        reg_b.freeze();
        let mut b = MessageBufferSet::new(Arc::new(reg_b)).unwrap();

        a.emit(handle_a, GlobalMsg(1)).unwrap();
        a.tick_finalise(&mut BoundaryContext::empty(), &[handle_a.channel_id()])
            .unwrap();
        b.tick_finalise(&mut BoundaryContext::empty(), &[handle_b.channel_id()])
            .unwrap();

        assert_eq!(a.brute_force(handle_a).unwrap().count(), 1);
        assert_eq!(b.brute_force(handle_b).unwrap().count(), 0);
    }

    #[test]
    fn handle_from_wrong_registry_is_rejected() {
        let mut alloc_a = ChannelAllocator::new();
        let mut reg_a = MessageRegistry::new();
        let handle_a = reg_a
            .register_brute_force::<GlobalMsg>(&mut alloc_a, Capacity::unbounded(4))
            .unwrap();
        reg_a.freeze();

        let mut alloc_b = ChannelAllocator::new();
        let mut reg_b = MessageRegistry::new();
        reg_b
            .register_brute_force::<GlobalMsg>(&mut alloc_b, Capacity::unbounded(4))
            .unwrap();
        reg_b.freeze();
        let buffers_b = MessageBufferSet::new(Arc::new(reg_b)).unwrap();

        let err = buffers_b.emit(handle_a, GlobalMsg(1)).unwrap_err();
        assert!(matches!(
            err,
            ECSError::Messaging(MessagingError::WrongRegistry)
        ));
    }
}
