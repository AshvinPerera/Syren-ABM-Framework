//! [`MessageRegistry`] - the compile-time (pre-freeze) catalogue of all
//! message types, their specialisations, and their per-type metadata.
//!
//! # Lifecycle
//!
//! 1. Create a [`MessageRegistry`] with [`MessageRegistry::new`].
//! 2. Register each message type with the appropriate `register_*` method,
//!    passing a `&mut ChannelAllocator` shared with the rest of the engine so
//!    that channel IDs are globally unique.
//! 3. Call [`MessageRegistry::freeze`] before handing the registry to
//!    [`MessageBufferSet`](crate::messaging::MessageBufferSet).  Further
//!    registrations after freezing return [`MessagingError::RegistryFrozen`].

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::channel_allocator::ChannelAllocator;
use crate::engine::entity::Entity;
use crate::engine::types::ChannelID;

#[cfg(feature = "messaging_gpu")]
use crate::engine::types::GPUResourceID;

use super::error::MessagingError;
use super::message::{
    BruteForceMessage, BucketMessage, Message, MessageHandle, MessageTypeID, SpatialMessage,
    TargetedMessage,
};

// -----------------------------------------------------------------------------
// Spatial configuration
// -----------------------------------------------------------------------------

/// Configuration for a spatial message type: defines the 2-D grid used to
/// bucket messages by position.
#[derive(Clone, Copy, Debug)]
pub struct SpatialConfig {
    /// World width in world-space units.
    pub width: f32,
    /// World height in world-space units.
    pub height: f32,
    /// Size of each grid cell (must be > 0).
    pub cell_size: f32,
}

impl SpatialConfig {
    /// Returns the number of cells along the X axis.
    #[inline]
    pub fn cols(&self) -> u32 {
        (self.width / self.cell_size).ceil() as u32
    }

    /// Returns the number of cells along the Y axis.
    #[inline]
    pub fn rows(&self) -> u32 {
        (self.height / self.cell_size).ceil() as u32
    }

    /// Total number of cells in the grid.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.cols() as usize * self.rows() as usize
    }

    /// Converts a world-space `(x, y)` coordinate to a flat cell index,
    /// clamped to the grid bounds.
    #[inline]
    pub fn cell_id_of(&self, x: f32, y: f32) -> u32 {
        let col = ((x / self.cell_size) as u32).min(self.cols() - 1);
        let row = ((y / self.cell_size) as u32).min(self.rows() - 1);
        row * self.cols() + col
    }

    /// Returns the range of cells (inclusive on both ends) covered by a
    /// circle centred at `(cx, cy)` with radius `r`.
    ///
    /// Returns `(col_lo, col_hi, row_lo, row_hi)`.
    #[inline]
    pub fn cell_range_for_radius(&self, cx: f32, cy: f32, r: f32) -> (u32, u32, u32, u32) {
        let cols = self.cols();
        let rows = self.rows();
        let col_lo = ((cx - r) / self.cell_size).floor().max(0.0) as u32;
        let col_hi = ((cx + r) / self.cell_size).ceil().min((cols - 1) as f32) as u32;
        let row_lo = ((cy - r) / self.cell_size).floor().max(0.0) as u32;
        let row_hi = ((cy + r) / self.cell_size).ceil().min((rows - 1) as f32) as u32;
        (col_lo, col_hi, row_lo, row_hi)
    }
}

// -----------------------------------------------------------------------------
// Capacity
// -----------------------------------------------------------------------------

/// Per-tick message capacity.
///
/// `initial` preallocates storage. `max`, when present, enforces a hard
/// per-tick cap during boundary finalisation.
#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    /// Initial allocation hint.
    pub initial: usize,
    /// Optional hard per-tick cap.
    pub max: Option<usize>,
}

impl Capacity {
    /// Creates a capacity with no hard maximum.
    pub fn unbounded(initial: usize) -> Self {
        Self { initial, max: None }
    }

    /// Creates a capacity with a hard maximum.
    pub fn bounded(initial: usize, max: usize) -> Self {
        Self {
            initial,
            max: Some(max),
        }
    }
}

// -----------------------------------------------------------------------------
// Specialisation tag
// -----------------------------------------------------------------------------

/// Which specialisation a message type was registered under.
#[derive(Clone, Copy, Debug)]
pub enum Specialisation {
    /// Linear-scan; every consumer sees every message.
    BruteForce,
    /// Counting-sort by integer key; consumers read a single bucket.
    Bucket {
        /// Maximum number of distinct bucket keys.
        max_buckets: u32,
    },
    /// Counting-sort by 2-D position; consumers do radius queries.
    Spatial(SpatialConfig),
    /// HashMap inbox keyed on full [`Entity`]; each recipient has their own
    /// contiguous slice.
    Targeted,
}

impl Specialisation {
    /// Returns the human-readable name of the variant, for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Specialisation::BruteForce => "BruteForce",
            Specialisation::Bucket { .. } => "Bucket",
            Specialisation::Spatial(_) => "Spatial",
            Specialisation::Targeted => "Targeted",
        }
    }
}

// -----------------------------------------------------------------------------
// Type-erased function pointers
// -----------------------------------------------------------------------------

type BucketKeyFn = unsafe fn(*const u8) -> u32;
type PositionFn = unsafe fn(*const u8) -> (f32, f32);
type RecipientFn = unsafe fn(*const u8) -> Entity;

/// Type-erased function pointers captured at registration time where the
/// concrete type `M` is known, used during the counting-sort scatter pass.
///
/// # Safety
///
/// Each function pointer receives a `*const u8` that must point to a valid,
/// initialised item of the type that was registered.  Callers in the buffer-set
/// module are responsible for ensuring this invariant.
pub(crate) struct ErasedFns {
    /// For `BucketMessage`: extracts the integer bucket key from a raw item.
    pub bucket_key: Option<BucketKeyFn>,
    /// For `SpatialMessage`: extracts the `(x, y)` world position.
    pub position: Option<PositionFn>,
    /// For `TargetedMessage`: extracts the recipient [`Entity`].
    pub recipient: Option<RecipientFn>,
}

// -----------------------------------------------------------------------------
// Descriptor
// -----------------------------------------------------------------------------

/// Metadata for a registered message type.
pub struct MessageDescriptor {
    /// Rust `TypeId` of the concrete message type.
    pub type_id: TypeId,
    /// Human-readable type name for diagnostics.
    pub type_name: &'static str,
    /// `size_of::<M>()`.
    pub item_size: usize,
    /// `align_of::<M>()`.
    pub item_align: usize,
    /// Which specialisation was chosen at registration.
    pub specialisation: Specialisation,
    /// Pre-tick buffer capacity hint.
    pub capacity: Capacity,
    /// Whether the message layout is GPU-safe.
    pub gpu_safe: bool,
    /// GPU resource backing this message type, when registered for GPU use.
    #[cfg(feature = "messaging_gpu")]
    pub gpu_resource_id: Option<GPUResourceID>,
    /// The [`ChannelID`] allocated for this message type's read/write
    /// channel.
    pub channel_id: ChannelID,
    /// Dense index used to address per-type slots in buffer arrays.
    pub(crate) message_type_id: MessageTypeID,
    /// Type-erased accessors captured at registration.
    pub(crate) erased_fns: ErasedFns,
}

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

/// The compile-time catalogue of message types.
///
/// Once [`freeze`](MessageRegistry::freeze) is called no further registrations
/// are accepted. The registry is typically wrapped in an [`std::sync::Arc`] and shared
/// between [`MessageBufferSet`](crate::messaging::MessageBufferSet) and any
/// system that needs to look up `MessageTypeID`s.
pub struct MessageRegistry {
    registry_id: u64,
    descriptors: Vec<MessageDescriptor>,
    by_type: HashMap<TypeId, MessageTypeID>,
    frozen: bool,
}

static NEXT_REGISTRY_ID: AtomicU64 = AtomicU64::new(1);

impl MessageRegistry {
    /// Creates an empty, unfrozen registry.
    pub fn new() -> Self {
        MessageRegistry {
            registry_id: NEXT_REGISTRY_ID.fetch_add(1, Ordering::Relaxed),
            descriptors: Vec::new(),
            by_type: HashMap::new(),
            frozen: false,
        }
    }

    // -------------------------------------------------------------------------
    // Registration helpers
    // -------------------------------------------------------------------------

    fn check_not_registered<M: Message>(&self) -> Result<(), MessagingError> {
        if self.by_type.contains_key(&TypeId::of::<M>()) {
            return Err(MessagingError::AlreadyRegistered(std::any::type_name::<M>()));
        }
        if self.frozen {
            return Err(MessagingError::RegistryFrozen);
        }
        Ok(())
    }

    fn push_descriptor<M: Message>(&mut self, desc: MessageDescriptor) -> MessageHandle<M> {
        let mtid = desc.message_type_id;
        let channel_id = desc.channel_id;
        self.by_type.insert(desc.type_id, mtid);
        self.descriptors.push(desc);
        MessageHandle::new(mtid, self.registry_id, channel_id)
    }

    // -------------------------------------------------------------------------
    // Public registration API
    // -------------------------------------------------------------------------

    /// Registers a [`BruteForceMessage`] type.
    ///
    /// # Errors
    ///
    /// Returns [`MessagingError::AlreadyRegistered`] if `M` was already
    /// registered, or [`MessagingError::RegistryFrozen`] if the registry is
    /// frozen.
    pub fn register_brute_force<M: BruteForceMessage>(
        &mut self,
        allocator: &mut ChannelAllocator,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, MessagingError> {
        self.check_not_registered::<M>()?;
        let mtid = MessageTypeID(self.descriptors.len() as u32);
        let channel_id = allocator
            .alloc()
            .map_err(|_| MessagingError::ChannelAllocationOverflow)?;
        let desc = MessageDescriptor {
            type_id: TypeId::of::<M>(),
            type_name: std::any::type_name::<M>(),
            item_size: std::mem::size_of::<M>(),
            item_align: std::mem::align_of::<M>(),
            specialisation: Specialisation::BruteForce,
            capacity,
            gpu_safe: M::GPU_SAFE,
            #[cfg(feature = "messaging_gpu")]
            gpu_resource_id: None,
            channel_id,
            message_type_id: mtid,
            erased_fns: ErasedFns {
                bucket_key: None,
                position: None,
                recipient: None,
            },
        };
        Ok(self.push_descriptor::<M>(desc))
    }

    /// Registers a [`BucketMessage`] type with `max_buckets` distinct keys.
    ///
    /// # Errors
    ///
    /// In addition to registration errors, returns
    /// [`MessagingError::BucketKeyOutOfRange`] if `max_buckets == 0`.
    pub fn register_bucket<M: BucketMessage>(
        &mut self,
        allocator: &mut ChannelAllocator,
        max_buckets: u32,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, MessagingError> {
        self.check_not_registered::<M>()?;
        if max_buckets == 0 {
            return Err(MessagingError::InvalidBucketConfig);
        }
        let mtid = MessageTypeID(self.descriptors.len() as u32);
        let channel_id = allocator
            .alloc()
            .map_err(|_| MessagingError::ChannelAllocationOverflow)?;
        let desc = MessageDescriptor {
            type_id: TypeId::of::<M>(),
            type_name: std::any::type_name::<M>(),
            item_size: std::mem::size_of::<M>(),
            item_align: std::mem::align_of::<M>(),
            specialisation: Specialisation::Bucket { max_buckets },
            capacity,
            gpu_safe: M::GPU_SAFE,
            #[cfg(feature = "messaging_gpu")]
            gpu_resource_id: None,
            channel_id,
            message_type_id: mtid,
            erased_fns: ErasedFns {
                bucket_key: Some(|ptr| {
                    // SAFETY: ptr points to a valid M; called only during
                    // scatter pass after items are fully written.
                    unsafe { (*(ptr as *const M)).bucket_key() }
                }),
                position: None,
                recipient: None,
            },
        };
        Ok(self.push_descriptor::<M>(desc))
    }

    /// Registers a [`SpatialMessage`] type.
    ///
    /// # Errors
    ///
    /// Returns [`MessagingError::InvalidSpatialConfig`] if `cell_size <= 0`,
    /// `width <= 0`, or `height <= 0`.
    pub fn register_spatial<M: SpatialMessage>(
        &mut self,
        allocator: &mut ChannelAllocator,
        config: SpatialConfig,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, MessagingError> {
        self.check_not_registered::<M>()?;
        if config.cell_size <= 0.0 || config.width <= 0.0 || config.height <= 0.0 {
            return Err(MessagingError::InvalidSpatialConfig {
                cell_size: config.cell_size,
                width: config.width,
                height: config.height,
            });
        }
        let mtid = MessageTypeID(self.descriptors.len() as u32);
        let channel_id = allocator
            .alloc()
            .map_err(|_| MessagingError::ChannelAllocationOverflow)?;
        let desc = MessageDescriptor {
            type_id: TypeId::of::<M>(),
            type_name: std::any::type_name::<M>(),
            item_size: std::mem::size_of::<M>(),
            item_align: std::mem::align_of::<M>(),
            specialisation: Specialisation::Spatial(config),
            capacity,
            gpu_safe: M::GPU_SAFE,
            #[cfg(feature = "messaging_gpu")]
            gpu_resource_id: None,
            channel_id,
            message_type_id: mtid,
            erased_fns: ErasedFns {
                bucket_key: None,
                position: Some(|ptr| {
                    // SAFETY: ptr points to a valid M.
                    unsafe { (*(ptr as *const M)).position() }
                }),
                recipient: None,
            },
        };
        Ok(self.push_descriptor::<M>(desc))
    }

    /// Registers a [`TargetedMessage`] type.
    pub fn register_targeted<M: TargetedMessage>(
        &mut self,
        allocator: &mut ChannelAllocator,
        capacity: Capacity,
    ) -> Result<MessageHandle<M>, MessagingError> {
        self.check_not_registered::<M>()?;
        let mtid = MessageTypeID(self.descriptors.len() as u32);
        let channel_id = allocator
            .alloc()
            .map_err(|_| MessagingError::ChannelAllocationOverflow)?;
        let desc = MessageDescriptor {
            type_id: TypeId::of::<M>(),
            type_name: std::any::type_name::<M>(),
            item_size: std::mem::size_of::<M>(),
            item_align: std::mem::align_of::<M>(),
            specialisation: Specialisation::Targeted,
            capacity,
            gpu_safe: M::GPU_SAFE,
            #[cfg(feature = "messaging_gpu")]
            gpu_resource_id: None,
            channel_id,
            message_type_id: mtid,
            erased_fns: ErasedFns {
                bucket_key: None,
                position: None,
                recipient: Some(|ptr| {
                    // SAFETY: ptr points to a valid M.
                    unsafe { (*(ptr as *const M)).recipient() }
                }),
            },
        };
        Ok(self.push_descriptor::<M>(desc))
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /// Freezes the registry.  After this point, `register_*` calls return
    /// [`MessagingError::RegistryFrozen`].
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Marks a descriptor as GPU-backed and records its world resource ID.
    #[cfg(feature = "messaging_gpu")]
    pub(crate) fn attach_gpu_resource<M: Message>(
        &mut self,
        handle: MessageHandle<M>,
        resource_id: GPUResourceID,
    ) {
        let desc = &mut self.descriptors[handle.message_type_id.index()];
        desc.gpu_safe = true;
        desc.gpu_resource_id = Some(resource_id);
    }

    /// Returns `true` if the registry has been frozen.
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    /// Returns all registered descriptors in registration order.
    #[inline]
    pub fn descriptors(&self) -> &[MessageDescriptor] {
        &self.descriptors
    }

    /// Returns the descriptor for message type `M`, or `None` if not
    /// registered.
    #[inline]
    pub fn descriptor_of<M: Message>(&self) -> Option<&MessageDescriptor> {
        let mtid = self.by_type.get(&TypeId::of::<M>())?;
        self.descriptors.get(mtid.index())
    }

    /// Returns the typed handle for message type `M`, or `None` if not registered.
    #[inline]
    pub fn handle_of<M: Message>(&self) -> Option<MessageHandle<M>> {
        let mtid = self.by_type.get(&TypeId::of::<M>()).copied()?;
        let desc = self.descriptors.get(mtid.index())?;
        Some(MessageHandle::new(mtid, self.registry_id, desc.channel_id))
    }

    #[inline]
    pub(crate) fn registry_id(&self) -> u64 {
        self.registry_id
    }

    /// Returns the descriptor for the given [`MessageTypeID`].
    ///
    /// # Panics
    ///
    /// Panics if `mtid` was not produced by this registry (index
    /// out of bounds).
    #[inline]
    pub(crate) fn descriptor(&self, mtid: MessageTypeID) -> &MessageDescriptor {
        &self.descriptors[mtid.index()]
    }
}

impl Default for MessageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct TestBucket(u32);
    impl Message for TestBucket {}
    impl BucketMessage for TestBucket {
        fn bucket_key(&self) -> u32 {
            self.0
        }
    }

    #[test]
    fn bucket_zero_config_is_rejected() {
        let mut alloc = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let err = registry
            .register_bucket::<TestBucket>(&mut alloc, 0, Capacity::unbounded(1))
            .unwrap_err();
        assert!(matches!(err, MessagingError::InvalidBucketConfig));
    }

    #[test]
    fn typed_handle_has_channel_id() {
        let mut alloc = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let handle = registry
            .register_bucket::<TestBucket>(&mut alloc, 4, Capacity::unbounded(1))
            .unwrap();
        assert_eq!(handle.channel_id(), 0);
        assert!(registry.handle_of::<TestBucket>().is_some());
    }
}
