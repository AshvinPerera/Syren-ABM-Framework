//! ECS System Abstractions
//!
//! This module defines the core *system execution model* used by the engine.
//!
//! A **system** is a unit of logic that operates over the ECS world. Systems:
//! - declare which components they read and write,
//! - are scheduled based on access conflicts,
//! - may be executed sequentially or in parallel,
//! - operate through a controlled [`ECSReference`] rather than direct world access.
//!
//! ## Design Goals
//!
//! The system abstraction is designed to:
//!
//! - **Enable parallel scheduling**
//!   by statically declaring component access (`read` / `write`) via [`AccessSets`],
//!   and channel ordering (`produces` / `consumes`) via [`ChannelSet`].
//!
//! - **Decouple logic from storage**
//!   so systems operate on *views* of the world rather than concrete data layouts.
//!
//! - **Support lightweight system definitions**
//!   through function-backed systems (`FnSystem`) without requiring boilerplate
//!   types for every system.
//!
//! ## Scheduling Model
//!
//! Systems are scheduled by the engine using their declared access sets:
//!
//! - Systems with *non-conflicting* access may run in parallel.
//! - Systems with conflicting writes are serialized relative to one another.
//! - Systems where one produces a channel the other consumes are placed in
//!   strictly ordered stages (producer first).
//! - Ordering is stabilised using system IDs.
//!
//! The scheduler is free to group systems into execution stages based on this
//! information.
//!
//! ## System Trait
//!
//! The [`System`] trait defines the minimal interface required for execution:
//!
//! - [`System::id`] provides a stable identifier.
//! - [`System::access`] declares component read/write requirements.
//! - [`System::run`] executes the system logic.
//!
//! All systems must be `Send + Sync` to allow execution on worker threads.
//!
//! ## Function-backed Systems
//!
//! [`FnSystem`] provides a convenient way to define systems using closures or
//! functions. This is the preferred mechanism for most gameplay and simulation
//! logic, as it avoids unnecessary type definitions while remaining fully
//! schedulable and parallel-safe.
//!
//! ## Thread Safety
//!
//! Systems do **not** receive direct mutable access to the world. Instead, they
//! operate through [`ECSReference`], which provides controlled entry points into
//! ECS execution phases.
//!
//! Correctness is enforced at runtime via borrow tracking and execution-phase discipline;
//! the scheduler optimises parallelism.
//!
//! ## Intended Usage
//!
//! This module is intended to be used in conjunction with:
//! - the scheduler (`scheduler` module),
//! - query construction (`query` module),
//! - and deferred command processing (`manager` module).
//!
//! Together, these components form the execution layer of the ECS.

use crate::engine::component::Signature;
use crate::engine::error::{ECSError, ECSResult, ExecutionError, InvalidAccessReason};
use crate::engine::manager::ECSReference;
#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;
use crate::engine::types::{ChannelID, ComponentID, SystemID};

use smallvec::SmallVec;

/// Bitset of [`ChannelID`]s for non-component scheduling dependencies.
///
/// Compact, deterministic, and cheap to intersect. Storage grows in 64-bit
/// words as needed; empty sets allocate nothing inline beyond the `SmallVec`
/// header. Two words inline covers channels 0-127, which is sufficient for
/// most simulations without heap allocation.
///
/// `ChannelSet` is the channel analogue of [`Signature`] for components.
/// It is attached to [`AccessSets::produces`] and [`AccessSets::consumes`]
/// to express ordering dependencies between systems beyond what component
/// read/write conflicts capture.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChannelSet {
    bits: SmallVec<[u64; 2]>,
}

impl ChannelSet {
    /// Creates a new, empty channel set.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `id` as present in this set.
    ///
    /// Grows the backing storage automatically if `id` exceeds 127.
    #[inline]
    pub fn insert(&mut self, id: ChannelID) {
        let word = (id as usize) / 64;
        let bit = (id as usize) % 64;
        if self.bits.len() <= word {
            self.bits.resize(word + 1, 0);
        }
        self.bits[word] |= 1u64 << bit;
    }

    /// Returns `true` if `id` is present in this set.
    #[inline]
    pub fn contains(&self, id: ChannelID) -> bool {
        let word = (id as usize) / 64;
        let bit = (id as usize) % 64;
        self.bits
            .get(word)
            .map_or(false, |w| (w & (1u64 << bit)) != 0)
    }

    /// Returns `true` if this set is empty (no channels present).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Returns `true` if this set and `other` share at least one channel ID.
    #[inline]
    pub fn intersects(&self, other: &ChannelSet) -> bool {
        let n = self.bits.len().min(other.bits.len());
        for i in 0..n {
            if (self.bits[i] & other.bits[i]) != 0 {
                return true;
            }
        }
        false
    }

    /// Merges all channel IDs from `other` into `self` (bitwise OR in place).
    #[inline]
    pub fn or_in_place(&mut self, other: &ChannelSet) {
        if other.bits.len() > self.bits.len() {
            self.bits.resize(other.bits.len(), 0);
        }
        for (a, &b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= b;
        }
    }

    /// Iterates all channel IDs present in this set, in ascending order.
    ///
    /// Channel IDs are `u32`; the `w * 64 + b` combination is guaranteed to
    /// fit in `u32` as long as word index `w < (u32::MAX / 64)` ~= 67M words
    /// ~= 4B channels. In practice `w` is bounded by the allocator's issued
    /// count, so overflow is impossible under any realistic simulation. The
    /// `debug_assert!` documents and enforces this invariant during testing
    /// without costing anything in release.
    pub fn iter(&self) -> impl Iterator<Item = ChannelID> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, &word)| {
            debug_assert!(
                w <= (u32::MAX as usize) / 64,
                "ChannelSet word index exceeds u32 channel-id space"
            );
            let base = (w as u32).saturating_mul(64);
            (0u32..64).filter_map(move |b| {
                if (word & (1u64 << b)) != 0 {
                    Some(base + b)
                } else {
                    None
                }
            })
        })
    }
}

/// Directional ordering constraint derived from channel produces/consumes.
///
/// Returned by [`AccessSets::channel_ordering`] when two access sets have a
/// channel dependency. The scheduler uses this to ensure producers always run
/// in an earlier stage than their consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelOrder {
    /// `self` must be placed in an earlier stage than `other`.
    SelfBeforeOther,
    /// `other` must be placed in an earlier stage than `self`.
    OtherBeforeSelf,
}

/// Declares the component access set of a system.
#[derive(Clone, Debug, Default)]
pub struct AccessSets {
    /// Components read by the system.
    pub read: Signature,
    /// Components written by the system.
    pub write: Signature,

    /// Channels whose data this system produces this tick.
    ///
    /// Common sources: messages emitted by the system, environment keys written.
    /// A system that produces channel C must run in a strictly earlier stage than
    /// any system that consumes C.
    pub produces: ChannelSet,

    /// Channels whose data this system consumes this tick.
    ///
    /// Common sources: messages read, environment keys read via a keyed handle.
    /// A consumer must always run in a strictly later stage than all producers
    /// of the same channel.
    pub consumes: ChannelSet,
}

impl AccessSets {
    /// Returns `true` if this access set conflicts with `other`, meaning the two
    /// systems cannot be placed in the same execution stage.
    ///
    /// Conflict sources (any one is sufficient):
    ///
    /// 1. **Component write/write**: both systems write the same component.
    /// 2. **Component write/read**: one writes a component the other reads.
    /// 3. **Channel ordering**: one system produces a channel the other consumes -
    ///    they must be in strictly ordered stages, so same-stage placement is
    ///    forbidden.
    ///
    /// Note: two systems that both *produce* the same channel are compatible
    /// (deterministic thread-local merge), so producer-producer is not a conflict.
    #[inline]
    pub fn conflicts_with(&self, other: &AccessSets) -> bool {
        if self.component_conflict(other) {
            return true;
        }
        // Channel ordering: if either direction produces what the other consumes,
        // they must be in different stages (producer first).
        if self.produces.intersects(&other.consumes) {
            return true;
        }
        if other.produces.intersects(&self.consumes) {
            return true;
        }
        false
    }

    /// Component-only conflict check (W intersection W, W intersection R, R intersection W).
    ///
    /// Extracted so `conflicts_with` can chain channel checks without
    /// duplicating the bitset loop.
    #[inline]
    pub(crate) fn component_conflict(&self, other: &AccessSets) -> bool {
        for ((a_w, a_r), (b_w, b_r)) in self
            .write
            .components
            .iter()
            .zip(self.read.components.iter())
            .zip(
                other
                    .write
                    .components
                    .iter()
                    .zip(other.read.components.iter()),
            )
        {
            if (a_w & b_w) != 0 {
                return true;
            } // W intersection W
            if (a_w & b_r) != 0 {
                return true;
            } // W intersection R
            if (a_r & b_w) != 0 {
                return true;
            } // R intersection W
        }
        false
    }

    /// Returns the directional ordering required between `self` and `other`
    /// based on channel dependencies, or `None` if there is no such dependency.
    ///
    /// Component conflicts are handled separately by the scheduler's packing
    /// logic and are not reflected here.
    pub fn channel_ordering(&self, other: &AccessSets) -> Option<ChannelOrder> {
        if self.produces.intersects(&other.consumes) {
            return Some(ChannelOrder::SelfBeforeOther);
        }
        if other.produces.intersects(&self.consumes) {
            return Some(ChannelOrder::OtherBeforeSelf);
        }
        None
    }

    /// Validates that this access set is internally consistent.
    ///
    /// A system's own access set must not alias itself. Two self-aliases
    /// would cause the scheduler to silently mis-pack the system:
    ///
    /// 1. **Component read/write self-alias** - the same component appearing
    ///    in both `read` and `write`. Declaring this weakens the write lock
    ///    contract at iteration time and conflates conflict semantics.
    /// 2. **Channel produce/consume self-alias** - the same channel in both
    ///    `produces` and `consumes`. The stage packer would place the system
    ///    in a stage where it both produces and consumes the channel, so the
    ///    consume would observe the channel before its own produce had been
    ///    finalised at a boundary. This is always a bug in the system's
    ///    declaration.
    ///
    /// Called by [`Scheduler::add_boxed`](crate::engine::scheduler::Scheduler::add_boxed)
    /// at registration time so that malformed systems are rejected early
    /// rather than producing silently-wrong schedules.
    pub fn validate(&self) -> ECSResult<()> {
        // Component read/write self-alias.
        for (i, (rw, ww)) in self
            .read
            .components
            .iter()
            .zip(self.write.components.iter())
            .enumerate()
        {
            let overlap = rw & ww;
            if overlap != 0 {
                let bit = overlap.trailing_zeros();
                let cid: ComponentID = ((i as u32) * 64 + bit) as ComponentID;
                return Err(ECSError::Execute(ExecutionError::InvalidQueryAccess {
                    component_id: cid,
                    reason: InvalidAccessReason::ReadAndWrite,
                }));
            }
        }

        // Channel produces/consumes self-alias.
        if self.produces.intersects(&self.consumes) {
            let offender = self
                .produces
                .iter()
                .find(|ch| self.consumes.contains(*ch))
                .unwrap_or(0);
            return Err(ECSError::Execute(ExecutionError::SelfChannelAlias {
                channel_id: offender,
            }));
        }

        Ok(())
    }
}

/// Execution backend for a system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemBackend {
    /// Standard Rust / Rayon execution on CPU.
    CPU,
    /// GPU dispatch.
    GPU,
}

/// GPU capability trait (feature-gated).
/// A GPU system is still a `System`, but additionally provides WGSL.

#[cfg(feature = "gpu")]
pub trait GpuSystem {
    /// WGSL source
    fn shader(&self) -> &'static str;

    /// Entry point name (default "main")
    fn entry_point(&self) -> &'static str {
        "main"
    }

    /// Workgroup size (default 256)
    fn workgroup_size(&self) -> u32 {
        256
    }

    /// GPU resources read or used by this kernel.
    fn uses_resources(&self) -> &[GPUResourceID] {
        &[]
    }

    /// GPU resources that this kernel may write.
    fn writes_resources(&self) -> &[GPUResourceID] {
        &[]
    }
}

/// A unit of executable logic operating on the ECS world.
///
/// A `System` represents a scheduled computation that:
/// - declares which components it reads and writes,
/// - can be ordered and parallelized based on access conflicts,
/// - is executed with a shared reference to the ECS world.
///
/// Systems must be `Send + Sync` so they can be scheduled and executed
/// in parallel across threads.

pub trait System: Send + Sync {
    /// Human-readable name (used for debugging/profiling).
    ///
    /// The default implementation returns `std::any::type_name_of_val(self)`,
    /// which is `&'static str` and coerces to the `&str` return type. Concrete
    /// implementations may return a borrow from `&self` to support dynamic
    /// names without leaking memory.
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name_of_val(self)
    }

    /// Returns the unique identifier of this system.
    fn id(&self) -> SystemID;

    /// Returns a reference to the component access sets required by this system.
    fn access(&self) -> &AccessSets;

    /// Returns which backend this system should run on.
    /// Defaults to [`SystemBackend::CPU`].
    #[inline]
    fn backend(&self) -> SystemBackend {
        SystemBackend::CPU
    }

    /// Executes the system logic against the ECS world.
    fn run(&self, world: ECSReference<'_>) -> ECSResult<()>;

    /// GPU capability hook.
    /// A GPU system should override this to return `Some(self)` (as `&dyn GpuSystem`)
    /// and also return `SystemBackend::GPU` from `backend()`.
    #[cfg(feature = "gpu")]
    #[inline]
    fn gpu(&self) -> Option<&dyn GpuSystem> {
        None
    }
}

/// A concrete [`System`] backed by a function or closure.
///
/// `FnSystem` allows systems to be defined inline using a function or
/// closure, without requiring a custom system type.
///
/// The function must return `ECSResult<()>` so that
/// execution failures can be propagated through the scheduler.
pub struct FnSystem<F>
where
    F: Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static,
{
    id: SystemID,
    name: &'static str,
    access: AccessSets,
    f: F,
}

impl<F> FnSystem<F>
where
    F: Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static,
{
    /// Creates a new function-backed system.
    ///
    /// # Parameters
    /// - `id`: Unique identifier for the system.
    /// - `name`: Human-readable name, useful for debugging and profiling.
    /// - `access`: Declared component access used for scheduling.
    /// - `f`: The function or closure executed when the system runs.
    pub fn new(id: SystemID, name: &'static str, access: AccessSets, f: F) -> Self {
        Self {
            id,
            name,
            access,
            f,
        }
    }
}

impl<F> System for FnSystem<F>
where
    F: Fn(ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn id(&self) -> SystemID {
        self.id
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, world: ECSReference<'_>) -> ECSResult<()> {
        (self.f)(world)
    }
}
