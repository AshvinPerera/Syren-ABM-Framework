//! GPU-facing message resources and binding metadata.
//!
//! Message registration defines the scheduler channel, payload layout, capacity,
//! and specialisation. GPU registration adds one world-owned resource per
//! message type so shaders can emit into raw buffers and consume finalised
//! buffers through stable group(1) bindings.

use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};

use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::types::{ChannelID, GPUResourceID};
use crate::gpu::{
    BoundaryGpuDispatch, BoundaryKernelDesc, GPUBindingDesc, GPUContext, GPUResource,
};

use super::error::MessagingError;
use super::message::{Message, MessageHandle};
use super::registry::{Capacity, SpatialConfig, Specialisation};

/// Marker for messages whose memory layout is valid in GPU storage buffers.
///
/// # Safety
///
/// Implementors must use a stable representation such as `#[repr(C)]` and
/// ensure every field has the same layout in WGSL as it does in Rust. Prefer
/// deriving `bytemuck::Pod` and `bytemuck::Zeroable` on simple scalar/vector
/// payload structs.
pub unsafe trait GpuMessage: Message + Pod + Zeroable {}

/// GPU emission strategy for a message resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuEmissionMode {
    /// Deterministic fixed-slot emission. Shaders write payloads to stable
    /// slots and mark `fixed_valid_flags[slot] = 1`.
    FixedSlots,
    /// Atomic append emission. This is opt-in because the relative order of
    /// concurrent atomic appends is not a deterministic economic ordering.
    AppendUnordered,
}

/// GPU finalise routing policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFinalisePolicy {
    /// Choose CPU or GPU finalise from producer/consumer backends and message
    /// volume. CPU-only small channels stay on the CPU.
    Adaptive {
        /// Minimum merged message count before CPU-produced GPU-consumed
        /// channels prefer GPU finalise.
        min_gpu_messages: usize,
    },
    /// Always use GPU finalise when the message layout supports it.
    AlwaysGpu,
    /// Always use CPU finalise and upload the final/index buffers for GPU
    /// consumers.
    AlwaysCpu,
}

/// Options used when registering a GPU-backed message type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMessageOptions {
    /// How shaders emit raw GPU messages.
    pub emission_mode: GpuEmissionMode,
    /// How boundary finalise chooses CPU versus GPU work.
    pub finalise_policy: GpuFinalisePolicy,
    /// Number of stable slots available for fixed-slot emission.
    pub fixed_slot_count: u32,
    /// Bounded number of messages one stable slot may emit. The initial
    /// implementation supports `1`; larger values are reserved for compatible
    /// future kernels.
    pub messages_per_slot: u32,
}

impl GpuMessageOptions {
    /// Deterministic defaults for economic agent-based models.
    pub fn deterministic(capacity: usize) -> Self {
        Self {
            emission_mode: GpuEmissionMode::FixedSlots,
            finalise_policy: GpuFinalisePolicy::Adaptive {
                min_gpu_messages: 1024,
            },
            fixed_slot_count: capacity as u32,
            messages_per_slot: 1,
        }
    }
}

/// GPU key metadata for bucketed messages.
///
/// # Safety
///
/// `BUCKET_KEY_WORD_OFFSET` must point to a `u32` field within the `#[repr(C)]`
/// payload layout used by WGSL.
pub unsafe trait GpuBucketMessage: GpuMessage {
    /// Offset, in 32-bit words from the start of the payload, of the bucket key.
    const BUCKET_KEY_WORD_OFFSET: u32;
}

/// GPU key metadata for spatial messages.
///
/// # Safety
///
/// The offsets must point to `f32` fields within the `#[repr(C)]` payload
/// layout used by WGSL.
pub unsafe trait GpuSpatialMessage: GpuMessage {
    /// Offset, in 32-bit words, of the x coordinate.
    const X_WORD_OFFSET: u32;
    /// Offset, in 32-bit words, of the y coordinate.
    const Y_WORD_OFFSET: u32;
}

/// GPU layout for the packed targeted recipient key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuTargetKeyLayout {
    /// A contiguous `u64` key beginning at this 32-bit word offset.
    U64Word {
        /// Offset of the low 32 bits; the high 32 bits follow immediately.
        word_offset: u32,
    },
    /// Separate low/high `u32` key words.
    U32Pair {
        /// Offset of the low 32 bits.
        low_word_offset: u32,
        /// Offset of the high 32 bits.
        high_word_offset: u32,
    },
}

/// GPU key metadata for targeted messages.
///
/// # Safety
///
/// The key layout must expose the full packed [`Entity`](crate::Entity) value,
/// including shard and version bits.
pub unsafe trait GpuTargetedMessage: GpuMessage {
    /// Layout of the full packed recipient key.
    const TARGET_KEY_LAYOUT: GpuTargetKeyLayout;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuKeyMetadata {
    None,
    Bucket {
        key_word_offset: u32,
    },
    Spatial {
        x_word_offset: u32,
        y_word_offset: u32,
    },
    Targeted {
        layout: GpuTargetKeyLayout,
    },
}

/// Resource IDs owned by one GPU message type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMessageResourceIds {
    /// World-owned resource that stores all buffers for this message type.
    pub resource: GPUResourceID,
}

/// Stable group(1) binding offsets for one GPU message resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMessageBindings {
    /// CPU-upload raw stream binding.
    pub raw_cpu_stream: u32,
    /// GPU append and fixed-slot raw stream binding.
    pub raw_gpu_stream: u32,
    /// Per-slot validity flags for fixed-slot emission.
    pub fixed_valid_flags: u32,
    /// Finalised sorted/linear message buffer binding.
    pub final_messages: u32,
    /// Index metadata binding.
    pub index_metadata: u32,
    /// Control buffer binding.
    ///
    /// The control buffer stores descriptor fields, overflow state, and the
    /// append counter. Shader helpers below document the word offsets.
    pub control: u32,
    /// Descriptor/params binding. Alias for [`GpuMessageBindings::control`].
    pub params: u32,
    /// Append counter binding. Alias for [`GpuMessageBindings::control`].
    pub append_counter: u32,
    /// Overflow status binding. Alias for [`GpuMessageBindings::control`].
    pub overflow_status: u32,
}

impl GpuMessageBindings {
    /// Bindings local to one resource. The dispatch layer offsets these by the
    /// resource's position in `GpuSystem::uses_resources()`.
    pub const LOCAL: Self = Self {
        raw_cpu_stream: 0,
        raw_gpu_stream: 1,
        fixed_valid_flags: 2,
        final_messages: 3,
        index_metadata: 4,
        control: 5,
        params: 5,
        append_counter: 5,
        overflow_status: 5,
    };
}

/// Typed handle returned by `ModelBuilder::register_gpu_*_message`.
#[derive(Debug)]
pub struct GpuMessageHandle<M: GpuMessage> {
    cpu: MessageHandle<M>,
    channel_id: ChannelID,
    specialisation: Specialisation,
    resource_ids: GpuMessageResourceIds,
    bindings: GpuMessageBindings,
    _marker: PhantomData<fn() -> M>,
}

impl<M: GpuMessage> GpuMessageHandle<M> {
    pub(crate) fn new(
        cpu: MessageHandle<M>,
        specialisation: Specialisation,
        resource_id: GPUResourceID,
    ) -> Self {
        Self {
            cpu,
            channel_id: cpu.channel_id(),
            specialisation,
            resource_ids: GpuMessageResourceIds {
                resource: resource_id,
            },
            bindings: GpuMessageBindings::LOCAL,
            _marker: PhantomData,
        }
    }

    /// CPU message handle for existing CPU APIs.
    #[inline]
    pub fn cpu(self) -> MessageHandle<M> {
        self.cpu
    }

    /// Scheduler channel ID.
    #[inline]
    pub fn channel_id(self) -> ChannelID {
        self.channel_id
    }

    /// Message specialisation metadata.
    #[inline]
    pub fn specialisation(&self) -> Specialisation {
        self.specialisation
    }

    /// GPU resource IDs to declare in `uses_resources()`/`writes_resources()`.
    #[inline]
    pub fn resource_ids(&self) -> GpuMessageResourceIds {
        self.resource_ids
    }

    /// Binding offsets within this message resource.
    #[inline]
    pub fn bindings(&self) -> GpuMessageBindings {
        self.bindings
    }
}

impl<M: GpuMessage> Clone for GpuMessageHandle<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: GpuMessage> Copy for GpuMessageHandle<M> {}

/// WGSL helpers for append emission.
pub const WGSL_APPEND_HELPERS: &str = r#"
struct SyrenGpuMessageControl {
    item_size: u32,
    capacity: u32,
    final_count: u32,
    specialisation: u32,
    overflow: atomic<u32>,
    append_counter: atomic<u32>,
    fixed_slot_count: u32,
    _reserved: u32,
};

fn syren_append_slot(control: ptr<storage, SyrenGpuMessageControl, read_write>) -> u32 {
    let slot = atomicAdd(&(*control).append_counter, 1u);
    let capacity = (*control).capacity;
    if (slot >= capacity) {
        atomicStore(&(*control).overflow, 1u);
        return 0xffffffffu;
    }
    return slot;
}
"#;

/// WGSL helper notes for fixed-slot one-message-per-agent emission.
pub const WGSL_FIXED_SLOT_HELPERS: &str = r#"
// Fixed-slot emit convention:
// - group(0) params word 0 is entity_len.
// - group(0) params word 1 is archetype_base.
// - agent_slot = archetype_base + local_row.
// - raw_gpu_stream[agent_slot] contains the payload.
// - fixed_valid_flags[agent_slot] is set to 1u for valid payloads and 0u otherwise.
// - control.overflow is set to 1u if the shader computes a slot >= control.capacity.
"#;

/// WGSL helper notes for indexed final-message reads.
pub const WGSL_INDEXED_READ_HELPERS: &str = r#"
// Final read convention:
// - final_messages is a packed array<M>.
// - index_metadata is u32 prefix data for bucket/spatial, and packed
//   (entity_low, entity_high, start, end) rows for targeted messages.
// - control.final_count is the number of valid entries in final_messages.
"#;

const WGSL_FINALISE_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read> raw_cpu: array<u32>;
@group(0) @binding(1) var<storage, read> raw_gpu: array<u32>;
@group(0) @binding(2) var<storage, read> valid_flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> final_messages: array<u32>;
@group(0) @binding(4) var<storage, read_write> index_metadata: array<u32>;
@group(0) @binding(5) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> scratch: array<u32>;

fn cw(i: u32) -> u32 {
    return atomicLoad(&control[i]);
}

fn cstore(i: u32, v: u32) {
    atomicStore(&control[i], v);
}

fn copy_cpu(src_item: u32, dst_item: u32, item_words: u32) {
    var w = 0u;
    loop {
        if (w >= item_words) { break; }
        final_messages[dst_item * item_words + w] = raw_cpu[src_item * item_words + w];
        w = w + 1u;
    }
}

fn copy_gpu(src_item: u32, dst_item: u32, item_words: u32) {
    var w = 0u;
    loop {
        if (w >= item_words) { break; }
        final_messages[dst_item * item_words + w] = raw_gpu[src_item * item_words + w];
        w = w + 1u;
    }
}

fn copy_gpu_to_final(src_item: u32, dst_item: u32, item_words: u32) {
    copy_gpu(src_item, dst_item, item_words);
}

fn key_bucket_cpu(item: u32, item_words: u32) -> u32 {
    return raw_cpu[item * item_words + cw(13u)];
}

fn key_bucket_gpu(item: u32, item_words: u32) -> u32 {
    return raw_gpu[item * item_words + cw(13u)];
}

fn key_spatial_from_word(word: u32) -> f32 {
    return bitcast<f32>(word);
}

fn key_spatial_cpu(item: u32, item_words: u32) -> u32 {
    let cols = cw(12u);
    let rows = cw(13u);
    let cell_size = bitcast<f32>(cw(14u));
    let x = key_spatial_from_word(raw_cpu[item * item_words + cw(16u)]);
    let y = key_spatial_from_word(raw_cpu[item * item_words + cw(17u)]);
    let col = min(u32(max(x / cell_size, 0.0)), cols - 1u);
    let row = min(u32(max(y / cell_size, 0.0)), rows - 1u);
    return row * cols + col;
}

fn key_spatial_gpu(item: u32, item_words: u32) -> u32 {
    let cols = cw(12u);
    let rows = cw(13u);
    let cell_size = bitcast<f32>(cw(14u));
    let x = key_spatial_from_word(raw_gpu[item * item_words + cw(16u)]);
    let y = key_spatial_from_word(raw_gpu[item * item_words + cw(17u)]);
    let col = min(u32(max(x / cell_size, 0.0)), cols - 1u);
    let row = min(u32(max(y / cell_size, 0.0)), rows - 1u);
    return row * cols + col;
}

fn counting_key_cpu(item: u32, item_words: u32, spatial: bool) -> u32 {
    if (spatial) {
        return key_spatial_cpu(item, item_words);
    }
    return key_bucket_cpu(item, item_words);
}

fn counting_key_gpu(item: u32, item_words: u32, spatial: bool) -> u32 {
    if (spatial) {
        return key_spatial_gpu(item, item_words);
    }
    return key_bucket_gpu(item, item_words);
}

fn target_key_low_cpu(item: u32, item_words: u32) -> u32 {
    return raw_cpu[item * item_words + cw(13u)];
}

fn target_key_high_cpu(item: u32, item_words: u32) -> u32 {
    return raw_cpu[item * item_words + cw(14u)];
}

fn target_key_low_gpu(item: u32, item_words: u32) -> u32 {
    return raw_gpu[item * item_words + cw(13u)];
}

fn target_key_high_gpu(item: u32, item_words: u32) -> u32 {
    return raw_gpu[item * item_words + cw(14u)];
}

fn key_less(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> bool {
    return (a_high < b_high) || (a_high == b_high && a_low < b_low);
}

fn swap_final_items(a: u32, b: u32, item_words: u32) {
    var w = 0u;
    loop {
        if (w >= item_words) { break; }
        let ai = a * item_words + w;
        let bi = b * item_words + w;
        let t = final_messages[ai];
        final_messages[ai] = final_messages[bi];
        final_messages[bi] = t;
        w = w + 1u;
    }
}

fn gpu_count_fixed(capacity: u32) -> u32 {
    let fixed_slots = min(cw(6u), capacity);
    var count = 0u;
    var i = 0u;
    loop {
        if (i >= fixed_slots) { break; }
        if (valid_flags[i] != 0u) {
            count = count + 1u;
        }
        i = i + 1u;
    }
    return count;
}

fn gpu_count(capacity: u32) -> u32 {
    if (cw(7u) == 1u) {
        return cw(5u);
    }
    return gpu_count_fixed(capacity);
}

fn for_each_gpu_brute(dst_start: u32, item_words: u32, capacity: u32) -> u32 {
    var dst = dst_start;
    if (cw(7u) == 1u) {
        var i = 0u;
        let count = min(cw(5u), capacity);
        loop {
            if (i >= count) { break; }
            copy_gpu_to_final(i, dst, item_words);
            dst = dst + 1u;
            i = i + 1u;
        }
        return dst;
    }
    let fixed_slots = min(cw(6u), capacity);
    var slot = 0u;
    loop {
        if (slot >= fixed_slots) { break; }
        if (valid_flags[slot] != 0u) {
            copy_gpu_to_final(slot, dst, item_words);
            dst = dst + 1u;
        }
        slot = slot + 1u;
    }
    return dst;
}

fn finalise_brute(cpu_count: u32, item_words: u32, total: u32, capacity: u32) {
    var i = 0u;
    loop {
        if (i >= cpu_count) { break; }
        copy_cpu(i, i, item_words);
        i = i + 1u;
    }
    _ = for_each_gpu_brute(cpu_count, item_words, capacity);
    index_metadata[0] = 0u;
    index_metadata[1] = total;
    cstore(2u, total);
    cstore(10u, 2u);
}

fn finalise_counting(cpu_count: u32, item_words: u32, total: u32, capacity: u32, spatial: bool) {
    var bucket_count = cw(12u);
    if (spatial) {
        bucket_count = cw(15u);
    }
    var k = 0u;
    loop {
        if (k > bucket_count) { break; }
        index_metadata[k] = 0u;
        scratch[k] = 0u;
        k = k + 1u;
    }

    var i = 0u;
    loop {
        if (i >= cpu_count) { break; }
        let key = counting_key_cpu(i, item_words, spatial);
        if (key >= bucket_count) {
            cstore(4u, 1u);
            cstore(11u, 3u);
            return;
        }
        index_metadata[key + 1u] = index_metadata[key + 1u] + 1u;
        i = i + 1u;
    }

    if (cw(7u) == 1u) {
        i = 0u;
        let count = min(cw(5u), capacity);
        loop {
            if (i >= count) { break; }
            let key = counting_key_gpu(i, item_words, spatial);
            if (key >= bucket_count) {
                cstore(4u, 1u);
                cstore(11u, 3u);
                return;
            }
            index_metadata[key + 1u] = index_metadata[key + 1u] + 1u;
            i = i + 1u;
        }
    } else {
        let fixed_slots = min(cw(6u), capacity);
        var slot = 0u;
        loop {
            if (slot >= fixed_slots) { break; }
            if (valid_flags[slot] != 0u) {
                let key = counting_key_gpu(slot, item_words, spatial);
                if (key >= bucket_count) {
                    cstore(4u, 1u);
                    cstore(11u, 3u);
                    return;
                }
                index_metadata[key + 1u] = index_metadata[key + 1u] + 1u;
            }
            slot = slot + 1u;
        }
    }

    var running = 0u;
    k = 0u;
    loop {
        if (k >= bucket_count) { break; }
        let count = index_metadata[k + 1u];
        index_metadata[k] = running;
        scratch[k] = running;
        running = running + count;
        k = k + 1u;
    }
    index_metadata[bucket_count] = running;

    i = 0u;
    loop {
        if (i >= cpu_count) { break; }
        let key = counting_key_cpu(i, item_words, spatial);
        let dst = scratch[key];
        copy_cpu(i, dst, item_words);
        scratch[key] = dst + 1u;
        i = i + 1u;
    }

    if (cw(7u) == 1u) {
        i = 0u;
        let count = min(cw(5u), capacity);
        loop {
            if (i >= count) { break; }
            let key = counting_key_gpu(i, item_words, spatial);
            let dst = scratch[key];
            copy_gpu(i, dst, item_words);
            scratch[key] = dst + 1u;
            i = i + 1u;
        }
    } else {
        let fixed_slots = min(cw(6u), capacity);
        var slot = 0u;
        loop {
            if (slot >= fixed_slots) { break; }
            if (valid_flags[slot] != 0u) {
                let key = counting_key_gpu(slot, item_words, spatial);
                let dst = scratch[key];
                copy_gpu(slot, dst, item_words);
                scratch[key] = dst + 1u;
            }
            slot = slot + 1u;
        }
    }

    cstore(2u, total);
    cstore(10u, bucket_count + 1u);
}

fn finalise_targeted(cpu_count: u32, item_words: u32, total: u32, capacity: u32) {
    var dst = 0u;
    var i = 0u;
    loop {
        if (i >= cpu_count) { break; }
        scratch[dst] = target_key_low_cpu(i, item_words);
        scratch[capacity + dst] = target_key_high_cpu(i, item_words);
        copy_cpu(i, dst, item_words);
        dst = dst + 1u;
        i = i + 1u;
    }

    if (cw(7u) == 1u) {
        i = 0u;
        let count = min(cw(5u), capacity);
        loop {
            if (i >= count) { break; }
            scratch[dst] = target_key_low_gpu(i, item_words);
            scratch[capacity + dst] = target_key_high_gpu(i, item_words);
            copy_gpu(i, dst, item_words);
            dst = dst + 1u;
            i = i + 1u;
        }
    } else {
        let fixed_slots = min(cw(6u), capacity);
        var slot = 0u;
        loop {
            if (slot >= fixed_slots) { break; }
            if (valid_flags[slot] != 0u) {
                scratch[dst] = target_key_low_gpu(slot, item_words);
                scratch[capacity + dst] = target_key_high_gpu(slot, item_words);
                copy_gpu(slot, dst, item_words);
                dst = dst + 1u;
            }
            slot = slot + 1u;
        }
    }

    i = 1u;
    loop {
        if (i >= total) { break; }
        var j = i;
        loop {
            if (j == 0u) { break; }
            let cur_low = scratch[j];
            let cur_high = scratch[capacity + j];
            let prev_low = scratch[j - 1u];
            let prev_high = scratch[capacity + j - 1u];
            if (!key_less(cur_low, cur_high, prev_low, prev_high)) { break; }
            scratch[j] = prev_low;
            scratch[capacity + j] = prev_high;
            scratch[j - 1u] = cur_low;
            scratch[capacity + j - 1u] = cur_high;
            swap_final_items(j, j - 1u, item_words);
            j = j - 1u;
        }
        i = i + 1u;
    }

    var row = 0u;
    if (total > 0u) {
        var start = 0u;
        var low = scratch[0];
        var high = scratch[capacity];
        i = 1u;
        loop {
            if (i > total) { break; }
            if (i == total || scratch[i] != low || scratch[capacity + i] != high) {
                let base = row * 4u;
                index_metadata[base] = low;
                index_metadata[base + 1u] = high;
                index_metadata[base + 2u] = start;
                index_metadata[base + 3u] = i;
                row = row + 1u;
                if (i < total) {
                    start = i;
                    low = scratch[i];
                    high = scratch[capacity + i];
                }
            }
            i = i + 1u;
        }
    }

    cstore(2u, total);
    cstore(10u, row * 4u);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let item_words = cw(0u) / 4u;
    let capacity = cw(1u);
    let cpu_count = cw(8u);
    var gcount = gpu_count(capacity);
    if (cw(4u) != 0u || gcount > capacity) {
        cstore(4u, 1u);
        cstore(11u, 1u);
        return;
    }
    let total = cpu_count + gcount;
    cstore(9u, gcount);
    if (total > capacity) {
        cstore(4u, 1u);
        cstore(11u, 2u);
        return;
    }

    let spec = cw(3u);
    if (spec == 0u) {
        finalise_brute(cpu_count, item_words, total, capacity);
    } else if (spec == 1u) {
        finalise_counting(cpu_count, item_words, total, capacity, false);
    } else if (spec == 2u) {
        finalise_counting(cpu_count, item_words, total, capacity, true);
    } else {
        finalise_targeted(cpu_count, item_words, total, capacity);
    }
}
"#;

const BINDINGS: [GPUBindingDesc; 6] = [
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: false },
    GPUBindingDesc { read_only: false },
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: false },
];

const FINALISE_BINDINGS: [GPUBindingDesc; 7] = [
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: true },
    GPUBindingDesc { read_only: false },
    GPUBindingDesc { read_only: false },
    GPUBindingDesc { read_only: false },
    GPUBindingDesc { read_only: false },
];

const CONTROL_WORDS: usize = 24;
const CONTROL_BYTES: usize = CONTROL_WORDS * std::mem::size_of::<u32>();
const CONTROL_ITEM_SIZE: usize = 0;
const CONTROL_CAPACITY: usize = 1;
const CONTROL_FINAL_COUNT: usize = 2;
const CONTROL_SPECIALISATION: usize = 3;
const CONTROL_OVERFLOW: usize = 4;
const CONTROL_APPEND_COUNTER: usize = 5;
const CONTROL_FIXED_SLOT_COUNT: usize = 6;
const CONTROL_EMISSION_MODE: usize = 7;
const CONTROL_CPU_RAW_COUNT: usize = 8;
const CONTROL_GPU_VALID_COUNT: usize = 9;
const CONTROL_INDEX_WORD_COUNT: usize = 10;
const CONTROL_ERROR_CODE: usize = 11;
const CONTROL_KEY0: usize = 12;
const CONTROL_KEY1: usize = 13;
const CONTROL_KEY2: usize = 14;
const CONTROL_KEY3: usize = 15;
const CONTROL_KEY4: usize = 16;
const CONTROL_KEY5: usize = 17;

pub(crate) struct GpuFinaliseOutput {
    pub(crate) final_bytes: Option<Vec<u8>>,
    pub(crate) index_words: Option<Vec<u32>>,
}

#[derive(Debug)]
struct GpuBuffers {
    raw_cpu: wgpu::Buffer,
    raw_gpu: wgpu::Buffer,
    fixed_valid_flags: wgpu::Buffer,
    final_messages: wgpu::Buffer,
    index_metadata: wgpu::Buffer,
    control: wgpu::Buffer,
    scratch: wgpu::Buffer,
    raw_bytes: usize,
    index_bytes: usize,
    valid_bytes: usize,
    scratch_bytes: usize,
}

/// World-owned GPU resource backing one registered message type.
#[derive(Debug)]
pub struct GpuMessageResource {
    name: String,
    item_size: usize,
    capacity: usize,
    specialisation: Specialisation,
    options: GpuMessageOptions,
    key_metadata: GpuKeyMetadata,
    raw_cpu_bytes: Vec<u8>,
    final_bytes: Vec<u8>,
    index_words: Vec<u32>,
    control_words: [u32; CONTROL_WORDS],
    downloaded_gpu_raw: Vec<u8>,
    buffers: Option<GpuBuffers>,
}

impl GpuMessageResource {
    /// Creates a resource shell for a registered GPU message.
    pub(crate) fn new(
        type_name: &'static str,
        item_size: usize,
        capacity: Capacity,
        specialisation: Specialisation,
        options: GpuMessageOptions,
        key_metadata: GpuKeyMetadata,
    ) -> Result<Self, MessagingError> {
        let capacity = capacity.max.unwrap_or(capacity.initial).max(1);
        let mut options = options;
        if options.fixed_slot_count == 0 {
            options.fixed_slot_count = capacity as u32;
        }
        if options.messages_per_slot != 1 {
            return Err(MessagingError::InvalidGpuMessageLayout {
                type_name,
                reason: "messages_per_slot values other than 1 are not supported yet".to_string(),
            });
        }
        validate_gpu_layout(type_name, item_size, specialisation, key_metadata)?;

        let mut control_words = [0u32; CONTROL_WORDS];
        control_words[CONTROL_ITEM_SIZE] = item_size as u32;
        control_words[CONTROL_CAPACITY] = capacity as u32;
        control_words[CONTROL_SPECIALISATION] = specialisation_code(specialisation);
        control_words[CONTROL_FIXED_SLOT_COUNT] = options.fixed_slot_count.min(capacity as u32);
        control_words[CONTROL_EMISSION_MODE] = emission_mode_code(options.emission_mode);
        write_key_metadata(&mut control_words, specialisation, key_metadata);

        Ok(Self {
            name: format!("GpuMessageResource<{type_name}>"),
            item_size,
            capacity,
            specialisation,
            options,
            key_metadata,
            raw_cpu_bytes: Vec::new(),
            final_bytes: Vec::new(),
            index_words: vec![0],
            control_words,
            downloaded_gpu_raw: Vec::new(),
            buffers: None,
        })
    }

    /// Clears per-tick GPU emission state.
    pub(crate) fn begin_tick(&mut self) {
        self.raw_cpu_bytes.clear();
        self.final_bytes.clear();
        self.index_words.clear();
        self.index_words.push(0);
        self.downloaded_gpu_raw.clear();
        self.control_words[CONTROL_FINAL_COUNT] = 0;
        self.control_words[CONTROL_OVERFLOW] = 0;
        self.control_words[CONTROL_APPEND_COUNTER] = 0;
        self.control_words[CONTROL_FIXED_SLOT_COUNT] =
            self.options.fixed_slot_count.min(self.capacity as u32);
        self.control_words[CONTROL_EMISSION_MODE] = emission_mode_code(self.options.emission_mode);
        self.control_words[CONTROL_CPU_RAW_COUNT] = 0;
        self.control_words[CONTROL_GPU_VALID_COUNT] = 0;
        self.control_words[CONTROL_INDEX_WORD_COUNT] = 1;
        self.control_words[CONTROL_ERROR_CODE] = 0;
        write_key_metadata(
            &mut self.control_words,
            self.specialisation,
            self.key_metadata,
        );
    }

    /// Publishes the CPU-emitted raw stream for shader diagnostics and custom
    /// GPU-side finalisation.
    pub(crate) fn publish_cpu_raw(&mut self, raw_bytes: &[u8]) {
        self.raw_cpu_bytes.clear();
        self.raw_cpu_bytes.extend_from_slice(raw_bytes);
        self.control_words[CONTROL_CPU_RAW_COUNT] = (raw_bytes.len() / self.item_size) as u32;
    }

    /// Publishes a CPU-finalised view for GPU consumers.
    pub(crate) fn publish_finalised(&mut self, final_bytes: &[u8], index_words: &[u32]) {
        self.final_bytes.clear();
        self.final_bytes.extend_from_slice(final_bytes);
        self.index_words.clear();
        self.index_words.extend_from_slice(index_words);
        if self.index_words.is_empty() {
            self.index_words.push(0);
        }
        self.control_words[CONTROL_FINAL_COUNT] = (self.final_bytes.len() / self.item_size) as u32;
        self.control_words[CONTROL_INDEX_WORD_COUNT] = self.index_words.len() as u32;
    }

    /// Capacity in messages.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the configured GPU emission mode.
    #[inline]
    pub fn emission_mode(&self) -> GpuEmissionMode {
        self.options.emission_mode
    }

    /// Returns whether this resource has enough metadata for GPU-native
    /// finalise for its registered specialisation.
    pub(crate) fn supports_gpu_finalise(&self) -> bool {
        match self.specialisation {
            Specialisation::BruteForce => true,
            Specialisation::Bucket { .. } => {
                matches!(self.key_metadata, GpuKeyMetadata::Bucket { .. })
            }
            Specialisation::Spatial(_) => {
                matches!(self.key_metadata, GpuKeyMetadata::Spatial { .. })
            }
            Specialisation::Targeted => {
                matches!(self.key_metadata, GpuKeyMetadata::Targeted { .. })
            }
        }
    }

    /// Returns whether the adaptive policy should use GPU finalise for this
    /// boundary.
    pub(crate) fn prefers_gpu_finalise(
        &self,
        cpu_raw_count: usize,
        gpu_producer: bool,
        gpu_consumer: bool,
    ) -> bool {
        if !self.supports_gpu_finalise() {
            return false;
        }
        match self.options.finalise_policy {
            GpuFinalisePolicy::AlwaysCpu => false,
            GpuFinalisePolicy::AlwaysGpu => true,
            GpuFinalisePolicy::Adaptive { min_gpu_messages } => {
                gpu_producer || (gpu_consumer && cpu_raw_count >= min_gpu_messages)
            }
        }
    }

    /// Downloads raw GPU emissions into a CPU raw buffer for the CPU fallback
    /// finalise path.
    pub(crate) fn download_gpu_raw_into(
        &mut self,
        dispatch: &BoundaryGpuDispatch<'_>,
        raw: &mut super::aligned_buffer::AlignedBuffer,
    ) -> ECSResult<()> {
        self.ensure_buffers(dispatch.context());
        let ctx = dispatch.context();
        let buffers = self.buffers.as_ref().unwrap();
        let control = read_control_words(ctx, &buffers.control)?;
        let overflow = control[CONTROL_OVERFLOW];
        let append_count = control[CONTROL_APPEND_COUNTER] as usize;

        if overflow != 0 {
            return Err(MessagingError::GpuEmitCapacityExceeded {
                resource: self.name.clone(),
                len: append_count.max(self.capacity + 1),
                max: self.capacity,
            }
            .into());
        }

        match self.options.emission_mode {
            GpuEmissionMode::AppendUnordered => {
                if append_count > self.capacity {
                    return Err(MessagingError::GpuEmitCapacityExceeded {
                        resource: self.name.clone(),
                        len: append_count,
                        max: self.capacity,
                    }
                    .into());
                }
                let byte_len = align_to_4(append_count * self.item_size);
                self.downloaded_gpu_raw.clear();
                if byte_len > 0 {
                    let bytes = read_buffer(ctx, &buffers.raw_gpu, byte_len)?;
                    self.downloaded_gpu_raw
                        .extend_from_slice(&bytes[..append_count * self.item_size]);
                    raw.extend_from_bytes(&self.downloaded_gpu_raw);
                }
            }
            GpuEmissionMode::FixedSlots => {
                let fixed_slots = control[CONTROL_FIXED_SLOT_COUNT] as usize;
                if fixed_slots > self.capacity {
                    return Err(MessagingError::GpuEmitCapacityExceeded {
                        resource: self.name.clone(),
                        len: fixed_slots,
                        max: self.capacity,
                    }
                    .into());
                }
                let flag_bytes = read_buffer(
                    ctx,
                    &buffers.fixed_valid_flags,
                    fixed_slots * std::mem::size_of::<u32>(),
                )?;
                let flags: &[u32] = bytemuck::cast_slice(&flag_bytes);
                let raw_bytes = read_buffer(ctx, &buffers.raw_gpu, self.capacity * self.item_size)?;
                self.downloaded_gpu_raw.clear();
                for (slot, &flag) in flags.iter().enumerate() {
                    if flag == 0 {
                        continue;
                    }
                    let start = slot * self.item_size;
                    self.downloaded_gpu_raw
                        .extend_from_slice(&raw_bytes[start..start + self.item_size]);
                }
                raw.extend_from_bytes(&self.downloaded_gpu_raw);
            }
        }
        Ok(())
    }

    /// Runs GPU-native finalise and optionally downloads final/index data for
    /// CPU consumers.
    pub(crate) fn finalise_gpu(
        &mut self,
        dispatch: &mut BoundaryGpuDispatch<'_>,
        needs_cpu_download: bool,
    ) -> ECSResult<GpuFinaliseOutput> {
        self.ensure_buffers(dispatch.context());
        self.upload_cpu_raw_for_finalise(dispatch.context());
        let buffers = self.buffers.as_ref().unwrap();

        let entries = [
            entry(0, &buffers.raw_cpu),
            entry(1, &buffers.raw_gpu),
            entry(2, &buffers.fixed_valid_flags),
            entry(3, &buffers.final_messages),
            entry(4, &buffers.index_metadata),
            entry(5, &buffers.control),
            entry(6, &buffers.scratch),
        ];
        dispatch.dispatch(BoundaryKernelDesc {
            label: "syren_gpu_message_finalise",
            shader: WGSL_FINALISE_KERNEL,
            entry_point: "main",
            bindings: &FINALISE_BINDINGS,
            entries: &entries,
            workgroups_x: 1,
            workgroups_y: 1,
            workgroups_z: 1,
        })?;

        let control = read_control_words(dispatch.context(), &buffers.control)?;
        let overflow = control[CONTROL_OVERFLOW];
        let error_code = control[CONTROL_ERROR_CODE];
        let final_count = control[CONTROL_FINAL_COUNT] as usize;
        let index_word_count = control[CONTROL_INDEX_WORD_COUNT] as usize;
        if overflow != 0 || error_code != 0 || final_count > self.capacity {
            return Err(MessagingError::GpuFinaliseFailed {
                resource: self.name.clone(),
                reason: format!(
                    "overflow={overflow}, error_code={error_code}, final_count={final_count}, capacity={}",
                    self.capacity
                ),
            }
            .into());
        }

        let mut output = GpuFinaliseOutput {
            final_bytes: None,
            index_words: None,
        };
        if needs_cpu_download {
            let final_len = final_count * self.item_size;
            let final_bytes = if final_len == 0 {
                Vec::new()
            } else {
                let bytes = read_buffer(
                    dispatch.context(),
                    &buffers.final_messages,
                    align_to_4(final_len),
                )?;
                bytes[..final_len].to_vec()
            };
            let index_bytes = index_word_count * std::mem::size_of::<u32>();
            let index_words = if index_word_count == 0 {
                Vec::new()
            } else {
                let bytes = read_buffer(dispatch.context(), &buffers.index_metadata, index_bytes)?;
                bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
            };
            self.final_bytes = final_bytes.clone();
            self.index_words = index_words.clone();
            output.final_bytes = Some(final_bytes);
            output.index_words = Some(index_words);
        }
        self.control_words[CONTROL_FINAL_COUNT] = final_count as u32;
        self.control_words[CONTROL_INDEX_WORD_COUNT] = index_word_count as u32;
        Ok(output)
    }

    fn upload_cpu_raw_for_finalise(&mut self, ctx: &GPUContext) {
        self.ensure_buffers(ctx);
        let buffers = self.buffers.as_ref().unwrap();
        write_padded(ctx, &buffers.raw_cpu, &self.raw_cpu_bytes);
        write_control_word(
            ctx,
            &buffers.control,
            CONTROL_ITEM_SIZE,
            self.item_size as u32,
        );
        write_control_word(
            ctx,
            &buffers.control,
            CONTROL_CAPACITY,
            self.capacity as u32,
        );
        write_control_word(ctx, &buffers.control, CONTROL_FINAL_COUNT, 0);
        write_control_word(
            ctx,
            &buffers.control,
            CONTROL_SPECIALISATION,
            specialisation_code(self.specialisation),
        );
        write_control_word(
            ctx,
            &buffers.control,
            CONTROL_EMISSION_MODE,
            emission_mode_code(self.options.emission_mode),
        );
        write_control_word(
            ctx,
            &buffers.control,
            CONTROL_CPU_RAW_COUNT,
            (self.raw_cpu_bytes.len() / self.item_size) as u32,
        );
        write_control_word(ctx, &buffers.control, CONTROL_GPU_VALID_COUNT, 0);
        write_control_word(ctx, &buffers.control, CONTROL_INDEX_WORD_COUNT, 0);
        write_control_word(ctx, &buffers.control, CONTROL_ERROR_CODE, 0);
        let mut words = self.control_words;
        write_key_metadata(&mut words, self.specialisation, self.key_metadata);
        for (idx, word) in words.iter().enumerate().skip(CONTROL_KEY0) {
            write_control_word(ctx, &buffers.control, idx, *word);
        }
    }

    fn ensure_buffers(&mut self, ctx: &GPUContext) {
        let raw_bytes = align_to_4(self.capacity * self.item_size).max(4);
        let index_bytes = self.index_capacity_bytes().max(4);
        let valid_bytes = align_to_4(self.capacity * std::mem::size_of::<u32>()).max(4);
        let scratch_bytes = self.scratch_capacity_bytes().max(4);
        let recreate = self.buffers.as_ref().map_or(true, |b| {
            b.raw_bytes < raw_bytes
                || b.index_bytes < index_bytes
                || b.valid_bytes < valid_bytes
                || b.scratch_bytes < scratch_bytes
        });

        if !recreate {
            return;
        }

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let raw_cpu = create_buffer(ctx, &format!("{}::raw_cpu", self.name), raw_bytes, storage);
        let raw_gpu = create_buffer(ctx, &format!("{}::raw_gpu", self.name), raw_bytes, storage);
        let fixed_valid_flags = create_buffer(
            ctx,
            &format!("{}::fixed_valid_flags", self.name),
            valid_bytes,
            storage,
        );
        let final_messages =
            create_buffer(ctx, &format!("{}::final", self.name), raw_bytes, storage);
        let index_metadata =
            create_buffer(ctx, &format!("{}::index", self.name), index_bytes, storage);
        let control = create_buffer(
            ctx,
            &format!("{}::control", self.name),
            CONTROL_BYTES,
            storage,
        );
        let scratch = create_buffer(
            ctx,
            &format!("{}::scratch", self.name),
            scratch_bytes,
            storage,
        );

        self.buffers = Some(GpuBuffers {
            raw_cpu,
            raw_gpu,
            fixed_valid_flags,
            final_messages,
            index_metadata,
            control,
            scratch,
            raw_bytes,
            index_bytes,
            valid_bytes,
            scratch_bytes,
        });
    }

    fn index_capacity_bytes(&self) -> usize {
        match self.specialisation {
            Specialisation::BruteForce => 4,
            Specialisation::Bucket { max_buckets } => (max_buckets as usize + 1) * 4,
            Specialisation::Spatial(cfg) => (cfg.total_cells() + 1) * 4,
            Specialisation::Targeted => self.capacity * 4 * 4,
        }
    }

    fn scratch_capacity_bytes(&self) -> usize {
        let words = match self.specialisation {
            Specialisation::BruteForce => 1,
            Specialisation::Bucket { max_buckets } => max_buckets as usize + 1,
            Specialisation::Spatial(cfg) => cfg.total_cells() + 1,
            Specialisation::Targeted => self.capacity * 2,
        };
        align_to_4(words * std::mem::size_of::<u32>())
    }
}

impl GPUResource for GpuMessageResource {
    fn name(&self) -> &str {
        &self.name
    }

    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        self.ensure_buffers(ctx);
        Ok(())
    }

    fn upload(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        self.ensure_buffers(ctx);
        let buffers = self.buffers.as_ref().unwrap();
        write_padded(ctx, &buffers.raw_cpu, &self.raw_cpu_bytes);
        write_padded(ctx, &buffers.final_messages, &self.final_bytes);
        context_write_u32(ctx, &buffers.index_metadata, &self.index_words);
        context_write_u32(ctx, &buffers.control, &self.control_words);
        zero_buffer(ctx, &buffers.fixed_valid_flags, buffers.valid_bytes);
        zero_buffer(ctx, &buffers.scratch, buffers.scratch_bytes);
        Ok(())
    }

    fn download(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        self.ensure_buffers(ctx);
        let buffers = self.buffers.as_ref().unwrap();
        let control_bytes = read_buffer(ctx, &buffers.control, CONTROL_BYTES)?;
        let control_word = |idx: usize| -> u32 {
            let start = idx * 4;
            u32::from_le_bytes(control_bytes[start..start + 4].try_into().unwrap())
        };
        let overflow = control_word(CONTROL_OVERFLOW);
        let gpu_count = control_word(CONTROL_APPEND_COUNTER) as usize;
        if overflow != 0 || gpu_count > self.capacity {
            return Err(ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!(
                    "gpu message '{}' exceeded capacity {} with {} attempted writes",
                    self.name, self.capacity, gpu_count
                )
                .into(),
            }));
        }
        let count = gpu_count;
        let byte_len = align_to_4(count * self.item_size);
        self.downloaded_gpu_raw.clear();
        if byte_len > 0 {
            let bytes = read_buffer(ctx, &buffers.raw_gpu, byte_len)?;
            self.downloaded_gpu_raw
                .extend_from_slice(&bytes[..count * self.item_size]);
        }
        Ok(())
    }

    fn automatic_download(&self) -> bool {
        false
    }

    fn bindings(&self) -> &[GPUBindingDesc] {
        &BINDINGS
    }

    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        let buffers = self.buffers.as_ref().ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("gpu message resource '{}' was not created", self.name).into(),
            })
        })?;
        out.push(entry(base, &buffers.raw_cpu));
        out.push(entry(base + 1, &buffers.raw_gpu));
        out.push(entry(base + 2, &buffers.fixed_valid_flags));
        out.push(entry(base + 3, &buffers.final_messages));
        out.push(entry(base + 4, &buffers.index_metadata));
        out.push(entry(base + 5, &buffers.control));
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn specialisation_code(specialisation: Specialisation) -> u32 {
    match specialisation {
        Specialisation::BruteForce => 0,
        Specialisation::Bucket { .. } => 1,
        Specialisation::Spatial(_) => 2,
        Specialisation::Targeted => 3,
    }
}

fn emission_mode_code(mode: GpuEmissionMode) -> u32 {
    match mode {
        GpuEmissionMode::FixedSlots => 0,
        GpuEmissionMode::AppendUnordered => 1,
    }
}

fn validate_gpu_layout(
    type_name: &'static str,
    item_size: usize,
    specialisation: Specialisation,
    key_metadata: GpuKeyMetadata,
) -> Result<(), MessagingError> {
    if item_size % 4 != 0 {
        return Err(MessagingError::InvalidGpuMessageLayout {
            type_name,
            reason: "GPU-native finalise requires item_size to be a multiple of 4".to_string(),
        });
    }

    let words = (item_size / 4) as u32;
    let in_bounds = |offset: u32| offset < words;
    let pair_in_bounds = |lo: u32, hi: u32| lo < words && hi < words;

    match (specialisation, key_metadata) {
        (Specialisation::BruteForce, GpuKeyMetadata::None) => Ok(()),
        (Specialisation::Bucket { .. }, GpuKeyMetadata::None) => Ok(()),
        (Specialisation::Bucket { .. }, GpuKeyMetadata::Bucket { key_word_offset })
            if in_bounds(key_word_offset) =>
        {
            Ok(())
        }
        (Specialisation::Spatial(_), GpuKeyMetadata::None) => Ok(()),
        (
            Specialisation::Spatial(_),
            GpuKeyMetadata::Spatial {
                x_word_offset,
                y_word_offset,
            },
        ) if pair_in_bounds(x_word_offset, y_word_offset) => Ok(()),
        (Specialisation::Targeted, GpuKeyMetadata::None) => Ok(()),
        (
            Specialisation::Targeted,
            GpuKeyMetadata::Targeted {
                layout: GpuTargetKeyLayout::U64Word { word_offset },
            },
        ) if word_offset + 1 < words => Ok(()),
        (
            Specialisation::Targeted,
            GpuKeyMetadata::Targeted {
                layout:
                    GpuTargetKeyLayout::U32Pair {
                        low_word_offset,
                        high_word_offset,
                    },
            },
        ) if pair_in_bounds(low_word_offset, high_word_offset) => Ok(()),
        _ => Err(MessagingError::InvalidGpuMessageLayout {
            type_name,
            reason: format!(
                "key metadata does not match {} message layout or is out of bounds",
                specialisation.name()
            ),
        }),
    }
}

fn write_key_metadata(
    control_words: &mut [u32; CONTROL_WORDS],
    specialisation: Specialisation,
    key_metadata: GpuKeyMetadata,
) {
    for word in &mut control_words[CONTROL_KEY0..] {
        *word = 0;
    }
    match (specialisation, key_metadata) {
        (Specialisation::Bucket { max_buckets }, GpuKeyMetadata::Bucket { key_word_offset }) => {
            control_words[CONTROL_KEY0] = max_buckets;
            control_words[CONTROL_KEY1] = key_word_offset;
        }
        (Specialisation::Bucket { max_buckets }, _) => {
            control_words[CONTROL_KEY0] = max_buckets;
        }
        (
            Specialisation::Spatial(cfg),
            GpuKeyMetadata::Spatial {
                x_word_offset,
                y_word_offset,
            },
        ) => {
            control_words[CONTROL_KEY0] = cfg.cols();
            control_words[CONTROL_KEY1] = cfg.rows();
            control_words[CONTROL_KEY2] = cfg.cell_size.to_bits();
            control_words[CONTROL_KEY3] = cfg.total_cells() as u32;
            control_words[CONTROL_KEY4] = x_word_offset;
            control_words[CONTROL_KEY5] = y_word_offset;
        }
        (Specialisation::Spatial(cfg), _) => {
            control_words[CONTROL_KEY0] = cfg.cols();
            control_words[CONTROL_KEY1] = cfg.rows();
            control_words[CONTROL_KEY2] = cfg.cell_size.to_bits();
            control_words[CONTROL_KEY3] = cfg.total_cells() as u32;
        }
        (
            Specialisation::Targeted,
            GpuKeyMetadata::Targeted {
                layout: GpuTargetKeyLayout::U64Word { word_offset },
            },
        ) => {
            control_words[CONTROL_KEY0] = 0;
            control_words[CONTROL_KEY1] = word_offset;
            control_words[CONTROL_KEY2] = word_offset + 1;
        }
        (
            Specialisation::Targeted,
            GpuKeyMetadata::Targeted {
                layout:
                    GpuTargetKeyLayout::U32Pair {
                        low_word_offset,
                        high_word_offset,
                    },
            },
        ) => {
            control_words[CONTROL_KEY0] = 1;
            control_words[CONTROL_KEY1] = low_word_offset;
            control_words[CONTROL_KEY2] = high_word_offset;
        }
        _ => {}
    }
}

#[inline]
fn align_to_4(bytes: usize) -> usize {
    (bytes + 3) & !3
}

fn create_buffer(
    ctx: &GPUContext,
    label: &str,
    bytes: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn write_padded(ctx: &GPUContext, buffer: &wgpu::Buffer, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    if bytes.len() % 4 == 0 {
        ctx.queue.write_buffer(buffer, 0, bytes);
    } else {
        let mut padded = bytes.to_vec();
        padded.resize(align_to_4(bytes.len()), 0);
        ctx.queue.write_buffer(buffer, 0, &padded);
    }
}

fn context_write_u32(ctx: &GPUContext, buffer: &wgpu::Buffer, words: &[u32]) {
    if words.is_empty() {
        ctx.queue.write_buffer(buffer, 0, &[0, 0, 0, 0]);
    } else {
        ctx.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(words));
    }
}

fn write_control_word(ctx: &GPUContext, buffer: &wgpu::Buffer, word_index: usize, word: u32) {
    ctx.queue.write_buffer(
        buffer,
        (word_index * std::mem::size_of::<u32>()) as u64,
        bytemuck::bytes_of(&word),
    );
}

fn zero_buffer(ctx: &GPUContext, buffer: &wgpu::Buffer, bytes: usize) {
    if bytes == 0 {
        return;
    }
    let zeros = vec![0u8; bytes];
    ctx.queue.write_buffer(buffer, 0, &zeros);
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_buffer(ctx: &GPUContext, source: &wgpu::Buffer, bytes: usize) -> ECSResult<Vec<u8>> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("syren_gpu_message_readback"),
        size: bytes.max(4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("syren_gpu_message_readback_encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes.max(4) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("wgpu poll failed before message readback: {e:?}").into(),
            })
        })?;

    let slice = staging.slice(..bytes.max(4) as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("wgpu poll failed during message readback: {e:?}").into(),
            })
        })?;
    receiver.recv().ok().transpose().map_err(|_| {
        ECSError::from(ExecutionError::GpuDispatchFailed {
            message: "failed to map gpu message readback".into(),
        })
    })?;

    let mapped = slice.get_mapped_range();
    let out = mapped[..bytes].to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

fn read_control_words(ctx: &GPUContext, source: &wgpu::Buffer) -> ECSResult<[u32; CONTROL_WORDS]> {
    let bytes = read_buffer(ctx, source, CONTROL_BYTES)?;
    let words: &[u32] = bytemuck::cast_slice(&bytes);
    let mut out = [0u32; CONTROL_WORDS];
    out.copy_from_slice(&words[..CONTROL_WORDS]);
    Ok(out)
}

/// Targeted index row as visible to GPU consumers.
pub fn targeted_index_words(entity_raw: u64, start: u32, end: u32) -> [u32; 4] {
    [entity_raw as u32, (entity_raw >> 32) as u32, start, end]
}

/// Spatial params tuple for resource descriptors.
pub fn spatial_params(config: SpatialConfig) -> [u32; 4] {
    [
        config.cols(),
        config.rows(),
        config.cell_size.to_bits(),
        config.total_cells() as u32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::{ECSError, ExecutionError};
    use crate::gpu::GPUResourceRegistry;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    struct TestBucketMsg {
        bucket: u32,
        value: u32,
    }

    fn skip_if_no_gpu(result: ECSResult<()>) {
        match result {
            Ok(()) => {}
            Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) => {
                eprintln!("skipping GPU messaging kernel test: {message}");
            }
            Err(err) => panic!("{err:?}"),
        }
    }

    #[test]
    fn gpu_bucket_finalise_matches_cpu_index_order() {
        let mut registry = GPUResourceRegistry::new();
        let resource = GpuMessageResource::new(
            "TestBucketMsg",
            std::mem::size_of::<TestBucketMsg>(),
            Capacity::bounded(4, 8),
            Specialisation::Bucket { max_buckets: 3 },
            GpuMessageOptions::deterministic(8),
            GpuKeyMetadata::Bucket { key_word_offset: 0 },
        )
        .unwrap();
        let id = registry.register(resource).unwrap();

        let messages = [
            TestBucketMsg {
                bucket: 1,
                value: 10,
            },
            TestBucketMsg {
                bucket: 0,
                value: 20,
            },
            TestBucketMsg {
                bucket: 1,
                value: 30,
            },
        ];

        let result = crate::gpu::with_boundary_dispatch(&mut registry, |dispatch, resources| {
            let resource = resources
                .get_mut_typed::<GpuMessageResource>(id)
                .expect("registered message resource");
            resource.publish_cpu_raw(bytemuck::cast_slice(&messages));
            let output = resource.finalise_gpu(dispatch, true)?;
            let final_bytes = output.final_bytes.expect("downloaded final bytes");
            let sorted: &[TestBucketMsg] = bytemuck::cast_slice(&final_bytes);
            assert_eq!(
                sorted,
                &[
                    TestBucketMsg {
                        bucket: 0,
                        value: 20
                    },
                    TestBucketMsg {
                        bucket: 1,
                        value: 10
                    },
                    TestBucketMsg {
                        bucket: 1,
                        value: 30
                    },
                ]
            );
            assert_eq!(output.index_words.as_deref(), Some(&[0, 1, 3, 3][..]));
            Ok(())
        });

        skip_if_no_gpu(result);
    }
}
