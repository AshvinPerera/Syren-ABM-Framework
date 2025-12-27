use std::mem::{align_of, size_of};
use std::sync::Once;

use abm_framework::engine::archetype::Archetype;
use abm_framework::engine::component::{
    component_id_of, freeze_components, register_component, Bundle, Signature,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::storage::{cast_slice, Attribute, TypeErasedAttribute};
use abm_framework::engine::types::{ArchetypeID, ChunkID, CHUNK_CAP};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity {
    dx: f32,
    dy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct A(u64);

#[derive(Clone, Copy, Debug, PartialEq)]
struct B(u32);

static INIT: Once = Once::new();

fn init_registry() {
    INIT.call_once(|| {
        register_component::<Position>().unwrap();
        register_component::<Velocity>().unwrap();
        register_component::<A>().unwrap();
        register_component::<B>().unwrap();
        freeze_components().unwrap();
    });
}

// Helper to push into Attribute<T> for both `rollback` and non-rollback builds.
fn attr_push<T: Send + Sync + 'static>(attr: &mut Attribute<T>, value: T) -> (ChunkID, u32) {
    #[cfg(feature = "rollback")]
    {
        let ((c, r), _action) = attr.push(value).unwrap();
        (c, r)
    }
    #[cfg(not(feature = "rollback"))]
    {
        let (c, r) = attr.push(value).unwrap();
        (c, r)
    }
}

#[test]
fn attribute_chunk_is_contiguous_and_aligned() {
    // Uses the real container from storage.rs: Attribute<T> (chunked MaybeUninit storage)
    let mut attr: Attribute<Position> = Attribute::default();

    // Fill first chunk completely
    for i in 0..CHUNK_CAP {
        let (c, r) = attr_push(&mut attr, Position { x: i as f32, y: 0.0 });
        assert_eq!(c, 0);
        assert_eq!(r as usize, i);
    }

    // Verify chunk_bytes API (TypeErasedAttribute) is real and returns expected size
    let (ptr, bytes) = attr
        .chunk_bytes(0, CHUNK_CAP)
        .expect("chunk 0 should exist");

    assert_eq!(bytes, CHUNK_CAP * size_of::<Position>());

    // Alignment check
    assert_eq!(
        (ptr as usize) % align_of::<Position>(),
        0,
        "chunk base pointer must be aligned for Position"
    );

    // Verify contiguity/stride using cast_slice helper from storage.rs
    let slice: &[Position] = unsafe { cast_slice(ptr, bytes) };
    assert_eq!(slice.len(), CHUNK_CAP);

    let base = slice.as_ptr() as usize;
    let stride = size_of::<Position>();

    for i in 0..CHUNK_CAP {
        let pi = unsafe { slice.as_ptr().add(i) as usize };
        assert_eq!(
            pi,
            base + i * stride,
            "row {i} not at expected byte offset within chunk"
        );
    }
}

#[test]
fn attribute_crosses_chunk_boundary_as_expected() {
    let mut attr: Attribute<u64> = Attribute::default();

    // Write CHUNK_CAP + 1 values → forces chunk 1 allocation
    for i in 0..(CHUNK_CAP + 1) {
        let (c, r) = attr_push(&mut attr, i as u64);
        if i < CHUNK_CAP {
            assert_eq!(c, 0);
            assert_eq!(r as usize, i);
        } else {
            assert_eq!(c, 1);
            assert_eq!(r as usize, 0);
        }
    }

    // chunk 0 should report CHUNK_CAP elements when asked for that length
    let (_p0, b0) = attr.chunk_bytes(0, CHUNK_CAP).unwrap();
    assert_eq!(b0, CHUNK_CAP * size_of::<u64>());

    // chunk 1 has only 1 initialized element; request length=1
    let (_p1, b1) = attr.chunk_bytes(1, 1).unwrap();
    assert_eq!(b1, 1 * size_of::<u64>());
}

#[test]
fn archetype_borrow_exposes_soa_columns_with_independent_addresses() {
    init_registry();

    // Build a signature for Position + Velocity using the real Signature bitset type
    let pos_id = component_id_of::<Position>().unwrap();
    let vel_id = component_id_of::<Velocity>().unwrap();

    let mut sig = Signature::default();
    sig.set(pos_id);
    sig.set(vel_id);

    // Archetype::new signature is real: new(archetype_id, signature) -> ECSResult<Self>
    let mut arch = Archetype::new(0 as ArchetypeID, sig).unwrap();

    // spawn_on requires &mut EntityShards, ShardID, and a DynamicBundle (Bundle implements it)
    let mut shards = EntityShards::new(1);

    // Spawn enough to ensure chunk 0 has some data
    for i in 0..1024usize {
        let mut b = Bundle::new();
        b.insert(pos_id, Position { x: i as f32, y: 1.0 });
        b.insert(vel_id, Velocity { dx: 0.5, dy: i as f32 });
        arch.spawn_on(&mut shards, 0, b).unwrap();
    }

    let borrow = arch.borrow_chunk_for(0, &[pos_id, vel_id], &[]).unwrap();
    assert!(borrow.length > 0);

    // borrow.reads returns Vec<(*const u8, usize)> in the same order as read_ids
    let (pos_ptr, pos_bytes) = borrow.reads[0];
    let (vel_ptr, vel_bytes) = borrow.reads[1];

    // Columns must not alias (SoA separation)
    assert_ne!(
        pos_ptr as usize, vel_ptr as usize,
        "Position and Velocity columns should not start at same address"
    );

    // Byte sizes should match `borrow.length * size_of::<T>()`
    assert_eq!(pos_bytes, borrow.length * size_of::<Position>());
    assert_eq!(vel_bytes, borrow.length * size_of::<Velocity>());

    // Validate contiguity inside the chunk slices via cast_slice
    let pos_slice: &[Position] = unsafe { cast_slice(pos_ptr, pos_bytes) };
    let vel_slice: &[Velocity] = unsafe { cast_slice(vel_ptr, vel_bytes) };

    assert_eq!(pos_slice.len(), borrow.length);
    assert_eq!(vel_slice.len(), borrow.length);

    // Stride check: address(i+1) - address(i) == size_of::<T>()
    let pos_base = pos_slice.as_ptr() as usize;
    let vel_base = vel_slice.as_ptr() as usize;

    for i in 0..borrow.length {
        let pi = unsafe { pos_slice.as_ptr().add(i) as usize };
        let vi = unsafe { vel_slice.as_ptr().add(i) as usize };
        assert_eq!(pi, pos_base + i * size_of::<Position>());
        assert_eq!(vi, vel_base + i * size_of::<Velocity>());
    }

    // Alignment check for both columns
    assert_eq!((pos_ptr as usize) % align_of::<Position>(), 0);
    assert_eq!((vel_ptr as usize) % align_of::<Velocity>(), 0);
}

#[test]
fn archetype_bytes_per_row_matches_component_sizes() {
    init_registry();

    let a = component_id_of::<A>().unwrap();
    let b = component_id_of::<B>().unwrap();

    let mut sig = Signature::default();
    sig.set(a);
    sig.set(b);

    let mut arch = Archetype::new(0 as ArchetypeID, sig).unwrap();
    let mut shards = EntityShards::new(1);

    for i in 0..256u32 {
        let mut bundle = Bundle::new();
        bundle.insert(a, A(i as u64));
        bundle.insert(b, B(i));
        arch.spawn_on(&mut shards, 0, bundle).unwrap();
    }

    let borrow = arch.borrow_chunk_for(0, &[a, b], &[]).unwrap();
    let len = borrow.length;

    let bytes_a = borrow.reads[0].1;
    let bytes_b = borrow.reads[1].1;

    assert_eq!(bytes_a / len, std::mem::size_of::<A>());
    assert_eq!(bytes_b / len, std::mem::size_of::<B>());
}

#[test]
fn archetype_chunk_pointer_is_stable_across_borrows() {
    init_registry();

    let pos_id = component_id_of::<Position>().unwrap();

    let mut sig = Signature::default();
    sig.set(pos_id);

    let mut arch = Archetype::new(1 as ArchetypeID, sig).unwrap();
    let mut shards = EntityShards::new(1);

    for i in 0..(CHUNK_CAP / 2) {
        let mut b = Bundle::new();
        b.insert(pos_id, Position { x: i as f32, y: 0.0 });
        arch.spawn_on(&mut shards, 0, b).unwrap();
    }

    let b1 = arch.borrow_chunk_for(0, &[pos_id], &[]).unwrap();
    let p1 = b1.reads[0].0 as usize;
    drop(b1);

    let b2 = arch.borrow_chunk_for(0, &[pos_id], &[]).unwrap();
    let p2 = b2.reads[0].0 as usize;

    assert_eq!(p1, p2, "chunk pointer moved between borrows");
}
