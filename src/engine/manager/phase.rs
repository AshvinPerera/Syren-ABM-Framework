//! RAII phase guards for the ECS read/write phase lock.
//!
//! These zero-cost token types prove at compile time that the caller holds the
//! appropriate phase lock before accessing `ECSData`.

use std::sync::{RwLockReadGuard, RwLockWriteGuard};

/// RAII guard representing the ECS read phase.
/// Holding this guard guarantees that no exclusive (structural) access exists.
#[allow(dead_code)]
pub struct PhaseRead<'a>(pub(super) RwLockReadGuard<'a, ()>);
/// RAII guard representing the ECS write (exclusive) phase.
/// Holding this guard guarantees no other readers or writers exist.
#[allow(dead_code)]
pub struct PhaseWrite<'a>(pub(super) RwLockWriteGuard<'a, ()>);
