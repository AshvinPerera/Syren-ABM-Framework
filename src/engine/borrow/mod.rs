//! # Borrow Tracking Module
//!
//! This module implements a **per-component read/write borrow tracker**
//! that enforces Rust-like borrowing rules at runtime across dynamically
//! scheduled systems.
//!
//! ## Purpose
//!
//! - Multiple systems may **read** the same component type concurrently.
//! - Only one system may **write** to a component type at a time.
//! - No system may read a component type while another system writes it.
//!
//! This is achieved without OS locks by using **atomic state machines**.
//!
//! ## State Encoding
//!
//! Each component ID maps to one `AtomicUsize` with the following meaning:
//!
//! | State | Meaning |
//! |------:|--------|
//! | `0` | Unlocked |
//! | `1` | Write-locked (exclusive writer) |
//! | `>= 2` | Read-locked (`state - 1` active readers) |
//!
//! ## Synchronization Strategy
//!
//! - Uses atomic operations with acquire/release ordering.
//! - The same thread re-acquiring the same component will deadlock.
//! - Both `acquire_read` and `acquire_write` enforce a bounded spin limit
//!   (default [`DEFAULT_SPIN_LIMIT`]) and return
//!   [`ExecutionError::BorrowConflict`] if the limit is exceeded, converting
//!   potential hangs into actionable errors.
//!
//! ## RAII Integration
//!
//! The [`BorrowGuard`] type provides RAII-style acquisition and release
//! of multiple component borrows for the full lifetime of a query or system.
//!
//! `clear()` uses a **dirty bitset** to avoid resetting all 4096 atomics on
//! every stage boundary. Only components that were actually borrowed during
//! the stage are touched during clear.

mod guard;
mod tracker;

#[cfg(test)]
mod tests;

pub use guard::BorrowGuard;
pub use tracker::BorrowTracker;
