//! Fast thread-local pseudo-random number generation utilities.
//!
//! This module provides a lightweight, lock-free source of pseudo-random
//! numbers intended for performance-critical paths such as simulations,
//! ECS scheduling, randomized iteration, or procedural generation.
//!
//! # Design
//!
//! The generator is implemented as a **thread-local xorshift64\*** RNG:
//!
//! - Each thread owns its own independent RNG state via `thread_local!`.
//! - The state is stored in a `Cell<u64>` to allow mutation without borrowing.
//! - No global state, locks, or atomics are used.
//!
//! The initial seed is a fixed, non-zero constant, ensuring deterministic
//! behavior *per thread* across executions unless the thread creation order
//! changes.
//!
//! # Performance characteristics
//!
//! - **O(1)** per call
//! - No heap allocation
//! - No synchronization or contention
//! - Suitable for very hot loops
//!
//! # Determinism
//!
//! The RNG is deterministic within a single thread: given the same sequence
//! of calls, it will produce the same output sequence. Different threads
//! evolve their states independently.
//!
//! # Safety and correctness
//!
//! This module does not use `unsafe` code. Interior mutability is handled
//! via `Cell`, and thread-local isolation guarantees data-race freedom.
//!
//! # Non-goals
//!
//! - This generator is **not cryptographically secure**.
//! - It should not be used for security-sensitive randomness.
//!
//! # Intended use cases
//!
//! - Simulation and modeling
//! - Randomized algorithms and heuristics
//! - ECS shuffling and load balancing
//! - Procedural generation
//!
//! For cryptographic or statistically rigorous randomness, prefer
//! `rand`-crate generators instead.

use std::cell::Cell;
use std::thread_local;


thread_local! {static TL_RNG: Cell<u64> = Cell::new(0x9E37_79B9_7F4A_7C15);}

/// Returns a fast, thread-local pseudo-random `u64`.
///
/// ## Behavior
/// This function generates a new pseudo-random value using a **xorshift64\***
/// algorithm backed by a thread-local state. Each thread maintains its own
/// independent RNG state, eliminating contention and synchronization overhead.
///
/// The generator is deterministic per thread and is seeded with a fixed
/// non-zero constant at thread initialization.
///
/// ## Guarantees
/// * **Lock-free:** No global synchronization or atomics.
/// * **Thread-safe:** Each thread has independent state via `thread_local!`.
/// * **Fast:** Suitable for hot paths and simulation workloads.
///
/// ## Non-goals
/// * This is **not cryptographically secure**.
/// * Output quality is sufficient for simulation, sampling, and randomized
///   iteration-not security-sensitive contexts.
///
/// ## Example
/// ```
/// let x = tl_rand_u64();
/// let y = tl_rand_u64();
/// assert_ne!(x, y);
/// ```

#[inline]
pub fn tl_rand_u64() -> u64 {
    TL_RNG.with(|c| {
        let mut x = c.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        c.set(x);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    })
}