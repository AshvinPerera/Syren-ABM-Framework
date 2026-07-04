//! Deterministic, seed-keyed pseudo-random number generation.
//!
//! # Why not a thread-local RNG?
//!
//! Systems run on Rayon worker threads, and *which* worker processes which
//! chunk of agents is decided by work stealing — it varies from run to run.
//! Any RNG whose state lives on the worker thread therefore produces
//! different per-agent draws on every execution, even with a fixed seed.
//!
//! Reproducible models instead derive randomness from *simulation
//! coordinates* — values that are identical across runs regardless of thread
//! assignment: the simulation seed, the current tick, the system id, and a
//! caller-chosen salt (typically an agent index, entity id, or chunk/row
//! pair).
//!
//! [`DetRng`] packages that pattern: construct one from the current
//! [`RunContext`] plus a salt, then draw as many values as needed. Two
//! constructions with identical inputs yield identical sequences on any
//! machine and any thread count.
//!
//! The engine's own activation-order shuffles use the same keyed
//! [`splitmix64`] mixer (see `engine::manager::query_executor`).
//!
//! # Example
//!
//! ```text
//! // Inside a system, with `ctx = ecs.run_context()` captured *before*
//! // entering a parallel for_each:
//! let mut rng = DetRng::from_context(ctx, entity.to_raw());
//! let draw = rng.next_f64();
//! ```
//!
//! # Non-goals
//!
//! The generator (splitmix64) is statistically solid for simulation use but
//! is **not cryptographically secure**.

use crate::engine::activation::RunContext;

/// One application of the splitmix64 state advance + output mix.
///
/// Used both as a one-shot bit mixer for seed derivation and as the stream
/// function behind [`DetRng`].
#[inline]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Deterministic pseudo-random generator keyed by simulation coordinates.
///
/// See the [module documentation](self) for the reproducibility rationale.
/// The sequence is the canonical splitmix64 stream over the derived seed.
#[derive(Clone, Copy, Debug)]
pub struct DetRng {
    state: u64,
}

impl DetRng {
    /// Creates a generator keyed by the deterministic run context and a
    /// caller-chosen salt.
    ///
    /// Use a salt that identifies the logical draw site — an entity's raw id,
    /// an agent index, or a `(chunk, row)` encoding — so distinct agents get
    /// independent streams within the same system and tick.
    #[inline]
    pub fn from_context(context: RunContext, salt: u64) -> Self {
        // Fold each coordinate through the mixer so structurally similar
        // inputs (adjacent ticks, adjacent salts) land far apart.
        let mut state = splitmix64(context.simulation_seed);
        state = splitmix64(state ^ context.tick);
        state = splitmix64(state ^ (context.system_id as u64));
        state = splitmix64(state ^ salt);
        Self { state }
    }

    /// Creates a generator from a raw seed.
    #[inline]
    pub fn from_seed(seed: u64) -> Self {
        Self {
            state: splitmix64(seed),
        }
    }

    /// Returns the next pseudo-random `u64` in the stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniformly distributed `f64` in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 53 mantissa bits scaled by 2^-53.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a uniformly distributed `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 mantissa bits scaled by 2^-24.
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Returns a value in `[0, upper)` via fixed-point scaling.
    ///
    /// Returns `0` when `upper == 0`. The scaling introduces a bias of at
    /// most `upper / 2^64`, negligible for simulation-scale ranges.
    #[inline]
    pub fn next_below(&mut self, upper: u64) -> u64 {
        ((self.next_u64() as u128 * upper as u128) >> 64) as u64
    }

    /// Returns a `usize` index in `[0, upper)`; `0` when `upper == 0`.
    #[inline]
    pub fn next_index(&mut self, upper: usize) -> usize {
        self.next_below(upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_context_and_salt_reproduce_sequences() {
        let ctx = RunContext {
            simulation_seed: 42,
            tick: 7,
            system_id: 3,
        };
        let mut a = DetRng::from_context(ctx, 1001);
        let mut b = DetRng::from_context(ctx, 1001);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_salts_produce_distinct_streams() {
        let ctx = RunContext {
            simulation_seed: 42,
            tick: 7,
            system_id: 3,
        };
        let mut a = DetRng::from_context(ctx, 0);
        let mut b = DetRng::from_context(ctx, 1);
        let same = (0..16).filter(|_| a.next_u64() == b.next_u64()).count();
        assert_eq!(same, 0);
    }

    #[test]
    fn unit_floats_stay_in_range() {
        let mut rng = DetRng::from_seed(9);
        for _ in 0..1024 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            let y = rng.next_f32();
            assert!((0.0..1.0).contains(&y));
        }
    }

    #[test]
    fn next_below_respects_bounds() {
        let mut rng = DetRng::from_seed(11);
        assert_eq!(rng.next_below(0), 0);
        for _ in 0..1024 {
            assert!(rng.next_below(10) < 10);
            assert!(rng.next_index(3) < 3);
        }
    }
}
