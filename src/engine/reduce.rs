//! Reduction primitives for global ECS observations.
//!
//! This module defines **pure, thread-local accumulator types** intended for
//! use with the ECS reduction APIs (e.g. [`ECSReference::reduce_abstraction`]
//! and [`ECSReference::reduce_read`]).
//!
//! ## Purpose
//! Reductions compute **summary statistics** over large populations of entities
//! without mutating ECS state. Common use cases include:
//!
//! * population counts,
//! * totals and aggregates,
//! * minima / maxima,
//! * means and variances,
//! * convergence and equilibrium checks,
//! * diagnostics and model instrumentation.
//!
//! ## Execution model
//! A reduction proceeds in two phases:
//!
//! 1. **Parallel accumulation**
//!    * Each worker thread processes a disjoint subset of archetype chunks.
//!    * Each thread maintains its own accumulator value.
//!
//! 2. **Deterministic combination**
//!    * Thread-local accumulators are merged using an associative `combine`
//!      operation.
//!    * Combination order is deterministic and independent of thread count.
//!
//! This model scales efficiently on multicore CPUs and maps directly to
//! GPU-style block reductions.
//!
//! ## Design principles
//! The accumulator types in this module are intentionally:
//!
//! * **Plain data containers** - no ECS references or side effects.
//! * **Copy / Clone** - easy to move between threads.
//! * **Execution-agnostic** - usable on CPU, GPU, or in offline analysis.
//!
//! They do not depend on the scheduler, query system, or ECS internals.
//!
//! ## Provided accumulators
//!
//! * [`Count`] - counts entities.
//! * [`Sum`] - accumulates floating-point totals.
//! * [`MinMax`] - tracks minimum and maximum values.
//! * [`Welford`] - computes mean and variance using a numerically stable
//!   online algorithm. Supports parallel combination via [`Welford::combine`].
//!
//!
//! ## Usage example
//!
//! ```ignore
//! // `ecs` is an ECSReference obtained from ECSManager::world_ref().
//! let query = ecs.query()?.read::<Wealth>()?.build()?;
//!
//! let total = ecs.reduce_read::<Wealth, Sum>(
//!     query,
//!     Sum::default,
//!     |acc, w| acc.0 += w.amount,
//!     |a, b| a.0 += b.0,
//! )?;
//! ```
//!
//! ## Safety
//! Accumulators are used exclusively within the ECS reduction APIs, which
//! enforce:
//!
//! * read-only phase discipline,
//! * iteration scope tracking,
//! * runtime borrow checking,
//! * disjoint chunk processing.

/// Accumulator that counts the number of entities processed.
///
/// ## Semantics
/// Each call to the reduction fold function typically increments the internal
/// counter by one, yielding the total number of entities matching a query.
///
/// ## Typical use cases
/// * Population size
/// * Cardinality of a query
/// * Participation counts

#[derive(Clone, Copy, Debug, Default)]
pub struct Count(pub u64);

/// Accumulator that computes a floating-point sum.
///
/// ## Semantics
/// Values are accumulated using standard floating-point addition. For large
/// populations or numerically sensitive models, users may prefer more stable
/// accumulation strategies (e.g. pairwise summation or Welford-based methods).
///
/// ## Typical use cases
/// * Total wealth
/// * Aggregate production
/// * Market volume

#[derive(Clone, Copy, Debug, Default)]
pub struct Sum(pub f64);

/// Accumulator that tracks minimum and maximum values.
///
/// ## Semantics
/// The accumulator maintains the smallest and largest values observed during
/// reduction. The default initializer sets:
/// * `min` to positive infinity
/// * `max` to negative infinity
///
/// allowing the first observed value to establish both bounds.
///
/// ## Typical use cases
/// * Price ranges
/// * Income bounds

#[derive(Clone, Copy, Debug)]
pub struct MinMax {
    /// Smallest observed value.
    pub min: f64,

    /// Largest observed value.
    pub max: f64,
}

impl Default for MinMax {
    fn default() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

/// Accumulator implementing Welford's online algorithm for mean and variance.
///
/// ## Semantics
/// This accumulator computes the mean and (sample) variance of a stream of
/// values in a numerically stable, single-pass manner.
///
/// It supports:
/// * incremental updates via [`Welford::push`],
/// * parallel combination of partial accumulators via [`Welford::combine`],
/// * stable results independent of iteration order.
///
/// ## Typical use cases
/// * Average income or wealth
/// * Inequality and dispersion metrics
/// * Convergence diagnostics
///
/// ## References
/// * Welford, B. P. (1962). *Note on a method for calculating corrected sums of
///   squares and products*.
/// * Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
///   pairwise algorithm for computing sample variances*.

#[derive(Clone, Copy, Debug, Default)]
pub struct Welford {
    /// Number of samples processed.
    pub n: u64,

    /// Running mean.
    pub mean: f64,

    /// Sum of squared deviations from the mean.
    pub m2: f64,
}

impl Welford {
    /// Incorporates a new sample into the running statistics.
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Returns the unbiased sample variance.
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        }
    }

    /// Merges another `Welford` accumulator into this one.
    ///
    /// This implements the parallel/pairwise combination formula from
    /// Chan, Golub & LeVeque (1979), which is numerically stable and
    /// produces identical results regardless of how partitions are combined.
    ///
    /// ## Usage
    /// This method is intended to be used as the `combine` function in
    /// [`ECSReference::reduce_read`] and [`ECSReference::reduce_abstraction`]:
    ///
    /// ```ignore
    /// ecs.reduce_read::<Value, Welford>(
    ///     query,
    ///     Welford::default,
    ///     |acc, v| acc.push(v.0),
    ///     |a, b| a.combine(b),
    /// )?;
    /// ```
    pub fn combine(&mut self, other: Welford) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other;
            return;
        }

        let combined_n = self.n + other.n;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * (other.n as f64 / combined_n as f64);
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.n as f64 * other.n as f64 / combined_n as f64);

        self.n = combined_n;
        self.mean = combined_mean;
        self.m2 = combined_m2;
    }
}
