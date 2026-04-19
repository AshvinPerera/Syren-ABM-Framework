//! The central simulation-wide parameter store.
//!
//! ## Design
//!
//! [`Environment`] is a frozen, typed key-value store that holds simulation-wide
//! parameters shared across all agents and systems. Its schema (the set of
//! registered keys and their types) is fixed at construction time by
//! [`EnvironmentBuilder`](super::builder::EnvironmentBuilder) and cannot be
//! extended afterwards. Individual values may be updated via [`Environment::set`]
//! at any point in the simulation.
//!
//! ## Schema immutability
//!
//! The schema is frozen by construction: [`Environment`] exposes no method to
//! add new keys. After [`EnvironmentBuilder::build`] returns, only existing
//! keys may be read or written. Calling [`set`](Environment::set) with an
//! unregistered key returns [`EnvironmentError::KeyNotFound`].
//!
//! ## Thread safety
//!
//! Each entry is protected by its own [`RwLock`], so concurrent readers are
//! never blocked by one another. Only a write on the *same key* requires
//! exclusive access.
//!
//! ## Dirty tracking
//!
//! A shared `dirty_keys` set records which keys have been mutated since the
//! last GPU upload. [`Environment::dirty_keys`] returns a snapshot and
//! [`Environment::clear_dirty`] resets it. For zero-allocation dirty checks,
//! [`Environment::has_any_dirty`] performs a read-locked membership test
//! without cloning. These are `pub(crate)` — only the GPU uniform buffer
//! implementation and tests within this crate need them.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use super::error::{EnvironmentError, EnvironmentResult};

// ─────────────────────────────────────────────────────────────────────────────
// Internal storage entry
// ─────────────────────────────────────────────────────────────────────────────

/// One registered parameter slot.
struct Entry {
    /// The type-erased value.
    value: Box<dyn Any + Send + Sync>,
    /// The concrete [`TypeId`] of the value, used for runtime type checking.
    type_id: TypeId,
    /// Human-readable type name, used in error messages.
    type_name: &'static str,
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment
// ─────────────────────────────────────────────────────────────────────────────

/// The central simulation-wide parameter store.
///
/// Parameters are typed key-value pairs whose schema is fixed after
/// [`EnvironmentBuilder::build`](super::builder::EnvironmentBuilder::build) is
/// called. Values can be read with [`get`](Self::get) and written with
/// [`set`](Self::set) at any time.
///
/// # Schema immutability
///
/// The schema is frozen by construction: `Environment` exposes no method to
/// insert new keys. Immutability is enforced by the absence of any insertion
/// API rather than by a runtime flag. Calling [`set`](Self::set) with an
/// unregistered key returns [`EnvironmentError::KeyNotFound`].
///
/// # Thread safety
///
/// Internally, each value is protected by an individual [`RwLock`]. Concurrent
/// reads on distinct keys (or the same key) never block one another. A write on
/// key `k` is exclusive only to that key.
///
/// # Dirty tracking
///
/// Every successful call to [`set`](Self::set) inserts the key into a shared
/// `dirty_keys` set. This set is consumed by the GPU uniform-buffer upload
/// path to decide which uniform fields need re-packing.
pub struct Environment {
    /// Per-entry storage, keyed by parameter name.
    ///
    /// Schema is frozen at construction — no new entries are ever inserted after
    /// [`EnvironmentBuilder::build`] returns.
    entries: HashMap<String, RwLock<Entry>>,

    /// Dirty set for GPU uniform buffer.
    ///
    /// Keys whose values have changed since the last call to
    /// [`clear_dirty`](Self::clear_dirty). Uses a [`HashSet`] so that
    /// repeated mutations of the same key do not cause unbounded growth.
    dirty_keys: RwLock<HashSet<String>>,
}

// Verify the compiler agrees Environment is Send + Sync.
// If a non-Send/Sync field is ever added, this will fail at compile time.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Environment>();
};

impl Environment {
    /// Constructs an [`Environment`] from a pre-validated schema.
    ///
    /// Called exclusively by [`EnvironmentBuilder::build`]; not part of the
    /// public API because callers should go through the builder.
    pub(super) fn from_schema(
        schema: Vec<(String, Box<dyn Any + Send + Sync>, TypeId, &'static str)>,
    ) -> Self {
        let mut entries = HashMap::with_capacity(schema.len());
        for (key, value, type_id, type_name) in schema {
            entries.insert(
                key,
                RwLock::new(Entry { value, type_id, type_name }),
            );
        }
        Self {
            entries,
            dirty_keys: RwLock::new(HashSet::new()),
        }
    }

    /// Returns the number of registered parameters.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no parameters have been registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if a parameter with the given key exists.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Reads a typed value from the environment.
    ///
    /// # Errors
    ///
    /// - [`EnvironmentError::KeyNotFound`] — the key was never registered.
    /// - [`EnvironmentError::TypeMismatch`] — the key exists but was registered
    ///   with a different type.
    ///
    /// # Panics
    ///
    /// Panics if the internal [`RwLock`] is poisoned (a thread panicked while
    /// holding a write lock on this entry).
    pub fn get<T: Any + Clone + Send + Sync>(&self, key: &str) -> EnvironmentResult<T> {
        let entry_lock = self.entries.get(key).ok_or_else(|| {
            EnvironmentError::KeyNotFound(key.to_owned())
        })?;

        let entry = entry_lock.read().expect("environment entry lock poisoned");

        let requested_type_id = TypeId::of::<T>();
        if entry.type_id != requested_type_id {
            return Err(EnvironmentError::TypeMismatch {
                key: key.to_owned(),
                expected: entry.type_name,
                actual: std::any::type_name::<T>(),
            });
        }

        // SAFETY: we just verified TypeId equality.
        let value = entry
            .value
            .downcast_ref::<T>()
            .expect("TypeId matched but downcast failed — this is a bug");

        Ok(value.clone())
    }

    /// Writes a typed value to the environment and marks the key dirty for
    /// GPU uniform upload.
    ///
    /// # Errors
    ///
    /// - [`EnvironmentError::KeyNotFound`] — the key was never registered.
    /// - [`EnvironmentError::TypeMismatch`] — the key exists but was registered
    ///   with a different type.
    ///
    /// # Panics
    ///
    /// Panics if any internal [`RwLock`] is poisoned.
    pub fn set<T: Any + Clone + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> EnvironmentResult<()> {
        let entry_lock = self.entries.get(key).ok_or_else(|| {
            EnvironmentError::KeyNotFound(key.to_owned())
        })?;

        {
            let mut entry = entry_lock.write().expect("environment entry lock poisoned");

            let requested_type_id = TypeId::of::<T>();
            if entry.type_id != requested_type_id {
                return Err(EnvironmentError::TypeMismatch {
                    key: key.to_owned(),
                    expected: entry.type_name,
                    actual: std::any::type_name::<T>(),
                });
            }

            // SAFETY: TypeId equality confirmed above.
            *entry
                .value
                .downcast_mut::<T>()
                .expect("TypeId matched but downcast_mut failed — this is a bug") = value;
        }

        // Mark dirty outside the entry lock to avoid holding both locks
        // simultaneously. Only allocate a `String` if the key is not already
        // in the dirty set.
        {
            let mut dirty = self.dirty_keys
                .write()
                .expect("dirty_keys lock poisoned");
            if !dirty.contains(key) {
                dirty.insert(key.to_owned());
            }
        }

        Ok(())
    }

    /// Returns `true` if any of the supplied keys appear in the dirty set.
    ///
    /// Unlike [`dirty_keys`](Self::dirty_keys), this does not clone the set —
    /// it performs a read-locked membership check with zero allocation.
    /// Preferred by the GPU uniform upload path for per-frame dirty checks.
    #[inline]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn has_any_dirty<'a>(
        &self,
        mut keys: impl Iterator<Item = &'a str>,
    ) -> bool {
        let dirty = self.dirty_keys
            .read()
            .expect("dirty_keys lock poisoned");
        keys.any(|k| dirty.contains(k))
    }

    /// Returns a snapshot of keys that have been mutated since the last
    /// [`clear_dirty`](Self::clear_dirty) call.
    ///
    /// This clones the entire dirty set. Prefer [`has_any_dirty`](Self::has_any_dirty)
    /// when only a membership check is needed.
    #[inline]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn dirty_keys(&self) -> HashSet<String> {
        self.dirty_keys
            .read()
            .expect("dirty_keys lock poisoned")
            .clone()
    }

    /// Clears the dirty-key set.
    ///
    /// Should be called by the GPU upload path after all pending uniform
    /// buffers have been uploaded.
    #[inline]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn clear_dirty(&self) {
        self.dirty_keys
            .write()
            .expect("dirty_keys lock poisoned")
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;
    use crate::environment::builder::EnvironmentBuilder;

    fn build_env() -> Arc<Environment> {
        EnvironmentBuilder::new()
            .register::<f32>("interest_rate", 0.05)
            .register::<u32>("world_width", 100)
            .register::<bool>("verbose", false)
            .build()
    }

    #[test]
    fn get_registered_value() {
        let env = build_env();
        let v: f32 = env.get("interest_rate").unwrap();
        assert!((v - 0.05f32).abs() < f32::EPSILON);
    }

    #[test]
    fn get_key_not_found() {
        let env = build_env();
        let err = env.get::<f32>("missing_key").unwrap_err();
        assert!(matches!(err, EnvironmentError::KeyNotFound(_)));
    }

    #[test]
    fn get_type_mismatch() {
        let env = build_env();
        // interest_rate was registered as f32; request f64.
        let err = env.get::<f64>("interest_rate").unwrap_err();
        assert!(matches!(err, EnvironmentError::TypeMismatch { .. }));
    }

    #[test]
    fn set_updates_value() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        let v: f32 = env.get("interest_rate").unwrap();
        assert!((v - 0.10f32).abs() < f32::EPSILON);
    }

    #[test]
    fn set_marks_key_dirty() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        let dirty = env.dirty_keys();
        assert!(dirty.contains("interest_rate"));
    }

    #[test]
    fn clear_dirty_empties_list() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.clear_dirty();
        assert!(env.dirty_keys().is_empty());
    }

    #[test]
    fn set_type_mismatch_returns_error() {
        let env = build_env();
        let err = env.set::<f64>("interest_rate", 0.10f64).unwrap_err();
        assert!(matches!(err, EnvironmentError::TypeMismatch { .. }));
    }

    #[test]
    fn set_missing_key_returns_error() {
        let env = build_env();
        let err = env.set::<f32>("nonexistent", 1.0).unwrap_err();
        assert!(matches!(err, EnvironmentError::KeyNotFound(_)));
    }

    #[test]
    fn contains_key() {
        let env = build_env();
        assert!(env.contains_key("world_width"));
        assert!(!env.contains_key("planet_radius"));
    }

    #[test]
    fn bool_roundtrip() {
        let env = build_env();
        let v: bool = env.get("verbose").unwrap();
        assert!(!v);
        env.set("verbose", true).unwrap();
        assert!(env.get::<bool>("verbose").unwrap());
    }

    #[test]
    fn len_and_is_empty() {
        let env = build_env();
        assert_eq!(env.len(), 3);
        assert!(!env.is_empty());

        let empty = EnvironmentBuilder::new().build();
        assert!(empty.is_empty());
    }

    #[test]
    fn dirty_keys_are_deduplicated() {
        let env = build_env();
        // Set the same key 100 times.
        for _ in 0..100 {
            env.set::<f32>("interest_rate", 0.10).unwrap();
        }
        let dirty = env.dirty_keys();
        // HashSet guarantees "interest_rate" appears at most once.
        assert_eq!(dirty.len(), 1);
        assert!(dirty.contains("interest_rate"));
    }

    #[test]
    fn dirty_tracks_multiple_distinct_keys() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();
        let dirty = env.dirty_keys();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains("interest_rate"));
        assert!(dirty.contains("world_width"));
    }

    #[test]
    fn has_any_dirty_returns_true_for_dirty_key() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        assert!(env.has_any_dirty(["interest_rate"].iter().copied()));
    }

    #[test]
    fn has_any_dirty_returns_false_for_clean_keys() {
        let env = build_env();
        assert!(!env.has_any_dirty(["interest_rate"].iter().copied()));
    }

    #[test]
    fn has_any_dirty_ignores_untracked_keys() {
        let env = build_env();
        env.set::<u32>("world_width", 200).unwrap();
        // Only "world_width" is dirty; "interest_rate" is not.
        assert!(!env.has_any_dirty(["interest_rate"].iter().copied()));
        assert!(env.has_any_dirty(["interest_rate", "world_width"].iter().copied()));
    }

    #[test]
    fn concurrent_reads_do_not_block() {
        use std::thread;
        let env = build_env();
        let env2 = Arc::clone(&env);

        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = env2.get::<f32>("interest_rate").unwrap();
            }
        });

        for _ in 0..1000 {
            let _ = env.get::<u32>("world_width").unwrap();
        }

        handle.join().unwrap();
    }
}
