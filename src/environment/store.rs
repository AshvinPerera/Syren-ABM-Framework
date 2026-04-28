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
//! Every successful call to [`set`](Environment::set) inserts the entry's
//! [`ChannelID`](crate::engine::types::ChannelID) into a shared
//! `dirty_channels` set. Using a [`HashSet`] ensures repeated mutations of the
//! same key do not cause unbounded growth. The dirty set is consumed by
//! [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary) at the end of
//! each tick to coordinate GPU uniform uploads and scheduler dependencies.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::engine::types::ChannelID;

use super::error::{EnvironmentError, EnvironmentResult};
use super::handle::EnvKey;

// -----------------------------------------------------------------------------
// Internal storage entry
// -----------------------------------------------------------------------------

/// One registered parameter slot.
///
/// The `channel_id` is immutable from registration onwards and lives outside
/// the [`RwLock`] so that [`channel_of`](Environment::channel_of) and
/// [`env_key`](Environment::env_key) never block on writers of the value.
/// Only the mutable parts (the boxed value and its runtime type metadata) live
/// inside the lock.
struct EntrySlot {
    /// Scheduling channel assigned to this key at build time. Immutable.
    channel_id: ChannelID,
    /// Mutable value plus its runtime type metadata, behind an `RwLock` so
    /// concurrent readers do not block one another.
    value: RwLock<EntryValue>,
}

/// Mutable contents of an [`EntrySlot`].
struct EntryValue {
    /// The type-erased value.
    value: Box<dyn Any + Send + Sync>,
    /// The concrete [`TypeId`] of the value, used for runtime type checking.
    type_id: TypeId,
    /// Human-readable type name, used in error messages.
    type_name: &'static str,
}

// -----------------------------------------------------------------------------
// Environment
// -----------------------------------------------------------------------------

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
/// Every successful call to [`set`](Self::set) inserts the key's
/// [`ChannelID`] into a shared `dirty_channels` set. This set is drained by
/// [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary) each tick to
/// coordinate GPU uniform uploads and system scheduling.
pub struct Environment {
    /// Per-entry storage, keyed by parameter name.
    ///
    /// Schema is frozen at construction - no new entries are ever inserted after
    /// [`EnvironmentBuilder::build`] returns. Each [`EntrySlot`] holds an
    /// immutable [`ChannelID`] outside any lock, so name-to-channel queries
    /// never contend with value writers.
    entries: HashMap<String, EntrySlot>,

    /// Channel IDs whose values have changed since the last tick.
    ///
    /// Uses a [`HashSet`] so repeated mutations of the same key do not cause
    /// unbounded growth.
    dirty_channels: RwLock<HashSet<ChannelID>>,
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
    /// `channel_ids` must be the same length as `schema` and in the same order.
    /// Each element is the [`ChannelID`] assigned to the corresponding schema
    /// entry by the [`ChannelAllocator`](crate::engine::channel_allocator::ChannelAllocator)
    /// used during the build.
    ///
    /// Called exclusively by
    /// [`EnvironmentBuilder::build`](super::builder::EnvironmentBuilder::build)
    /// and
    /// [`EnvironmentBuilder::build_with_allocator`](super::builder::EnvironmentBuilder::build_with_allocator);
    /// not part of the public API.
    pub(super) fn from_schema(
        schema: Vec<(String, Box<dyn Any + Send + Sync>, TypeId, &'static str)>,
        channel_ids: Vec<ChannelID>,
    ) -> Self {
        debug_assert_eq!(
            schema.len(),
            channel_ids.len(),
            "schema and channel_ids must be the same length"
        );

        let mut entries = HashMap::with_capacity(schema.len());
        for ((key, value, type_id, type_name), channel_id) in schema.into_iter().zip(channel_ids) {
            entries.insert(
                key,
                EntrySlot {
                    channel_id,
                    value: RwLock::new(EntryValue {
                        value,
                        type_id,
                        type_name,
                    }),
                },
            );
        }
        Self {
            entries,
            dirty_channels: RwLock::new(HashSet::new()),
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

    /// Returns the [`ChannelID`] assigned to `key`, or `None` if the key is
    /// not registered.
    ///
    /// Use this to obtain the ID needed for
    /// [`AccessSets::produces`](crate::AccessSets::produces) /
    /// [`AccessSets::consumes`](crate::AccessSets::consumes) declarations, or
    /// to construct an [`EnvKey`] for a key that was registered with a
    /// runtime-owned name.
    ///
    /// Lock-free: the channel ID lives outside the per-entry [`RwLock`], so
    /// this never blocks on a concurrent value writer.
    #[inline]
    pub fn channel_of(&self, key: &str) -> Option<ChannelID> {
        self.entries.get(key).map(|slot| slot.channel_id)
    }

    /// Returns every [`ChannelID`] owned by this environment, sorted in
    /// ascending order.
    ///
    /// Used by [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary)
    /// to declare its set of owned channels to the engine so that
    /// `finalise` is dispatched only when one of them appears in a boundary
    /// stage.
    pub fn all_channel_ids(&self) -> Vec<ChannelID> {
        let mut ids: Vec<ChannelID> = self.entries.values().map(|s| s.channel_id).collect();
        ids.sort_unstable();
        ids
    }

    /// Returns a typed handle for `key`, or `None` if the key is not
    /// registered.
    ///
    /// `key` must be a `&'static str` (a string literal) because [`EnvKey`]
    /// stores it as `&'static str` for zero-allocation access inside
    /// [`set`](Self::set) / [`get`](Self::get) call sites.
    ///
    /// The type parameter `T` is a compile-time marker only. It does **not**
    /// verify that `T` matches the registered type of the key; that check
    /// happens at runtime inside [`get`](Self::get) and [`set`](Self::set).
    ///
    /// Returns `None` if no key with that name was registered.
    #[inline]
    pub fn env_key<T: Any + Clone + Send + Sync>(&self, key: &'static str) -> Option<EnvKey<T>> {
        let channel_id = self.channel_of(key)?;
        Some(EnvKey::new(key, channel_id))
    }

    /// Reads a typed value from the environment.
    ///
    /// # Errors
    ///
    /// - [`EnvironmentError::KeyNotFound`] - the key was never registered.
    /// - [`EnvironmentError::TypeMismatch`] - the key exists but was registered
    ///   with a different type.
    ///
    /// # Panics
    ///
    /// Panics if the internal [`RwLock`] is poisoned (a thread panicked while
    /// holding a write lock on this entry).
    pub fn get<T: Any + Clone + Send + Sync>(&self, key: &str) -> EnvironmentResult<T> {
        let slot = self
            .entries
            .get(key)
            .ok_or_else(|| EnvironmentError::KeyNotFound(key.to_owned()))?;

        let entry = slot
            .value
            .read()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment entry",
            })?;

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
            .expect("TypeId matched but downcast failed - this is a bug");

        Ok(value.clone())
    }

    /// Writes a typed value to the environment and marks the key's channel
    /// dirty.
    ///
    /// The [`ChannelID`] inserted into the dirty set is the one assigned at
    /// build time. This allows [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary)
    /// to correlate dirty entries with scheduler channels without any string
    /// comparisons.
    ///
    /// # Errors
    ///
    /// - [`EnvironmentError::KeyNotFound`] - the key was never registered.
    /// - [`EnvironmentError::TypeMismatch`] - the key exists but was registered
    ///   with a different type.
    ///
    /// # Panics
    ///
    /// Panics if any internal [`RwLock`] is poisoned.
    pub fn set<T: Any + Clone + Send + Sync>(&self, key: &str, value: T) -> EnvironmentResult<()> {
        let slot = self
            .entries
            .get(key)
            .ok_or_else(|| EnvironmentError::KeyNotFound(key.to_owned()))?;

        // Channel ID is immutable and lives outside the lock - read it
        // directly without acquiring the inner RwLock.
        let channel_id = slot.channel_id;

        // Take the inner write lock only for the value update.
        {
            let mut entry = slot
                .value
                .write()
                .map_err(|_| EnvironmentError::LockPoisoned {
                    what: "environment entry",
                })?;

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
                .expect("TypeId matched but downcast_mut failed - this is a bug") = value;
        }

        // Mark the channel dirty after releasing the entry lock so the two
        // are never held simultaneously.
        self.dirty_channels
            .write()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment dirty channels",
            })?
            .insert(channel_id);

        Ok(())
    }

    /// Returns `true` if any of the supplied [`ChannelID`]s appear in the
    /// dirty set.
    ///
    /// Performs a read-locked membership check with zero allocation. Preferred
    /// over [`dirty_channel_ids`](Self::dirty_channel_ids) when only a boolean
    /// answer is needed.
    #[inline]
    pub(crate) fn has_any_dirty_channels(
        &self,
        channels: impl Iterator<Item = ChannelID>,
    ) -> EnvironmentResult<bool> {
        let dirty = self
            .dirty_channels
            .read()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment dirty channels",
            })?;
        Ok(channels.into_iter().any(|id| dirty.contains(&id)))
    }

    /// Returns `true` if `id` is currently in the dirty set.
    ///
    /// Single-channel membership probe used by
    /// [`EnvironmentBoundary::finalise`](super::boundary::EnvironmentBoundary)
    /// to compute the precise intersection of `channels  intersection  env.dirty  intersection 
    /// uniform.owned` without constructing intermediate iterators.
    ///
    /// Currently only the GPU uniform-buffer integration needs this primitive,
    /// so the method is gated behind the `gpu` feature to keep the CPU-only
    /// build free of dead-code warnings. Drop the gate if a CPU caller is
    /// added.
    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn is_channel_dirty(&self, id: ChannelID) -> EnvironmentResult<bool> {
        Ok(self
            .dirty_channels
            .read()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment dirty channels",
            })?
            .contains(&id))
    }

    /// Returns a snapshot of [`ChannelID`]s whose values have been mutated
    /// since the last [`clear_dirty`](Self::clear_dirty) or
    /// [`clear_dirty_for_channels`](Self::clear_dirty_for_channels) call.
    ///
    /// Clones the entire set. Prefer [`has_any_dirty_channels`](Self::has_any_dirty_channels)
    /// when only a membership check is needed.
    #[inline]
    pub(crate) fn dirty_channel_ids(&self) -> EnvironmentResult<HashSet<ChannelID>> {
        Ok(self
            .dirty_channels
            .read()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment dirty channels",
            })?
            .clone())
    }

    /// Removes the specified [`ChannelID`]s from the dirty set.
    ///
    /// Called after a consumer (e.g. a GPU uniform buffer) has processed only
    /// the channels it owns, leaving the remaining channels dirty for other
    /// consumers.
    #[inline]
    pub(crate) fn clear_dirty_for_channels(&self, channels: &[ChannelID]) -> EnvironmentResult<()> {
        let mut dirty =
            self.dirty_channels
                .write()
                .map_err(|_| EnvironmentError::LockPoisoned {
                    what: "environment dirty channels",
                })?;
        for id in channels {
            dirty.remove(id);
        }
        Ok(())
    }

    /// Clears the entire dirty-channel set.
    ///
    /// Called by [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary)
    /// at the end of each tick after all consumers have processed their
    /// pending dirty channels.
    #[inline]
    pub(crate) fn clear_dirty(&self) -> EnvironmentResult<()> {
        self.dirty_channels
            .write()
            .map_err(|_| EnvironmentError::LockPoisoned {
                what: "environment dirty channels",
            })?
            .clear();
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn poison_dirty_channels_for_test(&self) {
        let _guard = self.dirty_channels.write().unwrap();
        panic!("poison environment dirty-channel lock");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::builder::EnvironmentBuilder;
    use std::sync::Arc;

    fn build_env() -> Arc<Environment> {
        EnvironmentBuilder::new()
            .register::<f32>("interest_rate", 0.05)
            .unwrap()
            .register::<u32>("world_width", 100)
            .unwrap()
            .register::<bool>("verbose", false)
            .unwrap()
            .build()
            .unwrap()
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
    fn set_marks_channel_dirty() {
        let env = build_env();
        let id = env.channel_of("interest_rate").unwrap();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        let dirty = env.dirty_channel_ids().unwrap();
        assert!(dirty.contains(&id));
    }

    #[test]
    fn clear_dirty_empties_set() {
        let env = build_env();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.clear_dirty().unwrap();
        assert!(env.dirty_channel_ids().unwrap().is_empty());
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

        let empty = EnvironmentBuilder::new().build().unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn dirty_channels_are_deduplicated() {
        let env = build_env();
        let id = env.channel_of("interest_rate").unwrap();
        // Set the same key 100 times.
        for _ in 0..100 {
            env.set::<f32>("interest_rate", 0.10).unwrap();
        }
        let dirty = env.dirty_channel_ids().unwrap();
        // HashSet guarantees the channel appears at most once.
        assert_eq!(dirty.len(), 1);
        assert!(dirty.contains(&id));
    }

    #[test]
    fn dirty_tracks_multiple_distinct_channels() {
        let env = build_env();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();
        let dirty = env.dirty_channel_ids().unwrap();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&id_rate));
        assert!(dirty.contains(&id_width));
    }

    #[test]
    fn has_any_dirty_channels_returns_true_for_dirty() {
        let env = build_env();
        let id = env.channel_of("interest_rate").unwrap();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        assert!(env.has_any_dirty_channels([id].into_iter()).unwrap());
    }

    #[test]
    fn has_any_dirty_channels_returns_false_for_clean() {
        let env = build_env();
        let id = env.channel_of("interest_rate").unwrap();
        assert!(!env.has_any_dirty_channels([id].into_iter()).unwrap());
    }

    #[test]
    fn has_any_dirty_channels_ignores_unrelated() {
        let env = build_env();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();
        env.set::<u32>("world_width", 200).unwrap();
        assert!(!env.has_any_dirty_channels([id_rate].into_iter()).unwrap());
        assert!(env
            .has_any_dirty_channels([id_rate, id_width].into_iter())
            .unwrap());
    }

    #[test]
    fn clear_dirty_for_channels_is_selective() {
        let env = build_env();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();
        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();
        env.clear_dirty_for_channels(&[id_rate]).unwrap();
        let dirty = env.dirty_channel_ids().unwrap();
        assert!(!dirty.contains(&id_rate));
        assert!(dirty.contains(&id_width));
    }

    #[test]
    fn channel_of_returns_none_for_unknown_key() {
        let env = build_env();
        assert!(env.channel_of("nonexistent").is_none());
    }

    #[test]
    fn channel_of_returns_distinct_ids_for_distinct_keys() {
        let env = build_env();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();
        let id_verbose = env.channel_of("verbose").unwrap();
        // All three must be distinct.
        assert_ne!(id_rate, id_width);
        assert_ne!(id_rate, id_verbose);
        assert_ne!(id_width, id_verbose);
    }

    #[test]
    fn env_key_roundtrip() {
        let env = build_env();
        let key = env.env_key::<f32>("interest_rate").unwrap();
        assert_eq!(key.name(), "interest_rate");
        assert_eq!(key.channel_id(), env.channel_of("interest_rate").unwrap());
    }

    #[test]
    fn env_key_returns_none_for_unknown_key() {
        let env = build_env();
        assert!(env.env_key::<f32>("nonexistent").is_none());
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

    /// Verify `channel_of` does not block on a concurrent value writer.
    ///
    /// The channel ID lives outside the per-entry `RwLock`, so even while
    /// another thread holds the value write lock for an extended period
    /// (simulated here by hammering `set` from a background thread),
    /// `channel_of` must continue to return promptly.
    #[test]
    fn channel_of_is_lock_free_against_writers() {
        use std::thread;
        use std::time::{Duration, Instant};

        let env = build_env();
        let env_writer = Arc::clone(&env);

        // Spawn a writer that contends on the value lock for ~50ms.
        let stop_at = Instant::now() + Duration::from_millis(50);
        let handle = thread::spawn(move || {
            while Instant::now() < stop_at {
                env_writer.set::<f32>("interest_rate", 0.07).unwrap();
            }
        });

        // While the writer is running, channel_of calls must stay fast.
        // We run many of them and verify the total is well under the writer's
        // lifetime - proving no per-call serialisation against the writer.
        let start = Instant::now();
        for _ in 0..100_000 {
            let _ = env.channel_of("interest_rate").unwrap();
        }
        let elapsed = start.elapsed();

        handle.join().unwrap();

        // 100k lock-free hashmap lookups should complete well within the
        // writer's 50ms lifetime. Generous bound to keep the test stable
        // under heavy CI load.
        assert!(
            elapsed < Duration::from_millis(500),
            "channel_of appears to serialise against writers: {:?}",
            elapsed
        );
    }

    #[test]
    fn poisoned_entry_lock_returns_structured_error() {
        use std::thread;

        let env = build_env();
        let env_for_thread = Arc::clone(&env);
        let _ = thread::spawn(move || {
            let slot = env_for_thread.entries.get("interest_rate").unwrap();
            let _guard = slot.value.write().unwrap();
            panic!("poison environment entry");
        })
        .join();

        let err = env.get::<f32>("interest_rate").unwrap_err();
        assert!(matches!(
            err,
            EnvironmentError::LockPoisoned {
                what: "environment entry"
            }
        ));
    }

    #[test]
    fn poisoned_dirty_channel_helpers_return_structured_errors() {
        use std::thread;

        let env = build_env();
        let id = env.channel_of("interest_rate").unwrap();
        let env_for_thread = Arc::clone(&env);
        let _ = thread::spawn(move || env_for_thread.poison_dirty_channels_for_test()).join();

        assert!(matches!(
            env.has_any_dirty_channels([id].into_iter()),
            Err(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
        assert!(matches!(
            env.dirty_channel_ids(),
            Err(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
        assert!(matches!(
            env.clear_dirty_for_channels(&[id]),
            Err(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
        assert!(matches!(
            env.clear_dirty(),
            Err(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
    }
}
