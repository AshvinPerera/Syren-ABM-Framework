//! Component signature bitset and iteration helpers.
//!
//! A [`Signature`] is a compact, fixed-size bitset that encodes which components
//! an entity or archetype possesses. Each bit position corresponds to a
//! [`ComponentID`], allowing set operations (union, intersection, subset checks)
//! to be performed with simple bitwise arithmetic across `SIGNATURE_SIZE` `u64`
//! words.
//!
//! # Core types
//!
//! - [`Signature`] — the primary bitset type; supports setting, clearing, and
//!   querying individual component bits, as well as subset checks via
//!   [`contains_all`](Signature::contains_all).
//!
//! # Free functions
//!
//! - [`build_signature`] — constructs a [`Signature`] from a slice of
//!   [`ComponentID`]s.
//! - [`iter_bits_from_words`] — low-level iterator over set bits in a raw word
//!   array, used internally by [`Signature::iterate_over_components`] and
//!   exposed for performance-critical callers that operate on raw words directly.
//!
//! # Capacity
//!
//! The maximum number of distinct components is `SIGNATURE_SIZE * 64`. Passing
//! a [`ComponentID`] at or beyond that limit will panic at the call site.

use std::hash::{Hash, Hasher};

use crate::engine::types::{ComponentID, SIGNATURE_SIZE};

/// Bitset representing a set of components.
///
/// ## Design
/// The internal representation is a fixed-size array of `u64` words covering
/// `COMPONENT_CAP` bits. External code should use the provided methods
/// (`set`, `has`, `clear`, `contains_all`, `iterate_over_components`) rather
/// than manipulating the raw word array directly.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Signature {
    /// Packed component bitset.
    ///
    /// Visible within the crate for performance-critical archetype migration
    /// code that operates on raw words. External users should prefer the
    /// typed accessor methods.
    pub(crate) components: [u64; SIGNATURE_SIZE],
}

impl Default for Signature {
    fn default() -> Self {
        Self {
            components: [0u64; SIGNATURE_SIZE],
        }
    }
}

impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

impl Eq for Signature {}

impl Hash for Signature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.components.hash(state);
    }
}

impl Signature {
    /// Sets the bit corresponding to `component_id`.
    ///
    /// # Panics
    /// Panics if `component_id >= COMPONENT_CAP`.
    #[inline]
    pub fn set(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        assert!(
            index < SIGNATURE_SIZE,
            "component_id {} exceeds COMPONENT_CAP",
            component_id
        );
        self.components[index] |= 1u64 << bits;
    }

    /// Clears the bit corresponding to `component_id`.
    ///
    /// # Panics
    /// Panics if `component_id >= COMPONENT_CAP`.
    #[inline]
    pub fn clear(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        assert!(
            index < SIGNATURE_SIZE,
            "component_id {} exceeds COMPONENT_CAP",
            component_id
        );
        self.components[index] &= !(1u64 << bits);
    }

    /// Returns `true` if `component_id` is present in this signature.
    ///
    /// # Panics
    /// Panics if `component_id >= COMPONENT_CAP`.
    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        assert!(
            index < SIGNATURE_SIZE,
            "component_id {} exceeds COMPONENT_CAP",
            component_id
        );
        (self.components[index] >> bits) & 1 == 1
    }

    /// Returns `true` if all components in `signature` are present.
    #[inline]
    pub fn contains_all(&self, signature: &Signature) -> bool {
        for (component_a, component_b) in self.components.iter().zip(signature.components.iter()) {
            if (component_a & component_b) != *component_b {
                return false;
            }
        }
        true
    }

    /// Iterates over all component IDs set in this signature.
    pub fn iterate_over_components(&self) -> impl Iterator<Item = ComponentID> + '_ {
        iter_bits_from_words(&self.components)
    }

    /// Returns the raw word array backing this signature.
    ///
    /// Prefer using the typed methods (`set`, `has`, `contains_all`, etc.)
    /// unless you need direct word-level access for performance reasons.
    #[inline]
    pub fn as_words(&self) -> &[u64; SIGNATURE_SIZE] {
        &self.components
    }
}

/// Iterates over component IDs set in a raw signature word array.
#[inline]
pub fn iter_bits_from_words<'a>(
    words: &'a [u64; SIGNATURE_SIZE],
) -> impl Iterator<Item = ComponentID> + 'a {
    words.iter().enumerate().flat_map(|(word_index, &word)| {
        let base = word_index * 64;
        let mut bits = word;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let tz = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some((base + tz) as ComponentID)
        })
    })
}

/// Bitwise-ORs every word of `src` into the corresponding word of `dst`.
///
/// Used to accumulate component access sets (read/write signatures) and
/// pending-download masks across the scheduler and GPU dispatch runtime.
#[inline]
pub(crate) fn or_signature_in_place(dst: &mut Signature, src: &Signature) {
    for (d, s) in dst.components.iter_mut().zip(src.components.iter()) {
        *d |= *s;
    }
}
