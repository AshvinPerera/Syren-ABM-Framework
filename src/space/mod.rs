//! First-class agent spaces: discrete grids and continuous 2-D space.
//!
//! # Overview
//!
//! ABM models constantly ask "who is near me?" over **agents** (the spatial
//! *message* grid answers it over messages). This module provides two
//! boundary resources that maintain a per-tick spatial index over agents:
//!
//! - [`ContinuousSpace2D`] - agents at real-valued positions, radius and
//!   k-nearest queries over a uniform cell grid;
//! - [`GridSpace2D`] - agents on discrete cells, occupancy queries,
//!   Moore/von-Neumann neighbourhoods, and [`GridClaims`] for deterministic
//!   parallel movement conflict resolution.
//!
//! Both support bounded and toroidal topologies via [`GridGeometry`].
//!
//! # Data flow (the messaging design language)
//!
//! A **reindex system** (auto-registered by
//! `ModelBuilder::register_continuous_space` / `register_grid_space`, or
//! hand-built at the ECS level) iterates the position component and stages
//! `(entity, position)` into per-worker buffers, declaring `produces` on the
//! space's channel. At the next boundary, `finalise` counting-sorts the
//! staged entries into a CSR index. Systems that query the space declare
//! `consumes` on the channel and therefore always observe a **complete,
//! frozen snapshot of positions as of the reindex stage** - the same mental
//! model as per-tick messages.
//!
//! # Determinism
//!
//! Which worker stages which agent depends on Rayon work stealing, so after
//! the counting sort each cell's occupants are sorted by entity id (skippable
//! via [`ContinuousSpace2D::with_determinism`] /
//! [`GridSpace2D::with_determinism`]). Query iteration order and
//! [`GridClaims`] winners are therefore identical for a fixed seed at any
//! thread count.

mod geometry;

/// Continuous 2-D space with radius and k-nearest agent queries.
pub mod continuous;
/// Discrete cell grid with occupancy queries and movement claims.
pub mod grid;

pub use continuous::ContinuousSpace2D;
pub use geometry::GridGeometry;
pub use grid::{GridClaims, GridSpace2D};

use crate::engine::types::{BoundaryID, ChannelID};

/// Read access to an agent's continuous position, implemented by the
/// position component a [`ContinuousSpace2D`] indexes.
pub trait SpacePosition {
    /// World-space coordinates of the agent.
    fn xy(&self) -> (f32, f32);
}

/// Read access to an agent's grid cell, implemented by the position
/// component a [`GridSpace2D`] indexes.
pub trait GridPosition {
    /// `(column, row)` cell of the agent.
    fn cell(&self) -> (u32, u32);
}

/// Identifiers of a registered space: the boundary resource id (for
/// [`ECSReference::boundary`](crate::ECSReference::boundary)) and the
/// scheduler channel its reindex system produces (declare `consumes` on
/// every system that queries the space).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpaceHandle {
    /// Boundary id for `ecs.boundary::<ContinuousSpace2D>(..)` /
    /// `ecs.boundary::<GridSpace2D>(..)`.
    pub boundary_id: BoundaryID,
    /// Channel produced by the reindex system.
    pub channel_id: ChannelID,
}

impl SpaceHandle {
    /// The scheduler channel querying systems must consume.
    #[inline]
    pub fn channel(&self) -> ChannelID {
        self.channel_id
    }
}

/// Errors raised by space configuration and registration.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceError {
    /// Geometry extents or cell size were non-positive or non-finite.
    InvalidGeometry {
        /// Configured width.
        width: f32,
        /// Configured height.
        height: f32,
        /// Configured cell size.
        cell_size: f32,
    },
}

impl std::fmt::Display for SpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpaceError::InvalidGeometry {
                width,
                height,
                cell_size,
            } => write!(
                f,
                "invalid space geometry: width {width}, height {height}, cell size {cell_size} (all must be positive and finite)"
            ),
        }
    }
}

impl std::error::Error for SpaceError {}
