//! The water's isosurface, extracted on the GPU into vertex and index buffers that a material
//! can draw straight from; see `surface.wgsl`. The extraction grid lives in the vessel's own
//! frame, in the canonical water's units, and only the blocks of it the water touches are
//! visited, so the surface costs the same however large the vessel is and holds still on water
//! at rest in it. Everything here is sized by the particle cap, so no water, however much or
//! however scattered, outgrows it.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};

use super::resolution::canonical;
use super::{MAX_PARTICLES, Resolution};
use crate::core::math::Vec3d;
use crate::core::units::Metres;

/// A particle's splat reaches two blocks along each axis, so a drop on its own marks eight, and
/// no water marks more per particle.
pub const MAX_BLOCKS: usize = 8 * MAX_PARTICLES;
pub const TABLE_SLOTS: usize = 2 * MAX_BLOCKS;
/// A drop on its own crosses the eight cells round one corner and the six edges leaving it, and
/// no water crosses more per particle: one vertex per crossed cell, one quad per crossed edge.
pub const MAX_VERTICES: usize = 8 * MAX_PARTICLES;
pub const CELL_SLOTS: usize = 2 * MAX_VERTICES;
pub const MAX_INDICES: usize = 6 * 6 * MAX_PARTICLES;
const ISO: f32 = 0.9;
/// The extraction grid's cell, and how far a particle's splat reaches, in spacings.
const CELL: f32 = 0.8 * canonical::SPACING;
const SPLAT_RADIUS: f32 = 1.6 * canonical::SPACING;

#[derive(Resource, Clone, Default, PartialEq, ExtractResource, ShaderType)]
pub struct SurfaceParams {
    pub cell: f32,
    pub inv_r2: f32,
    pub iso: f32,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_blocks: u32,
    pub table_mask: u32,
    pub cell_mask: u32,
    /// The cell the vertices come out relative to.
    pub anchor: IVec3,
}

impl SurfaceParams {
    pub fn new(resolution: Resolution, anchor: Vec3d) -> Self {
        let cell = cell(resolution) as f64;
        SurfaceParams {
            cell: CELL,
            inv_r2: 1.0 / (SPLAT_RADIUS * SPLAT_RADIUS),
            iso: ISO,
            max_vertices: MAX_VERTICES as u32,
            max_indices: MAX_INDICES as u32,
            max_blocks: MAX_BLOCKS as u32,
            table_mask: TABLE_SLOTS as u32 - 1,
            cell_mask: CELL_SLOTS as u32 - 1,
            anchor: IVec3::new(
                (anchor[0] / cell).round() as i32,
                (anchor[1] / cell).round() as i32,
                (anchor[2] / cell).round() as i32,
            ),
        }
    }

    /// The point of the vessel's frame the vertices are relative to, in metres.
    pub fn origin(&self, resolution: Resolution) -> Vec3d {
        let cell = cell(resolution) as f64;
        [
            self.anchor.x as f64 * cell,
            self.anchor.y as f64 * cell,
            self.anchor.z as f64 * cell,
        ]
    }
}

/// How far from the vessel's centre the grid reaches along each axis: a block's key holds ten
/// bits per axis about the centre.
pub fn grid_reach(resolution: Resolution) -> Metres {
    Metres(512.0 * 4.0 * cell(resolution))
}

/// The grid's cell in metres at this resolution.
fn cell(resolution: Resolution) -> f32 {
    CELL * resolution.spacing.0
}

#[derive(Clone)]
pub struct SurfaceBuffers {
    /// Two vec4 per vertex: position with foam, normal with the key of the vertex's cell.
    pub vertices: Handle<ShaderBuffer>,
    pub indices: Handle<ShaderBuffer>,
    /// Vertex count, index count, block count.
    pub counters: Handle<ShaderBuffer>,
    /// The keys of the blocks the water touches, hashed, and the block each slot names.
    pub table: Handle<ShaderBuffer>,
    pub table_index: Handle<ShaderBuffer>,
    /// The key of every block in use.
    pub blocks: Handle<ShaderBuffer>,
    /// Workgroup counts for the kernels run per block and per crossed cell.
    pub dispatch: Handle<ShaderBuffer>,
    /// The keys of the cells the surface crosses, hashed, and each one's vertex and corners.
    pub cell_table: Handle<ShaderBuffer>,
    pub cell_value: Handle<ShaderBuffer>,
}

impl SurfaceBuffers {
    pub fn handles(&self) -> [&Handle<ShaderBuffer>; 9] {
        [
            &self.vertices,
            &self.indices,
            &self.counters,
            &self.table,
            &self.table_index,
            &self.blocks,
            &self.dispatch,
            &self.cell_table,
            &self.cell_value,
        ]
    }

    pub fn gpu(&self, assets: &RenderAssets<GpuShaderBuffer>) -> Option<Vec<Buffer>> {
        self.handles()
            .iter()
            .map(|h| assets.get(*h).map(|b| b.buffer.clone()))
            .collect()
    }
}

pub fn create_buffers(make: &mut impl FnMut(usize) -> Handle<ShaderBuffer>) -> SurfaceBuffers {
    SurfaceBuffers {
        vertices: make(MAX_VERTICES * 32),
        indices: make(MAX_INDICES * 4),
        counters: make(16),
        table: make(TABLE_SLOTS * 4),
        table_index: make(TABLE_SLOTS * 4),
        blocks: make(MAX_BLOCKS * 4),
        dispatch: make(32),
        cell_table: make(CELL_SLOTS * 4),
        cell_value: make(CELL_SLOTS * 4),
    }
}
