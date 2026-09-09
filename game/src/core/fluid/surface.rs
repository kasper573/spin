//! The water's isosurface, extracted on the GPU into vertex and index buffers that a material
//! can draw straight from; see `surface.wgsl`. The extraction grid lives in the vessel's own
//! frame and only the blocks of it the water touches exist, so the surface costs the same
//! however large the vessel is and holds still on water at rest in it.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};

use super::Resolution;

/// Cells per block edge; a block has this many cubed cells and one more cubed corners.
pub const BLOCK: usize = 4;
pub const CELLS_PER_BLOCK: usize = BLOCK * BLOCK * BLOCK;
pub const CORNERS_PER_BLOCK: usize = (BLOCK + 1) * (BLOCK + 1) * (BLOCK + 1);
/// Blocks the water may touch at once, and the slots of the table that finds them.
pub const MAX_BLOCKS: usize = 32768;
pub const TABLE_SLOTS: usize = 65536;
pub const MAX_VERTICES: usize = 200_000;
pub const MAX_INDICES: usize = 600_000;
const ISO: f32 = 0.9;

#[derive(Resource, Clone, Default, PartialEq, ExtractResource, ShaderType)]
pub struct SurfaceParams {
    pub cell: f32,
    pub inv_r2: f32,
    pub iso: f32,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_blocks: u32,
    pub table_mask: u32,
    pub pad: u32,
}

impl SurfaceParams {
    pub fn new(resolution: Resolution) -> Self {
        let spacing = resolution.spacing.0;
        let splat_radius = 1.6 * spacing;
        SurfaceParams {
            cell: 0.8 * spacing,
            inv_r2: 1.0 / (splat_radius * splat_radius),
            iso: ISO,
            max_vertices: MAX_VERTICES as u32,
            max_indices: MAX_INDICES as u32,
            max_blocks: MAX_BLOCKS as u32,
            table_mask: TABLE_SLOTS as u32 - 1,
            pad: 0,
        }
    }
}

#[derive(Clone)]
pub struct SurfaceBuffers {
    /// Density and foam at every corner of every block.
    pub corners: Handle<ShaderBuffer>,
    pub cell_vertex: Handle<ShaderBuffer>,
    /// Two vec4 per vertex: position with foam, normal.
    pub vertices: Handle<ShaderBuffer>,
    pub indices: Handle<ShaderBuffer>,
    /// Vertex count, index count, block count.
    pub counters: Handle<ShaderBuffer>,
    /// The keys of the blocks the water touches, hashed, and the block each slot names.
    pub table: Handle<ShaderBuffer>,
    pub table_index: Handle<ShaderBuffer>,
    /// The key of every block in use.
    pub blocks: Handle<ShaderBuffer>,
    /// Workgroup counts for the per-block kernels.
    pub dispatch: Handle<ShaderBuffer>,
}

impl SurfaceBuffers {
    pub fn handles(&self) -> [&Handle<ShaderBuffer>; 9] {
        [
            &self.corners,
            &self.cell_vertex,
            &self.vertices,
            &self.indices,
            &self.counters,
            &self.table,
            &self.table_index,
            &self.blocks,
            &self.dispatch,
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
        corners: make(MAX_BLOCKS * CORNERS_PER_BLOCK * 8),
        cell_vertex: make(MAX_BLOCKS * CELLS_PER_BLOCK * 4),
        vertices: make(MAX_VERTICES * 32),
        indices: make(MAX_INDICES * 4),
        counters: make(16),
        table: make(TABLE_SLOTS * 4),
        table_index: make(TABLE_SLOTS * 4),
        blocks: make(MAX_BLOCKS * 4),
        dispatch: make(16),
    }
}
