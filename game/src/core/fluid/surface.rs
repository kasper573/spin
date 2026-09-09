//! The water's isosurface, extracted on the GPU into vertex and index buffers that a material
//! can draw straight from; see `surface.wgsl`.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};

use super::{Grid, PARTICLE_SPACING};

const CELL: f32 = 0.8 * PARTICLE_SPACING;
const SPLAT_RADIUS: f32 = 1.6 * PARTICLE_SPACING;
const ISO: f32 = 0.9;
pub const MAX_VERTICES: usize = 200_000;
pub const MAX_INDICES: usize = 600_000;

/// The grid of `CELL`-sized cubes the surface is extracted on over the box reaching `extent`
/// from the origin each way, padded so splats near the walls stay inside it.
pub fn grid(extent: [f32; 3]) -> Grid {
    let pad = SPLAT_RADIUS + CELL;
    Grid::around(extent.map(|v| v + pad), CELL)
}

#[derive(Resource, Clone, Default, PartialEq, ExtractResource, ShaderType)]
pub struct SurfaceParams {
    pub origin: Vec4,
    pub dims: IVec4,
    pub iso: f32,
    pub inv_r2: f32,
    pub max_vertices: u32,
    pub max_indices: u32,
}

impl SurfaceParams {
    pub fn new(grid: &Grid) -> Self {
        let corners = (grid.dims[0] + 1) * (grid.dims[1] + 1) * (grid.dims[2] + 1);
        SurfaceParams {
            origin: Vec4::new(grid.min[0], grid.min[1], grid.min[2], grid.cell),
            dims: IVec4::new(grid.dims[0], grid.dims[1], grid.dims[2], corners),
            iso: ISO,
            inv_r2: 1.0 / (SPLAT_RADIUS * SPLAT_RADIUS),
            max_vertices: MAX_VERTICES as u32,
            max_indices: MAX_INDICES as u32,
        }
    }

    pub fn corners(&self) -> u32 {
        self.dims.w as u32
    }

    pub fn cells(&self) -> u32 {
        (self.dims.x * self.dims.y * self.dims.z) as u32
    }
}

#[derive(Clone)]
pub struct SurfaceBuffers {
    pub corners: Handle<ShaderBuffer>,
    pub cell_vertex: Handle<ShaderBuffer>,
    /// Two vec4 per vertex: position with foam, normal.
    pub vertices: Handle<ShaderBuffer>,
    pub indices: Handle<ShaderBuffer>,
    /// Vertex count, then index count.
    pub counters: Handle<ShaderBuffer>,
}

impl SurfaceBuffers {
    pub fn handles(&self) -> [&Handle<ShaderBuffer>; 5] {
        [
            &self.corners,
            &self.cell_vertex,
            &self.vertices,
            &self.indices,
            &self.counters,
        ]
    }

    pub fn gpu(&self, assets: &RenderAssets<GpuShaderBuffer>) -> Option<Vec<Buffer>> {
        self.handles()
            .iter()
            .map(|h| assets.get(*h).map(|b| b.buffer.clone()))
            .collect()
    }
}

pub fn create_buffers(
    make: &mut impl FnMut(usize) -> Handle<ShaderBuffer>,
    grid: &Grid,
) -> SurfaceBuffers {
    let corners = ((grid.dims[0] + 1) * (grid.dims[1] + 1) * (grid.dims[2] + 1)) as usize;
    SurfaceBuffers {
        corners: make(corners * 8),
        cell_vertex: make(grid.cells() * 4),
        vertices: make(MAX_VERTICES * 32),
        indices: make(MAX_INDICES * 4),
        counters: make(16),
    }
}
