//! The render-world half of the fluid: its buffers, one compute pipeline per kernel with only the
//! bindings that kernel touches (WebGPU allows few storage buffers per stage), and the frame's
//! dispatch order: thin if due, append, then per substep sort, fly, hand the motion to the grid,
//! let the pressure have its way with it, take it back and couple to the bodies, and finally
//! re-sort and extract the surface.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    Buffer, BufferId, BufferUsages, CachedComputePipelineId, CommandEncoder, ComputePassDescriptor,
    ComputePipelineDescriptor, DynamicUniformBuffer, PipelineCache, ShaderStages, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};

use super::frame::{
    ACCUMULATORS_PER_FRAME, FluidFrame, GpuBodies, Params, STAMP_SLOT, accumulators_of,
};
use super::surface::{self, MAX_MOTES, SurfaceBuffers, SurfaceParams, TABLE_SLOTS};
use super::{
    FluidReady, GRID_SLOTS, MAX_PARTICLES, MAX_SAMPLES, RELAXATIONS, SETTLINGS, TABLE_CELLS,
};
use crate::core::vessel::{VesselBinding, VesselLayout};

/// Threads per workgroup of the particle kernels, and of the scan's.
const WORKGROUP: u32 = 64;
/// The most lattice sites one placement of water may offer, and the threads that weigh them.
pub const SITES: usize = 4096;
const JOIN_THREADS: u32 = 256;
const SCAN_THREADS: usize = 256;
const ACCUMULATORS: usize = STAMP_SLOT + 4;

#[derive(Resource, Clone, ExtractResource)]
pub struct FluidBuffers {
    pub position: Handle<ShaderBuffer>,
    pub velocity: Handle<ShaderBuffer>,
    pub position_sorted: Handle<ShaderBuffer>,
    pub velocity_next: Handle<ShaderBuffer>,
    pub pred_a: Handle<ShaderBuffer>,
    pub pred_b: Handle<ShaderBuffer>,
    pub contact: Handle<ShaderBuffer>,
    pub cell_count: Handle<ShaderBuffer>,
    pub cell_start: Handle<ShaderBuffer>,
    pub slot: Handle<ShaderBuffer>,
    pub pending: Handle<ShaderBuffer>,
    /// The cell key of every particle, in sorted order.
    pub key: Handle<ShaderBuffer>,
    /// The scan's per-run totals.
    pub run_total: Handle<ShaderBuffer>,
    /// Lattice sites on offer to water being placed.
    pub sites: Handle<ShaderBuffer>,
    /// How each particle's velocity changes across it, three vec4 to a particle, and the same
    /// in sorted order.
    pub affine: Handle<ShaderBuffer>,
    pub affine_sorted: Handle<ShaderBuffer>,
    pub living: Handle<ShaderBuffer>,
    /// The grid: the keys of its cells, hashed; the slots in use; how many those are; the
    /// cells; what each has across its faces; and the workgroups that visit them.
    pub grid_keys: Handle<ShaderBuffer>,
    pub grid_list: Handle<ShaderBuffer>,
    pub grid_counters: Handle<ShaderBuffer>,
    pub grid_cells: Handle<ShaderBuffer>,
    pub grid_links: Handle<ShaderBuffer>,
    pub grid_dispatch: Handle<ShaderBuffer>,
    pub samples: Handle<ShaderBuffer>,
    pub boundary: Handle<ShaderBuffer>,
    pub sample_state: Handle<ShaderBuffer>,
    pub accum: Handle<ShaderBuffer>,
    pub surface: SurfaceBuffers,
}

pub fn create_buffers(assets: &mut Assets<ShaderBuffer>) -> FluidBuffers {
    let mut make = |bytes: usize| {
        let mut buffer = ShaderBuffer::with_size(bytes, default());
        buffer.buffer_description.usage |=
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::INDIRECT;
        assets.add(buffer)
    };
    let vec4s = |n: usize| n * 16;
    FluidBuffers {
        position: make(vec4s(MAX_PARTICLES)),
        velocity: make(vec4s(MAX_PARTICLES)),
        position_sorted: make(vec4s(MAX_PARTICLES)),
        velocity_next: make(vec4s(MAX_PARTICLES)),
        pred_a: make(vec4s(MAX_PARTICLES)),
        pred_b: make(vec4s(MAX_PARTICLES)),
        contact: make(vec4s(2 * MAX_PARTICLES)),
        cell_count: make(TABLE_CELLS * 4),
        cell_start: make((TABLE_CELLS + 1) * 4),
        slot: make(vec4s(MAX_PARTICLES)),
        pending: make(vec4s(2 * MAX_PARTICLES)),
        key: make(MAX_PARTICLES * 4),
        run_total: make(TABLE_CELLS.div_ceil(SCAN_THREADS) * 4),
        sites: make(vec4s(SITES)),
        affine: make(vec4s(6 * MAX_PARTICLES)),
        affine_sorted: make(vec4s(6 * MAX_PARTICLES)),
        living: make(4 * 4),
        grid_keys: make(GRID_SLOTS * 4),
        grid_list: make(2 * GRID_SLOTS * 4),
        grid_counters: make(4 * 4),
        grid_cells: make(vec4s(10 * GRID_SLOTS)),
        grid_links: make(GRID_SLOTS * 48),
        grid_dispatch: make(8 * 4),
        samples: make(vec4s(MAX_SAMPLES)),
        boundary: make(vec4s(2 * MAX_SAMPLES)),
        sample_state: make(vec4s(2 * MAX_SAMPLES)),
        accum: make(ACCUMULATORS * 4),
        surface: surface::create_buffers(&mut make),
    }
}

/// The water's kernels for the frame, which anything reading their results the same frame runs
/// after.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FluidStep;

/// The frame's uniforms and bind groups made, which anything binding [`FluidHulls`] the same
/// frame is prepared after.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FluidPrepare;

/// The bodies' hulls as the frame's steps have them, for water that is not this water to say
/// what it does to them where this water says it: the uniform the bodies are in and where in it
/// each step's are, how many samples the hulls have, the first of the frame's accumulators, and
/// what the water's drag and density are in the units the accumulators are in. No samples when
/// the frame reports no coupling.
#[derive(Resource)]
pub struct FluidHulls {
    pub bodies: Buffer,
    pub offsets: Vec<u32>,
    pub samples: u32,
    pub accumulators: u32,
    pub body_drag: f32,
    pub rest_density: f32,
}

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            prepare
                .in_set(RenderSystems::PrepareBindGroups)
                .in_set(FluidPrepare),
        )
        .add_systems(
            RenderGraph,
            dispatch.in_set(FluidStep).before(camera_driver),
        );
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Count,
    ScanRuns,
    ScanTotals,
    AddOffsets,
    Scatter,
    ScatterAffine,
    Bury,
    Tally,
    Thin,
    Predict,
    Inject,
    Join,
    Occupy,
    Muster,
    Gather,
    Walls,
    Level,
    MusterShores,
    LevelShores,
    Wetted,
    SmoothOnce,
    SmoothTwice,
    SmoothThrice,
    TellOnOnce,
    TellOnTwice,
    Slip,
    System,
    SettleRed,
    SettleBlack,
    RelaxRed,
    RelaxBlack,
    Project,
    Transfer,
    Weather,
    Place,
    Buoyancy,
    Drag,
    Shed,
    Fly,
    Mark,
    List,
    PrepareDispatch,
    Extract,
    PrepareQuads,
    Quads,
    Polish,
    PolishBack,
}

const PARTICLES: &str = "embedded://game/core/fluid/shaders/particles.wgsl";
const GRID: &str = "embedded://game/core/fluid/shaders/grid.wgsl";
const SORT: &str = "embedded://game/core/fluid/shaders/sort.wgsl";
const BODIES: &str = "embedded://game/core/fluid/shaders/bodies.wgsl";
const SURFACE: &str = "embedded://game/core/fluid/shaders/surface.wgsl";
const SPRAY: &str = "embedded://game/core/fluid/shaders/spray.wgsl";

/// Which bindings of each group a kernel uses. Binding 0 of groups 0, 2 and 3 is a uniform, the
/// rest are storage buffers; group 1 is the vessel's. Group 3 is the surface's, or the grid's
/// for the kernels that work the grid, which never touch the surface.
struct Spec {
    kernel: Kernel,
    shader: &'static str,
    entry: &'static str,
    particles: &'static [u32],
    vessel: bool,
    bodies: &'static [u32],
    surface: &'static [u32],
    grid: &'static [u32],
    /// (group, binding) pairs the kernel only reads, which the layout must say too.
    read_only: &'static [(usize, u32)],
    /// Threads per workgroup, as the kernel declares.
    workgroup: u32,
}

const NONE: &[(usize, u32)] = &[];
const PARTICLE_READS: &[(usize, u32)] = &[(0, 9), (0, 11), (0, 14), (2, 2), (2, 3)];
const GRID_READS: &[(usize, u32)] = &[(0, 4), (0, 9), (0, 12)];
const BODY_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 6), (0, 9), (0, 12), (2, 1)];
/// The surface is smoothed this many times back and forth, and once more into the buffer
/// it is drawn from.
const POLISH_PASSES: usize = 1;
const SURFACE_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 9), (0, 12)];
const SPRAY_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 4), (0, 9), (0, 12)];

const SPECS: [Spec; 47] = [
    Spec {
        kernel: Kernel::Count,
        shader: PARTICLES,
        entry: "count",
        particles: &[0, 1, 2, 8, 10],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::ScanRuns,
        shader: SORT,
        entry: "scan_runs",
        particles: &[0, 8, 9, 13],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: NONE,
        workgroup: SCAN_THREADS as u32,
    },
    Spec {
        kernel: Kernel::ScanTotals,
        shader: SORT,
        entry: "scan_totals",
        particles: &[0, 9, 13],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: NONE,
        workgroup: SCAN_THREADS as u32,
    },
    Spec {
        kernel: Kernel::AddOffsets,
        shader: SORT,
        entry: "add_offsets",
        particles: &[0, 9, 13],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: NONE,
        workgroup: SCAN_THREADS as u32,
    },
    Spec {
        kernel: Kernel::Scatter,
        shader: PARTICLES,
        entry: "scatter",
        particles: &[0, 1, 2, 3, 4, 9, 10, 12],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Bury,
        shader: PARTICLES,
        entry: "bury",
        particles: &[0, 3, 4, 9],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Tally,
        shader: PARTICLES,
        entry: "tally",
        particles: &[0, 9, 17],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: 1,
    },
    Spec {
        kernel: Kernel::ScatterAffine,
        shader: PARTICLES,
        entry: "scatter_affine",
        particles: &[0, 9, 10, 15, 16],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Thin,
        shader: PARTICLES,
        entry: "thin",
        particles: &[0, 1, 2, 3, 4, 15, 16],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Predict,
        shader: PARTICLES,
        entry: "predict",
        particles: &[0, 1, 2, 4, 5, 7, 15],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Inject,
        shader: PARTICLES,
        entry: "inject",
        particles: &[0, 1, 2, 11, 15],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Join,
        shader: PARTICLES,
        entry: "join",
        particles: &[0, 1, 2, 3, 9, 12, 14, 15],
        vessel: true,
        bodies: &[],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: JOIN_THREADS,
    },
    Spec {
        kernel: Kernel::Occupy,
        shader: GRID,
        entry: "occupy",
        particles: &[0, 5],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Muster,
        shader: GRID,
        entry: "muster",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[3, 6],
        read_only: GRID_READS,
        workgroup: 1,
    },
    Spec {
        kernel: Kernel::Gather,
        shader: GRID,
        entry: "gather",
        particles: &[0, 2, 5, 9, 12, 15],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Walls,
        shader: GRID,
        entry: "walls",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Level,
        shader: GRID,
        entry: "level",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::MusterShores,
        shader: GRID,
        entry: "muster_shores",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[3, 6],
        read_only: GRID_READS,
        workgroup: 1,
    },
    Spec {
        kernel: Kernel::LevelShores,
        shader: GRID,
        entry: "level_shores",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::TellOnOnce,
        shader: GRID,
        entry: "tell_on_once",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::TellOnTwice,
        shader: GRID,
        entry: "tell_on_twice",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Slip,
        shader: GRID,
        entry: "slip",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::System,
        shader: GRID,
        entry: "system",
        particles: &[0],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Wetted,
        shader: GRID,
        entry: "wetted",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::SmoothOnce,
        shader: GRID,
        entry: "smooth_once",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::SmoothTwice,
        shader: GRID,
        entry: "smooth_twice",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::SmoothThrice,
        shader: GRID,
        entry: "smooth_thrice",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::SettleRed,
        shader: GRID,
        entry: "settle_red",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::SettleBlack,
        shader: GRID,
        entry: "settle_black",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::RelaxRed,
        shader: GRID,
        entry: "relax_red",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::RelaxBlack,
        shader: GRID,
        entry: "relax_black",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Project,
        shader: GRID,
        entry: "project",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[],
        grid: &[1, 2, 3, 4, 5],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Transfer,
        shader: GRID,
        entry: "transfer",
        particles: &[0, 1, 2, 4, 5, 6, 7, 15],
        vessel: true,
        bodies: &[0],
        surface: &[],
        grid: &[1, 4],
        read_only: GRID_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Weather,
        shader: PARTICLES,
        entry: "weather",
        particles: &[0, 1, 2, 4, 5, 6, 7],
        vessel: true,
        bodies: &[0, 2, 3],
        surface: &[],
        grid: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Place,
        shader: BODIES,
        entry: "place",
        particles: &[0],
        vessel: true,
        bodies: &[0, 1, 2],
        surface: &[],
        grid: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Buoyancy,
        shader: BODIES,
        entry: "buoyancy",
        particles: &[0, 1, 2, 9, 12],
        vessel: true,
        bodies: &[0, 1, 2, 3, 4],
        surface: &[],
        grid: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Drag,
        shader: BODIES,
        entry: "drag",
        particles: &[0, 1, 2, 6, 9, 12],
        vessel: true,
        bodies: &[0, 1, 2, 4],
        surface: &[],
        grid: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Shed,
        shader: SPRAY,
        entry: "shed",
        particles: &[0, 1, 2, 4],
        vessel: true,
        bodies: &[],
        surface: &[13],
        grid: &[],
        read_only: SPRAY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Fly,
        shader: SPRAY,
        entry: "fly",
        particles: &[0, 1, 9, 12],
        vessel: true,
        bodies: &[],
        surface: &[13],
        grid: &[],
        read_only: SPRAY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Mark,
        shader: SURFACE,
        entry: "mark",
        particles: &[0, 1, 2, 9, 12],
        vessel: true,
        bodies: &[],
        surface: &[0, 3, 4, 11, 12],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::List,
        shader: SURFACE,
        entry: "list",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 3, 4, 5, 6],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::PrepareDispatch,
        shader: SURFACE,
        entry: "prepare_dispatch",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 3, 7],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Extract,
        shader: SURFACE,
        entry: "extract",
        particles: &[0, 1, 2, 9, 12],
        vessel: true,
        bodies: &[],
        surface: &[0, 1, 3, 6, 8, 9, 11],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::PrepareQuads,
        shader: SURFACE,
        entry: "prepare_quads",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 3, 7],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Quads,
        shader: SURFACE,
        entry: "quads",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 1, 2, 3, 4, 5, 6, 8, 9],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Polish,
        shader: SURFACE,
        entry: "polish",
        particles: &[0],
        vessel: true,
        bodies: &[],
        surface: &[0, 1, 3, 4, 5, 6, 8, 9, 10],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::PolishBack,
        shader: SURFACE,
        entry: "polish_back",
        particles: &[0],
        vessel: true,
        bodies: &[],
        surface: &[0, 1, 3, 4, 5, 6, 8, 9, 10],
        grid: &[],
        read_only: SURFACE_READS,
        workgroup: WORKGROUP,
    },
];

struct Pipeline {
    id: CachedComputePipelineId,
    layouts: [BindGroupLayoutDescriptor; 4],
    /// How many of the layouts the pipeline binds.
    groups: usize,
}

#[derive(Resource)]
struct Pipelines {
    items: Vec<Pipeline>,
    empty: BindGroupLayoutDescriptor,
}

#[derive(Resource, Default)]
struct Uniforms {
    params: DynamicUniformBuffer<Params>,
    bodies: DynamicUniformBuffer<GpuBodies>,
    surface: UniformBuffer<SurfaceParams>,
}

struct KernelGroups {
    particles: BindGroup,
    bodies: BindGroup,
    /// The surface's buffers, or the grid's for a kernel that works the grid.
    last: BindGroup,
}

/// Bind groups per kernel, and where in them a frame's parameters lie. The groups are kept
/// from frame to frame, and made anew only when a buffer they are `made_of` is replaced.
#[derive(Resource)]
struct BindGroups {
    kernels: Vec<KernelGroups>,
    empty: BindGroup,
    made_of: Vec<BufferId>,
    thin_offset: Option<u32>,
    inject_offset: u32,
    join_offset: u32,
    surface_offset: u32,
    params_offsets: Vec<u32>,
    bodies_offsets: Vec<u32>,
}

#[derive(Resource)]
struct RawBuffers {
    position: Buffer,
    velocity: Buffer,
    position_sorted: Buffer,
    velocity_next: Buffer,
    cell_count: Buffer,
    accum: Buffer,
    counters: Buffer,
    table: Buffer,
    cell_table: Buffer,
    dispatch: Buffer,
    motes: Buffer,
    affine: Buffer,
    affine_sorted: Buffer,
    grid_keys: Buffer,
    grid_counters: Buffer,
    grid_dispatch: Buffer,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    vessel: Res<VesselLayout>,
) {
    let empty = BindGroupLayoutDescriptor::new("fluid empty", &[]);
    let items = SPECS
        .iter()
        .map(|spec| {
            let reads = |group: usize| -> Vec<u32> {
                spec.read_only
                    .iter()
                    .filter(|(g, _)| *g == group)
                    .map(|(_, b)| *b)
                    .collect()
            };
            let layouts = [
                layout("fluid particles", spec.particles, &reads(0), true),
                if spec.vessel {
                    vessel.0.clone()
                } else {
                    empty.clone()
                },
                layout("fluid bodies", spec.bodies, &reads(2), true),
                if spec.grid.is_empty() {
                    layout("fluid surface", spec.surface, &reads(3), false)
                } else {
                    layout("fluid grid", spec.grid, &reads(3), false)
                },
            ];
            let groups = if !spec.surface.is_empty() || !spec.grid.is_empty() {
                4
            } else if !spec.bodies.is_empty() {
                3
            } else if spec.vessel {
                2
            } else {
                1
            };
            let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(spec.entry.into()),
                layout: layouts[..groups].to_vec(),
                shader: asset_server.load(spec.shader),
                entry_point: Some(spec.entry.into()),
                ..default()
            });
            Pipeline {
                id,
                layouts,
                groups,
            }
        })
        .collect();
    commands.insert_resource(Pipelines { items, empty });
}

/// A layout with the listed bindings: binding 0 a uniform buffer (dynamic when `dynamic`), the
/// rest storage buffers, read-only where listed.
fn layout(
    label: &'static str,
    bindings: &[u32],
    read_only: &[u32],
    dynamic: bool,
) -> BindGroupLayoutDescriptor {
    let entries: Vec<BindGroupLayoutEntry> = bindings
        .iter()
        .map(|&binding| {
            if binding == 0 {
                uniform_buffer_sized(dynamic, None).build(binding, ShaderStages::COMPUTE)
            } else {
                if read_only.contains(&binding) {
                    storage_buffer_read_only_sized(false, None)
                } else {
                    storage_buffer_sized(false, None)
                }
                .build(binding, ShaderStages::COMPUTE)
            }
        })
        .collect();
    BindGroupLayoutDescriptor::new(label, &entries)
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    pipelines: Res<Pipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<FluidBuffers>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    frame: Res<FluidFrame>,
    surface_params: Res<SurfaceParams>,
    mut uniforms: ResMut<Uniforms>,
    made: Option<ResMut<BindGroups>>,
) {
    let get = |handle: &Handle<ShaderBuffer>| gpu_buffers.get(handle).map(|b| b.buffer.clone());
    let all = |handles: &[&Handle<ShaderBuffer>]| {
        handles
            .iter()
            .map(|h| get(h))
            .collect::<Option<Vec<Buffer>>>()
    };
    let (Some(particles), Some(bodies), Some(surface), Some(grid)) = (
        all(&[
            &buffers.position,
            &buffers.velocity,
            &buffers.position_sorted,
            &buffers.velocity_next,
            &buffers.pred_a,
            &buffers.pred_b,
            &buffers.contact,
            &buffers.cell_count,
            &buffers.cell_start,
            &buffers.slot,
            &buffers.pending,
            &buffers.key,
            &buffers.run_total,
            &buffers.sites,
            &buffers.affine,
            &buffers.affine_sorted,
            &buffers.living,
        ]),
        all(&[
            &buffers.samples,
            &buffers.boundary,
            &buffers.sample_state,
            &buffers.accum,
        ]),
        all(&buffers.surface.handles()),
        all(&[
            &buffers.grid_keys,
            &buffers.grid_list,
            &buffers.grid_counters,
            &buffers.grid_cells,
            &buffers.grid_links,
            &buffers.grid_dispatch,
        ]),
    ) else {
        return;
    };
    if let Some(samples) = &frame.samples {
        queue.write_buffer(&bodies[0], 0, bytemuck::cast_slice(samples));
    }
    if !frame.pending.is_empty() {
        queue.write_buffer(&particles[10], 0, bytemuck::cast_slice(&frame.pending));
    }
    if !frame.sites.is_empty() {
        queue.write_buffer(&particles[13], 0, bytemuck::cast_slice(&frame.sites));
    }
    queue.write_buffer(
        &bodies[3],
        (STAMP_SLOT * 4) as u64,
        bytemuck::cast_slice(&[frame.ticket]),
    );

    queue.write_buffer(&particles[16], 12, bytemuck::bytes_of(&frame.ticket));
    uniforms.params.clear();
    uniforms.bodies.clear();
    let thin_offset = frame.thin.as_ref().map(|thin| uniforms.params.push(thin));
    let inject_offset = uniforms.params.push(&frame.inject);
    let join_offset = uniforms.params.push(&frame.join);
    let surface_offset = uniforms.params.push(&frame.surface);
    let params_offsets: Vec<u32> = frame
        .substeps
        .iter()
        .map(|s| uniforms.params.push(&s.params))
        .collect();
    let bodies_offsets: Vec<u32> = frame
        .substeps
        .iter()
        .map(|s| uniforms.bodies.push(&s.bodies.gpu))
        .collect();
    if bodies_offsets.is_empty() {
        uniforms.bodies.push(&GpuBodies::default());
    }
    uniforms.surface.set(surface_params.clone());
    uniforms.params.write_buffer(&device, &queue);
    uniforms.bodies.write_buffer(&device, &queue);
    uniforms.surface.write_buffer(&device, &queue);
    let (Some(params_binding), Some(bodies_binding), Some(surface_binding)) = (
        uniforms.params.binding(),
        uniforms.bodies.binding(),
        uniforms.surface.binding(),
    ) else {
        return;
    };
    if let Some(buffer) = uniforms.bodies.buffer() {
        let coupled = frame.substeps.first().filter(|_| frame.coupling);
        commands.insert_resource(FluidHulls {
            bodies: buffer.clone(),
            offsets: bodies_offsets.clone(),
            samples: coupled.map_or(0, |s| s.params.sample_count),
            accumulators: coupled.map_or(0, |s| s.params.accumulators),
            body_drag: coupled.map_or(0.0, |s| s.params.body_drag),
            rest_density: coupled.map_or(0.0, |s| s.params.rest_density),
        });
    }
    let made_of: Vec<BufferId> = [
        uniforms.params.buffer(),
        uniforms.bodies.buffer(),
        uniforms.surface.buffer(),
    ]
    .into_iter()
    .flatten()
    .chain(particles.iter().chain(&bodies).chain(&surface).chain(&grid))
    .map(Buffer::id)
    .collect();
    if let Some(mut made) = made
        && made.made_of == made_of
    {
        made.thin_offset = thin_offset;
        made.inject_offset = inject_offset;
        made.join_offset = join_offset;
        made.surface_offset = surface_offset;
        made.params_offsets = params_offsets;
        made.bodies_offsets = bodies_offsets;
        return;
    }

    let resource = |group: usize, binding: u32| -> BindingResource<'_> {
        match (group, binding) {
            (0, 0) => params_binding.clone(),
            (0, b) => particles[b as usize - 1].as_entire_binding(),
            (2, 0) => bodies_binding.clone(),
            (2, b) => bodies[b as usize - 1].as_entire_binding(),
            (3, 0) => surface_binding.clone(),
            (3, b) => surface[b as usize - 1].as_entire_binding(),
            (_, b) => grid[b as usize - 1].as_entire_binding(),
        }
    };
    // `buffers` names whose buffers the bindings are: a group's own, or the grid's as 4
    let build = |pipeline: &Pipeline, group: usize, buffers: usize, bindings: &[u32]| {
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .map(|&binding| BindGroupEntry {
                binding,
                resource: resource(buffers, binding),
            })
            .collect();
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.layouts[group]),
            &entries,
        )
    };
    let kernels = SPECS
        .iter()
        .zip(&pipelines.items)
        .map(|(spec, pipeline)| KernelGroups {
            particles: build(pipeline, 0, 0, spec.particles),
            bodies: build(pipeline, 2, 2, spec.bodies),
            last: if spec.grid.is_empty() {
                build(pipeline, 3, 3, spec.surface)
            } else {
                build(pipeline, 3, 4, spec.grid)
            },
        })
        .collect();
    let empty = device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipelines.empty),
        &[],
    );
    commands.insert_resource(BindGroups {
        kernels,
        empty,
        made_of,
        thin_offset,
        inject_offset,
        join_offset,
        surface_offset,
        params_offsets,
        bodies_offsets,
    });
    commands.insert_resource(RawBuffers {
        position: particles[0].clone(),
        velocity: particles[1].clone(),
        position_sorted: particles[2].clone(),
        velocity_next: particles[3].clone(),
        cell_count: particles[7].clone(),
        accum: bodies[3].clone(),
        counters: surface[2].clone(),
        table: surface[3].clone(),
        cell_table: surface[7].clone(),
        dispatch: surface[6].clone(),
        motes: surface[12].clone(),
        affine: particles[14].clone(),
        affine_sorted: particles[15].clone(),
        grid_keys: grid[0].clone(),
        grid_counters: grid[2].clone(),
        grid_dispatch: grid[5].clone(),
    });
}

/// How many threads a kernel runs: a count of them, or the workgroups an entry of the surface
/// kernels' own dispatch buffer names.
#[derive(Clone, Copy)]
enum Threads<'a> {
    Count(u32),
    Indirect(&'a Buffer, u64),
}

/// Where in the uniform buffers the parameters the kernels of a pass read lie.
#[derive(Clone, Copy)]
struct Offsets {
    params: u32,
    bodies: u32,
    vessel: u32,
}

struct Dispatch<'a> {
    cache: &'a PipelineCache,
    pipelines: &'a Pipelines,
    groups: &'a BindGroups,
    vessel: &'a VesselBinding,
}

impl Dispatch<'_> {
    fn ready(&self) -> bool {
        self.pipelines
            .items
            .iter()
            .all(|p| self.cache.get_compute_pipeline(p.id).is_some())
    }

    /// Run kernels one after another in a single pass: a pass of its own for each would have
    /// the driver keep a command buffer for each, a frame after frame of them.
    fn run(
        &self,
        encoder: &mut CommandEncoder,
        label: &'static str,
        offsets: Offsets,
        kernels: &[(Kernel, Threads)],
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            ..default()
        });
        for &(kernel, threads) in kernels {
            let index = SPECS.iter().position(|s| s.kernel == kernel).unwrap_or(0);
            let (spec, pipeline) = (&SPECS[index], &self.pipelines.items[index]);
            let Some(compute) = self.cache.get_compute_pipeline(pipeline.id) else {
                continue;
            };
            let groups = &self.groups.kernels[index];
            pass.set_pipeline(compute);
            pass.set_bind_group(0, &groups.particles, &[offsets.params]);
            if pipeline.groups > 1 {
                if spec.vessel {
                    pass.set_bind_group(1, &self.vessel.bind_group, &[offsets.vessel]);
                } else {
                    pass.set_bind_group(1, &self.groups.empty, &[]);
                }
            }
            if pipeline.groups > 2 {
                if spec.bodies.is_empty() {
                    pass.set_bind_group(2, &self.groups.empty, &[]);
                } else {
                    pass.set_bind_group(2, &groups.bodies, &[offsets.bodies]);
                }
            }
            if pipeline.groups > 3 {
                pass.set_bind_group(3, &groups.last, &[]);
            }
            match threads {
                Threads::Count(n) => {
                    pass.dispatch_workgroups(n.div_ceil(spec.workgroup).max(1), 1, 1)
                }
                Threads::Indirect(buffer, offset) => {
                    pass.dispatch_workgroups_indirect(buffer, offset)
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    groups: Option<Res<BindGroups>>,
    raw: Option<Res<RawBuffers>>,
    vessel: Option<Res<VesselBinding>>,
    frame: Res<FluidFrame>,
    ready: Res<FluidReady>,
) {
    let (Some(groups), Some(raw), Some(vessel)) = (groups, raw, vessel) else {
        return;
    };
    let d = Dispatch {
        cache: &cache,
        pipelines: &pipelines,
        groups: &groups,
        vessel: &vessel,
    };
    if !d.ready() || vessel.offsets.is_empty() {
        return;
    }
    ready.set();
    let vessel_now = *vessel.offsets.last().unwrap_or(&0);
    let encoder = render_context.command_encoder();
    let bytes = |n: u32| n as u64 * 16;
    let bin = |encoder: &mut CommandEncoder, at: Offsets, count: u32| {
        encoder.clear_buffer(&raw.cell_count, 0, None);
        let threads = Threads::Count(count);
        let cells = Threads::Count(TABLE_CELLS as u32);
        d.run(
            encoder,
            "fluid bin",
            at,
            &[
                (Kernel::Count, threads),
                (Kernel::ScanRuns, cells),
                (Kernel::ScanTotals, Threads::Count(1)),
                (Kernel::AddOffsets, cells),
                (Kernel::Scatter, threads),
                (Kernel::ScatterAffine, threads),
                (Kernel::Bury, threads),
                (Kernel::Tally, Threads::Count(1)),
            ],
        );
    };
    let sort = |encoder: &mut CommandEncoder, at: Offsets, count: u32| {
        bin(encoder, at, count);
        encoder.copy_buffer_to_buffer(&raw.position_sorted, 0, &raw.position, 0, bytes(count));
        encoder.copy_buffer_to_buffer(&raw.velocity_next, 0, &raw.velocity, 0, bytes(count));
        encoder.copy_buffer_to_buffer(&raw.affine_sorted, 0, &raw.affine, 0, 6 * bytes(count));
    };
    let now = |params: u32| Offsets {
        params,
        bodies: 0,
        vessel: vessel_now,
    };
    if let (Some(thin), Some(to)) = (&frame.thin, groups.thin_offset) {
        bin(encoder, now(to), thin.count);
        let remaining = Threads::Count(thin.pending);
        d.run(encoder, "fluid thin", now(to), &[(Kernel::Thin, remaining)]);
        // the spray is measured in the water's units, which thinning the water changes
        encoder.clear_buffer(&raw.motes, 0, None);
    }
    if frame.inject.pending > 0 {
        let joining = Threads::Count(frame.inject.pending);
        let at = now(groups.inject_offset);
        d.run(encoder, "fluid inject", at, &[(Kernel::Inject, joining)]);
    }
    if frame.join.pending > 0 {
        let at = now(groups.join_offset);
        bin(encoder, at, frame.join.count);
        let weighers = Threads::Count(JOIN_THREADS);
        d.run(encoder, "fluid join", at, &[(Kernel::Join, weighers)]);
    }
    if frame.coupling {
        let first = accumulators_of(frame.ticket) as u64 * 4;
        let bytes = ACCUMULATORS_PER_FRAME as u64 * 4;
        encoder.clear_buffer(&raw.accum, first, Some(bytes));
    }
    for (k, substep) in frame.substeps.iter().enumerate() {
        let at = Offsets {
            params: groups.params_offsets[k],
            bodies: groups.bodies_offsets[k],
            vessel: vessel.offsets.get(k).copied().unwrap_or(vessel_now),
        };
        let count = Threads::Count(substep.params.count);
        let samples = Threads::Count(substep.params.sample_count);
        let coupled = substep.params.sample_count > 0;
        if substep.params.count == 0 {
            if coupled {
                d.run(encoder, "fluid place", at, &[(Kernel::Place, samples)]);
            }
            continue;
        }
        let cells = Threads::Indirect(&raw.grid_dispatch, 0);
        let shores = Threads::Indirect(&raw.grid_dispatch, 16);
        sort(encoder, at, substep.params.count);
        encoder.clear_buffer(&raw.grid_keys, 0, None);
        encoder.clear_buffer(&raw.grid_counters, 0, None);
        let mut step = vec![(Kernel::Predict, count)];
        if coupled {
            step.push((Kernel::Place, samples));
        }
        step.extend([
            (Kernel::Occupy, count),
            (Kernel::Muster, Threads::Count(1)),
            (Kernel::Gather, cells),
            (Kernel::Walls, cells),
            (Kernel::Level, cells),
            (Kernel::MusterShores, Threads::Count(1)),
            (Kernel::LevelShores, shores),
            (Kernel::Wetted, cells),
            (Kernel::SmoothOnce, cells),
            (Kernel::SmoothTwice, cells),
            (Kernel::SmoothThrice, cells),
            (Kernel::System, cells),
        ]);
        for _ in 0..SETTLINGS {
            step.extend([(Kernel::SettleRed, cells), (Kernel::SettleBlack, cells)]);
        }
        for _ in 0..RELAXATIONS {
            step.extend([(Kernel::RelaxRed, cells), (Kernel::RelaxBlack, cells)]);
        }
        step.extend([
            (Kernel::Project, cells),
            (Kernel::TellOnOnce, cells),
            (Kernel::TellOnTwice, cells),
            (Kernel::Slip, cells),
            (Kernel::Transfer, count),
        ]);
        if coupled {
            step.push((Kernel::Buoyancy, samples));
        }
        step.push((Kernel::Weather, count));
        if coupled {
            step.push((Kernel::Drag, samples));
        }
        step.extend([
            (Kernel::Shed, count),
            (Kernel::Fly, Threads::Count(MAX_MOTES as u32)),
        ]);
        d.run(encoder, "fluid step", at, &step);
        encoder.copy_buffer_to_buffer(
            &raw.velocity_next,
            0,
            &raw.velocity,
            0,
            bytes(substep.params.count),
        );
    }
    if frame.changed {
        let at = now(groups.surface_offset);
        let count = Threads::Count(frame.surface.count);
        let blocks = Threads::Indirect(&raw.dispatch, 0);
        let cells = Threads::Indirect(&raw.dispatch, 16);
        let one = Threads::Count(1);
        sort(encoder, at, frame.surface.count);
        encoder.clear_buffer(&raw.counters, 0, None);
        encoder.clear_buffer(&raw.table, 0, None);
        encoder.clear_buffer(&raw.cell_table, 0, None);
        let mut surface = vec![
            (Kernel::Mark, count),
            (Kernel::List, Threads::Count(TABLE_SLOTS as u32)),
            (Kernel::PrepareDispatch, one),
            (Kernel::Extract, blocks),
            (Kernel::PrepareQuads, one),
            (Kernel::Quads, cells),
        ];
        for _ in 0..POLISH_PASSES {
            surface.extend([(Kernel::Polish, cells), (Kernel::PolishBack, cells)]);
        }
        surface.push((Kernel::Polish, cells));
        d.run(encoder, "fluid surface", at, &surface);
    }
}
