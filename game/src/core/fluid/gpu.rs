//! The render-world half of the fluid: its buffers, one compute pipeline per kernel with only the
//! bindings that kernel touches (WebGPU allows eight storage buffers per stage), and the frame's
//! dispatch order: thin if due, append, then per substep sort, predict, constrain, derive
//! velocities and couple to the bodies, and finally re-sort and extract the surface.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    Buffer, BufferUsages, CachedComputePipelineId, CommandEncoder, ComputePassDescriptor,
    ComputePipelineDescriptor, DynamicUniformBuffer, PipelineCache, ShaderStages, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};

use super::frame::{
    ACCUMULATORS_PER_FRAME, FluidFrame, GpuBodies, Params, STAMP_SLOT, accumulators_of,
};
use super::surface::{self, SurfaceBuffers, SurfaceParams, TABLE_SLOTS};
use super::{FluidReady, ITERATIONS, MAX_PARTICLES, MAX_SAMPLES, TABLE_CELLS};
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

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
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
    Thin,
    Predict,
    Inject,
    Join,
    Lambda,
    Delta,
    UpdateVelocities,
    Viscosity,
    Place,
    Buoyancy,
    Drag,
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
const SORT: &str = "embedded://game/core/fluid/shaders/sort.wgsl";
const BODIES: &str = "embedded://game/core/fluid/shaders/bodies.wgsl";
const SURFACE: &str = "embedded://game/core/fluid/shaders/surface.wgsl";

/// Which bindings of each group a kernel uses. Binding 0 of groups 0, 2 and 3 is a uniform, the
/// rest are storage buffers; group 1 is the vessel's.
struct Spec {
    kernel: Kernel,
    shader: &'static str,
    entry: &'static str,
    particles: &'static [u32],
    vessel: bool,
    bodies: &'static [u32],
    surface: &'static [u32],
    /// (group, binding) pairs the kernel only reads, which the layout must say too.
    read_only: &'static [(usize, u32)],
    /// Threads per workgroup, as the kernel declares.
    workgroup: u32,
}

const NONE: &[(usize, u32)] = &[];
const PARTICLE_READS: &[(usize, u32)] = &[(0, 9), (0, 11), (0, 14), (2, 2), (2, 3)];
const BODY_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 4), (0, 9), (0, 12), (2, 1)];
/// The surface is smoothed this many times back and forth, and once more into the buffer
/// it is drawn from.
const POLISH_PASSES: usize = 1;
const SURFACE_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 9), (0, 12)];

const SPECS: [Spec; 24] = [
    Spec {
        kernel: Kernel::Count,
        shader: PARTICLES,
        entry: "count",
        particles: &[0, 1, 2, 8, 10],
        vessel: true,
        bodies: &[],
        surface: &[],
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
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Thin,
        shader: PARTICLES,
        entry: "thin",
        particles: &[0, 1, 2, 3, 4],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Predict,
        shader: PARTICLES,
        entry: "predict",
        particles: &[0, 1, 2, 5, 7],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Inject,
        shader: PARTICLES,
        entry: "inject",
        particles: &[0, 1, 2, 11],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Join,
        shader: PARTICLES,
        entry: "join",
        particles: &[0, 1, 2, 3, 9, 12, 14],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: JOIN_THREADS,
    },
    Spec {
        kernel: Kernel::Lambda,
        shader: PARTICLES,
        entry: "lambda",
        particles: &[0, 5, 9, 12],
        vessel: true,
        bodies: &[0, 2],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Delta,
        shader: PARTICLES,
        entry: "delta",
        particles: &[0, 5, 6, 7, 9, 12],
        vessel: true,
        bodies: &[0, 2],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::UpdateVelocities,
        shader: PARTICLES,
        entry: "update_velocities",
        particles: &[0, 1, 2, 5, 7],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Viscosity,
        shader: PARTICLES,
        entry: "viscosity",
        particles: &[0, 1, 2, 4, 7, 9, 12],
        vessel: true,
        bodies: &[0, 2, 3],
        surface: &[],
        read_only: PARTICLE_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Place,
        shader: BODIES,
        entry: "place",
        particles: &[0],
        vessel: false,
        bodies: &[0, 1, 2],
        surface: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Buoyancy,
        shader: BODIES,
        entry: "buoyancy",
        particles: &[0, 1, 2, 9, 12],
        vessel: true,
        bodies: &[0, 2, 3, 4],
        surface: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Drag,
        shader: BODIES,
        entry: "drag",
        particles: &[0, 1, 2, 4, 9, 12],
        vessel: true,
        bodies: &[0, 2, 4],
        surface: &[],
        read_only: BODY_READS,
        workgroup: WORKGROUP,
    },
    Spec {
        kernel: Kernel::Mark,
        shader: SURFACE,
        entry: "mark",
        particles: &[0, 1, 9, 12],
        vessel: false,
        bodies: &[],
        surface: &[0, 3, 4, 11, 12],
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
    surface: BindGroup,
}

/// Bind groups for one frame: per kernel, one with the predicted positions read from buffer A
/// and written to B, and one the other way round.
#[derive(Resource)]
struct BindGroups {
    kernels: Vec<[KernelGroups; 2]>,
    empty: BindGroup,
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
                layout("fluid surface", spec.surface, &reads(3), false),
            ];
            let groups = if !spec.surface.is_empty() {
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
) {
    let get = |handle: &Handle<ShaderBuffer>| gpu_buffers.get(handle).map(|b| b.buffer.clone());
    let all = |handles: &[&Handle<ShaderBuffer>]| {
        handles
            .iter()
            .map(|h| get(h))
            .collect::<Option<Vec<Buffer>>>()
    };
    let (Some(particles), Some(bodies), Some(surface)) = (
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
        ]),
        all(&[
            &buffers.samples,
            &buffers.boundary,
            &buffers.sample_state,
            &buffers.accum,
        ]),
        all(&buffers.surface.handles()),
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

    let resource = |group: usize, binding: u32, variant: usize| -> BindingResource<'_> {
        let pred =
            |a: usize, b: usize| particles[if variant == 0 { a } else { b }].as_entire_binding();
        match (group, binding) {
            (0, 0) => params_binding.clone(),
            (0, 5) => pred(4, 5),
            (0, 6) => pred(5, 4),
            (0, b) => particles[b as usize - 1].as_entire_binding(),
            (2, 0) => bodies_binding.clone(),
            (2, b) => bodies[b as usize - 1].as_entire_binding(),
            (3, 0) => surface_binding.clone(),
            (_, b) => surface[b as usize - 1].as_entire_binding(),
        }
    };
    let build = |pipeline: &Pipeline, group: usize, bindings: &[u32], variant: usize| {
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .map(|&binding| BindGroupEntry {
                binding,
                resource: resource(group, binding, variant),
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
        .map(|(spec, pipeline)| {
            [0, 1].map(|variant| KernelGroups {
                particles: build(pipeline, 0, spec.particles, variant),
                bodies: build(pipeline, 2, spec.bodies, variant),
                surface: build(pipeline, 3, spec.surface, variant),
            })
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
    });
}

/// How many threads a kernel runs: a count of them, or the workgroups an entry of the surface
/// kernels' own dispatch buffer names.
#[derive(Clone, Copy)]
enum Threads<'a> {
    Count(u32),
    Indirect(&'a Buffer, u64),
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

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        encoder: &mut CommandEncoder,
        kernel: Kernel,
        variant: usize,
        params_offset: u32,
        bodies_offset: u32,
        vessel_offset: u32,
        threads: Threads,
    ) {
        let index = SPECS.iter().position(|s| s.kernel == kernel).unwrap_or(0);
        let (spec, pipeline) = (&SPECS[index], &self.pipelines.items[index]);
        let Some(compute) = self.cache.get_compute_pipeline(pipeline.id) else {
            return;
        };
        let groups = &self.groups.kernels[index][variant];
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(spec.entry),
            ..default()
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &groups.particles, &[params_offset]);
        if pipeline.groups > 1 {
            if spec.vessel {
                pass.set_bind_group(1, &self.vessel.bind_group, &[vessel_offset]);
            } else {
                pass.set_bind_group(1, &self.groups.empty, &[]);
            }
        }
        if pipeline.groups > 2 {
            if spec.bodies.is_empty() {
                pass.set_bind_group(2, &self.groups.empty, &[]);
            } else {
                pass.set_bind_group(2, &groups.bodies, &[bodies_offset]);
            }
        }
        if pipeline.groups > 3 {
            pass.set_bind_group(3, &groups.surface, &[]);
        }
        match threads {
            Threads::Count(n) => pass.dispatch_workgroups(n.div_ceil(spec.workgroup).max(1), 1, 1),
            Threads::Indirect(buffer, offset) => pass.dispatch_workgroups_indirect(buffer, offset),
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
    let bin = |encoder: &mut CommandEncoder, po: u32, vo: u32, count: u32| {
        encoder.clear_buffer(&raw.cell_count, 0, None);
        let threads = Threads::Count(count);
        d.run(encoder, Kernel::Count, 0, po, 0, vo, threads);
        let cells = Threads::Count(TABLE_CELLS as u32);
        d.run(encoder, Kernel::ScanRuns, 0, po, 0, vo, cells);
        d.run(encoder, Kernel::ScanTotals, 0, po, 0, vo, Threads::Count(1));
        d.run(encoder, Kernel::AddOffsets, 0, po, 0, vo, cells);
        d.run(encoder, Kernel::Scatter, 0, po, 0, vo, threads);
    };
    let sort = |encoder: &mut CommandEncoder, po: u32, vo: u32, count: u32| {
        bin(encoder, po, vo, count);
        encoder.copy_buffer_to_buffer(&raw.position_sorted, 0, &raw.position, 0, bytes(count));
        encoder.copy_buffer_to_buffer(&raw.velocity_next, 0, &raw.velocity, 0, bytes(count));
    };
    if let (Some(thin), Some(to)) = (&frame.thin, groups.thin_offset) {
        bin(encoder, to, vessel_now, thin.count);
        let remaining = Threads::Count(thin.pending);
        d.run(encoder, Kernel::Thin, 0, to, 0, vessel_now, remaining);
    }
    if frame.inject.pending > 0 {
        let joining = Threads::Count(frame.inject.pending);
        d.run(
            encoder,
            Kernel::Inject,
            0,
            groups.inject_offset,
            0,
            vessel_now,
            joining,
        );
    }
    if frame.join.pending > 0 {
        let jo = groups.join_offset;
        bin(encoder, jo, vessel_now, frame.join.count);
        let weighers = Threads::Count(JOIN_THREADS);
        d.run(encoder, Kernel::Join, 0, jo, 0, vessel_now, weighers);
    }
    if frame.coupling {
        let first = accumulators_of(frame.ticket) as u64 * 4;
        let bytes = ACCUMULATORS_PER_FRAME as u64 * 4;
        encoder.clear_buffer(&raw.accum, first, Some(bytes));
    }
    for (k, substep) in frame.substeps.iter().enumerate() {
        let (po, bo) = (groups.params_offsets[k], groups.bodies_offsets[k]);
        let vo = vessel.offsets.get(k).copied().unwrap_or(vessel_now);
        let count = Threads::Count(substep.params.count);
        let samples = Threads::Count(substep.params.sample_count);
        let coupled = substep.params.sample_count > 0;
        sort(encoder, po, vo, substep.params.count);
        d.run(encoder, Kernel::Predict, 0, po, bo, vo, count);
        if coupled {
            d.run(encoder, Kernel::Place, 0, po, bo, vo, samples);
        }
        let mut variant = 0;
        for _ in 0..ITERATIONS {
            d.run(encoder, Kernel::Lambda, variant, po, bo, vo, count);
            d.run(encoder, Kernel::Delta, variant, po, bo, vo, count);
            variant ^= 1;
        }
        d.run(
            encoder,
            Kernel::UpdateVelocities,
            variant,
            po,
            bo,
            vo,
            count,
        );
        if coupled {
            d.run(encoder, Kernel::Buoyancy, 0, po, bo, vo, samples);
        }
        d.run(encoder, Kernel::Viscosity, 0, po, bo, vo, count);
        if coupled {
            d.run(encoder, Kernel::Drag, 0, po, bo, vo, samples);
        }
        encoder.copy_buffer_to_buffer(
            &raw.velocity_next,
            0,
            &raw.velocity,
            0,
            bytes(substep.params.count),
        );
    }
    if frame.changed {
        let so = groups.surface_offset;
        let count = Threads::Count(frame.surface.count);
        let blocks = Threads::Indirect(&raw.dispatch, 0);
        let cells = Threads::Indirect(&raw.dispatch, 16);
        sort(encoder, so, vessel_now, frame.surface.count);
        encoder.clear_buffer(&raw.counters, 0, None);
        encoder.clear_buffer(&raw.table, 0, None);
        encoder.clear_buffer(&raw.cell_table, 0, None);
        d.run(encoder, Kernel::Mark, 0, so, 0, vessel_now, count);
        let slots = Threads::Count(TABLE_SLOTS as u32);
        d.run(encoder, Kernel::List, 0, so, 0, vessel_now, slots);
        let one = Threads::Count(1);
        d.run(encoder, Kernel::PrepareDispatch, 0, so, 0, vessel_now, one);
        d.run(encoder, Kernel::Extract, 0, so, 0, vessel_now, blocks);
        d.run(encoder, Kernel::PrepareQuads, 0, so, 0, vessel_now, one);
        d.run(encoder, Kernel::Quads, 0, so, 0, vessel_now, cells);
        for _ in 0..POLISH_PASSES {
            d.run(encoder, Kernel::Polish, 0, so, 0, vessel_now, cells);
            d.run(encoder, Kernel::PolishBack, 0, so, 0, vessel_now, cells);
        }
        d.run(encoder, Kernel::Polish, 0, so, 0, vessel_now, cells);
    }
}
