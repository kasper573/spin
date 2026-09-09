//! The render-world half of the fluid: its buffers, one compute pipeline per kernel with only the
//! bindings that kernel touches (WebGPU allows eight storage buffers per stage), and the frame's
//! dispatch order: append, then per substep sort, predict, constrain, derive velocities and
//! couple to the bodies, and finally re-sort and extract the surface.
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

use super::frame::{ACCUMULATORS_PER_BODY, FluidFrame, GpuBodies, Params, STAMP_SLOT};
use super::surface::{self, SurfaceBuffers, SurfaceParams};
use super::{FluidReady, Grid, ITERATIONS, MAX_BODIES, MAX_PARTICLES, MAX_SAMPLES};
use crate::core::vessel::{VesselBinding, VesselLayout};

const WORKGROUP: u32 = 64;
const ACCUMULATORS: usize = ACCUMULATORS_PER_BODY * MAX_BODIES;

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
    pub samples: Handle<ShaderBuffer>,
    pub boundary: Handle<ShaderBuffer>,
    pub sample_state: Handle<ShaderBuffer>,
    pub accum: Handle<ShaderBuffer>,
    pub surface: SurfaceBuffers,
}

pub fn create_buffers(
    assets: &mut Assets<ShaderBuffer>,
    grid: &Grid,
    surface_grid: &Grid,
) -> FluidBuffers {
    let mut make = |bytes: usize| {
        let mut buffer = ShaderBuffer::with_size(bytes, default());
        buffer.buffer_description.usage |= BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
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
        cell_count: make(grid.cells() * 4),
        cell_start: make((grid.cells() + 1) * 4),
        slot: make(MAX_PARTICLES * 4),
        pending: make(vec4s(2 * MAX_PARTICLES)),
        samples: make(vec4s(MAX_SAMPLES)),
        boundary: make(vec4s(2 * MAX_SAMPLES)),
        sample_state: make(vec4s(2 * MAX_SAMPLES)),
        accum: make(ACCUMULATORS * 4),
        surface: surface::create_buffers(&mut make, surface_grid),
    }
}

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(RenderGraph, dispatch.before(camera_driver));
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Count,
    Scan,
    Scatter,
    Predict,
    Inject,
    Lambda,
    Delta,
    UpdateVelocities,
    Viscosity,
    Place,
    Buoyancy,
    Drag,
    Density,
    PlaceVertices,
    PlaceQuads,
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
}

const NONE: &[(usize, u32)] = &[];
const PARTICLE_READS: &[(usize, u32)] = &[(0, 9), (0, 11), (2, 2), (2, 3)];
const BODY_READS: &[(usize, u32)] = &[(0, 1), (0, 2), (0, 4), (0, 9), (2, 1)];
const SURFACE_READS: &[(usize, u32)] = &[(0, 1), (0, 9)];

const SPECS: [Spec; 15] = [
    Spec {
        kernel: Kernel::Count,
        shader: PARTICLES,
        entry: "count",
        particles: &[0, 1, 2, 8, 10],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
    },
    Spec {
        kernel: Kernel::Scan,
        shader: SORT,
        entry: "scan",
        particles: &[0, 8, 9],
        vessel: false,
        bodies: &[],
        surface: &[],
        read_only: NONE,
    },
    Spec {
        kernel: Kernel::Scatter,
        shader: PARTICLES,
        entry: "scatter",
        particles: &[0, 1, 2, 3, 4, 9, 10],
        vessel: true,
        bodies: &[],
        surface: &[],
        read_only: PARTICLE_READS,
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
    },
    Spec {
        kernel: Kernel::Lambda,
        shader: PARTICLES,
        entry: "lambda",
        particles: &[0, 5, 9],
        vessel: true,
        bodies: &[0, 2],
        surface: &[],
        read_only: PARTICLE_READS,
    },
    Spec {
        kernel: Kernel::Delta,
        shader: PARTICLES,
        entry: "delta",
        particles: &[0, 5, 6, 7, 9],
        vessel: true,
        bodies: &[0, 2],
        surface: &[],
        read_only: PARTICLE_READS,
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
    },
    Spec {
        kernel: Kernel::Viscosity,
        shader: PARTICLES,
        entry: "viscosity",
        particles: &[0, 1, 2, 4, 7, 9],
        vessel: true,
        bodies: &[0, 2, 3],
        surface: &[],
        read_only: PARTICLE_READS,
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
    },
    Spec {
        kernel: Kernel::Buoyancy,
        shader: BODIES,
        entry: "buoyancy",
        particles: &[0, 1, 2, 9],
        vessel: true,
        bodies: &[0, 2, 3, 4],
        surface: &[],
        read_only: BODY_READS,
    },
    Spec {
        kernel: Kernel::Drag,
        shader: BODIES,
        entry: "drag",
        particles: &[0, 1, 2, 4, 9],
        vessel: false,
        bodies: &[0, 2, 4],
        surface: &[],
        read_only: BODY_READS,
    },
    Spec {
        kernel: Kernel::Density,
        shader: SURFACE,
        entry: "density",
        particles: &[0, 1, 9],
        vessel: false,
        bodies: &[],
        surface: &[0, 1],
        read_only: SURFACE_READS,
    },
    Spec {
        kernel: Kernel::PlaceVertices,
        shader: SURFACE,
        entry: "place_vertices",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 1, 2, 3, 5],
        read_only: SURFACE_READS,
    },
    Spec {
        kernel: Kernel::PlaceQuads,
        shader: SURFACE,
        entry: "place_quads",
        particles: &[0],
        vessel: false,
        bodies: &[],
        surface: &[0, 1, 2, 4, 5],
        read_only: SURFACE_READS,
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
    inject_offset: u32,
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
    counters: Buffer,
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
    let stamp = [frame.substeps.len() as u32, frame.ticket];
    queue.write_buffer(
        &bodies[3],
        (STAMP_SLOT * 4) as u64,
        bytemuck::cast_slice(&stamp),
    );

    uniforms.params.clear();
    uniforms.bodies.clear();
    let inject_offset = uniforms.params.push(&frame.inject);
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
        inject_offset,
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
        counters: surface[4].clone(),
    });
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
        threads: u32,
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
        pass.dispatch_workgroups(threads.div_ceil(WORKGROUP).max(1), 1, 1);
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
    surface_params: Res<SurfaceParams>,
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
    if frame.inject.pending > 0 {
        d.run(
            encoder,
            Kernel::Inject,
            0,
            groups.inject_offset,
            0,
            vessel_now,
            frame.inject.pending,
        );
    }
    let sort =
        |encoder: &mut CommandEncoder, params_offset: u32, vessel_offset: u32, count: u32| {
            encoder.clear_buffer(&raw.cell_count, 0, None);
            d.run(
                encoder,
                Kernel::Count,
                0,
                params_offset,
                0,
                vessel_offset,
                count,
            );
            d.run(encoder, Kernel::Scan, 0, params_offset, 0, vessel_offset, 1);
            d.run(
                encoder,
                Kernel::Scatter,
                0,
                params_offset,
                0,
                vessel_offset,
                count,
            );
            encoder.copy_buffer_to_buffer(&raw.position_sorted, 0, &raw.position, 0, bytes(count));
            encoder.copy_buffer_to_buffer(&raw.velocity_next, 0, &raw.velocity, 0, bytes(count));
        };
    for (k, substep) in frame.substeps.iter().enumerate() {
        let (po, bo) = (groups.params_offsets[k], groups.bodies_offsets[k]);
        let vo = vessel.offsets.get(k).copied().unwrap_or(vessel_now);
        let count = substep.params.count;
        let samples = substep.params.sample_count;
        sort(encoder, po, vo, count);
        d.run(encoder, Kernel::Predict, 0, po, bo, vo, count);
        if samples > 0 {
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
        if samples > 0 {
            d.run(encoder, Kernel::Buoyancy, 0, po, bo, vo, samples);
        }
        d.run(encoder, Kernel::Viscosity, 0, po, bo, vo, count);
        if samples > 0 {
            d.run(encoder, Kernel::Drag, 0, po, bo, vo, samples);
        }
        encoder.copy_buffer_to_buffer(&raw.velocity_next, 0, &raw.velocity, 0, bytes(count));
    }
    if frame.changed {
        let so = groups.surface_offset;
        sort(encoder, so, vessel_now, frame.surface.count);
        encoder.clear_buffer(&raw.counters, 0, None);
        d.run(
            encoder,
            Kernel::Density,
            0,
            so,
            0,
            vessel_now,
            surface_params.corners(),
        );
        d.run(
            encoder,
            Kernel::PlaceVertices,
            0,
            so,
            0,
            vessel_now,
            surface_params.cells(),
        );
        d.run(
            encoder,
            Kernel::PlaceQuads,
            0,
            so,
            0,
            vessel_now,
            surface_params.cells(),
        );
    }
}
