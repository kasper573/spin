//! The render-world half of the water on the ground: its buffers, a pipeline per kernel, and
//! the frame's dispatch: what is asked of the water first, then each step in a pass of its own,
//! then the water in flight that has come down, then how much water there is and the surface
//! to draw. Each step also says what the water does to the bodies wading in it.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    Buffer, BufferBinding, BufferId, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
    ComputePipelineDescriptor, DynamicUniformBuffer, PipelineCache, ShaderStages, ShaderType,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};

use super::{
    MAX_POURED, MAX_SHALLOWS_CELLS, ShallowsCell, ShallowsChart, ShallowsFrame, ShallowsReady,
};
use crate::core::fluid::{FluidBuffers, FluidHulls, FluidPrepare, FluidStep, GpuBodies};
use crate::core::vessel::{VesselBinding, VesselLayout};

/// The water's buffers, a cell of the chart to an entry of each but the last, which lists what
/// is poured.
#[derive(Resource, Clone, ExtractResource)]
pub struct ShallowsBuffers {
    pub bed: Handle<ShaderBuffer>,
    pub cells: Handle<ShaderBuffer>,
    pushed: Handle<ShaderBuffer>,
    pressing: Handle<ShaderBuffer>,
    giving: Handle<ShaderBuffer>,
    poured: Handle<ShaderBuffer>,
    landed: Handle<ShaderBuffer>,
    /// The cubic metres each row of the chart holds.
    pub rows: Handle<ShaderBuffer>,
    /// The water as a surface to draw, as the water in flight has its own: vertices, the
    /// triangles' corners among them, and how many of each there are.
    pub skin_vertices: Handle<ShaderBuffer>,
    pub skin_indices: Handle<ShaderBuffer>,
    pub skin_counters: Handle<ShaderBuffer>,
}

/// The ground water's kernels for the frame, which anything reading their results the same
/// frame runs after.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ShallowsStep;

pub fn create_buffers(assets: &mut Assets<ShaderBuffer>) -> ShallowsBuffers {
    let mut make = |bytes: usize| {
        let mut buffer = ShaderBuffer::with_size(bytes, default());
        buffer.buffer_description.usage |= BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        assets.add(buffer)
    };
    ShallowsBuffers {
        bed: make(MAX_SHALLOWS_CELLS * 4),
        cells: make(MAX_SHALLOWS_CELLS * ShallowsCell::BYTES),
        pushed: make(MAX_SHALLOWS_CELLS * 16),
        pressing: make(MAX_SHALLOWS_CELLS * 24),
        giving: make(MAX_SHALLOWS_CELLS * 8),
        poured: make(MAX_POURED * 16),
        landed: make(MAX_SHALLOWS_CELLS * 16),
        rows: make(MAX_SHALLOWS_CELLS * 4),
        skin_vertices: make(2 * MAX_SKIN_CORNERS * 48),
        skin_indices: make(MAX_SKIN_INDICES * 4),
        skin_counters: make(4 * 4),
    }
}

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            prepare
                .in_set(RenderSystems::PrepareBindGroups)
                .after(FluidPrepare),
        )
        .add_systems(
            RenderGraph,
            dispatch
                .in_set(ShallowsStep)
                .after(FluidStep)
                .before(camera_driver),
        );
}

/// The most corners a chart's cells have between them, however long and narrow it is, and the
/// most corners its triangles have.
const MAX_SKIN_CORNERS: usize = 2 * MAX_SHALLOWS_CELLS + 2;
pub const SKIN_INDICES_PER_CELL: usize = 12;
const MAX_SKIN_INDICES: usize = SKIN_INDICES_PER_CELL * MAX_SHALLOWS_CELLS;

const CHART: &str = "embedded://game/core/shallows/chart.wgsl";
const SHADER: &str = "embedded://game/core/shallows/shallows.wgsl";
const SKIN: &str = "embedded://game/core/shallows/skin.wgsl";
const ABSORB: &str = "embedded://game/core/shallows/absorb.wgsl";
const WADE: &str = "embedded://game/core/shallows/wade.wgsl";
const WORKGROUP: u32 = 64;
/// Sweeps of the pressure a step, each of both colours, from where the last step left it.
const SWEEPS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Lay,
    Stand,
    Pour,
    Push,
    Want,
    SweepRed,
    SweepBlack,
    YieldTo,
    Share,
    Rise,
    SkinCorners,
    SkinCells,
    Absorb,
    TakeInFlow,
    TakeInWater,
    Measure,
    Wade,
}

const KERNELS: [(Kernel, &str); 17] = [
    (Kernel::Lay, "lay"),
    (Kernel::Stand, "stand"),
    (Kernel::Pour, "pour"),
    (Kernel::Push, "push"),
    (Kernel::Want, "want"),
    (Kernel::SweepRed, "sweep_red"),
    (Kernel::SweepBlack, "sweep_black"),
    (Kernel::YieldTo, "yield_to"),
    (Kernel::Share, "share"),
    (Kernel::Rise, "rise"),
    (Kernel::SkinCorners, "skin_corners"),
    (Kernel::SkinCells, "skin_cells"),
    (Kernel::Absorb, "absorb"),
    (Kernel::TakeInFlow, "take_in_flow"),
    (Kernel::TakeInWater, "take_in_water"),
    (Kernel::Measure, "measure"),
    (Kernel::Wade, "wade"),
];

/// Which of the four shaders, each with a layout and a bind group of its own, a kernel is of:
/// the water's, the skin's, what brings the water in flight down, or what the bodies wade by.
fn family(kernel: Kernel) -> usize {
    match kernel {
        Kernel::SkinCorners | Kernel::SkinCells => 1,
        Kernel::Absorb | Kernel::TakeInFlow | Kernel::TakeInWater => 2,
        Kernel::Wade => 3,
        _ => 0,
    }
}

#[derive(ShaderType, Clone, Copy, Default)]
struct ShallowsUniform {
    size: UVec2,
    wraps: UVec2,
    cell: Vec2,
    low: Vec2,
    dt: f32,
    bed_friction: f32,
    pouring: u32,
    restoring: u32,
    standing: Vec4,
    particles: u32,
    particle_volume: f32,
    landing: f32,
    hull_samples: u32,
    accumulators: u32,
    body_drag: f32,
    rest_density: f32,
}

#[derive(Resource)]
struct Pipelines {
    ids: Vec<CachedComputePipelineId>,
    layouts: [BindGroupLayoutDescriptor; 4],
    // held for as long as the kernels that import it may be compiled
    _chart: Handle<Shader>,
}

#[derive(Resource, Default)]
struct Uniforms(DynamicUniformBuffer<ShallowsUniform>);

/// The water's bind groups, kept from frame to frame while the buffers they are `made_of`
/// stay; where in the uniforms the frame's asking and each of its steps lie; and where in the
/// bodies' uniform each step's bodies lie, of which there are none while no hull is wading.
#[derive(Resource)]
struct BindGroups {
    families: [BindGroup; 4],
    made_of: Vec<BufferId>,
    asking: u32,
    steps: Vec<u32>,
    hulls: Vec<u32>,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    vessel: Res<VesselLayout>,
) {
    // binding 0 the uniform, then storage buffers, read-only where marked
    let layout = |label: &'static str, read_only: &[bool]| {
        let mut entries: Vec<BindGroupLayoutEntry> =
            vec![uniform_buffer_sized(true, None).build(0, ShaderStages::COMPUTE)];
        entries.extend(read_only.iter().enumerate().map(|(i, &read_only)| {
            if read_only {
                storage_buffer_read_only_sized(false, None)
            } else {
                storage_buffer_sized(false, None)
            }
            .build(i as u32 + 1, ShaderStages::COMPUTE)
        }));
        BindGroupLayoutDescriptor::new(label, &entries)
    };
    let layouts = [
        layout(
            "shallows",
            &[false, false, false, false, false, true, false],
        ),
        layout("shallows skin", &[true, true, false, false, false]),
        layout("shallows absorb", &[true, false, false, true, false, false]),
        {
            let mut wade = layout("shallows wade", &[true, true, true, true, false]);
            wade.entries
                .push(uniform_buffer_sized(true, None).build(6, ShaderStages::COMPUTE));
            wade
        },
    ];
    let ids = KERNELS
        .iter()
        .map(|(kernel, entry)| {
            let family = family(*kernel);
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some((*entry).into()),
                layout: vec![layouts[family].clone(), vessel.0.clone()],
                shader: asset_server.load([SHADER, SKIN, ABSORB, WADE][family]),
                entry_point: Some((*entry).into()),
                ..default()
            })
        })
        .collect();
    commands.insert_resource(Pipelines {
        ids,
        layouts,
        _chart: asset_server.load(CHART),
    });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    pipelines: Res<Pipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    (buffers, flying, hulls): (
        Res<ShallowsBuffers>,
        Res<FluidBuffers>,
        Option<Res<FluidHulls>>,
    ),
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    frame: Res<ShallowsFrame>,
    mut uniforms: ResMut<Uniforms>,
    made: Option<ResMut<BindGroups>>,
) {
    let (Some(chart), Some(hulls)) = (frame.chart, hulls) else {
        commands.remove_resource::<BindGroups>();
        return;
    };
    let handles = [
        &buffers.bed,
        &buffers.cells,
        &buffers.pushed,
        &buffers.pressing,
        &buffers.giving,
        &buffers.poured,
        &buffers.skin_vertices,
        &buffers.skin_indices,
        &buffers.skin_counters,
        &buffers.landed,
        &flying.position,
        &flying.velocity,
        &buffers.rows,
        &flying.boundary,
        &flying.samples,
        &flying.accum,
    ];
    let Some(raw) = handles
        .iter()
        .map(|h| gpu_buffers.get(*h).map(|b| b.buffer.clone()))
        .collect::<Option<Vec<Buffer>>>()
    else {
        return;
    };
    if !frame.poured.is_empty() {
        queue.write_buffer(&raw[5], 0, bytemuck::cast_slice(&frame.poured));
    }
    if let Some(kept) = &frame.restored {
        queue.write_buffer(&raw[1], 0, bytemuck::cast_slice(kept));
    }
    let uniform = |dt: f32| ShallowsUniform {
        size: UVec2::from(chart.size),
        wraps: UVec2::new(chart.wraps[0] as u32, chart.wraps[1] as u32),
        cell: Vec2::new(chart.cell[0].0, chart.cell[1].0),
        low: Vec2::from(frame.low),
        dt,
        bed_friction: frame.bed_friction,
        pouring: frame.poured.len() as u32,
        restoring: frame.restored.is_some() as u32,
        standing: Vec4::from(frame.standing.unwrap_or_default()),
        particles: frame.flying.count,
        particle_volume: frame.flying.volume.0 / 1000.0,
        landing: frame.flying.landing.0,
        hull_samples: hulls.samples,
        accumulators: hulls.accumulators,
        body_drag: hulls.body_drag,
        rest_density: hulls.rest_density,
    };
    uniforms.0.clear();
    let asking = uniforms.0.push(&uniform(0.0));
    let steps: Vec<u32> = frame
        .steps
        .iter()
        .map(|dt| uniforms.0.push(&uniform(dt.0)))
        .collect();
    uniforms.0.write_buffer(&device, &queue);
    let Some(binding) = uniforms.0.binding() else {
        return;
    };
    let wading = if hulls.samples > 0 {
        hulls.offsets.clone()
    } else {
        Vec::new()
    };
    let made_of: Vec<BufferId> = raw
        .iter()
        .map(Buffer::id)
        .chain(uniforms.0.buffer().map(Buffer::id))
        .chain([hulls.bodies.id()])
        .collect();
    if let Some(mut made) = made
        && made.made_of == made_of
    {
        made.asking = asking;
        made.steps = steps;
        made.hulls = wading;
        return;
    }
    // the water's own buffers, then, for the skin, the ground and the cells with the skin's
    let bound = |buffers: &[usize], layout: &BindGroupLayoutDescriptor| {
        let mut entries = vec![BindGroupEntry {
            binding: 0,
            resource: binding.clone(),
        }];
        entries.extend(
            buffers
                .iter()
                .enumerate()
                .map(|(i, &buffer)| BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: raw[buffer].as_entire_binding(),
                }),
        );
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(layout),
            &entries,
        )
    };
    let mut wade = vec![BindGroupEntry {
        binding: 0,
        resource: binding.clone(),
    }];
    wade.extend(
        [0, 1, 13, 14, 15]
            .iter()
            .enumerate()
            .map(|(i, &buffer)| BindGroupEntry {
                binding: i as u32 + 1,
                resource: raw[buffer].as_entire_binding(),
            }),
    );
    wade.push(BindGroupEntry {
        binding: 6,
        resource: BindingResource::Buffer(BufferBinding {
            buffer: &hulls.bodies,
            offset: 0,
            size: Some(GpuBodies::min_size()),
        }),
    });
    let families = [
        bound(&[0, 1, 2, 3, 4, 5, 12], &pipelines.layouts[0]),
        bound(&[0, 1, 6, 7, 8], &pipelines.layouts[1]),
        bound(&[0, 1, 10, 11, 9, 3], &pipelines.layouts[2]),
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipelines.layouts[3]),
            &wade,
        ),
    ];
    commands.insert_resource(BindGroups {
        families,
        made_of,
        asking,
        steps,
        hulls: wading,
    });
}

fn dispatch(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    groups: Option<Res<BindGroups>>,
    vessel: Option<Res<VesselBinding>>,
    (frame, hulls, ready): (
        Res<ShallowsFrame>,
        Option<Res<FluidHulls>>,
        Res<ShallowsReady>,
    ),
    mut drawn_dry: Local<Option<ShallowsChart>>,
) {
    let Some(vessel) = vessel else {
        return;
    };
    let Some(computes) = pipelines
        .ids
        .iter()
        .map(|id| cache.get_compute_pipeline(*id))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let Some(&vessel_now) = vessel.offsets.last() else {
        return;
    };
    ready.set();
    let (Some(groups), Some(chart)) = (groups, frame.chart) else {
        return;
    };
    // dry ground of a chart whose skin was last drawn dry has nothing to be done to it
    if frame.dry && *drawn_dry == Some(chart) {
        return;
    }
    *drawn_dry = frame.dry.then_some(chart);
    let cells = (chart.cells() as u32).div_ceil(WORKGROUP);
    let corners = ((chart.size[0] + 1) * (chart.size[1] + 1)).div_ceil(WORKGROUP);
    let encoder = render_context.command_encoder();
    let rows = chart.size[1].div_ceil(WORKGROUP);
    let mut run = |label: &'static str, offsets: &[u32], vessel_at: u32, kernels: &[Kernel]| {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            ..default()
        });
        for kernel in kernels {
            let index = KERNELS
                .iter()
                .position(|(k, _)| k == kernel)
                .unwrap_or_default();
            pass.set_pipeline(computes[index]);
            pass.set_bind_group(0, &groups.families[family(*kernel)], offsets);
            pass.set_bind_group(1, &vessel.bind_group, &[vessel_at]);
            let threads = match kernel {
                Kernel::Pour => 1,
                Kernel::SkinCorners => corners,
                Kernel::Absorb => frame.flying.count.div_ceil(WORKGROUP),
                Kernel::Measure => rows,
                Kernel::Wade => hulls.as_ref().map_or(0, |h| h.samples).div_ceil(WORKGROUP),
                _ => cells,
            };
            pass.dispatch_workgroups(threads, 1, 1);
        }
    };
    let mut asked = vec![Kernel::Lay];
    if frame.standing.is_some() {
        asked.push(Kernel::Stand);
    }
    if !frame.poured.is_empty() {
        asked.push(Kernel::Pour);
    }
    run("shallows asked", &[groups.asking], vessel_now, &asked);
    let mut step = vec![Kernel::Push, Kernel::Want];
    step.extend([Kernel::SweepRed, Kernel::SweepBlack].repeat(SWEEPS));
    step.extend([Kernel::YieldTo, Kernel::Share, Kernel::Rise]);
    for (k, &params) in groups.steps.iter().enumerate() {
        let vessel_at = vessel.offsets.get(k).copied().unwrap_or(vessel_now);
        run("shallows step", &[params], vessel_at, &step);
        if let Some(&bodies) = groups.hulls.get(k) {
            run(
                "shallows wade",
                &[params, bodies],
                vessel_at,
                &[Kernel::Wade],
            );
        }
    }
    if frame.flying.count > 0 {
        run(
            "shallows absorb",
            &[groups.asking],
            vessel_now,
            &[Kernel::Absorb, Kernel::TakeInFlow, Kernel::TakeInWater],
        );
    }
    run(
        "shallows measure",
        &[groups.asking],
        vessel_now,
        &[Kernel::Measure],
    );
    run(
        "shallows skin",
        &[groups.asking],
        vessel_now,
        &[Kernel::SkinCorners, Kernel::SkinCells],
    );
}
