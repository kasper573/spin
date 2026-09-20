//! The render-world half of the water on the ground: its buffers, a pipeline per kernel, and
//! the frame's dispatch: what is asked of the water first, then each step in a pass of its own.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferId,
    BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    DynamicUniformBuffer, PipelineCache, ShaderStages, ShaderType,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};

use super::{MAX_POURED, MAX_SHALLOWS_CELLS, ShallowsCell, ShallowsFrame, ShallowsReady};
use crate::core::fluid::FluidStep;
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
    }
}

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniforms>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch
                .in_set(ShallowsStep)
                .after(FluidStep)
                .before(camera_driver),
        );
}

const SHADER: &str = "embedded://game/core/shallows/shallows.wgsl";
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
}

const KERNELS: [(Kernel, &str); 10] = [
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
];

#[derive(ShaderType, Clone, Copy, Default)]
struct ShallowsUniform {
    size: UVec2,
    wraps: UVec2,
    cell: Vec2,
    low: Vec2,
    dt: f32,
    bed_friction: f32,
    pouring: u32,
    standing: Vec4,
}

#[derive(Resource)]
struct Pipelines {
    ids: Vec<CachedComputePipelineId>,
    layout: BindGroupLayoutDescriptor,
}

#[derive(Resource, Default)]
struct Uniforms(DynamicUniformBuffer<ShallowsUniform>);

/// The water's bind group, kept from frame to frame while the buffers it is `made_of` stay, and
/// where in the uniforms the frame's asking and each of its steps lie.
#[derive(Resource)]
struct BindGroups {
    group: BindGroup,
    made_of: Vec<BufferId>,
    asking: u32,
    steps: Vec<u32>,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    vessel: Res<VesselLayout>,
) {
    let mut entries: Vec<BindGroupLayoutEntry> =
        vec![uniform_buffer_sized(true, None).build(0, ShaderStages::COMPUTE)];
    entries
        .extend((1..=5).map(|binding| {
            storage_buffer_sized(false, None).build(binding, ShaderStages::COMPUTE)
        }));
    entries.push(storage_buffer_read_only_sized(false, None).build(6, ShaderStages::COMPUTE));
    let layout = BindGroupLayoutDescriptor::new("shallows", &entries);
    let ids = KERNELS
        .iter()
        .map(|(_, entry)| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some((*entry).into()),
                layout: vec![layout.clone(), vessel.0.clone()],
                shader: asset_server.load(SHADER),
                entry_point: Some((*entry).into()),
                ..default()
            })
        })
        .collect();
    commands.insert_resource(Pipelines { ids, layout });
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    pipelines: Res<Pipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<ShallowsBuffers>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    frame: Res<ShallowsFrame>,
    mut uniforms: ResMut<Uniforms>,
    made: Option<ResMut<BindGroups>>,
) {
    let Some(chart) = frame.chart else {
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
    let uniform = |dt: f32| ShallowsUniform {
        size: UVec2::from(chart.size),
        wraps: UVec2::new(chart.wraps[0] as u32, chart.wraps[1] as u32),
        cell: Vec2::new(chart.cell[0].0, chart.cell[1].0),
        low: Vec2::from(frame.low),
        dt,
        bed_friction: frame.bed_friction,
        pouring: frame.poured.len() as u32,
        standing: Vec4::from(frame.standing.unwrap_or_default()),
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
    let made_of: Vec<BufferId> = raw
        .iter()
        .map(Buffer::id)
        .chain(uniforms.0.buffer().map(Buffer::id))
        .collect();
    if let Some(mut made) = made
        && made.made_of == made_of
    {
        made.asking = asking;
        made.steps = steps;
        return;
    }
    let mut entries = vec![BindGroupEntry {
        binding: 0,
        resource: binding,
    }];
    entries.extend(raw.iter().enumerate().map(|(i, buffer)| BindGroupEntry {
        binding: i as u32 + 1,
        resource: buffer.as_entire_binding(),
    }));
    let group = device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipelines.layout),
        &entries,
    );
    commands.insert_resource(BindGroups {
        group,
        made_of,
        asking,
        steps,
    });
}

fn dispatch(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    groups: Option<Res<BindGroups>>,
    vessel: Option<Res<VesselBinding>>,
    frame: Res<ShallowsFrame>,
    ready: Res<ShallowsReady>,
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
    let cells = (chart.cells() as u32).div_ceil(WORKGROUP);
    let encoder = render_context.command_encoder();
    let mut run = |label: &'static str, params: u32, vessel_at: u32, kernels: &[Kernel]| {
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
            pass.set_bind_group(0, &groups.group, &[params]);
            pass.set_bind_group(1, &vessel.bind_group, &[vessel_at]);
            let threads = if *kernel == Kernel::Pour { 1 } else { cells };
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
    run("shallows asked", groups.asking, vessel_now, &asked);
    let mut step = vec![Kernel::Push, Kernel::Want];
    step.extend([Kernel::SweepRed, Kernel::SweepBlack].repeat(SWEEPS));
    step.extend([Kernel::YieldTo, Kernel::Share, Kernel::Rise]);
    for (k, &params) in groups.steps.iter().enumerate() {
        let vessel_at = vessel.offsets.get(k).copied().unwrap_or(vessel_now);
        run("shallows step", params, vessel_at, &step);
    }
}
