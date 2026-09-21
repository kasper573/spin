//! The render world's half of the sheet: its buffers, a pipeline for each kernel of
//! `sheet.wgsl`, and the frame's order of them: the water carried over if the sheet was laid
//! anew, the water poured, then as many steps as the frame's time takes, each as long as the
//! GPU itself finds its quickest wave allows, and at last the surface to be drawn and the water
//! there is, counted.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
    BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    PipelineCache, ShaderStages, UniformBuffer,
};
use bevy::render::renderer::{
    RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue,
};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};

use super::{
    JETS_SHADER, JetsParams, SHADER, SHEET_CELLS, SHEET_CORNERS, SHEET_INDICES, SHEET_MOST_GRAINS,
    SHEET_WATCHED, SHEET_WATCHED_CORNERS, SheetFrame, SheetParams, SheetReady,
};

/// `jets.wgsl`'s counts: the drains, the points round a hoop's rim, and the hoops.
pub const DRAINS: usize = 2;
const RIM: usize = 16;
const HOOPS: usize = 512;
/// The corners of all the triangles the jets' surface can have.
pub const JET_INDICES: usize = 6 * RIM * HOOPS;

/// Threads to a workgroup of the kernels that go row by row.
const ROW_THREADS: u32 = 64;
/// And, each way, of those that go cell by cell.
const CELL_THREADS: u32 = 8;

#[derive(Resource, Clone, ExtractResource)]
pub struct SheetBuffers {
    bed: Handle<ShaderBuffer>,
    /// The water where a step begins, and where its first half leaves it.
    state: Handle<ShaderBuffer>,
    halfway: Handle<ShaderBuffer>,
    crossing_round: Handle<ShaderBuffer>,
    crossing_along: Handle<ShaderBuffer>,
    clock: Handle<ShaderBuffer>,
    threads: Handle<ShaderBuffer>,
    carried: Handle<ShaderBuffer>,
    /// What has run down the drains, the hoops of it in the air, and what is known of them.
    drained: Handle<ShaderBuffer>,
    hoops: Handle<ShaderBuffer>,
    flights: Handle<ShaderBuffer>,
    /// The jets' surface, kept as the sheet's own is below.
    pub jet_vertices: Handle<ShaderBuffer>,
    pub jet_indices: Handle<ShaderBuffer>,
    pub jet_counters: Handle<ShaderBuffer>,
    /// The surface, as the water's shader draws one: its vertices, the corners of its
    /// triangles, and how many of those there are, second of four counts.
    pub vertices: Handle<ShaderBuffer>,
    pub indices: Handle<ShaderBuffer>,
    pub counters: Handle<ShaderBuffer>,
    /// The water in each row of cells, in cubic metres.
    pub held: Handle<ShaderBuffer>,
    /// The water round the corners of the watched block: see `sheet.wgsl`.
    pub watched: Handle<ShaderBuffer>,
}

pub fn create_buffers(assets: &mut Assets<ShaderBuffer>) -> SheetBuffers {
    let mut make = |bytes: usize| {
        let mut buffer = ShaderBuffer::with_size(bytes, default());
        buffer.buffer_description.usage |=
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::INDIRECT;
        assets.add(buffer)
    };
    let cells = (SHEET_CELLS[0] * SHEET_CELLS[1]) as usize;
    let rows = SHEET_CELLS[1] as usize;
    let grains = ((SHEET_CELLS[0] + SHEET_CELLS[1]) * SHEET_MOST_GRAINS) as usize;
    SheetBuffers {
        bed: make(SHEET_CORNERS * 4),
        state: make(cells * 16),
        halfway: make(cells * 16),
        crossing_round: make(cells * 16),
        crossing_along: make(cells * 16),
        clock: make(12 + rows * 4),
        threads: make(16),
        carried: make((6 + grains) * 4),
        drained: make(cells * 16),
        hoops: make(HOOPS * (2 * RIM * 16 + 32)),
        flights: make(4 + 2 * DRAINS * 4),
        jet_vertices: make(JET_INDICES / 6 * 48),
        jet_indices: make(JET_INDICES * 4),
        jet_counters: make(16),
        vertices: make(SHEET_CORNERS * 48),
        indices: make(SHEET_INDICES * 4),
        counters: make(16),
        held: make((rows + 1) * 4),
        watched: make((1 + SHEET_WATCHED_CORNERS) * 16),
    }
}

/// The sheet's kernels for the frame, which anything reading what they leave the same frame
/// runs after.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SheetStep;

pub fn install(render_app: &mut SubApp) {
    render_app
        .init_resource::<Uniform>()
        .init_resource::<JetsUniform>()
        .init_resource::<Seen>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            RenderGraph,
            dispatch
                .in_set(SheetStep)
                .in_set(RenderGraphSystems::Render)
                .before(camera_driver),
        );
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    CarryRound,
    CarryAlong,
    Pour,
    Quickest,
    Pace,
    Cross,
    Drain,
    StepOnce,
    StepTwice,
    Raise,
    Face,
    Measure,
    Report,
    Press,
    Sink,
    Bear,
    Fly,
    Land,
    Weigh,
    Draw,
}

/// Which of `sheet.wgsl`'s bindings a kernel's layout holds: the carrying's, the solver's, the
/// pacing's, which alone writes how many threads the step's kernels are sent out with, or the
/// surface's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    Carrying,
    Jets,
    Solver,
    Pacing,
    Surface,
}

const KERNELS: [(Kernel, &str, Bound); 20] = [
    (Kernel::CarryRound, "carry_round", Bound::Carrying),
    (Kernel::CarryAlong, "carry_along", Bound::Carrying),
    (Kernel::Pour, "pour", Bound::Solver),
    (Kernel::Quickest, "quickest", Bound::Solver),
    (Kernel::Pace, "pace", Bound::Pacing),
    (Kernel::Cross, "cross", Bound::Solver),
    (Kernel::Drain, "drain", Bound::Solver),
    (Kernel::StepOnce, "step_once", Bound::Solver),
    (Kernel::StepTwice, "step_twice", Bound::Solver),
    (Kernel::Raise, "raise", Bound::Surface),
    (Kernel::Face, "face", Bound::Surface),
    (Kernel::Measure, "measure", Bound::Surface),
    (Kernel::Report, "report", Bound::Surface),
    (Kernel::Press, "press", Bound::Jets),
    (Kernel::Sink, "sink", Bound::Jets),
    (Kernel::Bear, "bear", Bound::Jets),
    (Kernel::Fly, "fly", Bound::Jets),
    (Kernel::Land, "land", Bound::Jets),
    (Kernel::Weigh, "weigh", Bound::Jets),
    (Kernel::Draw, "draw", Bound::Jets),
];

impl Bound {
    /// The bindings, and which of them are only read.
    fn bindings(self) -> (&'static [u32], &'static [u32]) {
        match self {
            Bound::Carrying => (&[0, 2, 3, 13], &[2, 13]),
            Bound::Jets => (&[0, 1, 3, 6, 10, 14, 15, 16, 17, 18, 19, 20], &[1]),
            Bound::Solver => (&[0, 1, 2, 3, 4, 5, 6], &[1, 2]),
            Bound::Pacing => (&[0, 6, 11], &[]),
            Bound::Surface => (&[0, 1, 2, 6, 7, 8, 9, 10, 12], &[1, 2]),
        }
    }

    fn layout(self) -> BindGroupLayoutDescriptor {
        let (bindings, read_only) = self.bindings();
        let entries: Vec<BindGroupLayoutEntry> = bindings
            .iter()
            .map(|&binding| {
                if binding == 0 {
                    uniform_buffer::<SheetParams>(false)
                } else if binding == 14 {
                    uniform_buffer::<JetsParams>(false)
                } else if read_only.contains(&binding) {
                    storage_buffer_read_only_sized(false, None)
                } else {
                    storage_buffer_sized(false, None)
                }
                .build(binding, ShaderStages::COMPUTE)
            })
            .collect();
        BindGroupLayoutDescriptor::new("sheet", &entries)
    }
}

#[derive(Resource)]
struct Pipelines(Vec<CachedComputePipelineId>);

fn init_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let ids = KERNELS
        .iter()
        .map(|(_, entry, bound)| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some((*entry).into()),
                layout: vec![bound.layout()],
                shader: assets.load(if *bound == Bound::Jets {
                    JETS_SHADER
                } else {
                    SHADER
                }),
                entry_point: Some((*entry).into()),
                ..default()
            })
        })
        .collect();
    commands.insert_resource(Pipelines(ids));
}

#[derive(Resource, Default)]
struct Uniform(UniformBuffer<SheetParams>);

#[derive(Resource, Default)]
struct JetsUniform(UniformBuffer<JetsParams>);

/// The floor, the carrying and the emptying the GPU has been brought up to.
#[derive(Resource, Default)]
struct Seen {
    floors: u32,
    carries: u32,
    emptied: u32,
}

/// The solver's bindings with the water read where a step begins and written halfway, and the
/// other way about; the carrying's, likewise; the pacing's; and the surface's, which reads the
/// water where steps begin.
#[derive(Resource)]
struct BindGroups {
    forth: BindGroup,
    back: BindGroup,
    carry_forth: BindGroup,
    carry_back: BindGroup,
    jets: BindGroup,
    pacing: BindGroup,
    surface: BindGroup,
    raw: Raw,
}

struct Raw {
    bed: Buffer,
    state: Buffer,
    halfway: Buffer,
    clock: Buffer,
    threads: Buffer,
    carried: Buffer,
    drained: Buffer,
    hoops: Buffer,
    flights: Buffer,
    jet_counters: Buffer,
    counters: Buffer,
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<SheetBuffers>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    frame: Res<SheetFrame>,
    (mut uniform, mut jets_uniform): (ResMut<Uniform>, ResMut<JetsUniform>),
    made: Option<Res<BindGroups>>,
) {
    uniform.0.set(frame.params);
    uniform.0.write_buffer(&device, &queue);
    jets_uniform.0.set(frame.jets);
    jets_uniform.0.write_buffer(&device, &queue);
    if made.is_some() {
        return;
    }
    let get = |handle: &Handle<ShaderBuffer>| gpu_buffers.get(handle).map(|b| b.buffer.clone());
    let (Some(params), Some(drains), Some(all)) = (
        uniform.0.binding(),
        jets_uniform.0.binding(),
        [
            &buffers.bed,
            &buffers.state,
            &buffers.halfway,
            &buffers.crossing_round,
            &buffers.crossing_along,
            &buffers.clock,
            &buffers.vertices,
            &buffers.indices,
            &buffers.counters,
            &buffers.held,
            &buffers.threads,
            &buffers.watched,
            &buffers.carried,
            &buffers.drained,
            &buffers.hoops,
            &buffers.flights,
            &buffers.jet_vertices,
            &buffers.jet_indices,
            &buffers.jet_counters,
        ]
        .into_iter()
        .map(get)
        .collect::<Option<Vec<Buffer>>>(),
    ) else {
        return;
    };
    let [
        bed,
        state,
        halfway,
        crossing_round,
        crossing_along,
        clock,
        vertices,
        indices,
        counters,
        held,
        threads,
        watched,
        carried,
        drained,
        hoops,
        flights,
        jet_vertices,
        jet_indices,
        jet_counters,
    ] = &all[..]
    else {
        return;
    };
    let group = |bound: Bound, entries: &[(u32, &Buffer)]| {
        let entries: Vec<BindGroupEntry> = std::iter::once(BindGroupEntry {
            binding: 0,
            resource: params.clone(),
        })
        .chain(entries.iter().map(|(binding, buffer)| BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        }))
        .collect();
        device.create_bind_group(
            "sheet",
            &cache.get_bind_group_layout(&bound.layout()),
            &entries,
        )
    };
    let solver = |before: &Buffer, after: &Buffer| {
        group(
            Bound::Solver,
            &[
                (1, bed),
                (2, before),
                (3, after),
                (4, crossing_round),
                (5, crossing_along),
                (6, clock),
            ],
        )
    };
    let carrying = |before: &Buffer, after: &Buffer| {
        group(Bound::Carrying, &[(2, before), (3, after), (13, carried)])
    };
    commands.insert_resource(BindGroups {
        forth: solver(state, halfway),
        back: solver(halfway, state),
        carry_forth: carrying(state, halfway),
        carry_back: carrying(halfway, state),
        jets: {
            let buffers = [
                (1, bed),
                (3, state),
                (6, clock),
                (10, held),
                (15, drained),
                (16, hoops),
                (17, flights),
                (18, jet_vertices),
                (19, jet_indices),
                (20, jet_counters),
            ];
            let entries: Vec<BindGroupEntry> = [(0, params.clone()), (14, drains.clone())]
                .into_iter()
                .chain(buffers.map(|(binding, buffer)| (binding, buffer.as_entire_binding())))
                .map(|(binding, resource)| BindGroupEntry { binding, resource })
                .collect();
            device.create_bind_group(
                "sheet",
                &cache.get_bind_group_layout(&Bound::Jets.layout()),
                &entries,
            )
        },
        pacing: group(Bound::Pacing, &[(6, clock), (11, threads)]),
        surface: group(
            Bound::Surface,
            &[
                (1, bed),
                (2, state),
                (6, clock),
                (7, vertices),
                (8, indices),
                (9, counters),
                (10, held),
                (12, watched),
            ],
        ),
        raw: Raw {
            bed: bed.clone(),
            state: state.clone(),
            halfway: halfway.clone(),
            clock: clock.clone(),
            threads: threads.clone(),
            carried: carried.clone(),
            drained: drained.clone(),
            hoops: hoops.clone(),
            flights: flights.clone(),
            jet_counters: jet_counters.clone(),
            counters: counters.clone(),
        },
    });
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    groups: Option<Res<BindGroups>>,
    frame: Res<SheetFrame>,
    queue: Res<RenderQueue>,
    mut seen: ResMut<Seen>,
    ready: Res<SheetReady>,
) {
    let Some(groups) = groups else {
        return;
    };
    let compiled: Option<Vec<_>> = pipelines
        .0
        .iter()
        .map(|id| cache.get_compute_pipeline(*id))
        .collect();
    let Some(compiled) = compiled else {
        return;
    };
    ready.set();
    let used = frame.params.used;
    if used.x == 0 || used.y == 0 {
        return;
    }
    if seen.floors != frame.floors {
        queue.write_buffer(&groups.raw.bed, 0, bytemuck::cast_slice(&frame.bed));
        seen.floors = frame.floors;
    }
    let carry = seen.carries != frame.carries && frame.wetted;
    seen.carries = frame.carries;
    if carry {
        queue.write_buffer(&groups.raw.carried, 0, bytemuck::cast_slice(&frame.carried));
    }
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    if seen.emptied != frame.emptied {
        encoder.clear_buffer(&groups.raw.state, 0, None);
        encoder.clear_buffer(&groups.raw.halfway, 0, None);
        encoder.clear_buffer(&groups.raw.clock, 0, None);
        encoder.clear_buffer(&groups.raw.drained, 0, None);
        encoder.clear_buffer(&groups.raw.hoops, 0, None);
        encoder.clear_buffer(&groups.raw.flights, 0, None);
        seen.emptied = frame.emptied;
    }
    encoder.clear_buffer(&groups.raw.counters, 0, None);
    encoder.clear_buffer(&groups.raw.jet_counters, 0, None);
    // a sheet nothing has been poured on has no water to step, draw or count
    if !frame.wetted {
        return;
    }
    let span = diagnostics.time_span(encoder, "sheet");
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("sheet"),
        ..default()
    });
    let pipeline = |kernel: Kernel| {
        let index = KERNELS.iter().position(|(k, ..)| *k == kernel).unwrap_or(0);
        compiled[index]
    };
    let corners = [
        (used.x + 1).div_ceil(CELL_THREADS),
        (used.y + 1).div_ceil(CELL_THREADS),
    ];
    if carry {
        for (half, kernel) in [
            (&groups.carry_forth, Kernel::CarryRound),
            (&groups.carry_back, Kernel::CarryAlong),
        ] {
            pass.set_pipeline(pipeline(kernel));
            pass.set_bind_group(0, half, &[]);
            pass.dispatch_workgroups(
                SHEET_CELLS[0].div_ceil(CELL_THREADS),
                SHEET_CELLS[1].div_ceil(CELL_THREADS),
                1,
            );
        }
    }
    let draining = frame
        .jets
        .drains
        .iter()
        .map(|drain| drain.cells.max_element())
        .max()
        .map_or(0, |cells| cells.div_ceil(CELL_THREADS));
    let poured = frame.params.pour_cells;
    pass.set_pipeline(pipeline(Kernel::Pour));
    pass.set_bind_group(0, &groups.back, &[]);
    pass.dispatch_workgroups(
        poured.x.div_ceil(CELL_THREADS).max(1),
        poured.y.div_ceil(CELL_THREADS).max(1),
        1,
    );
    for _ in 0..frame.steps {
        pass.set_pipeline(pipeline(Kernel::Quickest));
        pass.set_bind_group(0, &groups.forth, &[]);
        pass.dispatch_workgroups(used.y.div_ceil(ROW_THREADS), 1, 1);
        pass.set_pipeline(pipeline(Kernel::Pace));
        pass.set_bind_group(0, &groups.pacing, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        for (half, kernel) in [
            (&groups.forth, Kernel::StepOnce),
            (&groups.back, Kernel::StepTwice),
        ] {
            pass.set_bind_group(0, half, &[]);
            for kernel in [Kernel::Cross, Kernel::Drain, kernel] {
                pass.set_pipeline(pipeline(kernel));
                pass.dispatch_workgroups_indirect(&groups.raw.threads, 0);
            }
        }
        if draining > 0 {
            pass.set_bind_group(0, &groups.jets, &[]);
            pass.set_pipeline(pipeline(Kernel::Press));
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(pipeline(Kernel::Sink));
            pass.dispatch_workgroups(draining, draining, 1);
        }
    }
    pass.set_bind_group(0, &groups.jets, &[]);
    for (kernel, threads) in [
        (Kernel::Bear, 1),
        (Kernel::Fly, (HOOPS as u32).div_ceil(ROW_THREADS)),
        (Kernel::Land, 1),
        (Kernel::Weigh, 1),
        (Kernel::Draw, (HOOPS as u32).div_ceil(ROW_THREADS)),
    ] {
        pass.set_pipeline(pipeline(kernel));
        pass.dispatch_workgroups(threads, 1, 1);
    }
    pass.set_bind_group(0, &groups.surface, &[]);
    pass.set_pipeline(pipeline(Kernel::Raise));
    pass.dispatch_workgroups(corners[0], corners[1], 1);
    pass.set_pipeline(pipeline(Kernel::Face));
    pass.dispatch_workgroups(corners[0], corners[1], 1);
    pass.set_pipeline(pipeline(Kernel::Measure));
    pass.dispatch_workgroups(SHEET_CELLS[1].div_ceil(ROW_THREADS), 1, 1);
    pass.set_pipeline(pipeline(Kernel::Report));
    let watched = (SHEET_WATCHED + 1).div_ceil(CELL_THREADS);
    pass.dispatch_workgroups(watched, watched, 1);
    drop(pass);
    span.end(encoder);
}
