//! Position-based fluid (Macklin & Müller 2013) solved on the GPU with Akinci-style coupling to
//! rigid bodies, confined by a vessel the shaders describe. The CPU only keeps the particle count,
//! feeds in new particles and body poses each frame, and reads back what the water did to the
//! bodies. SI units throughout; the particle spacing fixes the kernel and the rest mass.
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::Shader;

use crate::core::math::Rng;
use crate::core::rigid::{Body, BodyShape, WaterCoupling};
use crate::core::units::Seconds;

mod frame;
mod gpu;
mod surface;

pub use frame::{Bodies, FluidFrame, GpuBodies, Params, Substep};
pub use gpu::FluidBuffers;

/// Set by the render world once every kernel has compiled and the vessel is bound; until then
/// nothing is handed to the GPU, so nothing is lost.
#[derive(Resource, Clone, Default)]
pub struct FluidReady(Arc<AtomicBool>);

impl FluidReady {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
pub use surface::{MAX_INDICES, SurfaceBuffers};

/// Rest spacing between particles (m); every kernel constant derives from it.
pub const PARTICLE_SPACING: f32 = 0.10;
pub const H: f32 = 2.0 * PARTICLE_SPACING;
pub const H2: f32 = H * H;
pub const REST_DENSITY: f32 = 1000.0;
pub const PARTICLE_MASS: f32 =
    REST_DENSITY * PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING;
pub const MAX_PARTICLES: usize = 32768;
/// Bodies the shaders reserve room for.
pub const MAX_BODIES: usize = 16;
/// Boundary samples over all solid bodies.
pub const MAX_SAMPLES: usize = 2048;
/// Substeps a single frame may run; beyond that the simulation falls behind real time.
pub const MAX_SUBSTEPS_PER_FRAME: usize = 4;
pub const POLY6: f32 = 315.0 / (64.0 * PI * H * H * H * H * H * H * H * H * H);
pub const MAX_SPEED: f32 = 15.0;
pub const MAX_SPEED_BODY: f64 = 15.0;
const SPIKY: f32 = -45.0 / (PI * H * H * H * H * H * H);
const W0: f32 = POLY6 * H2 * H2 * H2;
const EPS_LAMBDA: f32 = 0.02;
const ITERATIONS: usize = 3;
const SCORR_K: f32 = 0.001;
const SCORR_DQ: f32 = 0.3 * H;
const SCORR_WQ: f32 =
    POLY6 * (H2 - SCORR_DQ * SCORR_DQ) * (H2 - SCORR_DQ * SCORR_DQ) * (H2 - SCORR_DQ * SCORR_DQ);
const MAX_DELTA: f32 = 0.5 * PARTICLE_SPACING;
const WET_REF: f32 = 300.0;
const MARGIN: f32 = PARTICLE_SPACING * 0.5;
const SHADERS: [&str; 5] = [
    "embedded://game/core/fluid/shaders/common.wgsl",
    "embedded://game/core/fluid/shaders/particles.wgsl",
    "embedded://game/core/fluid/shaders/sort.wgsl",
    "embedded://game/core/fluid/shaders/bodies.wgsl",
    "embedded://game/core/fluid/shaders/surface.wgsl",
];

#[derive(Clone, Debug, PartialEq)]
pub struct FluidParams {
    pub viscosity: f32,
    pub wall_friction: f32,
    pub body_drag: f32,
    pub air: bool,
    /// Time constant for the vessel's air to drag free water along with its walls.
    pub air_tau: Seconds,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            viscosity: 0.15,
            wall_friction: 0.5,
            body_drag: 0.5,
            air: true,
            air_tau: Seconds(12.0),
        }
    }
}

/// One particle's state, for spawning and persistence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Visual agitation in [0, 1]; drives the foam rendering.
    pub foam: f32,
}

/// The axis-aligned cells the particles are sorted into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub min: [f32; 3],
    pub dims: [i32; 3],
    pub cell: f32,
}

impl Grid {
    pub fn new(min: [f32; 3], max: [f32; 3], cell: f32) -> Self {
        Grid {
            min,
            dims: [0, 1, 2].map(|a| ((max[a] - min[a]) / cell).ceil() as i32 + 1),
            cell,
        }
    }

    pub fn cells(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize
    }
}

/// The fluid as the CPU sees it: how many particles live on the GPU, what is about to join
/// them, the most recent copy read back, and what the water did to the bodies last frame.
#[derive(Resource)]
pub struct Fluid {
    grid: Grid,
    count: u32,
    pending: Vec<Particle>,
    changed: bool,
    shapes: Vec<frame::ShapeSamples>,
    samples: Option<Vec<[f32; 4]>>,
    snapshot: Snapshot,
    coupling: Option<Vec<WaterCoupling>>,
    /// The GPU's running totals as of the last readback; the totals never reset, so what arrives
    /// late or twice in a frame is still applied exactly once.
    totals: Vec<i32>,
    /// Frames handed to the GPU and the latest frame whose coupling has come back.
    issued: u32,
    reported: u32,
    rng: Rng,
}

#[derive(Default)]
struct Snapshot {
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    requested: u32,
    arrived: u32,
    parts: u8,
}

impl Fluid {
    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn is_full(&self) -> bool {
        self.count as usize >= MAX_PARTICLES
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    pub fn add(&mut self, p: Particle) -> bool {
        if self.is_full() {
            return false;
        }
        self.count += 1;
        self.pending.push(p);
        self.changed = true;
        true
    }

    /// Add up to `count` particles in a small cloud around a point, each nudged with a little
    /// random velocity on top of `velocity`. `place` gets the final say on every position.
    pub fn inject(
        &mut self,
        centre: [f32; 3],
        velocity: impl Fn([f32; 3]) -> [f32; 3],
        count: u32,
        mut place: impl FnMut([f32; 3]) -> [f32; 3],
    ) -> u32 {
        let spread = 2.5 * PARTICLE_SPACING;
        let jitter = 0.3;
        let mut added = 0;
        for _ in 0..count {
            let position = place([
                centre[0] + (self.rng.next_f32() - 0.5) * spread,
                centre[1] + (self.rng.next_f32() - 0.5) * spread,
                centre[2] + (self.rng.next_f32() - 0.5) * spread,
            ]);
            let v = velocity(position);
            let particle = Particle {
                position,
                velocity: [
                    v[0] + (self.rng.next_f32() - 0.5) * jitter,
                    v[1] + (self.rng.next_f32() - 0.5) * jitter,
                    v[2] + (self.rng.next_f32() - 0.5) * jitter,
                ],
                foam: 0.1,
            };
            if !self.add(particle) {
                break;
            }
            added += 1;
        }
        added
    }

    pub fn clear(&mut self) {
        self.count = 0;
        self.pending.clear();
        self.changed = true;
    }

    /// The bodies' shapes, whose boundary samples the water couples to. Bodies name them by
    /// index when a frame is packed.
    pub fn set_shapes(&mut self, shapes: &[BodyShape]) {
        let (layout, samples) = frame::sample_table(shapes);
        self.shapes = layout;
        self.samples = Some(samples);
    }

    /// Body state for one substep, as the shaders read it.
    pub fn pack(&self, bodies: &[Body], shapes: &[BodyShape]) -> Bodies {
        frame::pack(bodies, shapes, &self.shapes)
    }

    /// What the GPU should run this frame: append what joined, then step the substeps.
    pub fn frame(&mut self, params: &FluidParams, substeps: &[(Seconds, Bodies)]) -> FluidFrame {
        let count_before = self.count - self.pending.len() as u32;
        let pending = self
            .pending
            .drain(..)
            .flat_map(|p| {
                [
                    [p.position[0], p.position[1], p.position[2], p.foam],
                    [p.velocity[0], p.velocity[1], p.velocity[2], 0.0],
                ]
            })
            .collect::<Vec<_>>();
        self.issued = self.issued.wrapping_add(1);
        let frame = FluidFrame {
            ticket: self.issued,
            substeps: substeps
                .iter()
                .map(|(dt, bodies)| Substep {
                    params: Params::new(dt.0, params, self.count, bodies, &self.grid),
                    bodies: bodies.clone(),
                })
                .collect(),
            inject: Params::new(0.0, params, count_before, &Bodies::default(), &self.grid)
                .with_pending(pending.len() as u32 / 2),
            surface: Params::new(0.0, params, self.count, &Bodies::default(), &self.grid),
            pending,
            samples: self.samples.take(),
            changed: self.changed || !substeps.is_empty(),
        };
        self.changed = false;
        frame
    }

    /// The particles as of the last snapshot that arrived.
    pub fn particles(&self) -> impl Iterator<Item = Particle> + '_ {
        self.snapshot
            .positions
            .iter()
            .zip(&self.snapshot.velocities)
            .map(|(p, v)| Particle {
                position: [p[0], p[1], p[2]],
                velocity: [v[0], v[1], v[2]],
                foam: p[3],
            })
    }

    /// Ask the GPU for a copy of every particle; `snapshot_ready` tells when it has arrived.
    pub fn request_snapshot(&mut self, commands: &mut Commands, buffers: &FluidBuffers) -> u32 {
        self.snapshot.requested += 1;
        self.snapshot.parts = 0;
        if self.count == 0 {
            self.snapshot.positions.clear();
            self.snapshot.velocities.clear();
            self.snapshot.arrived = self.snapshot.requested;
            return self.snapshot.requested;
        }
        let bytes = self.count as u64 * 16;
        commands
            .spawn(Readback::buffer_range(buffers.position.clone(), 0, bytes))
            .observe(receive_positions);
        commands
            .spawn(Readback::buffer_range(buffers.velocity.clone(), 0, bytes))
            .observe(receive_velocities);
        self.snapshot.requested
    }

    /// Whether the snapshot with this ticket (or a later one) has arrived.
    pub fn snapshot_ready(&self, ticket: u32) -> bool {
        self.snapshot.arrived >= ticket
    }

    /// How many frames the GPU has yet to report the coupling for; the bodies should not run
    /// further ahead of the water than a frame or two.
    pub fn outstanding(&self) -> u32 {
        self.issued.wrapping_sub(self.reported)
    }

    /// What the water did to each body since the last call, once the GPU has reported it.
    pub fn take_coupling(&mut self) -> Option<Vec<WaterCoupling>> {
        self.coupling.take()
    }

    fn snapshot_part(&mut self, part: u8) {
        self.snapshot.parts |= part;
        if self.snapshot.parts == 3 {
            self.snapshot.arrived = self.snapshot.requested;
        }
    }
}

/// The GPU fluid inside the given bounds; the vessel plugin must supply the shaders' `vessel`
/// module and the render world's [`VesselLayout`](crate::core::vessel::VesselLayout).
pub struct FluidPlugin {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/common.wgsl");
        embedded_asset!(app, "shaders/particles.wgsl");
        embedded_asset!(app, "shaders/sort.wgsl");
        embedded_asset!(app, "shaders/bodies.wgsl");
        embedded_asset!(app, "shaders/surface.wgsl");
        let shaders = FluidShaders(
            SHADERS.map(|path| app.world().resource::<AssetServer>().load::<Shader>(path)),
        );
        let grid = Grid::new(self.min, self.max, H);
        let surface_grid = surface::grid(self.min, self.max);
        let buffers = gpu::create_buffers(
            &mut app.world_mut().resource_mut::<Assets<ShaderBuffer>>(),
            &grid,
            &surface_grid,
        );
        let ready = FluidReady::default();
        app.insert_resource(ready.clone())
            .insert_resource(Fluid {
                grid,
                count: 0,
                pending: Vec::new(),
                changed: true,
                shapes: Vec::new(),
                samples: None,
                snapshot: Snapshot::default(),
                coupling: None,
                totals: Vec::new(),
                issued: 0,
                reported: 0,
                rng: Rng::new(0x9E3779B97F4A7C15),
            })
            .insert_resource(buffers)
            .insert_resource(shaders)
            .insert_resource(surface::SurfaceParams::new(&surface_grid))
            .init_resource::<FluidFrame>()
            .add_plugins((
                ExtractResourcePlugin::<FluidBuffers>::default(),
                ExtractResourcePlugin::<FluidFrame>::default(),
                ExtractResourcePlugin::<surface::SurfaceParams>::default(),
            ))
            .add_systems(Startup, watch_impulses);
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(ready);
        gpu::install(render_app);
    }
}

#[derive(Resource)]
struct FluidShaders(#[allow(dead_code)] [Handle<Shader>; 5]);

fn watch_impulses(mut commands: Commands, buffers: Res<FluidBuffers>) {
    commands
        .spawn(Readback::buffer(buffers.accum.clone()))
        .observe(receive_coupling);
}

fn receive_coupling(event: On<ReadbackComplete>, mut fluid: ResMut<Fluid>) {
    let raw: Vec<i32> = event.to_shader_type();
    let (substeps, ticket) = match raw.get(frame::STAMP_SLOT..frame::STAMP_SLOT + 2) {
        Some([substeps, ticket]) => (*substeps as u32, *ticket as u32),
        _ => (0, fluid.reported),
    };
    fluid.reported = ticket;
    if substeps == 0 {
        return;
    }
    if fluid.totals.len() != raw.len() {
        fluid.totals = vec![0; raw.len()];
    }
    let delta: Vec<i32> = raw
        .iter()
        .zip(&fluid.totals)
        .map(|(now, before)| now.wrapping_sub(*before))
        .collect();
    fluid.totals = raw;
    let arrived = frame::decode_coupling(&delta);
    match &mut fluid.coupling {
        Some(pending) => {
            for (sum, more) in pending.iter_mut().zip(&arrived) {
                sum.add(more);
            }
        }
        None => fluid.coupling = Some(arrived),
    }
}

fn receive_positions(
    event: On<ReadbackComplete>,
    mut fluid: ResMut<Fluid>,
    mut commands: Commands,
) {
    fluid.snapshot.positions = event.to_shader_type();
    fluid.snapshot_part(1);
    commands.entity(event.entity).try_despawn();
}

fn receive_velocities(
    event: On<ReadbackComplete>,
    mut fluid: ResMut<Fluid>,
    mut commands: Commands,
) {
    fluid.snapshot.velocities = event.to_shader_type();
    fluid.snapshot_part(2);
    commands.entity(event.entity).try_despawn();
}
