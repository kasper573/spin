//! Position-based fluid (Macklin & Müller 2013) solved on the GPU with Akinci-style coupling to
//! rigid bodies, confined by a vessel the shaders describe. The CPU only keeps the particle count,
//! feeds in new particles and body poses each frame, and reads back what the water did to the
//! bodies. SI units throughout.
//!
//! The particles are binned by a hash of their grid cell and the surface is meshed only where
//! the water is, so neither the vessel's size nor the water's extent costs anything: the work is
//! the number of particles, which is capped. Water past the cap is made coarser instead, every
//! particle standing for twice as much of it, so any amount of water fits the same budget.
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
use crate::core::units::{Litres, MetresPerSecond, Seconds};

mod frame;
mod gpu;
mod resolution;
mod surface;

pub use frame::{Bodies, FluidFrame, GpuBodies, Params, Substep};
pub use gpu::FluidBuffers;
pub use resolution::{REST_DENSITY, Resolution};
pub use surface::{MAX_INDICES, SurfaceBuffers, SurfaceParams};

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

pub const MAX_PARTICLES: usize = 65536;
/// Slots of the cell table the particles are sorted into: the box in `common.wgsl`.
pub const TABLE_CELLS: usize = 64 * 32 * 64;
/// Bodies the shaders reserve room for.
pub const MAX_BODIES: usize = 16;
/// Boundary samples over all solid bodies.
pub const MAX_SAMPLES: usize = 2048;
/// Substeps a single frame may run; beyond that the simulation falls behind real time.
pub const MAX_SUBSTEPS_PER_FRAME: usize = 4;
const EPS_LAMBDA: f32 = 0.02;
const ITERATIONS: usize = 3;
const SCORR_K: f32 = 0.001;
const WET_REF: f32 = 300.0;
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
    /// A safety clamp on the particles' speed, well above anything the vessel's walls reach.
    pub max_speed: MetresPerSecond,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            viscosity: 0.15,
            wall_friction: 0.5,
            body_drag: 0.5,
            air: true,
            air_tau: Seconds(12.0),
            max_speed: MetresPerSecond(40.0),
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

/// The fluid as the CPU sees it: how many particles live on the GPU and how much water each
/// stands for, what is about to join them, the most recent copy read back, and what the water
/// did to the bodies last frame.
#[derive(Resource)]
pub struct Fluid {
    resolution: Resolution,
    count: u32,
    /// The particle count before a thinning this frame, if one is due.
    thin: Option<u32>,
    pending: Vec<Particle>,
    changed: bool,
    shapes: Vec<frame::ShapePoints>,
    layout: Vec<frame::ShapeSamples>,
    samples: Option<Vec<[f32; 4]>>,
    snapshot: Snapshot,
    coupling: Option<Vec<WaterCoupling>>,
    /// The GPU's running totals as of the last readback; the totals never reset, so what arrives
    /// late or twice in a frame is still applied exactly once.
    totals: Vec<i32>,
    /// Frames handed to the GPU and the latest frame whose coupling has come back.
    issued: u32,
    reported: u32,
    /// The simulated time and substeps of every frame handed to the GPU whose coupling is still
    /// to come back, by ticket.
    timeline: Vec<(u32, f64, f64)>,
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

impl Default for Fluid {
    fn default() -> Self {
        Fluid {
            resolution: Resolution::FINEST,
            count: 0,
            thin: None,
            pending: Vec::new(),
            changed: true,
            shapes: Vec::new(),
            layout: Vec::new(),
            samples: None,
            snapshot: Snapshot::default(),
            coupling: None,
            totals: Vec::new(),
            issued: 0,
            reported: 0,
            timeline: Vec::new(),
            rng: Rng::new(0x9E3779B97F4A7C15),
        }
    }
}

impl Fluid {
    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// How much water there is.
    pub fn litres(&self) -> Litres {
        Litres(self.count as f32 * self.resolution.litres_per_particle().0)
    }

    /// The surface extraction's parameters for the water as it is.
    pub fn surface(&self) -> SurfaceParams {
        SurfaceParams::new(self.resolution)
    }

    /// Add a particle of the current resolution. When the budget is full the water is made
    /// coarser first, which frees half of it; only a second fill within one frame is refused.
    pub fn add(&mut self, p: Particle) -> bool {
        if self.count as usize >= MAX_PARTICLES {
            if self.thin.is_some() {
                return false;
            }
            self.coarsen();
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
        let spread = 2.5 * self.resolution.spacing.0;
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

    /// Remove all water; what comes next starts out as fine as water gets.
    pub fn clear(&mut self) {
        self.restore(Resolution::FINEST);
    }

    /// Remove all water and take this resolution for what is added next, as when saved water
    /// is loaded back.
    pub fn restore(&mut self, resolution: Resolution) {
        self.count = 0;
        self.thin = None;
        self.pending.clear();
        self.set_resolution(resolution);
    }

    /// The bodies' shapes, whose boundary samples the water couples to. Bodies name them by
    /// index when a frame is packed.
    pub fn set_shapes(&mut self, shapes: &[BodyShape]) {
        self.shapes = shapes.iter().map(frame::ShapePoints::of).collect();
        self.reweight();
    }

    /// Body state for one substep, as the shaders read it.
    pub fn pack(&self, bodies: &[Body], shapes: &[BodyShape]) -> Bodies {
        frame::pack(bodies, shapes, &self.layout)
    }

    /// What the GPU should run this frame: thin the water if it is due, append what joined,
    /// then step the substeps.
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
        self.timeline.push((
            self.issued,
            substeps.iter().map(|(dt, _)| dt.0 as f64).sum(),
            substeps.len() as f64,
        ));
        let res = self.resolution;
        let still = Bodies::default();
        let frame = FluidFrame {
            ticket: self.issued,
            thin: self.thin.take().map(|before| {
                Params::new(0.0, params, res, before, &still).with_pending(count_before)
            }),
            substeps: substeps
                .iter()
                .map(|(dt, bodies)| Substep {
                    params: Params::new(dt.0, params, res, self.count, bodies),
                    bodies: bodies.clone(),
                })
                .collect(),
            inject: Params::new(0.0, params, res, count_before, &still)
                .with_pending(pending.len() as u32 / 2),
            surface: Params::new(0.0, params, res, self.count, &still),
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

    /// Every particle stands for twice the water from now on: the GPU keeps every other one of
    /// those it has, and whatever is waiting to join them joins the rest.
    fn coarsen(&mut self) {
        let waiting = self.pending.len() as u32;
        let on_gpu = self.count - waiting;
        self.thin = Some(on_gpu);
        self.count = on_gpu.div_ceil(2) + waiting;
        self.set_resolution(self.resolution.coarser());
    }

    fn set_resolution(&mut self, resolution: Resolution) {
        self.resolution = resolution;
        self.reweight();
        self.changed = true;
    }

    /// Boundary sample weights follow the kernel, so they are redone with the resolution.
    fn reweight(&mut self) {
        let (layout, samples) = frame::sample_table(&self.shapes, self.resolution);
        self.layout = layout;
        self.samples = Some(samples);
    }

    /// The simulated time and substeps of every frame up to this ticket, whose coupling has
    /// now come back.
    fn settle(&mut self, ticket: u32) -> (f64, f64) {
        let (mut seconds, mut substeps) = (0.0, 0.0);
        self.timeline.retain(|(issued, s, n)| {
            let reported = (ticket.wrapping_sub(*issued) as i32) >= 0;
            if reported {
                seconds += s;
                substeps += n;
            }
            !reported
        });
        (seconds, substeps)
    }

    fn snapshot_part(&mut self, part: u8) {
        self.snapshot.parts |= part;
        if self.snapshot.parts == 3 {
            self.snapshot.arrived = self.snapshot.requested;
        }
    }
}

/// The GPU fluid; the vessel plugin must supply the shaders' `vessel` module and the render
/// world's [`VesselLayout`](crate::core::vessel::VesselLayout).
pub struct FluidPlugin;

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
        let buffers =
            gpu::create_buffers(&mut app.world_mut().resource_mut::<Assets<ShaderBuffer>>());
        let ready = FluidReady::default();
        app.insert_resource(ready.clone())
            .init_resource::<Fluid>()
            .insert_resource(buffers)
            .insert_resource(shaders)
            .insert_resource(SurfaceParams::new(Resolution::FINEST))
            .init_resource::<FluidFrame>()
            .add_plugins((
                ExtractResourcePlugin::<FluidBuffers>::default(),
                ExtractResourcePlugin::<FluidFrame>::default(),
                ExtractResourcePlugin::<SurfaceParams>::default(),
            ))
            .add_systems(Startup, watch_impulses)
            .add_systems(PostUpdate, sync_surface);
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(ready);
        gpu::install(render_app);
    }
}

#[derive(Resource)]
struct FluidShaders(#[allow(dead_code)] [Handle<Shader>; 5]);

/// Hand the render world the surface parameters once they change.
fn sync_surface(fluid: Res<Fluid>, mut surface: ResMut<SurfaceParams>) {
    let wanted = fluid.surface();
    if *surface != wanted {
        *surface = wanted;
    }
}

fn watch_impulses(mut commands: Commands, buffers: Res<FluidBuffers>) {
    commands
        .spawn(Readback::buffer(buffers.accum.clone()))
        .observe(receive_coupling);
}

fn receive_coupling(event: On<ReadbackComplete>, mut fluid: ResMut<Fluid>) {
    let raw: Vec<i32> = event.to_shader_type();
    let ticket = raw
        .get(frame::STAMP_SLOT)
        .map_or(fluid.reported, |t| *t as u32);
    fluid.reported = ticket;
    let (seconds, taken) = fluid.settle(ticket);
    if taken == 0.0 {
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
    let mut arrived = frame::decode_coupling(&delta);
    for coupling in &mut arrived {
        coupling.seconds = seconds;
        coupling.substeps = taken;
    }
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
