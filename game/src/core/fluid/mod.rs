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
pub use surface::{MAX_BLOCKS, MAX_INDICES, MAX_VERTICES, SurfaceBuffers, SurfaceParams};

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
    /// What the water did to the bodies that they have yet to be given.
    coupling: Option<Vec<WaterCoupling>>,
    /// Σ midpoint·seconds over what the pending coupling covers, for its age when taken.
    measured: f64,
    /// Simulated seconds of water stepped so far.
    stepped: f64,
    /// The GPU's running totals as of the last readback; the totals never reset, so what arrives
    /// late or twice in a frame is still applied exactly once.
    totals: Vec<i32>,
    /// Frames handed to the GPU and the latest frame whose coupling has come back.
    issued: u32,
    reported: u32,
    /// Every frame handed to the GPU whose coupling is still to come back.
    timeline: Vec<Issue>,
    /// Whether a readback of the coupling is in flight.
    awaiting: bool,
    rng: Rng,
}

/// A frame handed to the GPU that couples water and bodies: the simulated time and substeps it
/// covers, and the midpoint of that time on the water's clock (so its coupling can be aged).
struct Issue {
    ticket: u32,
    seconds: f64,
    substeps: f64,
    midpoint: f64,
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
            measured: 0.0,
            stepped: 0.0,
            timeline: Vec::new(),
            awaiting: false,
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
    /// then step the substeps. Without water there is nothing to step, extract or read back,
    /// and a frame that couples nothing counts as reported at once.
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
        let stepping = self.count > 0 && !substeps.is_empty();
        let seconds: f64 = substeps.iter().map(|(dt, _)| dt.0 as f64).sum();
        let coupling = stepping && substeps.iter().any(|(_, b)| b.sample_count > 0);
        if coupling {
            self.timeline.push(Issue {
                ticket: self.issued,
                seconds,
                substeps: substeps.len() as f64,
                midpoint: self.stepped + seconds / 2.0,
            });
        }
        if stepping {
            self.stepped += seconds;
        }
        let res = self.resolution;
        let still = Bodies::default();
        let frame = FluidFrame {
            ticket: self.issued,
            thin: self.thin.take().map(|before| {
                Params::new(0.0, params, res, before, &still).with_pending(count_before)
            }),
            substeps: if stepping {
                substeps
                    .iter()
                    .map(|(dt, bodies)| Substep {
                        params: Params::new(dt.0, params, res, self.count, bodies),
                        bodies: bodies.clone(),
                    })
                    .collect()
            } else {
                Vec::new()
            },
            inject: Params::new(0.0, params, res, count_before, &still)
                .with_pending(pending.len() as u32 / 2),
            surface: Params::new(0.0, params, res, self.count, &still),
            pending,
            samples: self.samples.take(),
            changed: self.changed || stepping,
            coupling,
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
            .spawn((
                Readback::buffer_range(buffers.position.clone(), 0, bytes),
                ReadOnce,
            ))
            .observe(receive_positions);
        commands
            .spawn((
                Readback::buffer_range(buffers.velocity.clone(), 0, bytes),
                ReadOnce,
            ))
            .observe(receive_velocities);
        self.snapshot.requested
    }

    /// Whether the snapshot with this ticket (or a later one) has arrived.
    pub fn snapshot_ready(&self, ticket: u32) -> bool {
        self.snapshot.arrived >= ticket
    }

    /// How many frames the GPU has yet to report the coupling for.
    pub fn outstanding(&self) -> u32 {
        self.timeline.len() as u32
    }

    /// What the water did to each body over the next `seconds` of their time, aged by how far
    /// the water has stepped since the middle of the time it was measured over. A report is
    /// spent over as many seconds as it covers, so one that came back late is not a lump.
    pub fn take_coupling(&mut self, seconds: f64) -> Option<Vec<WaterCoupling>> {
        let mut pool = self.coupling.take()?;
        let covered = pool.first().map_or(0.0, |c| c.seconds);
        if covered <= 0.0 {
            return None;
        }
        let age = self.stepped - self.measured / covered;
        let share = (seconds / covered).min(1.0);
        let taken = pool
            .iter()
            .map(|c| WaterCoupling {
                age,
                ..c.scaled(share)
            })
            .collect();
        if share < 1.0 {
            for c in &mut pool {
                *c = c.scaled(1.0 - share);
            }
            self.measured *= 1.0 - share;
            self.coupling = Some(pool);
        } else {
            self.measured = 0.0;
        }
        Some(taken)
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

    /// Every frame up to this ticket, whose coupling has now come back, as one issue: their
    /// time and substeps summed, with the midpoint of that time.
    fn settle(&mut self, ticket: u32) -> Issue {
        let mut settled = Issue {
            ticket,
            seconds: 0.0,
            substeps: 0.0,
            midpoint: 0.0,
        };
        self.timeline.retain(|issue| {
            let reported = (ticket.wrapping_sub(issue.ticket) as i32) >= 0;
            if reported {
                settled.seconds += issue.seconds;
                settled.substeps += issue.substeps;
                settled.midpoint += issue.midpoint * issue.seconds;
            }
            !reported
        });
        if settled.seconds > 0.0 {
            settled.midpoint /= settled.seconds;
        }
        settled
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
            .add_systems(Startup, spawn_watcher)
            .add_systems(PostUpdate, (watch_impulses, sync_surface))
            .add_systems(Last, stop_rereading);
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

/// Marks a `Readback` to be performed once. Bevy reads an entity's buffer back every frame the
/// component is on it, and the answer takes a few frames to come, so left alone a readback
/// would be issued again and again until it arrived, and a GPU that falls behind would be
/// buried under ever more of them. The marker has the readback dropped the frame after it is
/// issued, while the entity and its observer stay to receive the answer.
#[derive(Component)]
pub struct ReadOnce;

fn stop_rereading(
    mut commands: Commands,
    issued: Query<Entity, (With<Readback>, With<Issued>)>,
    fresh: Query<Entity, (With<ReadOnce>, Without<Issued>)>,
) {
    for entity in &issued {
        commands
            .entity(entity)
            .remove::<(Readback, ReadOnce, Issued)>();
    }
    for entity in &fresh {
        commands.entity(entity).insert(Issued);
    }
}

/// A `ReadOnce` at the end of the frame it was asked in, which the render world reads back.
#[derive(Component)]
struct Issued;

/// The entity whose readback brings the coupling back from the GPU.
#[derive(Component)]
struct CouplingWatcher;

fn spawn_watcher(mut commands: Commands) {
    commands.spawn(CouplingWatcher).observe(receive_coupling);
}

/// Ask for the coupling while any frame's is still to come back, one readback at a time: the
/// watcher asks again once the answer is back, so what is in flight never piles up, and nothing
/// is read back at all while there is nothing to report.
fn watch_impulses(
    mut commands: Commands,
    buffers: Res<FluidBuffers>,
    mut fluid: ResMut<Fluid>,
    watcher: Single<Entity, With<CouplingWatcher>>,
) {
    if !fluid.awaiting && fluid.outstanding() > 0 {
        commands
            .entity(*watcher)
            .insert((Readback::buffer(buffers.accum.clone()), ReadOnce));
        fluid.awaiting = true;
    }
}

fn receive_coupling(event: On<ReadbackComplete>, mut fluid: ResMut<Fluid>) {
    fluid.awaiting = false;
    let raw: Vec<i32> = event.to_shader_type();
    let ticket = raw
        .get(frame::STAMP_SLOT)
        .map_or(fluid.reported, |t| *t as u32);
    if (ticket.wrapping_sub(fluid.reported) as i32) > 0 {
        fluid.reported = ticket;
    }
    let settled = fluid.settle(ticket);
    if settled.substeps == 0.0 {
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
        coupling.seconds = settled.seconds;
        coupling.substeps = settled.substeps;
    }
    fluid.measured += settled.midpoint * settled.seconds;
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
