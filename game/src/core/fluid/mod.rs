//! Position-based fluid (Macklin & Müller 2013) solved on the GPU with Akinci-style coupling to
//! rigid bodies, confined by a vessel the shaders describe. The CPU only keeps the particle count,
//! feeds in new particles and body poses each frame, and reads back what the water did to the
//! bodies.
//!
//! The water lives in its vessel's own frame, where water at rest stays put however fast the
//! vessel turns, and the GPU only ever simulates the canonical water of [`Resolution`]: the
//! CPU scales metres and seconds to its units on the way in and back on the way out. The
//! particles are binned by a hash of their grid cell and the surface is meshed only where the
//! water is, so neither the vessel's size nor the water's extent costs anything: the work is
//! the number of particles, which is capped. Water past the cap is made coarser instead, every
//! particle standing for twice as much of it, so any amount of water fits the same budget, and
//! a vessel of any size sets a floor on how fine its water is, so any size runs the same.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::Shader;

use crate::core::math::{Rng, Vec3d};
use crate::core::rigid::{Body, BodyShape, WaterCoupling};
use crate::core::units::{Litres, MetresPerSecond, Seconds};
use crate::core::vessel::WaterFrame;

mod frame;
mod gpu;
mod resolution;
mod surface;

pub use frame::{Bodies, FluidFrame, GpuBodies, Params, Substep};
pub use gpu::{FluidBuffers, FluidStep};
pub use resolution::{REST_DENSITY, Resolution, SPACINGS_FROM_AXIS, STEP_RATE, canonical};
pub use surface::{
    MAX_BLOCKS, MAX_DROPLETS, MAX_INDICES, MAX_VERTICES, SurfaceBuffers, SurfaceParams, grid_reach,
};

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
/// The random nudge new water gets, in the canonical water's units.
/// How far off its lattice site, in spacings, water is put down, so that no two rows of it are
/// ever exactly in line.
const JITTER: f64 = 0.05;
/// How many sites a placement offers per particle, and at least, so that water placed onto
/// water finds free room round it.
const SITES_PER_PARTICLE: usize = 64;
const LEAST_SITES: usize = 512;

/// The offsets of the lattice sites in a ball about a site, nearest first: what every
/// placement of water is offered, so that water placed onto water finds free room around it.
fn lattice_ball() -> &'static [[i32; 3]] {
    static BALL: std::sync::OnceLock<Vec<[i32; 3]>> = std::sync::OnceLock::new();
    BALL.get_or_init(|| {
        let reach = (3.0 * gpu::SITES as f64 / (4.0 * std::f64::consts::PI))
            .cbrt()
            .ceil() as i32
            + 1;
        let mut ball = Vec::new();
        for i in -reach..=reach {
            for j in -reach..=reach {
                for k in -reach..=reach {
                    ball.push([i, j, k]);
                }
            }
        }
        let d2 = |o: &[i32; 3]| o[0] * o[0] + o[1] * o[1] + o[2] * o[2];
        ball.sort_by_key(d2);
        ball.truncate(gpu::SITES);
        ball
    })
}
/// Resolutions remembered by the frame they took effect, for reading back what an older frame
/// left on the GPU.
const GENERATIONS_KEPT: usize = 64;
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
    /// Time constant for the vessel's air to bring free water to rest in the vessel's frame, in
    /// seconds of the water's own clock, so that big water is slow to settle as it is to fall.
    pub air_tau: Seconds,
    /// A safety clamp on the particles' speed through the vessel, well above anything they
    /// could be thrown at.
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

/// One particle's state in the vessel's frame, for spawning and persistence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    pub position: Vec3d,
    pub velocity: Vec3d,
    /// Visual agitation in [0, 1]; drives the foam rendering.
    pub foam: f32,
}

/// The fluid as the CPU sees it: how many particles live on the GPU and how much water each
/// stands for, what is about to join them, the most recent copy read back, and what the water
/// did to the bodies last frame.
#[derive(Resource)]
pub struct Fluid {
    resolution: Resolution,
    /// The finest the water may be, set by the size of its vessel.
    floor: Resolution,
    /// The point in the vessel's frame the surface is extracted about.
    anchor: Vec3d,
    count: u32,
    /// The particle count before a thinning this frame and their resolution, if one is due.
    thin: Option<(u32, Resolution)>,
    pending: Vec<Particle>,
    /// Water placed this frame: how many particles, and the lattice sites offered to them.
    joining: u32,
    sites: Vec<[f32; 4]>,
    changed: bool,
    shapes: Vec<frame::ShapePoints>,
    layout: Vec<frame::ShapeSamples>,
    samples: Option<Vec<[f32; 4]>>,
    snapshot: Snapshot,
    /// What the water did to the bodies that they have yet to be given.
    coupling: Option<Vec<WaterCoupling>>,
    /// Simulated seconds of water stepped so far.
    stepped: f64,
    /// Frames handed to the GPU and the latest frame whose coupling has come back.
    issued: u32,
    reported: u32,
    /// Every frame handed to the GPU whose coupling is still to come back.
    timeline: Vec<Issue>,
    /// Each resolution the water has had, by the first frame simulated at it.
    generations: Vec<(u32, Resolution)>,
    /// Whether a readback of the coupling is in flight.
    awaiting: bool,
    rng: Rng,
}

/// A frame handed to the GPU that couples water and bodies: the simulated time and substeps it
/// covers, the midpoint of that time on the water's clock, and the resolution and frame its
/// accumulators are in.
struct Issue {
    ticket: u32,
    seconds: f64,
    substeps: f64,
    midpoint: f64,
    resolution: Resolution,
    frame: WaterFrame,
}

#[derive(Default)]
struct Snapshot {
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    /// The frame the copy was taken after.
    ticket: u32,
    requested: u32,
    arrived: u32,
    parts: u8,
}

impl Default for Fluid {
    fn default() -> Self {
        Fluid {
            resolution: Resolution::FINEST,
            floor: Resolution::FINEST,
            anchor: [0.0; 3],
            count: 0,
            thin: None,
            pending: Vec::new(),
            joining: 0,
            sites: Vec::new(),
            changed: true,
            shapes: Vec::new(),
            layout: Vec::new(),
            samples: None,
            snapshot: Snapshot::default(),
            coupling: None,
            issued: 0,
            reported: 0,
            stepped: 0.0,
            timeline: Vec::new(),
            generations: vec![(0, Resolution::FINEST)],
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

    /// How much water each particle stands for.
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// How much water there is.
    pub fn litres(&self) -> Litres {
        Litres(self.count as f32 * self.resolution.litres_per_particle().0)
    }

    /// The real time each step of the water covers at its resolution.
    pub fn step(&self) -> Seconds {
        self.resolution.step()
    }

    /// The surface extraction's parameters for the water as it is.
    pub fn surface(&self) -> SurfaceParams {
        SurfaceParams::new(self.resolution, self.anchor)
    }

    /// The point in the vessel's frame the surface's vertices are relative to, and the metres
    /// each unit of them is.
    pub fn surface_origin(&self) -> (Vec3d, f64) {
        (
            self.surface().origin(self.resolution),
            self.resolution.length(),
        )
    }

    /// Extract the surface about this point of the vessel's frame from now on, so that its
    /// vertices are small near it.
    pub fn set_anchor(&mut self, anchor: Vec3d) {
        let origin = |anchor| SurfaceParams::new(self.resolution, anchor).origin(self.resolution);
        if origin(self.anchor) != origin(anchor) {
            self.anchor = anchor;
            self.changed = true;
        }
    }

    /// Add a particle of the current resolution. When the budget is full the water is made
    /// coarser first, which frees half of it; only a second fill within one frame is refused.
    pub fn add(&mut self, p: Particle) -> bool {
        if !self.reserve() {
            return false;
        }
        self.pending.push(p);
        true
    }

    /// Put down up to `count` particles of water about a point of the vessel's frame, at rest
    /// in it: each takes a site of the water's rest lattice, the free sites nearest the point
    /// first, so that water is added as gently as can be and water placed onto water spreads
    /// round it rather than into it. Once a frame has offered all the sites it can, the rest
    /// take the nearest sites free or not. `place` gets the final say on every site. Returns
    /// how many were added.
    pub fn inject(
        &mut self,
        centre: Vec3d,
        count: u32,
        mut place: impl FnMut(Vec3d) -> Vec3d,
    ) -> u32 {
        let length = self.resolution.length();
        let base = centre.map(|x| (x / length).floor());
        let mut site = |offset: &[i32; 3], rng: &mut Rng| {
            let mut nudge = |x: f64| x + 0.5 + (rng.next_f32() as f64 - 0.5) * JITTER;
            place([
                nudge(base[0] + offset[0] as f64) * length,
                nudge(base[1] + offset[1] as f64) * length,
                nudge(base[2] + offset[2] as f64) * length,
            ])
        };
        let room = gpu::SITES - self.sites.len();
        let offered = (count as usize * SITES_PER_PARTICLE)
            .max(LEAST_SITES)
            .min(room);
        let mut added = 0;
        if offered >= count as usize {
            while added < count && self.reserve() {
                added += 1;
            }
            self.joining += added;
            let length = self.resolution.length();
            for offset in lattice_ball().iter().take(offered) {
                let p = site(offset, &mut self.rng).map(|x| (x / length) as f32);
                self.sites.push([p[0], p[1], p[2], 0.0]);
            }
        } else {
            for offset in lattice_ball().iter().take(count as usize) {
                let position = site(offset, &mut self.rng);
                let particle = Particle {
                    position,
                    velocity: [0.0; 3],
                    foam: 0.0,
                };
                if !self.add(particle) {
                    break;
                }
                added += 1;
            }
        }
        added
    }

    /// Make room in the count for one more particle, coarsening the water if it is full.
    fn reserve(&mut self) -> bool {
        if self.count as usize >= MAX_PARTICLES {
            if self.thin.is_some() {
                return false;
            }
            self.coarsen();
        }
        self.count += 1;
        self.changed = true;
        true
    }

    /// Remove all water; what comes next starts out as fine as the vessel allows.
    pub fn clear(&mut self) {
        self.restore(self.floor);
    }

    /// Remove all water and take this resolution for what is added next, as when saved water
    /// is loaded back.
    pub fn restore(&mut self, resolution: Resolution) {
        self.count = 0;
        self.thin = None;
        self.pending.clear();
        self.joining = 0;
        self.sites.clear();
        self.set_resolution(resolution.at_least(self.floor));
    }

    /// The finest the water may be from now on. Water finer than that is made coarser, a step
    /// a frame, until it is not.
    pub fn set_floor(&mut self, floor: Resolution) {
        self.floor = floor;
        if self.resolution.spacing.0 < floor.spacing.0 {
            if self.count == 0 {
                self.set_resolution(floor);
            } else if self.thin.is_none() {
                self.coarsen();
            }
        }
    }

    /// The bodies' shapes, whose boundary samples the water couples to. Bodies name them by
    /// index when a frame is packed.
    pub fn set_shapes(&mut self, shapes: &[BodyShape]) {
        self.shapes = shapes.iter().map(frame::ShapePoints::of).collect();
        self.reweight();
    }

    /// Body state for one substep, as the shaders read it: in the water's frame, in the
    /// canonical units.
    pub fn pack(&self, bodies: &[Body], shapes: &[BodyShape], frame: &WaterFrame) -> Bodies {
        frame::pack(bodies, shapes, &self.layout, frame, self.resolution)
    }

    /// What the GPU should run this frame: thin the water if it is due, append what joined,
    /// then step the substeps. Without water there is nothing to step, extract or read back,
    /// and a frame that couples nothing counts as reported at once.
    pub fn frame(
        &mut self,
        params: &FluidParams,
        substeps: &[(Seconds, Bodies)],
        frame: &WaterFrame,
    ) -> FluidFrame {
        let count_before = self.count - self.pending.len() as u32 - self.joining;
        let joined = count_before + self.pending.len() as u32;
        let joining = std::mem::take(&mut self.joining);
        let sites = std::mem::take(&mut self.sites);
        let res = self.resolution;
        let (length, time) = (res.length(), res.time());
        let pending = self
            .pending
            .drain(..)
            .flat_map(|p| {
                let x = p.position.map(|x| (x / length) as f32);
                let v = p.velocity.map(|v| (v * time / length) as f32);
                [[x[0], x[1], x[2], p.foam], [v[0], v[1], v[2], 0.0]]
            })
            .collect::<Vec<_>>();
        self.issued = self.issued.wrapping_add(1);
        let ticket = self.issued;
        let stepping = self.count > 0 && !substeps.is_empty();
        let seconds: f64 = substeps.iter().map(|(dt, _)| dt.0 as f64).sum();
        let coupling = stepping && substeps.iter().any(|(_, b)| b.sample_count > 0);
        if coupling {
            self.timeline.push(Issue {
                ticket,
                seconds,
                substeps: substeps.len() as f64,
                midpoint: self.stepped + seconds / 2.0,
                resolution: res,
                frame: *frame,
            });
        }
        if stepping {
            self.stepped += seconds;
        }
        let still = Bodies::default();
        let frame = FluidFrame {
            ticket,
            thin: self.thin.take().map(|(before, from)| {
                Params::new(Seconds(0.0), params, res, before, &still)
                    .with_pending(count_before)
                    .thinning(from, res)
            }),
            substeps: if stepping {
                substeps
                    .iter()
                    .map(|(dt, bodies)| Substep {
                        params: Params::new(*dt, params, res, self.count, bodies).for_frame(ticket),
                        bodies: bodies.clone(),
                    })
                    .collect()
            } else {
                Vec::new()
            },
            inject: Params::new(Seconds(0.0), params, res, count_before, &still)
                .with_pending(pending.len() as u32 / 2),
            join: Params::new(Seconds(0.0), params, res, joined, &still)
                .with_pending(joining)
                .with_candidates(sites.len() as u32),
            surface: Params::new(Seconds(0.0), params, res, self.count, &still),
            pending,
            sites,
            samples: self.samples.take(),
            changed: self.changed || stepping,
            coupling,
        };
        self.changed = false;
        frame
    }

    /// The particles as of the last snapshot that arrived, in the vessel's frame.
    pub fn particles(&self) -> impl Iterator<Item = Particle> + '_ {
        let res = self.resolution_at(self.snapshot.ticket);
        let (length, time) = (res.length(), res.time());
        self.snapshot
            .positions
            .iter()
            .zip(&self.snapshot.velocities)
            .map(move |(p, v)| Particle {
                position: [p[0], p[1], p[2]].map(|x| x as f64 * length),
                velocity: [v[0], v[1], v[2]].map(|v| v as f64 * length / time),
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
            self.snapshot.ticket = self.issued;
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
        commands
            .spawn((
                Readback::buffer_range(buffers.accum.clone(), frame::STAMP_SLOT as u64 * 4, 4),
                ReadOnce,
            ))
            .observe(receive_stamp);
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

    /// What the water did to each body over the next `seconds` of their time. A report is
    /// spent over as many seconds as it covers, so one that came back late is not a lump.
    pub fn take_coupling(&mut self, seconds: f64) -> Option<Vec<WaterCoupling>> {
        let mut pool = self.coupling.take()?;
        let covered = pool.first().map_or(0.0, |c| c.seconds);
        if covered <= 0.0 {
            return None;
        }
        let share = (seconds / covered).min(1.0);
        let taken = pool.iter().map(|c| c.scaled(share)).collect();
        if share < 1.0 {
            for c in &mut pool {
                *c = c.scaled(1.0 - share);
            }
            self.coupling = Some(pool);
        }
        Some(taken)
    }

    /// Every particle stands for twice the water from now on: the GPU keeps every other one of
    /// those it has, and whatever is waiting to join them joins the rest.
    fn coarsen(&mut self) {
        let waiting = self.pending.len() as u32 + self.joining;
        let on_gpu = self.count - waiting;
        self.thin = Some((on_gpu, self.resolution));
        self.count = on_gpu.div_ceil(2) + waiting;
        let before = self.resolution.length();
        self.set_resolution(self.resolution.coarser());
        let scale = (before / self.resolution.length()) as f32;
        for site in &mut self.sites {
            for x in &mut site[..3] {
                *x *= scale;
            }
        }
    }

    fn set_resolution(&mut self, resolution: Resolution) {
        self.resolution = resolution;
        self.reweight();
        self.changed = true;
        let from = self.issued.wrapping_add(1);
        match self.generations.last_mut() {
            Some(last) if last.0 == from => last.1 = resolution,
            _ => self.generations.push((from, resolution)),
        }
        if self.generations.len() > GENERATIONS_KEPT {
            self.generations.remove(0);
        }
    }

    /// The resolution the water had in the frame with this ticket.
    fn resolution_at(&self, ticket: u32) -> Resolution {
        self.generations
            .iter()
            .rev()
            .find(|(from, _)| (ticket.wrapping_sub(*from) as i32) >= 0)
            .map_or(self.resolution, |(_, resolution)| *resolution)
    }

    /// Boundary sample weights follow the kernel, so they are redone with the resolution.
    fn reweight(&mut self) {
        let (layout, samples) = frame::sample_table(&self.shapes, self.resolution);
        self.layout = layout;
        self.samples = Some(samples);
    }

    /// Every frame up to this ticket, whose coupling has now come back, taken off the
    /// timeline: what each one's accumulators say, summed, with their time and substeps and
    /// the midpoint of that time.
    fn settle(&mut self, ticket: u32, raw: &[i32]) -> (Vec<WaterCoupling>, Issue) {
        let mut settled = Issue {
            ticket,
            seconds: 0.0,
            substeps: 0.0,
            midpoint: 0.0,
            resolution: self.resolution,
            frame: WaterFrame {
                origin: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let mut sum: Vec<WaterCoupling> = Vec::new();
        self.timeline.retain(|issue| {
            let reported = (ticket.wrapping_sub(issue.ticket) as i32) >= 0;
            if reported {
                settled.seconds += issue.seconds;
                settled.substeps += issue.substeps;
                settled.midpoint += issue.midpoint * issue.seconds;
                let first = frame::accumulators_of(issue.ticket) as usize;
                if let Some(slots) = raw.get(first..first + frame::ACCUMULATORS_PER_FRAME) {
                    let arrived = frame::decode_coupling(slots, issue.resolution, &issue.frame);
                    if sum.is_empty() {
                        sum = arrived;
                    } else {
                        for (total, more) in sum.iter_mut().zip(&arrived) {
                            total.add(more);
                        }
                    }
                }
            }
            !reported
        });
        if settled.seconds > 0.0 {
            settled.midpoint /= settled.seconds;
        }
        (sum, settled)
    }

    fn snapshot_part(&mut self, part: u8) {
        self.snapshot.parts |= part;
        if self.snapshot.parts == 7 {
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
            .insert_resource(Fluid::default().surface())
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

/// Every frame's accumulators the readback covers, each frame's in its own slots and units,
/// gathered into what the bodies are owed.
fn receive_coupling(event: On<ReadbackComplete>, mut fluid: ResMut<Fluid>) {
    fluid.awaiting = false;
    let raw: Vec<i32> = event.to_shader_type();
    let ticket = raw
        .get(frame::STAMP_SLOT)
        .map_or(fluid.reported, |t| *t as u32);
    if (ticket.wrapping_sub(fluid.reported) as i32) > 0 {
        fluid.reported = ticket;
    }
    let (mut arrived, settled) = fluid.settle(ticket, &raw);
    if settled.substeps == 0.0 || arrived.is_empty() {
        return;
    }
    for coupling in &mut arrived {
        coupling.seconds = settled.seconds;
        coupling.substeps = settled.substeps;
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

fn receive_stamp(event: On<ReadbackComplete>, mut fluid: ResMut<Fluid>, mut commands: Commands) {
    let stamp: Vec<u32> = event.to_shader_type();
    fluid.snapshot.ticket = stamp.first().copied().unwrap_or(fluid.issued);
    fluid.snapshot_part(4);
    commands.entity(event.entity).try_despawn();
}
