//! Water as particles that a grid keeps from being squeezed (an affine particle-in-cell
//! solver: see `grid.wgsl`), solved on the GPU with Akinci-style coupling to rigid bodies,
//! confined by a vessel the shaders describe. The CPU only keeps the particle count, feeds in
//! new particles and body poses each frame, and reads back what the water did to the bodies.
//!
//! The water lives in its vessel's own frame, where water at rest stays put however fast the
//! vessel turns, and the GPU only ever simulates the canonical water of [`Resolution`]: the
//! CPU scales metres and seconds to its units on the way in and back on the way out. The
//! particles are binned by a hash of their grid cell, the grid exists only where the water is
//! and the surface is meshed only there, so neither the vessel's size nor the water's extent
//! costs anything: the work is the number of particles, which is capped. Water past the cap is
//! made coarser instead, every particle standing for twice as much of it, so any amount of
//! water fits the same budget, and a vessel of any size sets a floor on how fine its water is,
//! so any size runs the same.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::Shader;

use crate::core::math::{Rng, Vec3d, norm};
use crate::core::rigid::{Body, BodyShape, WaterCoupling};
use crate::core::units::{KilogramsPerCubicMetre, Litres, Metres, MetresPerSecond, Seconds};
use crate::core::vessel::WaterFrame;

mod frame;
mod gpu;
mod resolution;
mod surface;

pub use frame::{Bodies, FluidFrame, GpuBodies, Params, Substep};
pub use gpu::{FluidBuffers, FluidStep};
pub use resolution::{REST_DENSITY, Resolution, SPACINGS_FROM_ORIGIN, STEP_RATE, canonical};
pub use surface::{
    GRID_REACH, MAX_BLOCKS, MAX_DROPLETS, MAX_INDICES, MAX_MOTES, MAX_VERTICES, SurfaceBuffers,
    SurfaceParams, grid_reach, surface_cell,
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
/// Slots of the table the grid's cells are kept in: every cell round every particle, which
/// for water that lies together is a few to a particle.
pub const GRID_SLOTS: usize = 4 * MAX_PARTICLES;
/// Substeps a single frame may run; beyond that the simulation falls behind real time.
pub const MAX_SUBSTEPS_PER_FRAME: usize = 4;
/// What a rough bed takes of the dynamic pressure of the water running over it: Manning's
/// roughness of earth and short grass, under a particle's depth of water.
const BED_FRICTION: f32 = 0.008;
/// Water's surface tension, in newtons a metre, and the Weber number past which the air tears
/// a drop apart: what it leaves whole is as wide as has that number at the speed it meets.
const SURFACE_TENSION: f64 = 0.072;
/// The air's pressure over its density, at the temperature of a room, in metres squared per
/// second squared: what tells how hard air of a given density presses.
const AIR_STIFFNESS: f64 = 84_400.0;
const SHATTERING_WEBER: f64 = 12.0;
/// How long the air takes to tear a drop apart, counted in the time the drop takes to cross
/// its own width through the air, scaled by the root of how much denser than the air it is.
const BREAKUP_TIME: f32 = 5.0;
/// The narrowest drops the air tears water to.
const FINEST_DROP: Metres = Metres(0.001);

/// Sweeps of the pressure over each colour of its checkerboard in a step. The pressure starts
/// from what the water carried out of the last step, so the sweeps only have to follow what
/// changed since, which reaches a cell further with each.
const RELAXATIONS: usize = 24;
/// Sweeps of the push that shifts water back to its own density. What they leave undone in a
/// step is still there to be done in the next, so they need not reach far in any one.
const SETTLINGS: usize = 8;
const WET_REF: f32 = 300.0;
/// The random nudge new water gets, in the canonical water's units.
/// How far off its lattice site, in spacings, water is put down, so that no two rows of it are
/// ever exactly in line.
const JITTER: f64 = 0.01;
/// How many sites a placement offers per particle, and at least, so that water placed onto
/// water finds free room round it.
const SITES_PER_PARTICLE: usize = 64;
const LEAST_SITES: usize = 512;
/// How many sites are looked through at first for each one sought: a placement by a wall finds
/// half of them outside the vessel.
const ROOM_SOUGHT: usize = 2;

/// The offsets of at least `holding` lattice sites in a ball about a site, nearest first: what
/// a placement of water is offered, so that water placed onto water finds free room around it.
/// The ball every ordinary placement asks for is made once.
fn lattice_ball(holding: usize) -> std::borrow::Cow<'static, [[i32; 3]]> {
    static USUAL: std::sync::OnceLock<Vec<[i32; 3]>> = std::sync::OnceLock::new();
    const USUAL_HOLDS: usize = ROOM_SOUGHT * gpu::SITES;
    if holding <= USUAL_HOLDS {
        USUAL.get_or_init(|| ball_of(USUAL_HOLDS)).as_slice().into()
    } else {
        ball_of(holding).into()
    }
}

fn ball_of(holding: usize) -> Vec<[i32; 3]> {
    let reach = (3.0 * holding as f64 / (4.0 * std::f64::consts::PI))
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
    ball.sort_by_key(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
    ball.truncate(holding);
    ball
}

/// Resolutions remembered by the frame they took effect, for reading back what an older frame
/// left on the GPU.
const GENERATIONS_KEPT: usize = 64;
const SHADERS: [&str; 8] = [
    "embedded://game/core/fluid/shaders/common.wgsl",
    "embedded://game/core/fluid/shaders/views.wgsl",
    "embedded://game/core/fluid/shaders/particles.wgsl",
    "embedded://game/core/fluid/shaders/grid.wgsl",
    "embedded://game/core/fluid/shaders/sort.wgsl",
    "embedded://game/core/fluid/shaders/bodies.wgsl",
    "embedded://game/core/fluid/shaders/surface.wgsl",
    "embedded://game/core/fluid/shaders/spray.wgsl",
];

#[derive(Clone, Debug, PartialEq)]
pub struct FluidParams {
    pub body_drag: f32,
    /// How dense the air the vessel holds is, which drags on the water it meets: none in a
    /// vacuum.
    pub air_density: KilogramsPerCubicMetre,
    /// A safety clamp on the particles' speed through the vessel, well above anything they
    /// could be thrown at.
    pub max_speed: MetresPerSecond,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            body_drag: 0.5,
            air_density: KilogramsPerCubicMetre(1.2),
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
    /// The finest the water may be, set by its reach.
    floor: Resolution,
    /// How far from its frame's origin the water has been put, or been seen to reach, in metres.
    reach: f64,
    /// How many times the water has been emptied.
    emptied: u32,
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
            reach: 0.0,
            emptied: 0,
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
        SurfaceParams::new()
    }

    /// How far from its frame's origin the water has been put, or been seen to reach.
    pub fn reach(&self) -> Metres {
        Metres(self.reach as f32)
    }

    /// Water has been seen this far from its frame's origin. Single precision holds the water
    /// only so many of its spacings out, so water reaching farther is made coarser.
    pub fn reached(&mut self, distance: Metres) {
        let distance = distance.0 as f64;
        if distance > self.reach {
            self.reach = distance;
            self.floor = Resolution::finest_for(Metres(distance as f32));
        }
    }

    /// How many times the water has been emptied, which tells one water from the next.
    pub fn emptied(&self) -> u32 {
        self.emptied
    }

    /// Make the water coarser, a step a frame, while it is finer than its reach allows.
    pub fn keep_floor(&mut self) {
        if self.resolution.spacing.0 < self.floor.spacing.0 {
            if self.count == 0 {
                self.set_resolution(self.floor);
            } else if self.thin.is_none() {
                self.coarsen();
            }
        }
    }

    /// Add a particle of the current resolution. When the budget is full the water is made
    /// coarser first, which frees half of it; only a second fill within one frame is refused.
    pub fn add(&mut self, p: Particle) -> bool {
        if !self.reserve() {
            return false;
        }
        self.reached(Metres(norm(&p.position) as f32));
        self.pending.push(p);
        true
    }

    /// Put down up to `count` particles of water about a point of the vessel's frame, at rest
    /// in it: each takes a site of the water's rest lattice, the free sites nearest the point
    /// first, so that water is added as gently as can be and water placed onto water spreads
    /// round it rather than into it. Once a frame has offered all the sites it can, the rest
    /// take the nearest sites free or not. Only sites the vessel `has_room` at are offered:
    /// a site outside it is no site, and moving it inside would lay it on top of another.
    /// Water that would overfill the budget is made coarser first, and what is asked for with
    /// it, so that it is still as much water. Returns how many were added, of the particles
    /// the water is made of once they have been.
    pub fn inject(
        &mut self,
        centre: Vec3d,
        count: u32,
        mut has_room: impl FnMut(Vec3d) -> bool,
    ) -> u32 {
        let mut count = count;
        while self.count as usize + count as usize > MAX_PARTICLES && self.thin.is_none() {
            self.coarsen();
            count = count.div_ceil(2);
        }
        let pitch = self.resolution.lattice().0 as f64;
        let base = centre.map(|x| (x / pitch).floor());
        let room = gpu::SITES - self.sites.len();
        let offered = (count as usize * SITES_PER_PARTICLE)
            .max(LEAST_SITES)
            .min(room);
        let sought = offered.max(count as usize);
        let mut holding = ROOM_SOUGHT * sought;
        let mut sites: Vec<Vec3d> = Vec::new();
        let mut farthest = 0.0f64;
        // a ball twice the size that holds no more room has come to the end of the vessel's,
        // which only shows if a site has room or not whichever ball it is met in: so it is
        // the site that is asked about, and not where the water put down on it is nudged to
        loop {
            let found_before = sites.len();
            sites.clear();
            for offset in lattice_ball(holding).iter() {
                if sites.len() >= sought {
                    break;
                }
                let site = [0, 1, 2].map(|c| base[c] + offset[c] as f64 + 0.5);
                if has_room(site.map(|x| x * pitch)) {
                    let mut nudge = |x: f64| x + (self.rng.next_f32() as f64 - 0.5) * JITTER;
                    farthest = farthest.max(norm(&offset.map(|x| x as f64)));
                    sites.push(site.map(|x| nudge(x) * pitch));
                }
            }
            if sites.len() >= sought
                || (holding > ROOM_SOUGHT * sought && sites.len() == found_before)
            {
                break;
            }
            holding *= 2;
        }
        self.reached(Metres((norm(&centre) + (farthest + 1.0) * pitch) as f32));
        let count = count.min(sites.len() as u32);
        let mut added = 0;
        if offered >= count as usize {
            while added < count && self.reserve() {
                added += 1;
                self.joining += 1;
            }
            let length = self.resolution.length();
            for site in sites.iter().take(offered) {
                let p = site.map(|x| (x / length) as f32);
                self.sites.push([p[0], p[1], p[2], 0.0]);
            }
        } else {
            for &position in sites.iter().take(count as usize) {
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

    /// Remove all water; what comes next starts out as fine as water can be.
    pub fn clear(&mut self) {
        self.restore(Resolution::FINEST);
    }

    /// Remove all water and take this resolution for what is added next, as when saved water
    /// is loaded back.
    pub fn restore(&mut self, resolution: Resolution) {
        self.count = 0;
        self.thin = None;
        self.pending.clear();
        self.joining = 0;
        self.sites.clear();
        self.reach = 0.0;
        self.floor = Resolution::FINEST;
        self.emptied = self.emptied.wrapping_add(1);
        self.set_resolution(resolution);
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
        let mut serial = self.issued.wrapping_mul(40_503);
        let pending = self
            .pending
            .drain(..)
            .flat_map(|p| {
                let x = p.position.map(|x| (x / length) as f32);
                let v = p.velocity.map(|v| (v * time / length) as f32);
                serial = serial.wrapping_add(1);
                let told_apart = (serial.wrapping_mul(2_654_435_761) >> 22) as f32;
                [[x[0], x[1], x[2], p.foam], [v[0], v[1], v[2], told_apart]]
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
        let on_gpu = self.count - self.pending.len() as u32 - self.joining;
        // only water on the GPU is thinned there, which it can be once a frame
        if on_gpu > 0 {
            self.thin = Some((on_gpu, self.resolution));
        }
        // the water waiting to join is thinned as the water on the GPU is: every other
        // particle of it, and every other site of the lattice, which are the sites of a
        // lattice with twice the room to each
        let mut kept = false;
        self.pending.retain(|_| {
            kept = !kept;
            kept
        });
        self.sites.retain(|site| {
            let steps: f32 = site[..3]
                .iter()
                .map(|x| (x / canonical::LATTICE - 0.5).round())
                .sum();
            steps.rem_euclid(2.0) == 0.0
        });
        self.joining = self.joining.div_ceil(2).min(self.sites.len() as u32);
        self.count = on_gpu.div_ceil(2) + self.pending.len() as u32 + self.joining;
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
        embedded_asset!(app, "shaders/grid.wgsl");
        embedded_asset!(app, "shaders/sort.wgsl");
        embedded_asset!(app, "shaders/bodies.wgsl");
        embedded_asset!(app, "shaders/surface.wgsl");
        embedded_asset!(app, "shaders/views.wgsl");
        embedded_asset!(app, "shaders/spray.wgsl");
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
struct FluidShaders(#[allow(dead_code)] [Handle<Shader>; 8]);

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
