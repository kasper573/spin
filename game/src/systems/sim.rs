//! The running simulation: the drum and the avatar are stepped on the CPU and the
//! water follows on the GPU. The bodies are stepped by a period of [`SUBSTEP_RATE`] whatever the
//! display does, so that what they do is the same at any frame rate: a frame takes as many whole
//! substeps as its time holds, up to a few, and carries what is left over to the next; beyond
//! that, time stretches rather than the frame. The eye is drawn between where it was at the last
//! two substeps, as far on as the time left over, so that a display faster than the substeps
//! still sees it move smoothly. The water steps at its own
//! fixed pace whatever the display does, so it costs the same at any frame rate: the step of
//! its [`Resolution`](crate::core::fluid::Resolution), which is longer in proportion for
//! coarser water, whose waves are slower. Each of its steps is taken at the moment the bodies'
//! substeps pass its due time, with the drum and the bodies as they are then. Requested time
//! (the tests, the bench, scripts) always steps at exactly [`SUBSTEP_RATE`].
//!
//! The bodies are simulated in the drum's own frame about a site on its wall, so their
//! coordinates stay small however large the ring; the site follows the avatar as the scene
//! asks, and everything measured from the old site is carried over to the new one.
//!
//! The initial state is a ring world: ground all the way round the drum, spinning exactly fast
//! enough that the avatar standing on it weighs what it would on Earth, and nothing else.
use bevy::prelude::*;

use crate::core::avatar::{self, AvatarInput, Gyros, Thrusters};
use crate::core::fluid::{
    Bodies, Fluid, FluidFrame, FluidParams, FluidReady, MAX_SUBSTEPS_PER_FRAME,
};
use crate::core::math::{
    Quatd, Vec3d, norm, quat_about_y, quat_conjugate, quat_from_basis, quat_from_rotation_vector,
    quat_mul, quat_rotate, rotation_vector,
};
use crate::core::rigid::{self, Body, BodyParams, BodyShape, Ground, WaterCoupling};
use crate::core::sheet::Sheet;
use crate::core::units::{
    EARTH_GRAVITY, Hertz, Metres, MetresPerSecond, MetresPerSecondSquared, Radians,
    RadiansPerSecond, RadiansPerSecondSquared, Seconds, WorldTime,
};
use crate::core::vessel::Vessel;
use crate::systems::drum::{DEFAULT_RING, Drum, GROUND_DEPTH, Ring, SheetWindow, Shift};

/// The most the bodies are stepped by at once.
pub const SUBSTEP_RATE: Hertz = Hertz(120.0);
/// The speed clamps sit this far above the rim of the drum.
const SPEED_HEADROOM: f32 = 40.0;
/// How far the eye's pupil reaches from its middle: while the water's surface lies across it the
/// eye is no more under the water than over it, and sees as it last did.
const PUPIL: Metres = Metres(0.02);
/// What a frame's time may fall short of a whole number of substeps by and still take them all:
/// frames that together last a whole number of substeps then take exactly that many.
const SUBSTEP_SLACK: f64 = 1e-6;
const MAX_FRAME_TIME: Seconds = Seconds(0.1);
const RATE_WINDOW: Seconds = Seconds(1.0);
const AVATAR_SHAPE: usize = 0;

#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum SimSet {
    /// Input handling that mutates the simulation.
    Command,
    /// Advancing the simulation.
    Step,
    /// Everything that reads the stepped state.
    Observe,
}

/// The avatar's weight as the ground pushes back, in g, its speed over that ground (or through
/// the air), and whether it is a solid body with nothing under its feet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footing {
    pub weight: f32,
    pub ground_speed: f32,
    pub airborne: bool,
}

/// One step the water takes: the drum and the bodies as they were when it was due.
#[derive(Clone)]
pub struct SubstepRecord {
    pub dt: Seconds,
    pub spin: RadiansPerSecond,
    pub spin_rate: RadiansPerSecondSquared,
    pub angle: Radians,
    pub bodies: Bodies,
}

/// How far the eye as drawn is behind the eye as last stepped: the last substep's move and turn
/// of it, taken back by as much of a substep as the frame being drawn comes before the next one.
#[derive(Clone, Copy, Default)]
struct EyeLag {
    moved: Vec3d,
    turned: Vec3d,
    short: f64,
}

impl EyeLag {
    fn from_step((from, turned): (Vec3d, Quatd), (to, turning): (Vec3d, Quatd)) -> EyeLag {
        EyeLag {
            moved: [0, 1, 2].map(|k| to[k] - from[k]),
            turned: rotation_vector(&quat_mul(&turning, &quat_conjugate(&turned))),
            short: 0.0,
        }
    }

    fn drawn(&self, (p, q): (Vec3d, Quatd)) -> (Vec3d, Quatd) {
        let back = quat_from_rotation_vector(&self.turned.map(|c| -c * self.short));
        (
            [0, 1, 2].map(|k| p[k] - self.moved[k] * self.short),
            quat_mul(&back, &q),
        )
    }
}

#[derive(Resource)]
pub struct Simulation {
    pub drum: Drum,
    /// What the pilot asks of the avatar during the coming substeps.
    pub avatar_input: AvatarInput,
    /// How hard each of the avatar's thrusters is firing.
    pub thrusters: Thrusters,
    /// The attitude the avatar's gyros hold it to.
    pub gyros: Gyros,
    pub params: FluidParams,
    pub body_params: BodyParams,
    /// Simulated time since the last reset.
    pub time: WorldTime,
    /// Fraction of real time the simulation keeps up with (1 = full speed).
    pub rate: f32,
    /// The water's steps of the last frame.
    pub substeps: Vec<SubstepRecord>,
    /// The avatar.
    bodies: Vec<Body>,
    shapes: [BodyShape; 1],
    /// Real time not yet stepped, short of a substep.
    accumulator: f64,
    eye_lag: EyeLag,
    /// Simulated time since the water last stepped.
    water_due: f64,
    queued: f32,
    stepped: Seconds,
    window: (f32, f32),
    /// Whether the eye is under the water's surface.
    eye_under: bool,
}

impl Default for Simulation {
    fn default() -> Self {
        Simulation::new(DEFAULT_RING)
    }
}

/// Distance from the axis to the avatar's centre of mass when it stands on the initial ground.
pub fn standing_radius(ring: Ring) -> Metres {
    Metres(ring.floor_radius().0 - avatar::standing_height() as f32)
}

/// The spin at which the standing avatar's centre of mass is carried round at exactly one g.
pub fn standing_spin(ring: Ring) -> RadiansPerSecond {
    RadiansPerSecond((EARTH_GRAVITY.0 / standing_radius(ring).0 as f64).sqrt() as f32)
}

/// The artificial gravity felt at the standing avatar's centre of mass under `spin`.
pub fn standing_gravity(spin: RadiansPerSecond, ring: Ring) -> MetresPerSecondSquared {
    let w = spin.0 as f64;
    MetresPerSecondSquared(w * w * standing_radius(ring).0 as f64)
}

impl Simulation {
    /// The initial ring world at this size: ground all the way round, spinning at one g, the
    /// avatar standing on it with its thrusters equalized to that.
    pub fn new(ring: Ring) -> Simulation {
        let shapes = [avatar::shape()];
        let mut drum = Drum::new(ring);
        drum.spin = standing_spin(ring);
        drum.target_spin = drum.spin;
        let avatar = standing_avatar(&shapes[AVATAR_SHAPE]);
        let thrusters =
            Thrusters::with_power(avatar::equalized_thrust(standing_gravity(drum.spin, ring)));
        Simulation {
            gyros: Gyros::holding(&avatar),
            drum,
            avatar_input: AvatarInput::default(),
            thrusters,
            params: FluidParams::default(),
            body_params: BodyParams::default(),
            time: WorldTime(0.0),
            rate: 1.0,
            substeps: Vec::new(),
            bodies: vec![avatar],
            shapes,
            accumulator: 0.0,
            eye_lag: EyeLag::default(),
            water_due: 0.0,
            queued: 0.0,
            stepped: Seconds(0.0),
            window: (0.0, 0.0),
            eye_under: false,
        }
    }

    /// Raise (or, by a negative amount, lower) the ground within `radius` of a point of the
    /// frame.
    pub fn sculpt(&mut self, at: Vec3d, radius: f64, amount: f64) {
        self.drum.sculpt(at, radius, amount);
    }

    /// Change the ring's size about every body, which stays where it was about the axis: the
    /// wall moves, not the bodies, except that solid ones are kept inside it. The ground and
    /// the water stay where they were on the wall.
    pub fn resize(&mut self, ring: Ring, (sheet, window): (&mut Sheet, &mut SheetWindow)) {
        if ring == self.drum.ring {
            return;
        }
        let resized = self.drum.resize(ring);
        window.lay(&self.drum, sheet, Some(resized.ground));
        let carried = resized.frame;
        for body in &mut self.bodies {
            body.p = [
                body.p[0] + carried[0],
                body.p[1] + carried[1],
                body.p[2] + carried[2],
            ];
            if body.solid {
                let reach = self.shapes[body.shape].reach();
                body.p = self.drum.place_sphere_inside(body.p, reach);
            }
        }
    }
}

/// The avatar upright on the ground at the site, at rest on it and facing spinward, so that
/// it starts out standing rather than falling.
fn standing_avatar(shape: &BodyShape) -> Body {
    let p = [
        -(GROUND_DEPTH.0 as f64 + avatar::standing_height()),
        0.0,
        0.0,
    ];
    let up = [-1.0, 0.0, 0.0];
    let back = [0.0, 0.0, 1.0];
    let right = [
        up[1] * back[2] - up[2] * back[1],
        up[2] * back[0] - up[0] * back[2],
        up[0] * back[1] - up[1] * back[0],
    ];
    let q = quat_from_basis(&right, &up, &back);
    Body::new(AVATAR_SHAPE, shape, p, q, [0.0; 3], [0.0; 3])
}

impl Simulation {
    /// Advance by one frame of real time, or by queued time if any was requested. The water's
    /// coupling from the last frame lands on the bodies first, and with it what the sheet of
    /// water lying on the floor does to them.
    pub fn advance(&mut self, real: Seconds, fluid: &mut Fluid, lying: (&Sheet, &SheetWindow)) {
        let max_dt = SUBSTEP_RATE.period().0;
        self.substeps.clear();
        let requested = self.queued > 0.0;
        let (steps, dt) = if requested {
            let steps = ((self.queued / max_dt).round() as usize).min(MAX_SUBSTEPS_PER_FRAME);
            self.queued = (self.queued - steps as f32 * max_dt).max(0.0);
            if self.queued < max_dt * 0.5 {
                self.queued = 0.0;
            }
            (steps, max_dt)
        } else {
            self.accumulator = (self.accumulator + real.0 as f64).min(MAX_FRAME_TIME.0 as f64);
            let whole = ((self.accumulator + SUBSTEP_SLACK) / max_dt as f64) as usize;
            let steps = whole.min(MAX_SUBSTEPS_PER_FRAME);
            self.accumulator = (self.accumulator - steps as f64 * max_dt as f64).max(0.0);
            (steps, max_dt)
        };
        let coupling = if steps > 0 {
            let flying = fluid.take_coupling(steps as f64 * dt as f64);
            self.with_lying(flying, lying, steps as f64, dt as f64)
        } else {
            None
        };
        self.clamp_speeds();
        let sunk = lying.1.sunk(&self.drum, lying.0, self.stepped_eye().0);
        if sunk.0.abs() > PUPIL.0 {
            self.eye_under = sunk.0 > 0.0;
        }
        let water_step = fluid.step().0 as f64;
        for k in 0..steps {
            let eye = self.stepped_eye();
            avatar::drive(
                &mut self.bodies[0],
                &mut self.thrusters,
                &mut self.gyros,
                &self.avatar_input,
                &self.drum,
                dt as f64,
            );
            self.drum.advance(dt as f64);
            let water = if k == 0 { coupling.as_deref() } else { None };
            let before = self.bodies[0].p;
            rigid::step(
                dt as f64,
                &self.drum,
                &self.shapes,
                &mut self.bodies,
                water,
                &self.body_params,
            );
            if let Some(passage) = self.drum.passage(before, self.bodies[0].p) {
                self.bodies[0].carried_through(&passage);
                self.gyros.held = quat_mul(&passage.turn, &self.gyros.held);
                self.gyros.footing = self.gyros.footing.map(|n| passage.turned(n));
            }
            let stepped = self.stepped_eye();
            self.eye_lag = if self.drum.passage(eye.0, stepped.0).is_some() {
                EyeLag::default()
            } else {
                EyeLag::from_step(eye, stepped)
            };
            self.time = self.time.after(Seconds(dt));
            self.water_due += dt as f64;
            if self.water_due >= water_step * (1.0 - 1e-6) {
                self.water_due = (self.water_due - water_step).max(0.0);
                self.substeps.push(SubstepRecord {
                    dt: Seconds(water_step as f32),
                    spin: self.drum.spin,
                    spin_rate: self.drum.spin_rate,
                    angle: self.drum.angle,
                    bodies: fluid.pack(&self.bodies, &self.shapes, &self.drum.water_frame()),
                });
            }
        }
        // requested time is drawn as stepped, whatever real time the frame took
        self.eye_lag.short = if requested || real.0 <= 0.0 {
            0.0
        } else {
            (1.0 - self.accumulator / dt as f64).max(0.0)
        };
        self.window.0 += real.0;
        self.stepped = Seconds(steps as f32 * dt);
        self.window.1 += self.stepped.0;
        if self.window.0 >= RATE_WINDOW.0 {
            self.rate = (self.window.1 / self.window.0).min(1.0);
            self.window = (0.0, 0.0);
        }
    }

    /// What all the water does to the bodies over the frame: what the water in flight did, if
    /// it reported, and what the sheet does over the same time and as many substeps.
    fn with_lying(
        &self,
        flying: Option<Vec<WaterCoupling>>,
        (sheet, window): (&Sheet, &SheetWindow),
        steps: f64,
        dt: f64,
    ) -> Option<Vec<WaterCoupling>> {
        let mut all = flying;
        for (i, body) in self.bodies.iter().enumerate() {
            let Some(felt) = window.bearing(&self.drum, sheet, body, &self.shapes[body.shape])
            else {
                continue;
            };
            let all = all.get_or_insert_with(|| {
                let none = WaterCoupling {
                    seconds: steps * dt,
                    substeps: steps,
                    ..default()
                };
                vec![none; self.bodies.len()]
            });
            let had = &mut all[i];
            for k in 0..3 {
                had.buoyancy[k] += felt.buoyancy[k] * had.seconds;
                had.buoyancy_torque[k] += felt.buoyancy_torque[k] * had.seconds;
                had.flow[k] += felt.flow[k] * had.substeps;
            }
            had.coupling += felt.coupling * had.substeps;
            had.wet += felt.wet * had.substeps;
        }
        all
    }

    /// Keep the safety clamps on speed and spin above what the spinning rim reaches, whatever
    /// size and spin the ring is set to.
    fn clamp_speeds(&mut self) {
        let rim = self.drum.spin.0.abs().max(self.drum.target_spin.0.abs());
        let speed = MetresPerSecond(rim * self.drum.ring.radius.0 * 2.0 + SPEED_HEADROOM);
        self.params.max_speed = speed;
        self.body_params.max_speed = speed;
        self.body_params.max_spin = RadiansPerSecond(rim * 2.0 + 25.0);
    }

    /// Ask for exactly this much simulated time over the coming frames, real time aside.
    pub fn request(&mut self, seconds: Seconds) {
        self.queued += seconds.0;
    }

    /// Requested time not yet simulated.
    pub fn queued(&self) -> Seconds {
        Seconds(self.queued)
    }

    /// How much of the world's time the last frame stepped: what the player's hands work for
    /// over a frame, so that they do as much in the world's time however fast the machine is.
    pub fn stepped(&self) -> Seconds {
        self.stepped
    }

    /// Back to the initial ring world of the same size, but the settings, the target spin, the
    /// thrusters' power, the sites of the bodies and the water, and the avatar stay as they are.
    pub fn reset(&mut self) {
        let params = self.params.clone();
        let body_params = self.body_params.clone();
        let target = self.drum.target_spin;
        let (site, water) = (self.drum.site, self.drum.water);
        let power = self.thrusters.power;
        let gyros = self.gyros;
        let avatar = std::mem::take(&mut self.bodies).swap_remove(0);
        *self = Simulation::new(self.drum.ring);
        self.params = params;
        self.body_params = body_params;
        self.drum.target_spin = target;
        self.drum.site = site;
        self.drum.water = water;
        self.thrusters.power = power;
        self.gyros = gyros;
        self.bodies[0] = avatar;
    }

    /// Move the site to the wall below a point of the frame, and carry everything measured
    /// from the old site over to the new one.
    pub fn resite(&mut self, below: Vec3d) -> Shift {
        let shift = self.drum.resite(below);
        let turn = quat_about_y(shift.arc / self.drum.ring.radius.0 as f64);
        for body in &mut self.bodies {
            let p = self.drum.carried(body.p, shift);
            let q = quat_mul(&turn, &body.q);
            let v = self.drum.carried_vector(body.v, shift);
            let w = self.drum.carried_vector(body.w, shift);
            let ground = body.ground.map(|ground| Ground {
                point: self.drum.carried(ground.point, shift),
                normal: self.drum.carried_vector(ground.normal, shift),
                ..ground
            });
            body.place(p, q);
            body.v = v;
            body.w = w;
            body.ground = ground;
        }
        self.gyros.held = quat_mul(&turn, &self.gyros.held);
        self.gyros.footing = self.gyros.footing.map(|n| quat_rotate(&turn, &n));
        self.eye_lag.moved = self.drum.carried_vector(self.eye_lag.moved, shift);
        self.eye_lag.turned = quat_rotate(&turn, &self.eye_lag.turned);
        shift
    }

    /// Where the avatar's eye is seen from and how it is turned in the frame being drawn: as
    /// far between where it was at the last two substeps as real time has come since.
    pub fn eye(&self) -> (Vec3d, Quatd) {
        self.eye_lag.drawn(self.stepped_eye())
    }

    /// Where the avatar's eye is and how it is turned as last stepped. The eye goes through a
    /// portal on its own, ahead of the body it is set in or after it, so it is never anywhere
    /// but where it looks from.
    fn stepped_eye(&self) -> (Vec3d, Quatd) {
        let hull = self.avatar();
        let eye = avatar::eye(hull);
        match self.drum.passage(hull.p, eye) {
            Some(passage) => (passage.point, quat_mul(&passage.turn, &hull.q)),
            None => (eye, hull.q),
        }
    }

    /// Where the viewer is, which everything is drawn about: the avatar's centre, or its eye
    /// once that has gone through a portal ahead of the rest of it.
    pub fn viewer(&self) -> Vec3d {
        let hull = self.avatar();
        match self.drum.passage(hull.p, avatar::eye(hull)) {
            Some(passage) => passage.point,
            None => hull.p,
        }
    }

    pub fn shapes(&self) -> &[BodyShape] {
        &self.shapes
    }

    /// Whether the eye is under water: the avatar wet over its head, and within the drum at
    /// that, since the water is held in by the drum and a viewer outside it looks in through
    /// glass however much of the hull the water it is looking at happens to touch.
    pub fn submerged(&self) -> bool {
        self.eye_under && self.drum.holds(self.avatar().p)
    }

    pub fn avatar(&self) -> &Body {
        &self.bodies[0]
    }

    /// What the avatar's feet feel right now.
    pub fn footing(&self) -> Footing {
        let avatar = self.avatar();
        match avatar.ground {
            Some(ground) => Footing {
                weight: (ground.support.0 * avatar.inv_m / EARTH_GRAVITY.0) as f32,
                ground_speed: norm(&avatar.point_velocity(&ground.point)) as f32,
                airborne: false,
            },
            None => Footing {
                weight: 0.0,
                ground_speed: norm(&avatar.v) as f32,
                airborne: avatar.solid,
            },
        }
    }

    pub fn avatar_mut(&mut self) -> &mut Body {
        &mut self.bodies[0]
    }

    /// Add up to `count` particles of water around a point of the frame, at rest in the drum.
    /// Water put where there is none has its frame set down on the wall under it, so that it is
    /// as fine as water can be wherever it is put.
    pub fn inject(&mut self, fluid: &mut Fluid, centre: Vec3d, count: u32) -> u32 {
        if fluid.is_empty() {
            self.drum.settle_water(centre);
        }
        let drum = &self.drum;
        let margin = fluid.resolution().margin().0 as f64;
        fluid.inject(drum.to_water(centre), count, |p| drum.has_room(p, margin))
    }
}

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Simulation>()
            .configure_sets(
                Update,
                (SimSet::Command, SimSet::Step, SimSet::Observe).chain(),
            )
            .add_systems(Startup, register_shapes)
            .add_systems(Update, step.in_set(SimSet::Step));
    }
}

fn register_shapes(sim: Res<Simulation>, mut fluid: ResMut<Fluid>) {
    fluid.set_shapes(sim.shapes());
}

fn step(
    mut sim: ResMut<Simulation>,
    mut fluid: ResMut<Fluid>,
    mut frame: ResMut<FluidFrame>,
    ready: Res<FluidReady>,
    time: Res<Time>,
    (sheet, window): (Res<Sheet>, Res<SheetWindow>),
) {
    if !ready.get() {
        return;
    }
    fluid.keep_floor();
    sim.advance(Seconds(time.delta_secs()), &mut fluid, (&sheet, &window));
    let substeps: Vec<(Seconds, Bodies)> = sim
        .substeps
        .iter()
        .map(|r| (r.dt, r.bodies.clone()))
        .collect();
    *frame = fluid.frame(&sim.params, &substeps, &sim.drum.water_frame());
}
