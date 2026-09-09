//! The running simulation: the drum, the avatar and the rafts are stepped on the CPU and the
//! water follows on the GPU. Real time takes one substep per frame, no longer than a sixtieth
//! of a second, so a fast display gets smooth bodies and a slow frame is split into a few
//! substeps; beyond that, time stretches rather than the frame. The water steps at exactly
//! sixty hertz whatever the display does, so it costs the same at any frame rate: each of its
//! steps is taken at the moment the bodies' substeps pass a sixtieth, with the drum and the
//! bodies as they are then. Requested time (the tests, the bench, scripts) always steps at
//! exactly sixty hertz, so the water steps with every substep.
//!
//! The initial state is a ring world: ground all the way round the drum, spinning exactly fast
//! enough that the avatar standing on it weighs what it would on Earth, and nothing else.
use bevy::prelude::*;

use crate::core::avatar::{self, AvatarInput, Gyros, Thrusters};
use crate::core::fluid::{
    Bodies, Fluid, FluidFrame, FluidParams, FluidReady, MAX_BODIES, MAX_SUBSTEPS_PER_FRAME,
};
use crate::core::math::{basis_from_normal, norm, quat_from_basis};
use crate::core::rigid::{self, Body, BodyParams, BodyShape};
use crate::core::units::{
    EARTH_GRAVITY, Hertz, Metres, MetresPerSecond, MetresPerSecondSquared, Radians,
    RadiansPerSecond, Seconds,
};
use crate::core::vessel::Vessel;
use crate::systems::drum::{DEFAULT_RING, Drum, Ring};
use crate::systems::rafts;

/// The water's step rate, and the most the bodies are stepped by at once.
pub const SUBSTEP_RATE: Hertz = Hertz(60.0);
pub const MAX_RAFTS: usize = MAX_BODIES - 1;
/// The speed clamps sit this far above the rim of the drum.
const SPEED_HEADROOM: f32 = 40.0;
/// Shortest substep real time is split into; faster frames are gathered into one.
const MIN_SUBSTEP: Seconds = Seconds(1.0 / 240.0);
const MAX_FRAME_TIME: Seconds = Seconds(0.1);
/// How many frames the bodies may run ahead of the water's coupling before the simulation waits
/// for the GPU: a coupling this stale is still a fraction of a degree of the drum's turn, while
/// an unbounded backlog would let the bodies chase water that has long moved on.
const MAX_FRAMES_AHEAD: u32 = 3;
const RATE_WINDOW: Seconds = Seconds(1.0);
const AVATAR_SHAPE: usize = 0;
const RAFT_SHAPE: usize = 1;

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
    pub angle: Radians,
    pub bodies: Bodies,
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
    pub time: Seconds,
    /// Fraction of real time the simulation keeps up with (1 = full speed).
    pub rate: f32,
    /// The water's steps of the last frame.
    pub substeps: Vec<SubstepRecord>,
    /// The avatar first, then the rafts.
    bodies: Vec<Body>,
    shapes: [BodyShape; 2],
    accumulator: f32,
    /// Simulated time since the water last stepped.
    water_due: f64,
    queued: f32,
    window: (f32, f32),
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
        let shapes = [avatar::shape(), rafts::shape()];
        let mut drum = Drum::new(ring);
        drum.spin = standing_spin(ring);
        drum.target_spin = drum.spin;
        let avatar = standing_avatar(&shapes[AVATAR_SHAPE], &drum);
        let thrusters =
            Thrusters::with_power(avatar::equalized_thrust(standing_gravity(drum.spin, ring)));
        Simulation {
            gyros: Gyros::holding(&avatar),
            drum,
            avatar_input: AvatarInput::default(),
            thrusters,
            params: FluidParams::default(),
            body_params: BodyParams::default(),
            time: Seconds(0.0),
            rate: 1.0,
            substeps: Vec::new(),
            bodies: vec![avatar],
            shapes,
            accumulator: 0.0,
            water_due: 0.0,
            queued: 0.0,
            window: (0.0, 0.0),
        }
    }

    /// Make the ring another size. The water and the landscape stretch to fit on their own;
    /// whatever body the new walls would cut through is pulled inside them.
    pub fn resize(&mut self, ring: Ring) {
        if ring == self.drum.ring {
            return;
        }
        self.drum.resize(ring);
        for body in self.bodies.iter_mut().filter(|body| body.solid) {
            let reach = self.shapes[body.shape].reach();
            body.p = self.drum.place_sphere_inside(body.p, reach);
        }
    }
}

/// The avatar upright on the ground at wheel angle zero, facing spinward and moving with the
/// ground, so that it starts out standing rather than falling.
fn standing_avatar(shape: &BodyShape, drum: &Drum) -> Body {
    let r = standing_radius(drum.ring).0 as f64;
    let p = [r, 0.0, 0.0];
    let up = [-1.0, 0.0, 0.0];
    let ground_velocity = drum.wall_velocity(p);
    let forward = if ground_velocity[2] < 0.0 {
        [0.0, 0.0, -1.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let back = [-forward[0], -forward[1], -forward[2]];
    let right = [
        up[1] * back[2] - up[2] * back[1],
        up[2] * back[0] - up[0] * back[2],
        up[0] * back[1] - up[1] * back[0],
    ];
    let q = quat_from_basis(&right, &up, &back);
    Body::new(
        AVATAR_SHAPE,
        shape,
        p,
        q,
        ground_velocity,
        drum.angular_velocity(),
    )
}

impl Simulation {
    /// Advance by one frame of real time, or by queued time if any was requested. The water's
    /// coupling from the last frame lands on the bodies first.
    pub fn advance(&mut self, real: Seconds, fluid: &mut Fluid) {
        let max_dt = SUBSTEP_RATE.period().0;
        self.substeps.clear();
        if fluid.outstanding() > MAX_FRAMES_AHEAD {
            return;
        }
        let (steps, dt) = if self.queued > 0.0 {
            let steps = ((self.queued / max_dt).round() as usize).min(MAX_SUBSTEPS_PER_FRAME);
            self.queued = (self.queued - steps as f32 * max_dt).max(0.0);
            if self.queued < max_dt * 0.5 {
                self.queued = 0.0;
            }
            (steps, max_dt)
        } else {
            self.accumulator = (self.accumulator + real.0).min(MAX_FRAME_TIME.0);
            if self.accumulator < MIN_SUBSTEP.0 {
                (0, max_dt)
            } else {
                let steps =
                    ((self.accumulator / max_dt).ceil() as usize).clamp(1, MAX_SUBSTEPS_PER_FRAME);
                let dt = (self.accumulator / steps as f32).min(max_dt);
                self.accumulator = 0.0;
                (steps, dt)
            }
        };
        let coupling = if steps > 0 {
            fluid.take_coupling()
        } else {
            None
        };
        self.clamp_speeds();
        let water_step = max_dt as f64;
        for k in 0..steps {
            self.drum.advance(dt as f64);
            avatar::drive(
                &mut self.bodies[0],
                &mut self.thrusters,
                &mut self.gyros,
                &self.avatar_input,
                &self.drum,
                dt as f64,
            );
            let water = if k == 0 { coupling.as_deref() } else { None };
            rigid::step(
                dt as f64,
                &self.drum,
                &self.shapes,
                &mut self.bodies,
                water,
                &self.body_params,
            );
            self.time.0 += dt;
            self.water_due += dt as f64;
            if self.water_due >= water_step * (1.0 - 1e-6) {
                self.water_due = (self.water_due - water_step).max(0.0);
                self.substeps.push(SubstepRecord {
                    dt: Seconds(max_dt),
                    spin: self.drum.spin,
                    angle: self.drum.angle,
                    bodies: fluid.pack(&self.bodies, &self.shapes),
                });
            }
        }
        self.window.0 += real.0;
        self.window.1 += steps as f32 * dt;
        if self.window.0 >= RATE_WINDOW.0 {
            self.rate = (self.window.1 / self.window.0).min(1.0);
            self.window = (0.0, 0.0);
        }
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

    /// Back to the initial ring world of the same size, but the settings, the target spin, the
    /// thrusters' power and the avatar stay as they are.
    pub fn reset(&mut self) {
        let params = self.params.clone();
        let body_params = self.body_params.clone();
        let raft_shape = self.shapes[RAFT_SHAPE].friction;
        let target = self.drum.target_spin;
        let power = self.thrusters.power;
        let gyros = self.gyros;
        let avatar = std::mem::take(&mut self.bodies).swap_remove(0);
        *self = Simulation::new(self.drum.ring);
        self.params = params;
        self.body_params = body_params;
        self.shapes[RAFT_SHAPE].friction = raft_shape;
        self.drum.target_spin = target;
        self.thrusters.power = power;
        self.gyros = gyros;
        self.bodies[0] = avatar;
    }

    pub fn set_raft_friction(&mut self, friction: f64) {
        self.shapes[RAFT_SHAPE].friction = friction;
    }

    pub fn shapes(&self) -> &[BodyShape] {
        &self.shapes
    }

    pub fn avatar(&self) -> &Body {
        &self.bodies[0]
    }

    /// What the avatar's feet feel right now.
    pub fn footing(&self) -> Footing {
        let avatar = self.avatar();
        match avatar.ground {
            Some(ground) => {
                let wall = self.drum.wall_velocity(ground.point);
                let feet = avatar.point_velocity(&ground.point);
                Footing {
                    weight: (ground.support.0 * avatar.inv_m / EARTH_GRAVITY.0) as f32,
                    ground_speed: norm(&[feet[0] - wall[0], feet[1] - wall[1], feet[2] - wall[2]])
                        as f32,
                    airborne: false,
                }
            }
            None => {
                let air = self.drum.air_velocity(avatar.p).unwrap_or([0.0; 3]);
                Footing {
                    weight: 0.0,
                    ground_speed: norm(&[
                        avatar.v[0] - air[0],
                        avatar.v[1] - air[1],
                        avatar.v[2] - air[2],
                    ]) as f32,
                    airborne: avatar.solid,
                }
            }
        }
    }

    pub fn avatar_mut(&mut self) -> &mut Body {
        &mut self.bodies[0]
    }

    pub fn rafts(&self) -> &[Body] {
        &self.bodies[1..]
    }

    pub fn rafts_mut(&mut self) -> &mut [Body] {
        &mut self.bodies[1..]
    }

    pub fn clear_rafts(&mut self) {
        self.bodies.truncate(1);
    }

    /// Add up to `count` particles around a point inside the drum, moving with the glass.
    pub fn inject(&self, fluid: &mut Fluid, centre: [f32; 3], count: u32) -> u32 {
        let drum = &self.drum;
        fluid.inject(
            centre,
            |p| {
                drum.wall_velocity([p[0] as f64, p[1] as f64, p[2] as f64])
                    .map(|v| v as f32)
            },
            count,
            |p| drum.place_inside(p),
        )
    }

    /// Spawn a raft centred at `p`, lying flat against a surface with inward normal `n` and
    /// moving with the glass.
    pub fn spawn_raft(&mut self, p: [f64; 3], n: [f64; 3]) -> bool {
        if self.rafts().len() >= MAX_RAFTS {
            return false;
        }
        let q = basis_from_normal(&n);
        let (v, w) = (self.drum.wall_velocity(p), self.drum.angular_velocity());
        self.bodies
            .push(Body::new(RAFT_SHAPE, &self.shapes[RAFT_SHAPE], p, q, v, w));
        true
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
) {
    if !ready.get() {
        return;
    }
    sim.advance(Seconds(time.delta_secs()), &mut fluid);
    let substeps: Vec<(Seconds, Bodies)> = sim
        .substeps
        .iter()
        .map(|r| (r.dt, r.bodies.clone()))
        .collect();
    *frame = fluid.frame(&sim.params, &substeps);
}
