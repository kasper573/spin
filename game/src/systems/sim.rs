//! The running simulation: the drum, the shuttle and the rafts are stepped on the CPU and the
//! water follows on the GPU with the same substeps. Real time takes one substep per frame, no
//! longer than a sixtieth of a second, so a fast display gets smooth water and a slow frame is
//! split into a few substeps; beyond that, time stretches rather than the frame. Requested time
//! (the tests, the bench, scripts) always steps at exactly sixty hertz.
use bevy::prelude::*;

use crate::core::fluid::{
    Bodies, Fluid, FluidFrame, FluidParams, FluidReady, MAX_BODIES, MAX_SUBSTEPS_PER_FRAME,
    PARTICLE_MASS, REST_DENSITY,
};
use crate::core::math::basis_from_normal;
use crate::core::rigid::{self, Body, BodyParams, BodyShape};
use crate::core::shuttle::{self, ShuttleInput};
use crate::core::units::{Hertz, Litres, Radians, RadiansPerSecond, Seconds};
use crate::core::vessel::Vessel;
use crate::systems::drum::Drum;
use crate::systems::player;
use crate::systems::rafts;

pub const SUBSTEP_RATE: Hertz = Hertz(60.0);
pub const MAX_RAFTS: usize = MAX_BODIES - 1;
/// Shortest substep real time is split into; faster frames are gathered into one.
const MIN_SUBSTEP: Seconds = Seconds(1.0 / 240.0);
const MAX_FRAME_TIME: Seconds = Seconds(0.1);
/// How many frames the bodies may run ahead of the water's coupling before the simulation waits
/// for the GPU: a coupling this stale is still a fraction of a degree of the drum's turn, while
/// an unbounded backlog would let the bodies chase water that has long moved on.
const MAX_FRAMES_AHEAD: u32 = 3;
const RATE_WINDOW: Seconds = Seconds(1.0);
const SHUTTLE_SHAPE: usize = 0;
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

/// One substep the CPU took, for the GPU to take too.
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
    /// What the pilot asks of the shuttle during the coming substeps.
    pub shuttle_input: ShuttleInput,
    pub params: FluidParams,
    pub body_params: BodyParams,
    /// Simulated time since the last reset.
    pub time: Seconds,
    /// Fraction of real time the simulation keeps up with (1 = full speed).
    pub rate: f32,
    /// The substeps of the last frame.
    pub substeps: Vec<SubstepRecord>,
    /// The shuttle first, then the rafts.
    bodies: Vec<Body>,
    shapes: [BodyShape; 2],
    accumulator: f32,
    queued: f32,
    window: (f32, f32),
}

impl Default for Simulation {
    fn default() -> Self {
        let shapes = [shuttle::shape(), rafts::shape()];
        let mut shuttle = Body::new(
            SHUTTLE_SHAPE,
            &shapes[SHUTTLE_SHAPE],
            shuttle::centre_for_eye(player::START_EYE),
            [0.0, 0.0, 0.0, 1.0],
            [0.0; 3],
            [0.0; 3],
        );
        shuttle.solid = false;
        Simulation {
            drum: Drum::default(),
            shuttle_input: ShuttleInput::default(),
            params: FluidParams::default(),
            body_params: BodyParams::default(),
            time: Seconds(0.0),
            rate: 1.0,
            substeps: Vec::new(),
            bodies: vec![shuttle],
            shapes,
            accumulator: 0.0,
            queued: 0.0,
            window: (0.0, 0.0),
        }
    }
}

impl Simulation {
    /// Advance by one frame of real time, or by queued time if any was requested. The water's
    /// coupling from the last frame lands on the bodies first.
    pub fn advance(&mut self, real: Seconds, fluid: &mut Fluid) {
        let max_dt = SUBSTEP_RATE.period().0;
        let coupling = fluid.take_coupling();
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
        for k in 0..steps {
            self.drum.advance(dt as f64);
            shuttle::drive(
                &mut self.bodies[0],
                &self.shuttle_input,
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
            self.substeps.push(SubstepRecord {
                dt: Seconds(dt),
                spin: self.drum.spin,
                angle: self.drum.angle,
                bodies: fluid.pack(&self.bodies, &self.shapes),
            });
            self.time.0 += dt;
        }
        self.window.0 += real.0;
        self.window.1 += steps as f32 * dt;
        if self.window.0 >= RATE_WINDOW.0 {
            self.rate = (self.window.1 / self.window.0).min(1.0);
            self.window = (0.0, 0.0);
        }
    }

    /// Ask for exactly this much simulated time over the coming frames, real time aside.
    pub fn request(&mut self, seconds: Seconds) {
        self.queued += seconds.0;
    }

    /// Requested time not yet simulated.
    pub fn queued(&self) -> Seconds {
        Seconds(self.queued)
    }

    /// Empty the drum and flatten the landscape; the settings and the shuttle stay as they are.
    pub fn reset(&mut self) {
        let params = self.params.clone();
        let body_params = self.body_params.clone();
        let target = self.drum.target_spin;
        let shuttle = std::mem::take(&mut self.bodies).swap_remove(0);
        *self = Simulation::default();
        self.params = params;
        self.body_params = body_params;
        self.drum.target_spin = target;
        self.bodies[0] = shuttle;
    }

    pub fn water(fluid: &Fluid) -> Litres {
        Litres(fluid.len() as f32 * PARTICLE_MASS / REST_DENSITY * 1000.0)
    }

    pub fn shapes(&self) -> &[BodyShape] {
        &self.shapes
    }

    pub fn shuttle(&self) -> &Body {
        &self.bodies[0]
    }

    pub fn shuttle_mut(&mut self) -> &mut Body {
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
