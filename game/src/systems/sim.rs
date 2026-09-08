//! The running simulation: drum, water, the shuttle and the rafts stepped together at a fixed
//! substep, with a per-frame time budget so a heavy scene slows the simulation instead of the
//! frame rate.
use bevy::platform::time::Instant;
use bevy::prelude::*;

use crate::core::fluid::{Fluid, FluidParams, MAX_BODIES, PARTICLE_MASS, REST_DENSITY};
use crate::core::math::basis_from_normal;
use crate::core::rigid::{Body, BodyShape};
use crate::core::shuttle::{self, ShuttleInput};
use crate::core::units::{Hertz, Litres, Seconds};
use crate::core::vessel::Vessel;
use crate::systems::drum::{Drum, HALF_WIDTH, RADIUS};
use crate::systems::player;
use crate::systems::rafts;

pub const SUBSTEP_RATE: Hertz = Hertz(60.0);
pub const MAX_RAFTS: usize = MAX_BODIES - 1;
const FRAME_BUDGET: Seconds = Seconds(0.012);
const MAX_FRAME_TIME: Seconds = Seconds(0.1);
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

#[derive(Resource)]
pub struct Simulation {
    pub drum: Drum,
    pub fluid: Fluid,
    /// What the pilot asks of the shuttle during the coming substeps.
    pub shuttle_input: ShuttleInput,
    pub params: FluidParams,
    /// Simulated time since the last reset.
    pub time: Seconds,
    /// Fraction of real time the simulation keeps up with (1 = full speed).
    pub rate: f32,
    /// The shuttle first, then the rafts.
    bodies: Vec<Body>,
    shapes: [BodyShape; 2],
    accumulator: f32,
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
            fluid: Fluid::new(
                [-RADIUS, -HALF_WIDTH, -RADIUS],
                [RADIUS, HALF_WIDTH, RADIUS],
            ),
            shuttle_input: ShuttleInput::default(),
            params: FluidParams::default(),
            time: Seconds(0.0),
            rate: 1.0,
            bodies: vec![shuttle],
            shapes,
            accumulator: 0.0,
            window: (0.0, 0.0),
        }
    }
}

impl Simulation {
    pub fn substep(&mut self) {
        let dt = SUBSTEP_RATE.period().0;
        self.drum.advance(dt as f64);
        shuttle::drive(
            &mut self.bodies[0],
            &self.shuttle_input,
            &self.drum,
            dt as f64,
        );
        self.fluid
            .step(dt, &self.drum, &self.shapes, &mut self.bodies, &self.params);
        self.time.0 += dt;
    }

    /// Advance by one frame of real time, within the frame budget.
    pub fn advance(&mut self, real: Seconds) {
        let dt = SUBSTEP_RATE.period().0;
        self.accumulator = (self.accumulator + real.0).min(MAX_FRAME_TIME.0);
        let start = Instant::now();
        let mut simulated = 0.0;
        while self.accumulator >= dt {
            self.substep();
            self.accumulator -= dt;
            simulated += dt;
            if start.elapsed().as_secs_f32() > FRAME_BUDGET.0 {
                self.accumulator = 0.0;
                break;
            }
        }
        self.window.0 += real.0;
        self.window.1 += simulated;
        if self.window.0 >= RATE_WINDOW.0 {
            self.rate = (self.window.1 / self.window.0).min(1.0);
            self.window = (0.0, 0.0);
        }
    }

    /// Advance by exactly this much simulated time, however long it takes.
    pub fn advance_exact(&mut self, seconds: Seconds) {
        let steps = (seconds.0 * SUBSTEP_RATE.0).round() as u32;
        for _ in 0..steps {
            self.substep();
        }
    }

    /// Empty the drum and flatten the landscape; the settings and the shuttle stay as they are.
    pub fn reset(&mut self) {
        let params = self.params.clone();
        let target = self.drum.target_spin;
        let shuttle = std::mem::take(&mut self.bodies).swap_remove(0);
        *self = Simulation::default();
        self.params = params;
        self.drum.target_spin = target;
        self.bodies[0] = shuttle;
    }

    pub fn water(&self) -> Litres {
        Litres(self.fluid.len() as f32 * PARTICLE_MASS / REST_DENSITY * 1000.0)
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
    pub fn inject(&mut self, centre: [f32; 3], count: u32) -> u32 {
        let drum = &self.drum;
        self.fluid.inject(
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
            .add_systems(Update, step.in_set(SimSet::Step));
    }
}

fn step(mut sim: ResMut<Simulation>, time: Res<Time>) {
    sim.advance(Seconds(time.delta_secs()));
}
