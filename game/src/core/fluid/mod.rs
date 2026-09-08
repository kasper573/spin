//! Position-based fluid (Macklin & Müller 2013) with Akinci-style coupling to rigid boxes, confined
//! by a [`Vessel`]. SI units throughout; the particle spacing fixes the kernel and the rest mass.
use std::f32::consts::PI;

use crate::core::math::Rng;
use crate::core::rigid::{Body, BoxShape};
use crate::core::units::{Hertz, Seconds};
use crate::core::vessel::{Contact, Vessel};

mod grid;
mod solver;

use grid::Grid;

/// Rest spacing between particles (m); every kernel constant derives from it.
pub const PARTICLE_SPACING: f32 = 0.10;
pub const H: f32 = 2.0 * PARTICLE_SPACING;
pub const H2: f32 = H * H;
pub const REST_DENSITY: f32 = 1000.0;
pub const PARTICLE_MASS: f32 =
    REST_DENSITY * PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING;
pub const MAX_PARTICLES: usize = 16000;
pub const MAX_BODIES: usize = 12;

const MAX_NEIGHBOURS: usize = 56;
const MAX_BOUNDARY_NEIGHBOURS: usize = 48;
const MAX_BOUNDARY_SAMPLES: usize = 1200;
pub const POLY6: f32 = 315.0 / (64.0 * PI * H * H * H * H * H * H * H * H * H);
const SPIKY: f32 = -45.0 / (PI * H * H * H * H * H * H);
const W0: f32 = POLY6 * H2 * H2 * H2;
const EPS_LAMBDA: f32 = 0.02;
const ITERATIONS: usize = 3;
const SCORR_K: f32 = 0.001;
const SCORR_DQ: f32 = 0.3 * H;
const SCORR_WQ: f32 =
    POLY6 * (H2 - SCORR_DQ * SCORR_DQ) * (H2 - SCORR_DQ * SCORR_DQ) * (H2 - SCORR_DQ * SCORR_DQ);
pub const MAX_SPEED: f32 = 15.0;
pub const MAX_SPEED_BODY: f64 = 15.0;
const MAX_DELTA: f32 = 0.5 * PARTICLE_SPACING;
const WET_REF: f32 = 300.0;

#[derive(Clone, Debug, PartialEq)]
pub struct FluidParams {
    pub viscosity: f32,
    pub wall_friction: f32,
    pub body_friction: f64,
    pub restitution: f64,
    pub body_drag: f32,
    pub air: bool,
    /// Time constant for the vessel's air to drag free objects along with its walls.
    pub air_tau: Seconds,
    /// Rate at which a fully wetted body's spin relaxes toward the vessel's rotation.
    pub wet_spin_rate: Hertz,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            viscosity: 0.15,
            wall_friction: 0.5,
            body_friction: 0.45,
            restitution: 0.2,
            body_drag: 0.5,
            air: true,
            air_tau: Seconds(12.0),
            wet_spin_rate: Hertz(20.0),
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

/// Structure-of-arrays particle store plus the boundary samples of the bodies it couples to.
pub struct Fluid {
    n: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    lam: Vec<f32>,
    dx: Vec<f32>,
    dy: Vec<f32>,
    dz: Vec<f32>,
    foam: Vec<f32>,
    contact: Vec<Contact>,
    nb: Vec<u32>,
    nbc: Vec<u16>,
    bnb: Vec<u32>,
    bnbc: Vec<u16>,
    grid: Grid,
    boundary: Boundary,
    rng: Rng,
    scratch: Vec<f32>,
}

impl Fluid {
    /// A fluid whose particles stay within the given box (the vessel's bounds).
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        let f = || vec![0f32; MAX_PARTICLES];
        Fluid {
            n: 0,
            x: f(),
            y: f(),
            z: f(),
            vx: f(),
            vy: f(),
            vz: f(),
            px: f(),
            py: f(),
            pz: f(),
            lam: f(),
            dx: f(),
            dy: f(),
            dz: f(),
            foam: f(),
            contact: vec![Contact::default(); MAX_PARTICLES],
            nb: vec![0; MAX_PARTICLES * MAX_NEIGHBOURS],
            nbc: vec![0; MAX_PARTICLES],
            bnb: vec![0; MAX_PARTICLES * MAX_BOUNDARY_NEIGHBOURS],
            bnbc: vec![0; MAX_PARTICLES],
            grid: Grid::new(MAX_PARTICLES, min, max),
            boundary: Boundary::new(min, max),
            rng: Rng::new(0x9E3779B97F4A7C15),
            scratch: f(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn is_full(&self) -> bool {
        self.n >= MAX_PARTICLES
    }

    pub fn add(&mut self, p: Particle) -> bool {
        if self.is_full() {
            return false;
        }
        let i = self.n;
        self.n += 1;
        [self.x[i], self.y[i], self.z[i]] = p.position;
        [self.vx[i], self.vy[i], self.vz[i]] = p.velocity;
        self.contact[i] = Contact::default();
        self.foam[i] = p.foam;
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
        self.n = 0;
    }

    pub fn particle(&self, i: usize) -> Particle {
        Particle {
            position: [self.x[i], self.y[i], self.z[i]],
            velocity: [self.vx[i], self.vy[i], self.vz[i]],
            foam: self.foam[i],
        }
    }

    pub fn particles(&self) -> impl Iterator<Item = Particle> + '_ {
        (0..self.n).map(|i| self.particle(i))
    }

    /// Lay particles out in grid order so neighbours are neighbours in memory too.
    fn sort_by_cell(&mut self) {
        let n = self.n;
        let order = &self.grid.entries[..n];
        let scratch = &mut self.scratch;
        for array in [
            &mut self.x,
            &mut self.y,
            &mut self.z,
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
            &mut self.px,
            &mut self.py,
            &mut self.pz,
            &mut self.foam,
        ] {
            for (k, &i) in order.iter().enumerate() {
                scratch[k] = array[i as usize];
            }
            array[..n].copy_from_slice(&scratch[..n]);
        }
        let contacts: Vec<Contact> = order.iter().map(|&i| self.contact[i as usize]).collect();
        self.contact[..n].copy_from_slice(&contacts);
        for (k, entry) in self.grid.entries[..n].iter_mut().enumerate() {
            *entry = k as u32;
        }
    }

    /// Advance fluid and bodies together by one substep inside the vessel.
    pub fn step(
        &mut self,
        dt: f32,
        vessel: &impl Vessel,
        shape: &BoxShape,
        bodies: &mut [Body],
        params: &FluidParams,
    ) {
        solver::step(self, dt, vessel, shape, bodies, params);
    }
}

/// Boundary particles sampled from every body each substep (Akinci-style coupling).
struct Boundary {
    n: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    psi: Vec<f32>,
    body: Vec<u32>,
    rho: Vec<f32>,
    fvx: Vec<f32>,
    fvy: Vec<f32>,
    fvz: Vec<f32>,
    fx: Vec<f32>,
    fz: Vec<f32>,
    grid: Grid,
}

impl Boundary {
    fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        let f = || vec![0f32; MAX_BOUNDARY_SAMPLES];
        Boundary {
            n: 0,
            x: f(),
            y: f(),
            z: f(),
            vx: f(),
            vy: f(),
            vz: f(),
            psi: f(),
            body: vec![0; MAX_BOUNDARY_SAMPLES],
            rho: f(),
            fvx: f(),
            fvy: f(),
            fvz: f(),
            fx: f(),
            fz: f(),
            grid: Grid::new(MAX_BOUNDARY_SAMPLES, min, max),
        }
    }
}
