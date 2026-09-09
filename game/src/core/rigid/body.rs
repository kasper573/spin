use crate::core::math::{Quatd, Vec3d, add_scaled, cross, mat3mul, norm, quat_integrate};
use crate::core::units::{MetresPerSecond, Newtons, RadiansPerSecond};

/// What the water did to a body over the substeps of a frame: the buoyancy impulse and torque,
/// and the flow around the hull weighted by how strongly each wetted sample coupled to it, as a
/// fraction of the body's mass, summed over every substep. The flow sums let the body be relaxed
/// toward the water it is actually in rather than by a difference against a stale copy of its
/// own velocity, so the coupling stays stable whenever the readback lands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterCoupling {
    /// The velocity the buoyancy gave the body.
    pub buoyancy: Vec3d,
    /// The buoyancy's torque impulse per unit of the body's mass.
    pub buoyancy_torque: Vec3d,
    /// Σ k·v of the water around the hull.
    pub flow: Vec3d,
    /// Σ k: the coupled water's mass over the body's mass, summed over the substeps.
    pub coupling: f64,
    /// The wetted fraction of the hull, summed over the substeps.
    pub wet: f64,
    /// Simulated time the sums cover, and how many substeps it was taken in.
    pub seconds: f64,
    pub substeps: f64,
}

impl WaterCoupling {
    pub fn add(&mut self, other: &WaterCoupling) {
        for k in 0..3 {
            self.buoyancy[k] += other.buoyancy[k];
            self.buoyancy_torque[k] += other.buoyancy_torque[k];
            self.flow[k] += other.flow[k];
        }
        self.coupling += other.coupling;
        self.wet += other.wet;
        self.seconds += other.seconds;
        self.substeps += other.substeps;
    }

    /// The share `f` of the sums, covering that share of their time.
    pub fn scaled(&self, f: f64) -> WaterCoupling {
        WaterCoupling {
            buoyancy: self.buoyancy.map(|x| x * f),
            buoyancy_torque: self.buoyancy_torque.map(|x| x * f),
            flow: self.flow.map(|x| x * f),
            coupling: self.coupling * f,
            wet: self.wet * f,
            seconds: self.seconds * f,
            substeps: self.substeps * f,
        }
    }

    /// The length of the substeps the sums were taken in.
    fn substep(&self) -> f64 {
        self.seconds / self.substeps
    }

    /// The velocity of the water around the hull.
    fn flow_velocity(&self) -> Vec3d {
        self.flow.map(|f| f / self.coupling)
    }
}

const DEFAULT_FRICTION: f64 = 0.45;
const DEFAULT_RESTITUTION: f64 = 0.2;
/// The water's drag: a body relaxes toward the flow around it on this time scale divided by the
/// coupled water's mass over its own, so a hull gripped by its own mass of water is carried
/// along in a couple of seconds.
const DRAG_TAU: f64 = 2.5;

/// The wall a body stood on during the last substep: where it touched, which way is up there,
/// and how hard the wall pushed back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground {
    pub point: Vec3d,
    /// Unit normal pointing from the wall into the vessel.
    pub normal: Vec3d,
    pub support: Newtons,
}

/// A sphere of a hull, local to the centre of mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullSphere {
    pub centre: Vec3d,
    pub radius: f64,
}

/// What a body is made of, for contacts: spheres, the first of which is its bulk, whose centre
/// may sit away from the centre of mass so that whatever pushes on the hull, the ground, the
/// water or the air, turns it until the ballast hangs below. Any further spheres, such as a
/// head above the bulk, only collide. Friction acts through the centre of mass rather than at
/// the hull, so the spheres slide instead of rolling.
#[derive(Clone, Debug, PartialEq)]
pub struct Hull {
    pub spheres: Vec<HullSphere>,
}

impl Hull {
    /// The sphere the water and the air push on.
    pub fn bulk(&self) -> HullSphere {
        self.spheres[0]
    }

    /// Where the air pushes on the hull: its bulk's centre.
    pub fn centre_of_pressure(&self) -> Vec3d {
        self.bulk().centre
    }
}

/// Mass properties, material, fluid boundary samples and contact points shared by every body of
/// one kind.
pub struct BodyShape {
    pub hull: Hull,
    pub mass: f64,
    pub inv_inertia: Vec3d,
    /// Coulomb friction and bounciness against walls and other bodies.
    pub friction: f64,
    pub restitution: f64,
    /// Boundary sample points for fluid coupling.
    pub samples: Vec<[f32; 3]>,
    pub volume_per_sample: f64,
}

impl BodyShape {
    /// A sphere with its centre of mass `drop` below its centre and boundary samples on its
    /// surface.
    pub fn weighted_sphere(radius: f64, mass: f64, drop: f64, sample_count: usize) -> Self {
        let centre = [0.0, drop, 0.0];
        let about_axis = 0.4 * mass * radius * radius;
        let inv_inertia = [
            1.0 / (about_axis + mass * drop * drop),
            1.0 / about_axis,
            1.0 / (about_axis + mass * drop * drop),
        ];
        let samples = sphere_samples(radius, centre, sample_count);
        let volume = 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
        BodyShape {
            hull: Hull {
                spheres: vec![HullSphere { centre, radius }],
            },
            mass,
            inv_inertia,
            friction: DEFAULT_FRICTION,
            restitution: DEFAULT_RESTITUTION,
            volume_per_sample: volume / samples.len() as f64,
            samples,
        }
    }

    /// The same shape displacing this much water (m³) instead of its whole hull.
    pub fn displacing(mut self, volume: f64) -> BodyShape {
        self.volume_per_sample = volume / self.samples.len() as f64;
        self
    }

    /// With another sphere that collides, local to the centre of mass.
    pub fn with_sphere(mut self, centre: Vec3d, radius: f64) -> BodyShape {
        self.hull.spheres.push(HullSphere { centre, radius });
        self
    }

    /// Distance from the centre of mass to the farthest point of the hull.
    pub fn reach(&self) -> f64 {
        self.hull
            .spheres
            .iter()
            .map(|sphere| sphere.radius + norm(&sphere.centre))
            .fold(0.0, f64::max)
    }
}

pub struct Body {
    /// Index into the shapes the simulation stepped this body with.
    pub shape: usize,
    /// Solid bodies contact walls, other bodies and water; the rest only feel the air.
    pub solid: bool,
    pub inv_m: f64,
    inv_i: Vec3d,
    /// Centre of mass.
    pub p: Vec3d,
    pub q: Quatd,
    pub v: Vec3d,
    pub w: Vec3d,
    /// Rotation matrix (row-major) and world-space inverse inertia.
    pub m: [f64; 9],
    pub iw: [f64; 9],
    /// Fraction of boundary samples in water, in [0, 1].
    pub wet: f64,
    /// The vessel wall the body stood on during the last substep, if any.
    pub ground: Option<Ground>,
}

impl Body {
    pub fn new(shape: usize, of: &BodyShape, p: Vec3d, q: Quatd, v: Vec3d, w: Vec3d) -> Self {
        let inv_i = of.inv_inertia;
        let mut body = Body {
            shape,
            solid: true,
            inv_m: 1.0 / of.mass,
            inv_i,
            p,
            q,
            v,
            w,
            m: [0.0; 9],
            iw: [0.0; 9],
            wet: 0.0,
            ground: None,
        };
        body.update_rotation();
        body
    }

    /// Move the body somewhere else, at rest.
    pub fn place(&mut self, p: Vec3d, q: Quatd) {
        self.p = p;
        self.q = q;
        self.v = [0.0; 3];
        self.w = [0.0; 3];
        self.update_rotation();
    }

    #[inline]
    pub fn to_world(&self, l: &Vec3d) -> Vec3d {
        let m = &self.m;
        [
            m[0] * l[0] + m[1] * l[1] + m[2] * l[2] + self.p[0],
            m[3] * l[0] + m[4] * l[1] + m[5] * l[2] + self.p[1],
            m[6] * l[0] + m[7] * l[1] + m[8] * l[2] + self.p[2],
        ]
    }

    #[inline]
    pub fn to_local(&self, wp: &Vec3d) -> Vec3d {
        let m = &self.m;
        let r = [wp[0] - self.p[0], wp[1] - self.p[1], wp[2] - self.p[2]];
        [
            m[0] * r[0] + m[3] * r[1] + m[6] * r[2],
            m[1] * r[0] + m[4] * r[1] + m[7] * r[2],
            m[2] * r[0] + m[5] * r[1] + m[8] * r[2],
        ]
    }

    /// A local direction in world space.
    #[inline]
    pub fn rotate(&self, l: &Vec3d) -> Vec3d {
        mat3mul(&self.m, l)
    }

    #[inline]
    pub fn point_velocity(&self, wp: &Vec3d) -> Vec3d {
        let r = [wp[0] - self.p[0], wp[1] - self.p[1], wp[2] - self.p[2]];
        let c = cross(&self.w, &r);
        [c[0] + self.v[0], c[1] + self.v[1], c[2] + self.v[2]]
    }

    pub fn apply_impulse(&mut self, j: &Vec3d, at: &Vec3d) {
        add_scaled(&mut self.v, j, self.inv_m);
        let r = [at[0] - self.p[0], at[1] - self.p[1], at[2] - self.p[2]];
        add_scaled(&mut self.w, &mat3mul(&self.iw, &cross(&r, j)), 1.0);
    }

    /// An impulse through the centre of mass: no torque.
    pub fn apply_central_impulse(&mut self, j: &Vec3d) {
        add_scaled(&mut self.v, j, self.inv_m);
    }

    /// A torque impulse about the centre of mass.
    pub fn apply_angular_impulse(&mut self, l: &Vec3d) {
        add_scaled(&mut self.w, &mat3mul(&self.iw, l), 1.0);
    }

    /// Take what the water did over the time its coupling covers: the drag relaxes the body
    /// toward the flow at the rate its coupling sets, the buoyancy is limited to `max_accel` of
    /// acceleration. The water never spins the body: the hull slides through it.
    pub fn couple(&mut self, water: &WaterCoupling, max_accel: f64) {
        if water.substeps <= 0.0 || water.seconds <= 0.0 {
            return;
        }
        self.wet = (water.wet / water.substeps).clamp(0.0, 1.0);
        if water.coupling > 0.0 {
            let dt = water.substep();
            let flow = water.flow_velocity();
            let s_lin = 1.0 - (-water.coupling * dt / DRAG_TAU).exp();
            for (v, f) in self.v.iter_mut().zip(flow) {
                *v += (f - *v) * s_lin;
            }
        }

        let cap = max_accel * water.seconds;
        let mut dv = water.buoyancy;
        let mut l = water.buoyancy_torque;
        let dvm = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
        if dvm > cap {
            let s = cap / dvm;
            for k in 0..3 {
                dv[k] *= s;
                l[k] *= s;
            }
        }
        add_scaled(&mut self.v, &dv, 1.0);
        if self.inv_m > 0.0 {
            add_scaled(&mut self.w, &mat3mul(&self.iw, &l), 1.0 / self.inv_m);
        }
    }

    /// Move by a substep, with the speed and spin held under the safety clamps.
    pub fn integrate(&mut self, dt: f64, max_speed: MetresPerSecond, max_spin: RadiansPerSecond) {
        let sp = (self.v[0] * self.v[0] + self.v[1] * self.v[1] + self.v[2] * self.v[2]).sqrt();
        if sp > max_speed.0 as f64 {
            let s = max_speed.0 as f64 / sp;
            for k in 0..3 {
                self.v[k] *= s;
            }
        }
        let ws = (self.w[0] * self.w[0] + self.w[1] * self.w[1] + self.w[2] * self.w[2]).sqrt();
        if ws > max_spin.0 as f64 {
            let s = max_spin.0 as f64 / ws;
            for k in 0..3 {
                self.w[k] *= s;
            }
        }
        let v = self.v;
        add_scaled(&mut self.p, &v, dt);
        self.q = quat_integrate(&self.q, &self.w, dt);
        self.update_rotation();
    }

    fn update_rotation(&mut self) {
        let [x, y, z, w] = self.q;
        let m = &mut self.m;
        m[0] = 1.0 - 2.0 * (y * y + z * z);
        m[1] = 2.0 * (x * y - z * w);
        m[2] = 2.0 * (x * z + y * w);
        m[3] = 2.0 * (x * y + z * w);
        m[4] = 1.0 - 2.0 * (x * x + z * z);
        m[5] = 2.0 * (y * z - x * w);
        m[6] = 2.0 * (x * z - y * w);
        m[7] = 2.0 * (y * z + x * w);
        m[8] = 1.0 - 2.0 * (x * x + y * y);
        let i = self.inv_i;
        for a in 0..3 {
            for b in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += m[a * 3 + k] * i[k] * m[b * 3 + k];
                }
                self.iw[a * 3 + b] = s;
            }
        }
    }
}

/// Points spread evenly over a sphere's surface (a Fibonacci lattice).
fn sphere_samples(radius: f64, centre: Vec3d, count: usize) -> Vec<[f32; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let r = (1.0 - y * y).sqrt();
            let a = golden * i as f64;
            [
                (centre[0] + radius * r * a.cos()) as f32,
                (centre[1] + radius * y) as f32,
                (centre[2] + radius * r * a.sin()) as f32,
            ]
        })
        .collect()
}
