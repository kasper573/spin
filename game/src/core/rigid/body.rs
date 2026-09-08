use crate::core::fluid::{H2, MAX_SPEED_BODY, POLY6, REST_DENSITY};
use crate::core::math::{Quatd, Vec3d, add_scaled, cross, mat3mul, mat3solve};

/// What the water did to a body over a frame: the buoyancy impulse and torque, and the flow
/// around the hull weighted by how strongly each wetted sample coupled to it, as a fraction of
/// the body's mass. The flow sums let the body be relaxed toward the water it is actually in
/// rather than by a difference against a stale copy of its own velocity, so the coupling stays
/// stable whenever the readback lands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterCoupling {
    pub buoyancy: Vec3d,
    pub buoyancy_torque: Vec3d,
    /// Σ k·v of the water around the hull.
    pub flow: Vec3d,
    /// Σ k·(r × v) of the water around the hull, about the centre of mass.
    pub flow_moment: Vec3d,
    /// Σ k·r over the wetted samples.
    pub hull: Vec3d,
    /// Σ k·(xx, yy, zz, xy, yz, zx) over the wetted samples.
    pub hull_tensor: [f64; 6],
    /// Σ k: the coupling mass over the body's mass.
    pub coupling: f64,
    pub wet: f64,
}

impl WaterCoupling {
    pub fn add(&mut self, other: &WaterCoupling) {
        for k in 0..3 {
            self.buoyancy[k] += other.buoyancy[k];
            self.buoyancy_torque[k] += other.buoyancy_torque[k];
            self.flow[k] += other.flow[k];
            self.flow_moment[k] += other.flow_moment[k];
            self.hull[k] += other.hull[k];
        }
        for k in 0..6 {
            self.hull_tensor[k] += other.hull_tensor[k];
        }
        self.coupling += other.coupling;
        self.wet += other.wet;
    }

    /// The velocity of the water around the hull.
    fn flow_velocity(&self) -> Vec3d {
        self.flow.map(|f| f / self.coupling)
    }

    /// The spin that, together with `v`, would move the wetted hull with the water: solves
    /// Σ k·r × (v + ω × r) = Σ k·r × v_water for ω. The wetted samples of a floating hull lie
    /// nearly in a plane, which makes the system ill-conditioned, so it is regularised toward
    /// `prior`, the rotation the surrounding water has when nothing disturbs it.
    fn flow_spin(&self, v: &Vec3d, prior: &Vec3d) -> Option<Vec3d> {
        let t = &self.hull_tensor;
        let trace = t[0] + t[1] + t[2];
        let m = [
            2.0 * trace - t[0],
            -t[3],
            -t[5],
            -t[3],
            2.0 * trace - t[1],
            -t[4],
            -t[5],
            -t[4],
            2.0 * trace - t[2],
        ];
        let carried = cross(&self.hull, v);
        let rhs = [
            self.flow_moment[0] - carried[0] + trace * prior[0],
            self.flow_moment[1] - carried[1] + trace * prior[1],
            self.flow_moment[2] - carried[2] + trace * prior[2],
        ];
        mat3solve(&m, &rhs)
    }

    /// Σ k·|r|², the coupling's moment.
    fn coupling_moment(&self) -> f64 {
        self.hull_tensor[0] + self.hull_tensor[1] + self.hull_tensor[2]
    }
}

/// What a body is made of, for contacts. Positions are local to the body's centre of mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// A box centred on the centre of mass.
    Box { half: Vec3d },
    /// A sphere whose centre may sit away from the centre of mass, so it rights itself when
    /// resting on a surface. Friction acts through the centre of mass rather than at the hull, so
    /// the sphere slides instead of rolling.
    Sphere { radius: f64, centre: Vec3d },
}

/// Mass properties, fluid boundary samples and contact points shared by every body of one kind.
pub struct BodyShape {
    pub collider: Collider,
    pub mass: f64,
    pub inv_inertia: Vec3d,
    /// Boundary sample points [x, y, z, Ψ] for fluid coupling.
    pub samples: Vec<[f32; 4]>,
    pub volume_per_sample: f64,
    /// Contact sample points on the surface (boxes only; spheres contact analytically).
    pub points: Vec<Vec3d>,
}

impl BodyShape {
    /// A thin board: boundary samples on its mid-plane.
    pub fn board(size: Vec3d, density: f64, sample_spacing: f64) -> Self {
        let [a, b, c] = size;
        let mass = density * a * b * c;
        let inv_inertia = [
            12.0 / (mass * (b * b + c * c)),
            12.0 / (mass * (a * a + c * c)),
            12.0 / (mass * (a * a + b * b)),
        ];
        let samples = weighted(mid_plane_samples(a, c, sample_spacing));
        let volume_per_sample = a * b * c / samples.len() as f64;
        let half = [a / 2.0, b / 2.0, c / 2.0];
        BodyShape {
            collider: Collider::Box { half },
            mass,
            inv_inertia,
            samples,
            volume_per_sample,
            points: box_contact_points(half),
        }
    }

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
        let samples = weighted(sphere_samples(radius, centre, sample_count));
        let volume = 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
        BodyShape {
            collider: Collider::Sphere { radius, centre },
            mass,
            inv_inertia,
            volume_per_sample: volume / samples.len() as f64,
            samples,
            points: Vec::new(),
        }
    }

    /// Distance from the centre of mass to the farthest point of the hull.
    pub fn reach(&self) -> f64 {
        match self.collider {
            Collider::Box { half: [x, y, z] } => (x * x + y * y + z * z).sqrt(),
            Collider::Sphere { radius, centre } => {
                radius
                    + (centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2]).sqrt()
            }
        }
    }
}

pub struct Body {
    /// Index into the shapes the simulation stepped this body with.
    pub shape: usize,
    /// Solid bodies contact walls, other bodies and water; the rest only feel the air.
    pub solid: bool,
    pub inv_m: f64,
    inv_i: Vec3d,
    inv_i_max: f64,
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
}

impl Body {
    pub fn new(shape: usize, of: &BodyShape, p: Vec3d, q: Quatd, v: Vec3d, w: Vec3d) -> Self {
        let inv_i = of.inv_inertia;
        let mut body = Body {
            shape,
            solid: true,
            inv_m: 1.0 / of.mass,
            inv_i,
            inv_i_max: inv_i[0].max(inv_i[1]).max(inv_i[2]),
            p,
            q,
            v,
            w,
            m: [0.0; 9],
            iw: [0.0; 9],
            wet: 0.0,
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

    /// Take what the water did: the drag relaxes the body toward the flow by at most once over,
    /// the buoyancy is limited to `max_dv` of velocity change. `spin` is the rotation of the
    /// water at rest in the vessel.
    pub fn couple(&mut self, water: &WaterCoupling, max_dv: f64, spin: &Vec3d) {
        self.wet = water.wet.clamp(0.0, 1.0);
        if water.coupling > 0.0 {
            let flow = water.flow_velocity();
            let s_lin = water.coupling.min(1.0);
            for (v, f) in self.v.iter_mut().zip(flow) {
                *v += (f - *v) * s_lin;
            }
            if let Some(wanted) = water.flow_spin(&self.v, spin) {
                let s_ang = (water.coupling_moment() / self.inv_m * self.inv_i_max).min(1.0);
                for (w, f) in self.w.iter_mut().zip(wanted) {
                    *w += (f - *w) * s_ang;
                }
            }
        }

        let cap = max_dv / self.inv_m;
        let mut j = water.buoyancy;
        let mut l = water.buoyancy_torque;
        let jm = (j[0] * j[0] + j[1] * j[1] + j[2] * j[2]).sqrt();
        if jm > cap {
            let s = cap / jm;
            for k in 0..3 {
                j[k] *= s;
                l[k] *= s;
            }
        }
        add_scaled(&mut self.v, &j, self.inv_m);
        add_scaled(&mut self.w, &mat3mul(&self.iw, &l), 1.0);
    }

    /// The largest inverse moment of inertia about any principal axis.
    pub fn inv_inertia_max(&self) -> f64 {
        self.inv_i_max
    }

    pub fn integrate(&mut self, dt: f64) {
        let sp = (self.v[0] * self.v[0] + self.v[1] * self.v[1] + self.v[2] * self.v[2]).sqrt();
        if sp > MAX_SPEED_BODY {
            let s = MAX_SPEED_BODY / sp;
            for k in 0..3 {
                self.v[k] *= s;
            }
        }
        let ws = (self.w[0] * self.w[0] + self.w[1] * self.w[1] + self.w[2] * self.w[2]).sqrt();
        if ws > 25.0 {
            let s = 25.0 / ws;
            for k in 0..3 {
                self.w[k] *= s;
            }
        }
        let v = self.v;
        add_scaled(&mut self.p, &v, dt);
        let [qx, qy, qz, qw] = self.q;
        let w = self.w;
        self.q[0] += 0.5 * (w[0] * qw + w[1] * qz - w[2] * qy) * dt;
        self.q[1] += 0.5 * (w[1] * qw + w[2] * qx - w[0] * qz) * dt;
        self.q[2] += 0.5 * (w[2] * qw + w[0] * qy - w[1] * qx) * dt;
        self.q[3] += 0.5 * (-w[0] * qx - w[1] * qy - w[2] * qz) * dt;
        let l = (self.q[0] * self.q[0]
            + self.q[1] * self.q[1]
            + self.q[2] * self.q[2]
            + self.q[3] * self.q[3])
            .sqrt();
        for k in 0..4 {
            self.q[k] /= l;
        }
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

fn mid_plane_samples(len_x: f64, len_z: f64, spacing: f64) -> Vec<[f32; 3]> {
    let kx = (len_x / spacing).round() as usize;
    let kz = (len_z / spacing).round() as usize;
    let mut pts = Vec::new();
    for i in 0..=kx {
        for j in 0..=kz {
            pts.push([
                (-len_x / 2.0 + i as f64 * spacing) as f32,
                0.0,
                (-len_z / 2.0 + j as f64 * spacing) as f32,
            ]);
        }
    }
    pts
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

/// Akinci volume weights Ψ: each sample stands in for the rest density its neighbours don't cover.
fn weighted(points: Vec<[f32; 3]>) -> Vec<[f32; 4]> {
    points
        .iter()
        .map(|p| {
            let mut s = 0.0f32;
            for q in &points {
                let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < H2 {
                    let t = H2 - r2;
                    s += POLY6 * t * t * t;
                }
            }
            [p[0], p[1], p[2], REST_DENSITY / s]
        })
        .collect()
}

fn box_contact_points([hx, hy, hz]: Vec3d) -> Vec<Vec3d> {
    let mut pts = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                pts.push([sx * hx, sy * hy, sz * hz]);
            }
        }
    }
    for sy in [-1.0, 1.0] {
        pts.push([0.0, sy * hy, hz]);
        pts.push([0.0, sy * hy, -hz]);
        pts.push([hx, sy * hy, 0.0]);
        pts.push([-hx, sy * hy, 0.0]);
        pts.push([0.0, sy * hy, 0.0]);
    }
    pts
}
