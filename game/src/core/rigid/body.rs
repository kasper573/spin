use crate::core::fluid::{H2, MAX_SPEED_BODY, POLY6, REST_DENSITY};
use crate::core::math::{Quatd, Vec3d, add_scaled, cross, mat3mul};

/// A thin box every body shares: mass properties, boundary samples on its mid-plane and the
/// contact points on its surface.
pub struct BoxShape {
    pub half: Vec3d,
    pub mass: f64,
    pub inv_inertia: Vec3d,
    /// Boundary sample points [x, y, z, Ψ] for fluid coupling.
    pub samples: Vec<[f32; 4]>,
    pub volume_per_sample: f64,
    /// Contact sample points: corners, edge midpoints and face centres.
    pub points: Vec<Vec3d>,
}

impl BoxShape {
    pub fn new(size: Vec3d, density: f64, sample_spacing: f64) -> Self {
        let [a, b, c] = size;
        let mass = density * a * b * c;
        let inv_inertia = [
            12.0 / (mass * (b * b + c * c)),
            12.0 / (mass * (a * a + c * c)),
            12.0 / (mass * (a * a + b * b)),
        ];
        let samples = mid_plane_samples(a, c, sample_spacing);
        let volume_per_sample = a * b * c / samples.len() as f64;
        BoxShape {
            half: [a / 2.0, b / 2.0, c / 2.0],
            mass,
            inv_inertia,
            samples,
            volume_per_sample,
            points: contact_points([a / 2.0, b / 2.0, c / 2.0]),
        }
    }

    /// Distance from the centre to the farthest corner.
    pub fn reach(&self) -> f64 {
        let [x, y, z] = self.half;
        (x * x + y * y + z * z).sqrt()
    }
}

pub struct Body {
    pub half: Vec3d,
    pub inv_m: f64,
    inv_i: Vec3d,
    inv_i_max: f64,
    pub p: Vec3d,
    pub q: Quatd,
    pub v: Vec3d,
    pub w: Vec3d,
    /// Rotation matrix (row-major) and world-space inverse inertia.
    pub m: [f64; 9],
    pub iw: [f64; 9],
    acc_j: Vec3d,
    acc_l: Vec3d,
    drag_j: Vec3d,
    drag_l: Vec3d,
    drag_k: f64,
    drag_ka: f64,
    pub submerged: u32,
    /// Fraction of boundary samples in water, in [0, 1].
    pub wet: f64,
}

impl Body {
    pub fn new(shape: &BoxShape, p: Vec3d, q: Quatd, v: Vec3d, w: Vec3d) -> Self {
        let inv_i = shape.inv_inertia;
        let mut body = Body {
            half: shape.half,
            inv_m: 1.0 / shape.mass,
            inv_i,
            inv_i_max: inv_i[0].max(inv_i[1]).max(inv_i[2]),
            p,
            q,
            v,
            w,
            m: [0.0; 9],
            iw: [0.0; 9],
            acc_j: [0.0; 3],
            acc_l: [0.0; 3],
            drag_j: [0.0; 3],
            drag_l: [0.0; 3],
            drag_k: 0.0,
            drag_ka: 0.0,
            submerged: 0,
            wet: 0.0,
        };
        body.update_rotation();
        body
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

    /// Accumulate an impulse applied at a world point, to be applied once per substep.
    #[inline]
    pub fn accumulate(&mut self, j: Vec3d, at: Vec3d) {
        add_scaled(&mut self.acc_j, &j, 1.0);
        let r = [at[0] - self.p[0], at[1] - self.p[1], at[2] - self.p[2]];
        add_scaled(&mut self.acc_l, &cross(&r, &j), 1.0);
    }

    /// Accumulate a drag impulse from one particle–sample pair; `mw` is the pair's coupling mass,
    /// used to measure how far the summed drag would relax this body in a single substep.
    #[inline]
    pub fn accumulate_drag(&mut self, j: Vec3d, at: Vec3d, mw: f64) {
        add_scaled(&mut self.drag_j, &j, 1.0);
        let r = [at[0] - self.p[0], at[1] - self.p[1], at[2] - self.p[2]];
        add_scaled(&mut self.drag_l, &cross(&r, &j), 1.0);
        self.drag_k += mw * self.inv_m;
        self.drag_ka += mw * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * self.inv_i_max;
    }

    /// Apply the accumulated water impulses; drag relaxes the body at most once per substep and
    /// the rest is limited to `max_dv` of velocity change.
    pub fn apply_accumulated(&mut self, max_dv: f64) {
        let s_lin = if self.drag_k > 1.0 {
            1.0 / self.drag_k
        } else {
            1.0
        };
        let s_ang = if self.drag_ka > 1.0 {
            1.0 / self.drag_ka
        } else {
            1.0
        };
        let drag_j = self.drag_j;
        add_scaled(&mut self.v, &drag_j, self.inv_m * s_lin);
        add_scaled(&mut self.w, &mat3mul(&self.iw, &self.drag_l), s_ang);
        self.drag_j = [0.0; 3];
        self.drag_l = [0.0; 3];
        self.drag_k = 0.0;
        self.drag_ka = 0.0;

        let cap = max_dv / self.inv_m;
        let jm = (self.acc_j[0] * self.acc_j[0]
            + self.acc_j[1] * self.acc_j[1]
            + self.acc_j[2] * self.acc_j[2])
            .sqrt();
        if jm > cap {
            let s = cap / jm;
            for k in 0..3 {
                self.acc_j[k] *= s;
                self.acc_l[k] *= s;
            }
        }
        let acc_j = self.acc_j;
        add_scaled(&mut self.v, &acc_j, self.inv_m);
        add_scaled(&mut self.w, &mat3mul(&self.iw, &self.acc_l), 1.0);
        self.acc_j = [0.0; 3];
        self.acc_l = [0.0; 3];
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

/// Boundary samples on the box's mid-plane with Akinci volume weights Ψ.
fn mid_plane_samples(len_x: f64, len_z: f64, spacing: f64) -> Vec<[f32; 4]> {
    let kx = (len_x / spacing).round() as usize;
    let kz = (len_z / spacing).round() as usize;
    let mut pts: Vec<[f32; 4]> = Vec::new();
    for i in 0..=kx {
        for j in 0..=kz {
            pts.push([
                (-len_x / 2.0 + i as f64 * spacing) as f32,
                0.0,
                (-len_z / 2.0 + j as f64 * spacing) as f32,
                0.0,
            ]);
        }
    }
    let copy = pts.clone();
    for p in pts.iter_mut() {
        let mut s = 0.0f32;
        for q in &copy {
            let dx = p[0] - q[0];
            let dz = p[2] - q[2];
            let r2 = dx * dx + dz * dz;
            if r2 < H2 {
                let t = H2 - r2;
                s += POLY6 * t * t * t;
            }
        }
        p[3] = REST_DENSITY / s;
    }
    pts
}

fn contact_points([hx, hy, hz]: Vec3d) -> Vec<Vec3d> {
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
