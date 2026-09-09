pub type Vec3d = [f64; 3];
pub type Quatd = [f64; 4];

#[inline]
pub fn cross(a: &Vec3d, b: &Vec3d) -> Vec3d {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn dot(a: &Vec3d, b: &Vec3d) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Row-major 3x3 matrix times vector.
#[inline]
pub fn mat3mul(m: &[f64; 9], v: &Vec3d) -> Vec3d {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// a += b * s
#[inline]
pub fn add_scaled(a: &mut Vec3d, b: &Vec3d, s: f64) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y * s;
    }
}

#[inline]
pub fn norm(v: &Vec3d) -> f64 {
    dot(v, v).sqrt()
}

/// A vector turned about the y axis by `angle`, the way a positive spin about y turns things.
#[inline]
pub fn rotate_y(v: &Vec3d, angle: f64) -> Vec3d {
    let (s, c) = angle.sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

/// The quaternion of a turn about the y axis by `angle`.
pub fn quat_about_y(angle: f64) -> Quatd {
    [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()]
}

/// Hamilton product of two quaternions (x, y, z, w): the rotation `b` then `a`.
pub fn quat_mul(a: &Quatd, b: &Quatd) -> Quatd {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

pub fn quat_conjugate(q: &Quatd) -> Quatd {
    [-q[0], -q[1], -q[2], q[3]]
}

/// A vector turned by a unit quaternion.
pub fn quat_rotate(q: &Quatd, v: &Vec3d) -> Vec3d {
    let u = [q[0], q[1], q[2]];
    let uv = cross(&u, v);
    let uuv = cross(&u, &uv);
    let mut out = *v;
    add_scaled(&mut out, &uv, 2.0 * q[3]);
    add_scaled(&mut out, &uuv, 2.0);
    out
}

/// The attitude after turning at `w` for `dt`: exact for a steady turn, so a body that spins
/// evenly keeps time with a frame that turns evenly, however fast either does.
pub fn quat_integrate(q: &Quatd, w: &Vec3d, dt: f64) -> Quatd {
    let turn = quat_from_rotation_vector(&w.map(|w| w * dt));
    let mut out = quat_mul(&turn, q);
    let l = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    for x in &mut out {
        *x /= l;
    }
    out
}

/// The rotation by a rotation vector: about its direction, by its length in radians.
pub fn quat_from_rotation_vector(v: &Vec3d) -> Quatd {
    let angle = norm(v);
    if angle < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let s = (angle / 2.0).sin() / angle;
    [v[0] * s, v[1] * s, v[2] * s, (angle / 2.0).cos()]
}

/// The shortest rotation taking the unit vector `a` onto the unit vector `b`.
pub fn quat_between(a: &Vec3d, b: &Vec3d) -> Quatd {
    let c = cross(a, b);
    let w = 1.0 + dot(a, b);
    if w < 1e-9 {
        let helper = if a[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let axis = cross(a, &helper);
        let l = norm(&axis).max(1e-12);
        return [axis[0] / l, axis[1] / l, axis[2] / l, 0.0];
    }
    let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + w * w).sqrt();
    [c[0] / l, c[1] / l, c[2] / l, w / l]
}

/// The rotation a unit quaternion stands for as an axis scaled by its angle, the shorter way
/// round.
pub fn rotation_vector(q: &Quatd) -> Vec3d {
    let sign = if q[3] < 0.0 { -1.0 } else { 1.0 };
    let (x, y, z, w) = (q[0] * sign, q[1] * sign, q[2] * sign, q[3] * sign);
    let sin_half = (x * x + y * y + z * z).sqrt();
    if sin_half < 1e-12 {
        return [0.0; 3];
    }
    let angle = 2.0 * sin_half.atan2(w.min(1.0));
    [x, y, z].map(|c| c / sin_half * angle)
}

/// Quaternion (x, y, z, w) whose rotation maps local axes onto the given orthonormal basis.
pub fn quat_from_basis(ex: &Vec3d, ey: &Vec3d, ez: &Vec3d) -> Quatd {
    let (m0, m1, m2) = (ex[0], ey[0], ez[0]);
    let (m3, m4, m5) = (ex[1], ey[1], ez[1]);
    let (m6, m7, m8) = (ex[2], ey[2], ez[2]);
    let tr = m0 + m4 + m8;
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [(m7 - m5) / s, (m2 - m6) / s, (m3 - m1) / s, 0.25 * s]
    } else if m0 > m4 && m0 > m8 {
        let s = (1.0 + m0 - m4 - m8).sqrt() * 2.0;
        [0.25 * s, (m1 + m3) / s, (m2 + m6) / s, (m7 - m5) / s]
    } else if m4 > m8 {
        let s = (1.0 + m4 - m0 - m8).sqrt() * 2.0;
        [(m1 + m3) / s, 0.25 * s, (m5 + m7) / s, (m2 - m6) / s]
    } else {
        let s = (1.0 + m8 - m0 - m4).sqrt() * 2.0;
        [(m2 + m6) / s, (m5 + m7) / s, 0.25 * s, (m3 - m1) / s]
    }
}

/// Small deterministic RNG (xorshift64*), uniform in [0, 1).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        v as f32 / (1u64 << 24) as f32
    }
}
