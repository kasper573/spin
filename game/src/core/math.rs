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

/// Orientation of a board lying flat against a surface with normal n (the board's local y).
pub fn basis_from_normal(n: &Vec3d) -> Quatd {
    let helper = if n[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let mut ex = cross(&helper, n);
    let l = norm(&ex).max(1e-12);
    ex = [ex[0] / l, ex[1] / l, ex[2] / l];
    let ez = cross(&ex, n);
    quat_from_basis(&ex, n, &ez)
}

/// Small deterministic RNG (xorshift64*), uniform in [0, 1).
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
