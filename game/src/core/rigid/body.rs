use crate::core::math::{
    Quatd, Vec3d, add_scaled, cross, mat3mul, mat3solve, quat_integrate, quat_rotate,
};
use crate::core::units::{MetresPerSecond, Newtons, RadiansPerSecond};

/// What the water did to a body over the substeps of a frame: the buoyancy impulse and torque,
/// and the flow around the hull weighted by how strongly each wetted sample coupled to it, as a
/// fraction of the body's mass, summed over every substep. The flow sums let the body be relaxed
/// toward the water it is actually in rather than by a difference against a stale copy of its
/// own velocity, so the coupling stays stable whenever the readback lands.
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
    /// Σ k: the coupled water's mass over the body's mass, summed over the substeps.
    pub coupling: f64,
    /// The wetted fraction of the hull, summed over the substeps.
    pub wet: f64,
    /// Simulated time the sums cover, and how many substeps it was taken in.
    pub seconds: f64,
    pub substeps: f64,
    /// Simulated seconds from the middle of the time the sums cover to the moment they are
    /// applied; the water and the hull turn with the vessel meanwhile.
    pub age: f64,
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
        self.seconds += other.seconds;
        self.substeps += other.substeps;
    }

    /// The share `f` of the sums, covering that share of their time.
    pub fn scaled(&self, f: f64) -> WaterCoupling {
        WaterCoupling {
            buoyancy: self.buoyancy.map(|x| x * f),
            buoyancy_torque: self.buoyancy_torque.map(|x| x * f),
            flow: self.flow.map(|x| x * f),
            flow_moment: self.flow_moment.map(|x| x * f),
            hull: self.hull.map(|x| x * f),
            hull_tensor: self.hull_tensor.map(|x| x * f),
            coupling: self.coupling * f,
            wet: self.wet * f,
            seconds: self.seconds * f,
            substeps: self.substeps * f,
            age: self.age,
        }
    }

    /// The sums as they stand after everything they were taken from has turned by `q`.
    pub fn turned(&self, q: &Quatd) -> WaterCoupling {
        let t = &self.hull_tensor;
        let columns = [
            quat_rotate(q, &[t[0], t[3], t[5]]),
            quat_rotate(q, &[t[3], t[1], t[4]]),
            quat_rotate(q, &[t[5], t[4], t[2]]),
        ];
        let row = |i: usize| quat_rotate(q, &[columns[0][i], columns[1][i], columns[2][i]]);
        let (x, y, z) = (row(0), row(1), row(2));
        WaterCoupling {
            buoyancy: quat_rotate(q, &self.buoyancy),
            buoyancy_torque: quat_rotate(q, &self.buoyancy_torque),
            flow: quat_rotate(q, &self.flow),
            flow_moment: quat_rotate(q, &self.flow_moment),
            hull: quat_rotate(q, &self.hull),
            hull_tensor: [x[0], y[1], z[2], x[1], y[2], z[0]],
            ..*self
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

const DEFAULT_FRICTION: f64 = 0.45;
const DEFAULT_RESTITUTION: f64 = 0.2;
/// The water's drag: a body relaxes toward the flow around it on this time scale divided by the
/// coupled water's mass over its own, so a hull gripped by its own mass of water is carried
/// along in a couple of seconds and a light board rides the water almost at once.
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

/// What a body is made of, for contacts. Positions are local to the body's centre of mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// A box centred on the centre of mass.
    Box { half: Vec3d },
    /// A sphere whose centre may sit away from the centre of mass, so that whatever pushes on
    /// the hull, the ground, the water or the air, turns it until the ballast hangs below.
    /// Friction acts through the centre of mass rather than at the hull, so the sphere slides
    /// instead of rolling.
    Sphere { radius: f64, centre: Vec3d },
}

impl Collider {
    /// Where the air pushes on the hull: its geometric centre.
    pub fn centre_of_pressure(self) -> Vec3d {
        match self {
            Collider::Box { .. } => [0.0; 3],
            Collider::Sphere { centre, .. } => centre,
        }
    }
}

/// Mass properties, material, fluid boundary samples and contact points shared by every body of
/// one kind.
pub struct BodyShape {
    pub collider: Collider,
    pub mass: f64,
    pub inv_inertia: Vec3d,
    /// Coulomb friction and bounciness against walls and other bodies.
    pub friction: f64,
    pub restitution: f64,
    /// Boundary sample points for fluid coupling.
    pub samples: Vec<[f32; 3]>,
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
        let samples = mid_plane_samples(a, c, sample_spacing);
        let volume_per_sample = a * b * c / samples.len() as f64;
        let half = [a / 2.0, b / 2.0, c / 2.0];
        BodyShape {
            collider: Collider::Box { half },
            mass,
            inv_inertia,
            friction: DEFAULT_FRICTION,
            restitution: DEFAULT_RESTITUTION,
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
        let samples = sphere_samples(radius, centre, sample_count);
        let volume = 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
        BodyShape {
            collider: Collider::Sphere { radius, centre },
            mass,
            inv_inertia,
            friction: DEFAULT_FRICTION,
            restitution: DEFAULT_RESTITUTION,
            volume_per_sample: volume / samples.len() as f64,
            samples,
            points: Vec::new(),
        }
    }

    /// The same shape displacing this much water (m³) instead of its whole hull.
    pub fn displacing(mut self, volume: f64) -> BodyShape {
        self.volume_per_sample = volume / self.samples.len() as f64;
        self
    }

    /// Whether the water turns the body: it grips a board's faces and turns it with the flow,
    /// while a sphere slides through it the way it slides over the ground, leaving its spin to
    /// whatever else steers it.
    pub fn turns_in_water(&self) -> bool {
        matches!(self.collider, Collider::Box { .. })
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
            inv_i_max: inv_i[0].max(inv_i[1]).max(inv_i[2]),
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
    /// acceleration. `spin` is the rotation of the water at rest in the vessel; only a body
    /// that `turns` is spun by the water around its hull, the rest slide through it.
    pub fn couple(&mut self, water: &WaterCoupling, max_accel: f64, spin: &Vec3d, turns: bool) {
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
            if turns && let Some(wanted) = water.flow_spin(&self.v, spin) {
                let moment = water.coupling_moment() / self.inv_m * self.inv_i_max;
                let s_ang = 1.0 - (-moment * dt / DRAG_TAU).exp();
                for (w, f) in self.w.iter_mut().zip(wanted) {
                    *w += (f - *w) * s_ang;
                }
            }
        }

        let cap = max_accel * water.seconds / self.inv_m;
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
