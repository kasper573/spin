use std::f64::consts::PI;

use super::Ring;
use crate::core::units::Metres;

pub const SEGMENTS: usize = 512;
pub const ROWS: usize = 64;

const TWO_PI: f64 = PI * 2.0;
const DPHI: f64 = TWO_PI / SEGMENTS as f64;

/// Angle of a world point in the wheel's own frame.
#[inline]
pub fn wheel_angle(x: f64, z: f64, theta: f64) -> f64 {
    z.atan2(x) + theta
}

/// Sculptable terrain on the inside of the drum floor: a periodic heightfield in the wheel's frame,
/// measured inward from the glass, with a fixed number of segments round and rows across whatever
/// size the ring is. Height zero means bare glass.
pub struct Landscape {
    ring: Ring,
    heights: Vec<f32>,
    empty: bool,
    /// Bumped on every change so consumers can rebuild derived data lazily.
    version: u64,
}

impl Landscape {
    /// Bare glass all the way round.
    pub fn new(ring: Ring) -> Self {
        Landscape {
            ring,
            heights: vec![0.0; SEGMENTS * ROWS],
            empty: true,
            version: 0,
        }
    }

    /// Ground of one depth everywhere.
    pub fn flat(ring: Ring, depth: Metres) -> Self {
        let mut land = Landscape::new(ring);
        land.flatten(depth);
        land
    }

    pub fn ring(&self) -> Ring {
        self.ring
    }

    /// Stretch the same heights round a ring of another size.
    pub fn resize(&mut self, ring: Ring) {
        self.ring = ring;
        let max = ring.max_height().0;
        for h in &mut self.heights {
            *h = h.min(max);
        }
        self.version += 1;
    }

    /// Make the ground one depth everywhere.
    pub fn flatten(&mut self, depth: Metres) {
        let depth = depth.0.clamp(0.0, self.ring.max_height().0);
        self.heights.fill(depth);
        self.empty = depth <= 0.0;
        self.version += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.empty
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Heights in segment-major order.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn height_at(&self, segment: usize, row: usize) -> f32 {
        self.heights[segment * ROWS + row]
    }

    pub fn max_height(&self) -> f32 {
        self.heights.iter().copied().fold(0.0, f32::max)
    }

    pub fn load(&mut self, data: &[f32]) {
        let max = self.ring.max_height().0;
        let mut any = false;
        for (i, v) in self.heights.iter_mut().enumerate() {
            let s = data.get(i).copied().unwrap_or(0.0);
            let s = if s.is_finite() {
                s.clamp(0.0, max)
            } else {
                0.0
            };
            *v = s;
            any |= s > 0.0;
        }
        self.empty = !any;
        self.version += 1;
    }

    /// Raise (or lower, for negative `amount`) the terrain by up to `amount` metres inside a round
    /// brush centred at wheel angle `phi` and axial position `y`.
    pub fn sculpt(&mut self, phi: f64, y: f64, radius: f64, amount: f64) {
        let r = self.ring.radius.0 as f64;
        let half_width = self.ring.half_width.0 as f64;
        let dy = self.row_spacing();
        let max = self.ring.max_height().0;
        let dphi_max = radius / r;
        let i_lo = ((phi - dphi_max) / DPHI).floor() as i64;
        let i_hi = ((phi + dphi_max) / DPHI).ceil() as i64;
        let j_lo = (((y - radius) + half_width) / dy).floor().max(0.0) as usize;
        let j_hi = ((((y + radius) + half_width) / dy).ceil() as usize).min(ROWS - 1);
        let mut any = false;
        for i in i_lo..=i_hi {
            let seg = i.rem_euclid(SEGMENTS as i64) as usize;
            let dphi = i as f64 * DPHI - phi;
            let dx = dphi * r;
            for j in j_lo..=j_hi {
                let dy = -half_width + j as f64 * dy - y;
                let q = (dx * dx + dy * dy) / (radius * radius);
                if q >= 1.0 {
                    continue;
                }
                let w = (1.0 - q) * (1.0 - q);
                let h = &mut self.heights[seg * ROWS + j];
                *h = (*h + (amount * w) as f32).clamp(0.0, max);
                any = true;
            }
        }
        if any {
            self.empty = self.heights.iter().all(|h| *h <= 0.0);
            self.version += 1;
        }
    }

    /// Bilinear height and its derivatives with respect to wheel angle (rad) and axial position (m).
    #[inline]
    pub fn sample(&self, phi: f64, y: f64) -> (f64, f64, f64) {
        let dy_row = self.row_spacing();
        let u = (phi / DPHI).rem_euclid(SEGMENTS as f64);
        let i0 = u.floor() as usize % SEGMENTS;
        let fu = u - u.floor();
        let i1 = (i0 + 1) % SEGMENTS;
        let v = ((y + self.ring.half_width.0 as f64) / dy_row).clamp(0.0, ROWS as f64 - 1.0 - 1e-6);
        let j0 = v.floor() as usize;
        let fv = v - v.floor();
        let j1 = j0 + 1;
        let h = &self.heights;
        let h00 = h[i0 * ROWS + j0] as f64;
        let h10 = h[i1 * ROWS + j0] as f64;
        let h01 = h[i0 * ROWS + j1] as f64;
        let h11 = h[i1 * ROWS + j1] as f64;
        let hv = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv;
        let dphi = ((h10 - h00) * (1.0 - fv) + (h11 - h01) * fv) / DPHI;
        let dy = ((h01 - h00) * (1.0 - fu) + (h11 - h10) * fu) / dy_row;
        (hv, dphi, dy)
    }

    /// Signed penetration of a world point into the terrain (positive = inside) and the inward unit
    /// normal of the terrain surface. `margin` shifts the surface inward.
    #[inline]
    pub fn penetration(&self, x: f64, y: f64, z: f64, theta: f64, margin: f64) -> (f64, [f64; 3]) {
        let radius = self.ring.radius.0 as f64;
        let r = (x * x + z * z).sqrt();
        if r < 1e-6 {
            return (-radius, [0.0, 0.0, 0.0]);
        }
        let (h, dphi, dy) = self.sample(wheel_angle(x, z, theta), y);
        let f = r - (radius - h - margin);
        let g_phi = dphi / r;
        let gx = (x - g_phi * z) / r;
        let gz = (z + g_phi * x) / r;
        let gy = dy;
        let len = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-12);
        (f / len, [-gx / len, -gy / len, -gz / len])
    }

    /// The arc one segment spans round the ring: the finest feature the landscape can hold.
    pub fn segment_arc(&self) -> f64 {
        DPHI * self.ring.radius.0 as f64
    }

    /// Distance between rows along the axis.
    pub fn row_spacing(&self) -> f64 {
        2.0 * self.ring.half_width.0 as f64 / (ROWS as f64 - 1.0)
    }
}
