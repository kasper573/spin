//! The glass drum: a solid cylinder spinning about its axis (world y), with a sculptable landscape
//! on the inside of its floor. Implements the vessel the fluid and rafts live in.
mod gpu;
mod landscape;
mod render;

pub use gpu::{DrumFrame, DrumUniform};
pub use landscape::{Landscape, wheel_angle};
pub use render::DrumPlugin;

use crate::core::units::{Radians, RadiansPerSecond};
use crate::core::vessel::{Contact, Penetration, Penetrations, Vessel};

pub const RADIUS: f32 = 3.5;
pub const HALF_WIDTH: f32 = 0.6;
/// The glass shell's thickness, felt only from outside.
pub const GLASS_THICKNESS: f64 = 0.05;
/// Maximum spin-up acceleration of the drum (rad/s²).
const SPIN_ACCEL: f64 = 0.6;

pub struct Drum {
    pub spin: RadiansPerSecond,
    pub target_spin: RadiansPerSecond,
    /// Accumulated rotation; the glass at wheel angle φ sits at world angle φ − angle.
    pub angle: Radians,
    pub landscape: Landscape,
}

impl Default for Drum {
    fn default() -> Self {
        Drum {
            spin: RadiansPerSecond(0.0),
            target_spin: RadiansPerSecond(0.0),
            angle: Radians(0.0),
            landscape: Landscape::new(),
        }
    }
}

impl Drum {
    pub fn advance(&mut self, dt: f64) {
        let d = (self.target_spin.0 - self.spin.0) as f64;
        let max = SPIN_ACCEL * dt;
        self.spin.0 += d.clamp(-max, max) as f32;
        self.angle.0 += self.spin.0 as f64 * dt;
    }

    /// Whether a point is in the air the drum encloses.
    pub fn encloses(&self, p: [f64; 3]) -> bool {
        p[0] * p[0] + p[2] * p[2] < (RADIUS as f64) * (RADIUS as f64)
            && p[1].abs() < HALF_WIDTH as f64
    }

    /// Angle of a world point in the drum's own frame.
    pub fn wheel_angle(&self, x: f64, z: f64) -> f64 {
        wheel_angle(x, z, self.angle.0)
    }

    /// Pull a point inside the drum, clear of the caps and above the landscape.
    pub fn place_inside(&self, p: [f32; 3]) -> [f32; 3] {
        let margin = 0.1;
        let y = p[1].clamp(-HALF_WIDTH + margin, HALF_WIDTH - margin);
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        let (h, _, _) = self
            .landscape
            .sample(self.wheel_angle(p[0] as f64, p[2] as f64), y as f64);
        let limit = RADIUS - margin - h as f32;
        if r > limit {
            [p[0] * limit / r, y, p[2] * limit / r]
        } else {
            [p[0], y, p[2]]
        }
    }
}

impl Drum {
    /// A sphere inside the drum against the rim, the landscape and the caps.
    fn inner_sphere_penetrations(&self, c: [f64; 3], radius: f64) -> Penetrations {
        let mut out = Penetrations::default();
        let (depth, normal) = if self.landscape.is_empty() {
            let r = (c[0] * c[0] + c[2] * c[2]).sqrt().max(1e-9);
            (r + radius - RADIUS as f64, [-c[0] / r, 0.0, -c[2] / r])
        } else {
            self.landscape
                .penetration(c[0], c[1], c[2], self.angle.0, radius)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let cap = HALF_WIDTH as f64 - radius;
        if c[1] > cap {
            out.push(Penetration {
                depth: c[1] - cap,
                normal: [0.0, -1.0, 0.0],
            });
        } else if c[1] < -cap {
            out.push(Penetration {
                depth: -cap - c[1],
                normal: [0.0, 1.0, 0.0],
            });
        }
        out
    }

    /// A sphere outside the drum against the outer surface of the glass shell.
    fn outer_sphere_penetrations(&self, c: [f64; 3], radius: f64) -> Penetrations {
        let mut out = Penetrations::default();
        let outer_radius = RADIUS as f64 + GLASS_THICKNESS;
        let outer_half = HALF_WIDTH as f64 + GLASS_THICKNESS;
        let r = (c[0] * c[0] + c[2] * c[2]).sqrt().max(1e-9);
        let radial = [c[0] / r, 0.0, c[2] / r];
        let axial = [0.0, c[1].signum(), 0.0];
        let (dr, dy) = (r - outer_radius, c[1].abs() - outer_half);
        let (distance, normal) = if dr > 0.0 && dy > 0.0 {
            let d = (dr * dr + dy * dy).sqrt();
            (
                d,
                [radial[0] * dr / d, axial[1] * dy / d, radial[2] * dr / d],
            )
        } else if dr > 0.0 {
            (dr, radial)
        } else if dy > 0.0 {
            (dy, axial)
        } else if dr > dy {
            (dr, radial)
        } else {
            (dy, axial)
        };
        if distance < radius {
            out.push(Penetration {
                depth: radius - distance,
                normal,
            });
        }
        out
    }
}

impl Vessel for Drum {
    fn confine(&self, p: &mut [f32; 3], margin: f32) -> Contact {
        let mut contact = Contact::default();
        if self.landscape.is_empty() {
            let limit = RADIUS - margin;
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
            if r > limit {
                let s = limit / r;
                p[0] *= s;
                p[2] *= s;
                contact.push([-p[0] / limit, 0.0, -p[2] / limit]);
            }
        } else {
            for _ in 0..2 {
                let (pen, n) = self.landscape.penetration(
                    p[0] as f64,
                    p[1] as f64,
                    p[2] as f64,
                    self.angle.0,
                    margin as f64,
                );
                if pen <= 0.0 {
                    break;
                }
                p[0] += (n[0] * pen) as f32;
                p[1] += (n[1] * pen) as f32;
                p[2] += (n[2] * pen) as f32;
                if contact.is_empty() {
                    contact.push([n[0] as f32, n[1] as f32, n[2] as f32]);
                }
            }
        }
        let cap = HALF_WIDTH - margin;
        if p[1] > cap {
            p[1] = cap;
            contact.push([0.0, -1.0, 0.0]);
        } else if p[1] < -cap {
            p[1] = -cap;
            contact.push([0.0, 1.0, 0.0]);
        }
        contact
    }

    fn penetrations(&self, p: [f64; 3]) -> Penetrations {
        let mut out = Penetrations::default();
        let (depth, normal) = if self.landscape.is_empty() {
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt().max(1e-9);
            (r - RADIUS as f64, [-p[0] / r, 0.0, -p[2] / r])
        } else {
            self.landscape
                .penetration(p[0], p[1], p[2], self.angle.0, 0.0)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let hw = HALF_WIDTH as f64;
        if p[1] > hw {
            out.push(Penetration {
                depth: p[1] - hw,
                normal: [0.0, -1.0, 0.0],
            });
        } else if p[1] < -hw {
            out.push(Penetration {
                depth: -hw - p[1],
                normal: [0.0, 1.0, 0.0],
            });
        }
        out
    }

    fn sphere_penetrations(&self, centre: [f64; 3], radius: f64) -> Penetrations {
        if self.encloses(centre) {
            self.inner_sphere_penetrations(centre, radius)
        } else {
            self.outer_sphere_penetrations(centre, radius)
        }
    }

    fn wall_velocity(&self, p: [f64; 3]) -> [f64; 3] {
        let w = self.spin.0 as f64;
        [w * p[2], 0.0, -w * p[0]]
    }

    fn air_velocity(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        self.encloses(p).then(|| self.wall_velocity(p))
    }

    fn angular_velocity(&self) -> [f64; 3] {
        [0.0, self.spin.0 as f64, 0.0]
    }
}
