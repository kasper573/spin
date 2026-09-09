//! The glass drum: a solid cylinder spinning about its axis (world y), with a sculptable landscape
//! on the inside of its floor that starts as a layer of ground around the whole ring. Implements
//! the vessel the fluid and the bodies live in. Its size is a setting: the ring can be made
//! wider or narrower and its landscape stretches to fit.
mod gpu;
mod landscape;
mod render;

pub use gpu::{DrumFrame, DrumUniform};
pub use landscape::{Landscape, wheel_angle};
pub use render::DrumPlugin;

use serde::{Deserialize, Serialize};

use crate::core::units::{Metres, Radians, RadiansPerSecond};
use crate::core::vessel::{Contact, Penetration, Penetrations, Vessel};

/// The ring as it starts out.
pub const DEFAULT_RING: Ring = Ring {
    radius: Metres(10.5),
    half_width: Metres(6.0),
};
/// The largest ring the settings allow.
pub const LARGEST_RING: Ring = Ring {
    radius: Metres(499.5),
    half_width: Metres(499.5),
};
/// The ground that covers the glass all the way round in the initial state.
pub const GROUND_DEPTH: Metres = Metres(0.5);
/// The glass shell's thickness, felt only from outside.
pub const GLASS_THICKNESS: f64 = 0.1;
/// Maximum spin-up acceleration of the drum (rad/s²).
const SPIN_ACCEL: f64 = 0.6;

/// The drum's size: how far the glass is from the axis and how far each cap is from the middle.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub radius: Metres,
    pub half_width: Metres,
}

impl Ring {
    /// Distance from the axis to the top of the initial ground.
    pub fn floor_radius(self) -> Metres {
        Metres(self.radius.0 - GROUND_DEPTH.0)
    }

    /// The landscape may never be raised closer to the axis than this.
    pub fn max_height(self) -> Metres {
        Metres(self.radius.0 - 1.0)
    }

    /// Half the size of the box the drum fits in: radius across, half width along the axis.
    pub fn extent(self) -> [f32; 3] {
        [self.radius.0, self.half_width.0, self.radius.0]
    }

    /// Whether a point is in the air the drum encloses.
    pub fn encloses(self, p: [f64; 3]) -> bool {
        let r = self.radius.0 as f64;
        p[0] * p[0] + p[2] * p[2] < r * r && p[1].abs() < self.half_width.0 as f64
    }
}

pub struct Drum {
    pub ring: Ring,
    pub spin: RadiansPerSecond,
    pub target_spin: RadiansPerSecond,
    /// Accumulated rotation; the glass at wheel angle φ sits at world angle φ − angle.
    pub angle: Radians,
    pub landscape: Landscape,
}

impl Default for Drum {
    fn default() -> Self {
        Drum::new(DEFAULT_RING)
    }
}

impl Drum {
    /// A still drum of this size with the initial ground all the way round.
    pub fn new(ring: Ring) -> Drum {
        Drum {
            ring,
            spin: RadiansPerSecond(0.0),
            target_spin: RadiansPerSecond(0.0),
            angle: Radians(0.0),
            landscape: Landscape::flat(ring, GROUND_DEPTH),
        }
    }

    /// Make the drum another size; the landscape stretches with it.
    pub fn resize(&mut self, ring: Ring) {
        self.ring = ring;
        self.landscape.resize(ring);
    }

    /// Spin up or down toward the target and turn; the angle is kept to one turn so that its
    /// single-precision copies on the GPU stay exact however long the drum has been running.
    pub fn advance(&mut self, dt: f64) {
        let d = (self.target_spin.0 - self.spin.0) as f64;
        let max = SPIN_ACCEL * dt;
        self.spin.0 += d.clamp(-max, max) as f32;
        self.angle.0 = (self.angle.0 + self.spin.0 as f64 * dt).rem_euclid(std::f64::consts::TAU);
    }

    /// Whether a point is in the air the drum encloses.
    pub fn encloses(&self, p: [f64; 3]) -> bool {
        self.ring.encloses(p)
    }

    /// Angle of a world point in the drum's own frame.
    pub fn wheel_angle(&self, x: f64, z: f64) -> f64 {
        wheel_angle(x, z, self.angle.0)
    }

    /// Pull a point inside the drum, clear of the caps and above the landscape.
    pub fn place_inside(&self, p: [f32; 3]) -> [f32; 3] {
        let margin = 0.3;
        let half_width = self.ring.half_width.0;
        let y = p[1].clamp(-half_width + margin, half_width - margin);
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        let (h, _, _) = self
            .landscape
            .sample(self.wheel_angle(p[0] as f64, p[2] as f64), y as f64);
        let limit = self.ring.radius.0 - margin - h as f32;
        if r > limit {
            [p[0] * limit / r, y, p[2] * limit / r]
        } else {
            [p[0], y, p[2]]
        }
    }

    /// Pull a sphere's centre inside the drum, clear of the caps and no deeper than the initial
    /// ground, keeping its bearing from the axis.
    pub fn place_sphere_inside(&self, c: [f64; 3], radius: f64) -> [f64; 3] {
        let half_width = self.ring.half_width.0 as f64 - radius;
        let y = c[1].clamp(-half_width, half_width);
        let r = (c[0] * c[0] + c[2] * c[2]).sqrt();
        let limit = self.ring.floor_radius().0 as f64 - radius;
        if r > limit && r > 1e-9 {
            [c[0] * limit / r, y, c[2] * limit / r]
        } else {
            [c[0], y, c[2]]
        }
    }
}

impl Drum {
    /// A sphere inside the drum against the rim, the landscape and the caps.
    fn inner_sphere_penetrations(&self, c: [f64; 3], radius: f64) -> Penetrations {
        let mut out = Penetrations::default();
        let (depth, normal) = if self.landscape.is_empty() {
            let r = (c[0] * c[0] + c[2] * c[2]).sqrt().max(1e-9);
            (
                r + radius - self.ring.radius.0 as f64,
                [-c[0] / r, 0.0, -c[2] / r],
            )
        } else {
            self.landscape
                .penetration(c[0], c[1], c[2], self.angle.0, radius)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let cap = self.ring.half_width.0 as f64 - radius;
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
        let outer_radius = self.ring.radius.0 as f64 + GLASS_THICKNESS;
        let outer_half = self.ring.half_width.0 as f64 + GLASS_THICKNESS;
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
            let limit = self.ring.radius.0 - margin;
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
        let cap = self.ring.half_width.0 - margin;
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
            (r - self.ring.radius.0 as f64, [-p[0] / r, 0.0, -p[2] / r])
        } else {
            self.landscape
                .penetration(p[0], p[1], p[2], self.angle.0, 0.0)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let hw = self.ring.half_width.0 as f64;
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
