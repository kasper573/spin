//! How much water one particle stands for. Every kernel constant of the solver follows from the
//! rest spacing, so the water can be made coarser as a whole: twice the water per particle for
//! the same number of particles.
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::core::units::{Litres, Metres};

pub const REST_DENSITY: f32 = 1000.0;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Resolution {
    /// Rest spacing between particles.
    pub spacing: Metres,
}

impl Default for Resolution {
    fn default() -> Self {
        Resolution::FINEST
    }
}

impl Resolution {
    /// The finest water: what a fresh simulation starts with.
    pub const FINEST: Resolution = Resolution {
        spacing: Metres(0.32),
    };

    /// Every particle stands for twice the water.
    pub fn coarser(self) -> Resolution {
        Resolution {
            spacing: Metres(self.spacing.0 * 2f32.cbrt()),
        }
    }

    /// The water one particle stands for.
    pub fn litres_per_particle(self) -> Litres {
        let s = self.spacing.0;
        Litres(s * s * s * 1000.0)
    }

    /// Kernel support radius.
    pub fn h(self) -> f32 {
        2.0 * self.spacing.0
    }

    pub fn h_sq(self) -> f32 {
        self.h() * self.h()
    }

    pub fn mass(self) -> f32 {
        let s = self.spacing.0;
        REST_DENSITY * s * s * s
    }

    pub fn poly6(self) -> f32 {
        315.0 / (64.0 * PI * self.h().powi(9))
    }

    pub fn spiky(self) -> f32 {
        -45.0 / (PI * self.h().powi(6))
    }

    /// The poly6 kernel at zero distance.
    pub fn w_zero(self) -> f32 {
        self.poly6() * self.h_sq().powi(3)
    }

    /// The poly6 kernel at the artificial pressure's reference distance.
    pub fn scorr_wq(self) -> f32 {
        let dq = 0.3 * self.h();
        self.poly6() * (self.h_sq() - dq * dq).powi(3)
    }

    /// The most a constraint iteration may move a particle.
    pub fn max_delta(self) -> f32 {
        0.5 * self.spacing.0
    }

    /// How far particles keep from the walls.
    pub fn margin(self) -> f32 {
        0.5 * self.spacing.0
    }

    /// Akinci volume weights Ψ for boundary samples: each sample stands in for the rest density
    /// its neighbours don't cover.
    pub fn sample_weights(self, points: &[[f32; 3]]) -> Vec<[f32; 4]> {
        let (h_sq, poly6) = (self.h_sq(), self.poly6());
        points
            .iter()
            .map(|p| {
                let mut s = 0.0f32;
                for q in points {
                    let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 < h_sq {
                        let t = h_sq - r2;
                        s += poly6 * t * t * t;
                    }
                }
                [p[0], p[1], p[2], REST_DENSITY / s]
            })
            .collect()
    }
}
