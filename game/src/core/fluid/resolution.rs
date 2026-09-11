//! How much water one particle stands for, and the one water the GPU ever simulates. Every
//! resolution is the canonical water with its units scaled: the rest spacing is a unit of
//! length, and since this is water under gravity, its seconds are √(spacing / finest spacing)
//! real seconds, so that it flows the same at any scale. Single precision holds the water
//! finely only so many of its spacings from the origin of its frame, so how far the water
//! reaches from there sets a floor on how fine it can be: water put in one place is as fine
//! on a ring of any size, and water spread far is coarser.
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::core::units::{Hertz, Litres, Metres, Seconds};

pub const REST_DENSITY: f32 = 1000.0;
/// How far from the origin of its frame water may reach, in its own spacings.
pub const SPACINGS_FROM_ORIGIN: f32 = 1500.0;
/// How often the canonical water steps, in its own seconds.
pub const STEP_RATE: Hertz = Hertz(60.0);

/// The canonical water's constants, in its own units: particles a unit apart, a kernel two
/// units wide, and the rest density's worth of mass each.
pub mod canonical {
    use super::{PI, REST_DENSITY};

    pub const SPACING: f32 = 1.0;
    pub const H: f32 = 2.0;
    pub const H_SQ: f32 = H * H;
    pub const MASS: f32 = REST_DENSITY;
    pub const POLY6: f32 = 315.0 / (64.0 * PI * (H_SQ * H_SQ * H_SQ * H_SQ * H));
    pub const SPIKY: f32 = -45.0 / (PI * (H_SQ * H_SQ * H_SQ));
    /// The poly6 kernel at zero distance.
    pub const W_ZERO: f32 = POLY6 * H_SQ * H_SQ * H_SQ;
    /// The poly6 kernel at the artificial pressure's reference distance, 0.3 h.
    pub const SCORR_WQ: f32 =
        POLY6 * (H_SQ - 0.09 * H_SQ) * (H_SQ - 0.09 * H_SQ) * (H_SQ - 0.09 * H_SQ);
    /// The most a constraint iteration may move a particle.
    pub const MAX_DELTA: f32 = 0.5 * SPACING;
    /// How far particles keep from the walls.
    pub const MARGIN: f32 = 0.5 * SPACING;

    /// Akinci volume weights Ψ for boundary samples in canonical units: each sample stands in
    /// for the rest density its neighbours don't cover.
    pub fn sample_weights(points: &[[f32; 3]]) -> Vec<[f32; 4]> {
        points
            .iter()
            .map(|p| {
                let mut s = 0.0f32;
                for q in points {
                    let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 < H_SQ {
                        let t = H_SQ - r2;
                        s += POLY6 * t * t * t;
                    }
                }
                [p[0], p[1], p[2], REST_DENSITY / s]
            })
            .collect()
    }
}

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

    /// The finest water that may reach this far from the origin of its frame.
    pub fn finest_for(reach: Metres) -> Resolution {
        Resolution {
            spacing: Metres(
                Resolution::FINEST
                    .spacing
                    .0
                    .max(reach.0 / SPACINGS_FROM_ORIGIN),
            ),
        }
    }

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

    /// Metres per unit of the canonical water's length.
    pub fn length(self) -> f64 {
        self.spacing.0 as f64
    }

    /// Real seconds per second of the canonical water.
    pub fn time(self) -> f64 {
        (self.spacing.0 as f64 / Resolution::FINEST.spacing.0 as f64).sqrt()
    }

    /// The real time one step of this water covers.
    pub fn step(self) -> Seconds {
        Seconds((STEP_RATE.period().0 as f64 * self.time()) as f32)
    }

    /// How far particles keep from the walls.
    pub fn margin(self) -> Metres {
        Metres(canonical::MARGIN * self.spacing.0)
    }
}
