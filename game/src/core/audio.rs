//! Procedural thruster sound. Every thruster is the same jet, filtered noise under a hum, and its
//! level sets how loud it plays: silent when idle, a quarter as soon as it fires, full at full
//! level. What tells the thrusters apart is where they sit around the head: each is panned to
//! its side and muffled a little when it is behind.
use std::f32::consts::{FRAC_PI_4, TAU};
use std::num::NonZero;

use crate::core::math::{Rng, Vec3d};

pub const SAMPLE_RATE: NonZero<u32> = NonZero::new(44_100).unwrap();
/// Loudness of a voice as soon as its thruster fires at all.
const FLOOR: f32 = 0.25;
/// The hum under the jet (Hz), and the jet noise's cutoff (Hz).
const HUM: f32 = 80.0;
const JET: f32 = 900.0;
/// A jet straight behind the head is heard through the skull: this much quieter, cut off here.
const BEHIND_LEVEL: f32 = 0.8;
const BEHIND_CUTOFF: f32 = 1200.0;

/// Linear loudness of a thruster's voice at this level of thrust.
pub fn gain(level: f64) -> f32 {
    if level <= 0.0 {
        0.0
    } else {
        FLOOR + (1.0 - FLOOR) * level.min(1.0) as f32
    }
}

/// A thruster's voice at full loudness, one sample at a time. Every voice sounds the same; the
/// seed only keeps two jets from hissing in step.
#[derive(Clone, Debug)]
pub struct Voice {
    rng: Rng,
    phase: f32,
    jet: [f32; 2],
    smoothing: f32,
}

impl Voice {
    pub fn new(seed: u64) -> Voice {
        Voice {
            rng: Rng::new(0x5EED ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            phase: 0.0,
            jet: [0.0; 2],
            smoothing: lowpass(JET),
        }
    }

    pub fn sample(&mut self) -> f32 {
        let noise = self.rng.next_f32() * 2.0 - 1.0;
        self.jet[0] += (noise - self.jet[0]) * self.smoothing;
        self.jet[1] += (self.jet[0] - self.jet[1]) * self.smoothing;
        self.phase = (self.phase + HUM / SAMPLE_RATE.get() as f32) % 1.0;
        let hum = (TAU * self.phase).sin() * 0.5 + (2.0 * TAU * self.phase).sin() * 0.25;
        (self.jet[1] * 2.5 + hum * 0.35).clamp(-1.0, 1.0) * 0.6
    }
}

/// How a sound from somewhere around the head reaches the two ears: panned with constant power
/// by how far to the side it is, quieter and duller by how far behind.
#[derive(Clone, Debug)]
pub struct Placement {
    left: f32,
    right: f32,
    smoothing: f32,
    heard: f32,
}

impl Placement {
    /// For a sound at `at`, relative to the head in its level frame (x right, y up, z back).
    pub fn around(at: Vec3d) -> Placement {
        let [x, _, z] = at.map(|m| m as f32);
        let side = (x * x + z * z).sqrt();
        let (pan, behind) = if side > 1e-6 {
            (x / side, (z / side).max(0.0))
        } else {
            (0.0, 0.0)
        };
        let angle = (pan + 1.0) * FRAC_PI_4;
        let level = 1.0 - (1.0 - BEHIND_LEVEL) * behind;
        let cutoff = SAMPLE_RATE.get() as f32 / 2.0 * (1.0 - behind) + BEHIND_CUTOFF * behind;
        Placement {
            left: angle.cos() * level,
            right: angle.sin() * level,
            smoothing: lowpass(cutoff),
            heard: 0.0,
        }
    }

    /// The sample as the left and right ear hear it.
    pub fn hear(&mut self, sample: f32) -> [f32; 2] {
        self.heard += (sample - self.heard) * self.smoothing;
        [self.heard * self.left, self.heard * self.right]
    }

    /// Loudness at each ear for a sound at full level.
    pub fn ears(&self) -> [f32; 2] {
        [self.left, self.right]
    }
}

/// One-pole low-pass smoothing for this cutoff at the sample rate.
fn lowpass(cutoff: f32) -> f32 {
    (1.0 - (-TAU * cutoff / SAMPLE_RATE.get() as f32).exp()).min(1.0)
}
