//! Procedural thruster sound. Every pushing thruster is the same jet, filtered noise under a hum,
//! and every turning one the same lighter puff of air, and a thruster's level sets how loud it
//! plays: silent when idle, a quarter as soon as it fires, full at full level. What tells the
//! thrusters apart is where they sit around the head: the head shadows the far ear, so a jet to
//! one side is a little quieter and duller in the other ear, and one behind is duller in both.
use std::f32::consts::TAU;
use std::num::NonZero;

use crate::core::math::{Rng, Vec3d};

pub const SAMPLE_RATE: NonZero<u32> = NonZero::new(44_100).unwrap();
/// Loudness of a voice as soon as its thruster fires at all.
const FLOOR: f32 = 0.25;
/// The hum under the jet (Hz), and the jet noise's cutoff (Hz).
const HUM: f32 = 80.0;
const JET: f32 = 900.0;
/// The puff of a turning thruster: airier noise, no hum, and this much of the jet's loudness.
const PUFF: f32 = 2600.0;
const PUFF_LEVEL: f32 = 0.4;
/// A jet straight to one side reaches the far ear this much quieter, cut off here.
const SHADOW_LEVEL: f32 = 0.45;
const SHADOW_CUTOFF: f32 = 1800.0;
/// A jet straight behind the head reaches both ears this much quieter, cut off here.
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

/// A thruster's voice at full loudness, one sample at a time. Every jet sounds the same, and so
/// does every puff; the seed only keeps two from hissing in step.
#[derive(Clone, Debug)]
pub struct Voice {
    rng: Rng,
    phase: f32,
    jet: [f32; 2],
    smoothing: f32,
    hum: f32,
    level: f32,
}

impl Voice {
    /// The jet of a pushing thruster.
    pub fn new(seed: u64) -> Voice {
        Voice {
            rng: Rng::new(0x5EED ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            phase: 0.0,
            jet: [0.0; 2],
            smoothing: lowpass(JET),
            hum: 0.35,
            level: 0.6,
        }
    }

    /// The lighter puff of a turning thruster.
    pub fn puff(seed: u64) -> Voice {
        Voice {
            smoothing: lowpass(PUFF),
            hum: 0.0,
            level: 0.6 * PUFF_LEVEL,
            ..Voice::new(seed)
        }
    }

    pub fn sample(&mut self) -> f32 {
        let noise = self.rng.next_f32() * 2.0 - 1.0;
        self.jet[0] += (noise - self.jet[0]) * self.smoothing;
        self.jet[1] += (self.jet[0] - self.jet[1]) * self.smoothing;
        self.phase = (self.phase + HUM / SAMPLE_RATE.get() as f32) % 1.0;
        let hum = (TAU * self.phase).sin() * 0.5 + (2.0 * TAU * self.phase).sin() * 0.25;
        (self.jet[1] * 2.5 + hum * self.hum).clamp(-1.0, 1.0) * self.level
    }
}

/// How a sound from somewhere around the head reaches the two ears: each ear hears it a little
/// quieter and duller the more the head is in the way.
#[derive(Clone, Debug)]
pub struct Placement {
    ears: [Ear; 2],
}

#[derive(Clone, Debug)]
struct Ear {
    gain: f32,
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
        let nyquist = SAMPLE_RATE.get() as f32 / 2.0;
        let ear = |toward: f32| {
            let shadow = (pan * toward).max(0.0);
            let level =
                (1.0 - (1.0 - SHADOW_LEVEL) * shadow) * (1.0 - (1.0 - BEHIND_LEVEL) * behind);
            let cutoff = (nyquist + (SHADOW_CUTOFF - nyquist) * shadow)
                .min(nyquist + (BEHIND_CUTOFF - nyquist) * behind);
            Ear {
                gain: level,
                smoothing: lowpass(cutoff),
                heard: 0.0,
            }
        };
        Placement {
            ears: [ear(1.0), ear(-1.0)],
        }
    }

    /// The sample as the left and right ear hear it.
    pub fn hear(&mut self, sample: f32) -> [f32; 2] {
        let mut out = [0.0; 2];
        for (ear, heard) in self.ears.iter_mut().zip(&mut out) {
            ear.heard += (sample - ear.heard) * ear.smoothing;
            *heard = ear.heard * ear.gain;
        }
        out
    }

    /// Loudness at each ear for a sound at full level.
    pub fn ears(&self) -> [f32; 2] {
        [self.ears[0].gain, self.ears[1].gain]
    }
}

/// One-pole low-pass smoothing for this cutoff at the sample rate.
fn lowpass(cutoff: f32) -> f32 {
    (1.0 - (-TAU * cutoff / SAMPLE_RATE.get() as f32).exp()).min(1.0)
}
