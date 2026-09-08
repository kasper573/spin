//! Procedural thruster sound. Each thruster has a voice of its own, a jet of filtered noise under
//! a hum, pitched by the thruster, and its level sets how loud the voice plays: silent when idle,
//! a quarter as soon as it fires, full at full level.
use std::f32::consts::TAU;
use std::num::NonZero;

use crate::core::avatar::Thruster;
use crate::core::math::Rng;

pub const SAMPLE_RATE: NonZero<u32> = NonZero::new(44_100).unwrap();
/// Loudness of a voice as soon as its thruster fires at all.
const FLOOR: f32 = 0.25;
/// The lowest thruster's hum (Hz); each thruster is a fifth of an octave above the one before.
const HUM: f32 = 60.0;
/// The lowest thruster's jet noise cutoff (Hz); each thruster is a quarter of an octave above.
const JET: f32 = 500.0;

/// Linear loudness of a thruster's voice at this level of thrust.
pub fn gain(level: f64) -> f32 {
    if level <= 0.0 {
        0.0
    } else {
        FLOOR + (1.0 - FLOOR) * level.min(1.0) as f32
    }
}

/// A thruster's voice at full loudness, one sample at a time.
#[derive(Clone, Debug)]
pub struct Voice {
    rng: Rng,
    phase: f32,
    step: f32,
    jet: [f32; 2],
    smoothing: f32,
}

impl Voice {
    pub fn new(thruster: Thruster) -> Voice {
        let rank = thruster as usize as f32;
        let rate = SAMPLE_RATE.get() as f32;
        let cutoff = JET * 2f32.powf(rank / 4.0);
        Voice {
            rng: Rng::new(0x5EED + thruster as u64),
            phase: 0.0,
            step: HUM * 2f32.powf(rank / 5.0) / rate,
            jet: [0.0; 2],
            smoothing: 1.0 - (-TAU * cutoff / rate).exp(),
        }
    }

    pub fn sample(&mut self) -> f32 {
        let noise = self.rng.next_f32() * 2.0 - 1.0;
        self.jet[0] += (noise - self.jet[0]) * self.smoothing;
        self.jet[1] += (self.jet[0] - self.jet[1]) * self.smoothing;
        self.phase = (self.phase + self.step) % 1.0;
        let hum = (TAU * self.phase).sin() * 0.5 + (2.0 * TAU * self.phase).sin() * 0.25;
        (self.jet[1] * 2.5 + hum * 0.35).clamp(-1.0, 1.0) * 0.6
    }
}
