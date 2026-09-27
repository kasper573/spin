use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Seconds(pub f32);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Metres(pub f32);

/// The world's time since it began, in seconds. A world lives for days, and single precision
/// cannot count substeps that far: by three days a substep rounds to two.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct WorldTime(pub f64);

impl WorldTime {
    pub fn after(self, span: Seconds) -> WorldTime {
        WorldTime(self.0 + span.0 as f64)
    }

    pub fn since(self, earlier: WorldTime) -> Seconds {
        Seconds((self.0 - earlier.0) as f32)
    }

    /// What shaders are told of it: how far into a [`SHADER_CLOCK_PERIOD`] it is, which single
    /// precision keeps to a small part of a frame.
    pub fn wound(self) -> Seconds {
        Seconds(self.0.rem_euclid(SHADER_CLOCK_PERIOD.0 as f64) as f32)
    }
}

/// How often the time told to shaders winds back to nought. They are told this too, and make
/// every steady motion they draw a whole number of times in it, so that it is never seen to wind
/// back.
pub const SHADER_CLOCK_PERIOD: Seconds = Seconds(7200.0);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct MetresPerSecond(pub f32);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Radians(pub f64);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct RadiansPerSecond(pub f32);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct RadiansPerSecondSquared(pub f32);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Litres(pub f32);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct LitresPerSecond(pub f32);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hertz(pub f32);

impl Hertz {
    pub fn period(self) -> Seconds {
        Seconds(1.0 / self.0)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct MetresPerSecondSquared(pub f64);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Newtons(pub f64);

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct PixelsPerSecond(pub f32);

/// Standard gravity at the Earth's surface, the reference every "g" readout is measured against.
pub const EARTH_GRAVITY: MetresPerSecondSquared = MetresPerSecondSquared(9.80665);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Pascals(pub f64);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Kelvin(pub f64);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct KilogramsPerCubicMetre(pub f64);

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Nanometres(pub f64);
