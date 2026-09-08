//! The vehicle the viewer rides: a ballasted sphere that slides instead of rolling and so always
//! settles upright, with thrusters, attitude gyros, a flight assist that brakes against the
//! surrounding air, and a head that looks around independently of the hull.
use serde::{Deserialize, Serialize};

use crate::core::math::{Vec3d, add_scaled, norm};
use crate::core::rigid::{Body, BodyShape};
use crate::core::vessel::Vessel;

pub const RADIUS: f64 = 0.5;
const MASS: f64 = 120.0;
/// How far the centre of mass sits below the hull's centre; the lever that rights the shuttle.
const BALLAST: f64 = 0.2;
const BOUNDARY_SAMPLES: usize = 300;
/// The eye sits this far above the hull's centre, still inside the hull.
const EYE_HEIGHT: f64 = 0.35;
/// Peak thruster acceleration (m/s²).
pub const THRUST: f64 = 14.0;
/// The flight assist brakes relative motion against the air on this time scale, which also caps
/// the cruising speed at `THRUST` times it.
const ASSIST_TAU: f64 = 0.35;
/// The gyros settle the hull's spin onto the requested one on this time scale.
const GYRO_TAU: f64 = 0.3;
const LOOK_RATE: f64 = 0.0022;
pub const ROLL_RATE: f64 = 1.6;
const MAX_PITCH: f64 = 1.55;

/// What the pilot asks of the shuttle, in world space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShuttleInput {
    /// Requested acceleration; zero lets the flight assist brake.
    pub thrust: Vec3d,
    /// Requested hull spin relative to the surrounding air.
    pub spin: Vec3d,
}

/// Where the head points relative to the hull.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct Look {
    pub yaw: f64,
    pub pitch: f64,
}

impl Look {
    /// Mouse movement in pixels turns the head.
    pub fn turn(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx * LOOK_RATE;
        self.pitch = (self.pitch - dy * LOOK_RATE).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// The head direction that faces `target` from an upright hull whose eye is at `eye`.
    pub fn facing(eye: Vec3d, target: Vec3d) -> Look {
        let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let flat = (d[0] * d[0] + d[2] * d[2]).sqrt();
        Look {
            yaw: (-d[0]).atan2(-d[2]),
            pitch: d[1].atan2(flat).clamp(-MAX_PITCH, MAX_PITCH),
        }
    }
}

pub fn shape() -> BodyShape {
    BodyShape::weighted_sphere(RADIUS, MASS, BALLAST, BOUNDARY_SAMPLES)
}

/// The eye's offset from the centre of mass, in the hull's frame.
pub fn eye_offset() -> Vec3d {
    [0.0, BALLAST + EYE_HEIGHT, 0.0]
}

/// World position of the eye.
pub fn eye(body: &Body) -> Vec3d {
    body.to_world(&eye_offset())
}

/// Centre of mass of an upright hull whose eye is at `eye`.
pub fn centre_for_eye(eye: Vec3d) -> Vec3d {
    let o = eye_offset();
    [eye[0] - o[0], eye[1] - o[1], eye[2] - o[2]]
}

/// One substep of thrusters, flight assist and gyros. The assist brakes toward the surrounding
/// air, or toward the stars where there is none.
pub fn drive(body: &mut Body, input: &ShuttleInput, vessel: &impl Vessel, dt: f64) {
    let air = vessel.air_velocity(body.p);
    let rest = air.unwrap_or([0.0; 3]);
    let thrust = limited(input.thrust, THRUST);
    add_scaled(&mut body.v, &thrust, dt);
    let slip = [
        rest[0] - body.v[0],
        rest[1] - body.v[1],
        rest[2] - body.v[2],
    ];
    let brake = limited(slip.map(|s| s / ASSIST_TAU), THRUST);
    add_scaled(&mut body.v, &brake, dt);

    let air_spin = if air.is_some() {
        vessel.angular_velocity()
    } else {
        [0.0; 3]
    };
    let k = (dt / GYRO_TAU).min(1.0);
    for ((w, air), wanted) in body.w.iter_mut().zip(air_spin).zip(input.spin) {
        *w += (air + wanted - *w) * k;
    }
}

/// `v` scaled down to at most `max` long.
fn limited(v: Vec3d, max: f64) -> Vec3d {
    let len = norm(&v);
    if len > max {
        v.map(|x| x * max / len)
    } else {
        v
    }
}
