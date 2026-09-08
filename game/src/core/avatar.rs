//! The viewer's body: a ballasted sphere the height of a person, with legs that walk, run and
//! jump off whatever ground it stands on, thrusters and a flight assist for when it is a ghost,
//! gyros that keep it upright, and a head that looks around independently of the hull.
//!
//! Standing on the ground, the only force the body can exert on the world is through its feet:
//! the legs push along the ground up to what friction allows, a jump is one push along the
//! ground's normal, and in the air the body coasts. Ground here is whatever wall of the vessel
//! it touched last, so a ring's floor works as well as a flat one.
use serde::{Deserialize, Serialize};

use crate::core::math::{Vec3d, add_scaled, dot, norm};
use crate::core::rigid::{Body, BodyShape};
use crate::core::units::{EARTH_GRAVITY, Metres, MetresPerSecond};
use crate::core::vessel::Vessel;

pub const RADIUS: f64 = 0.5;
pub const MASS: f64 = 80.0;
/// How far the centre of mass sits below the hull's centre; the lever that rights the body.
const BALLAST: f64 = 0.2;
const BOUNDARY_SAMPLES: usize = 300;
/// Where the eye sits above the ground when standing.
pub const EYE_HEIGHT: Metres = Metres(1.7);
/// Ground speed the legs settle on when asked to walk, and when asked to run.
pub const WALK_SPEED: MetresPerSecond = MetresPerSecond(1.5);
pub const RUN_SPEED: MetresPerSecond = MetresPerSecond(4.0);
/// How high a jump lifts the centre of mass under one g.
pub const JUMP_HEIGHT: Metres = Metres(0.4);
/// Time scale on which the legs bring the body to the speed it wants.
const LEG_TAU: f64 = 0.25;
/// Coulomb friction between the feet and the ground, the most the legs can push per unit of the
/// ground's support.
const LEG_GRIP: f64 = 0.8;
/// Peak thruster acceleration of a ghost (m/s²).
pub const THRUST: f64 = 14.0;
/// The flight assist brakes relative motion against the air on this time scale, which also caps
/// the cruising speed at `THRUST` times it.
const ASSIST_TAU: f64 = 0.35;
/// The gyros settle the hull's spin onto the requested one on this time scale.
const GYRO_TAU: f64 = 0.3;
const LOOK_RATE: f64 = 0.0022;
pub const ROLL_RATE: f64 = 1.6;
const MAX_PITCH: f64 = 1.55;

/// What the pilot asks of the body, in world space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AvatarInput {
    /// Requested motion, at most unit length: the direction and fraction of full speed to walk,
    /// or of full thrust for a ghost.
    pub motion: Vec3d,
    pub run: bool,
    pub jump: bool,
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
    let mut shape = BodyShape::weighted_sphere(RADIUS, MASS, BALLAST, BOUNDARY_SAMPLES);
    shape.friction = 0.0;
    shape.restitution = 0.0;
    shape
}

/// Height of the centre of mass above the ground when standing upright.
pub fn standing_height() -> f64 {
    RADIUS - BALLAST
}

/// The eye's offset from the centre of mass, in the hull's frame.
pub fn eye_offset() -> Vec3d {
    [0.0, EYE_HEIGHT.0 as f64 - standing_height(), 0.0]
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

/// Speed a jump leaves the ground with: enough to rise `JUMP_HEIGHT` under one g.
pub fn jump_speed() -> f64 {
    (2.0 * EARTH_GRAVITY.0 * JUMP_HEIGHT.0 as f64).sqrt()
}

/// One substep of the legs or the thrusters, then the gyros.
pub fn drive(body: &mut Body, input: &AvatarInput, vessel: &impl Vessel, dt: f64) {
    if body.solid {
        walk(body, input, vessel, dt);
    } else {
        fly(body, input, vessel, dt);
    }
    let air_spin = if vessel.air_velocity(body.p).is_some() {
        vessel.angular_velocity()
    } else {
        [0.0; 3]
    };
    let k = (dt / GYRO_TAU).min(1.0);
    for ((w, air), wanted) in body.w.iter_mut().zip(air_spin).zip(input.spin) {
        *w += (air + wanted - *w) * k;
    }
}

/// Legs: while the body stands on something, push along the ground until the feet move over it
/// at the wanted speed, within what friction allows, and jump off it. In the air nothing can be
/// done.
fn walk(body: &mut Body, input: &AvatarInput, vessel: &impl Vessel, dt: f64) {
    let Some(ground) = body.ground else {
        return;
    };
    let n = ground.normal;
    let ground_velocity = vessel.wall_velocity(ground.point);
    let speed = if input.run { RUN_SPEED } else { WALK_SPEED }.0 as f64;
    let along = flatten(&limited(input.motion, 1.0), &n);
    let wanted = [
        ground_velocity[0] + along[0] * speed,
        ground_velocity[1] + along[1] * speed,
        ground_velocity[2] + along[2] * speed,
    ];
    let feet = body.point_velocity(&ground.point);
    let slip = flatten(
        &[
            wanted[0] - feet[0],
            wanted[1] - feet[1],
            wanted[2] - feet[2],
        ],
        &n,
    );
    let grip = LEG_GRIP * ground.support.0 * body.inv_m;
    let push = limited(slip.map(|s| s / LEG_TAU), grip);
    add_scaled(&mut body.v, &push, dt);
    if input.jump {
        let rising = dot(&feet, &n) - dot(&ground_velocity, &n);
        add_scaled(&mut body.v, &n, (jump_speed() - rising).max(0.0));
    }
}

/// Thrusters and the flight assist, which brakes toward the surrounding air, or toward the stars
/// where there is none.
fn fly(body: &mut Body, input: &AvatarInput, vessel: &impl Vessel, dt: f64) {
    let rest = vessel.air_velocity(body.p).unwrap_or([0.0; 3]);
    let thrust = limited(input.motion.map(|m| m * THRUST), THRUST);
    add_scaled(&mut body.v, &thrust, dt);
    let slip = [
        rest[0] - body.v[0],
        rest[1] - body.v[1],
        rest[2] - body.v[2],
    ];
    let brake = limited(slip.map(|s| s / ASSIST_TAU), THRUST);
    add_scaled(&mut body.v, &brake, dt);
}

/// `v` without its component along the unit normal `n`.
fn flatten(v: &Vec3d, n: &Vec3d) -> Vec3d {
    let d = dot(v, n);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
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
