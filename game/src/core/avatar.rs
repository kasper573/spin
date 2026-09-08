//! The viewer's body: a ballasted sphere the height of a person with eight thrusters, one along
//! each axis of the head's level frame and one at each shoulder for roll, legs that walk it over
//! whatever ground it stands on, gyros that keep it upright, and a head that looks around
//! independently of the hull.
//!
//! The thrusters spool up and down rather than switch, so their levels are state the hull carries.
//! Standing on the ground, the horizontal thrust is the legs' orders: they push along the ground,
//! within what friction allows, until the feet move over it at walking speed in the direction
//! thrust. Thrust along the ground's normal acts as it is, so thrusting up harder than the ground
//! pulls lifts the body off it, and in the air every thruster acts as it is. A ghost adds a flight
//! assist that brakes it against the surrounding air. Ground is whatever wall of the vessel the
//! body touched last, so a ring's floor works as well as a flat one.
use serde::{Deserialize, Serialize};

use crate::core::math::{Vec3d, add_scaled, dot, norm};
use crate::core::rigid::{Body, BodyShape, Ground};
use crate::core::units::{Metres, MetresPerSecond, MetresPerSecondSquared, Seconds};
use crate::core::vessel::Vessel;

pub const RADIUS: f64 = 0.5;
pub const MASS: f64 = 80.0;
/// How far the centre of mass sits below the hull's centre; the lever that rights the body.
const BALLAST: f64 = 0.2;
const BOUNDARY_SAMPLES: usize = 300;
/// Where the eye sits above the ground when standing.
pub const EYE_HEIGHT: Metres = Metres(1.7);
/// Ground speed the legs settle on when thrust along the ground.
pub const WALK_SPEED: MetresPerSecond = MetresPerSecond(1.5);
/// Time scale on which the legs bring the body to the speed it wants.
const LEG_TAU: f64 = 0.25;
/// Coulomb friction between the feet and the ground, the most the legs can push per unit of the
/// ground's support.
const LEG_GRIP: f64 = 0.8;
/// Peak acceleration of each thruster: 1.78 times the standing gravity, so thrusting straight up
/// from the floor nets 0.78 g upward.
pub const THRUST: MetresPerSecondSquared = MetresPerSecondSquared(17.5);
/// A thruster takes this long to go from idle to full, and as long to die down.
pub const SPOOL_TIME: Seconds = Seconds(0.3);
/// The flight assist brakes a ghost's motion relative to the air on this time scale, which also
/// caps its cruising speed at `THRUST` times it.
const ASSIST_TAU: f64 = 0.35;
/// The gyros settle the hull's spin onto the requested one on this time scale.
const GYRO_TAU: f64 = 0.3;
const LOOK_RATE: f64 = 0.0022;
/// Roll rate (rad/s) the roll thrusters ask of the gyros at full level.
pub const ROLL_RATE: f64 = 1.6;
const MAX_PITCH: f64 = 1.55;

/// The eight thrusters. The linear ones point along the axes of the head's level frame; the roll
/// pair sit at the shoulders, each lifting its own side.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thruster {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
    RollLeft,
    RollRight,
}

impl Thruster {
    pub const ALL: [Thruster; 8] = [
        Thruster::Forward,
        Thruster::Back,
        Thruster::Left,
        Thruster::Right,
        Thruster::Up,
        Thruster::Down,
        Thruster::RollLeft,
        Thruster::RollRight,
    ];

    /// The direction a linear thruster pushes in the level frame (x right, y up, z back); the
    /// roll pair push nothing along an axis.
    pub fn direction(self) -> Vec3d {
        match self {
            Thruster::Forward => [0.0, 0.0, -1.0],
            Thruster::Back => [0.0, 0.0, 1.0],
            Thruster::Left => [-1.0, 0.0, 0.0],
            Thruster::Right => [1.0, 0.0, 0.0],
            Thruster::Up => [0.0, 1.0, 0.0],
            Thruster::Down => [0.0, -1.0, 0.0],
            Thruster::RollLeft | Thruster::RollRight => [0.0; 3],
        }
    }
}

/// How hard each thruster is firing, 0 to 1, in the order of `Thruster::ALL`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct Thrusters {
    levels: [f64; 8],
}

impl Thrusters {
    pub fn level(&self, thruster: Thruster) -> f64 {
        self.levels[thruster as usize]
    }

    pub fn levels(&self) -> [f64; 8] {
        self.levels
    }

    /// Move every thruster toward what the input asks of it, as fast as it can spool.
    pub fn spool(&mut self, input: &AvatarInput, dt: f64) {
        let rate = dt / SPOOL_TIME.0 as f64;
        for (level, wanted) in self.levels.iter_mut().zip(input.levels) {
            *level += (wanted.clamp(0.0, 1.0) - *level).clamp(-rate, rate);
        }
    }

    /// The linear thrusters' net push as a multiple of one thruster's, in the level frame.
    pub fn net(&self) -> Vec3d {
        let mut net = [0.0; 3];
        for thruster in Thruster::ALL {
            add_scaled(&mut net, &thruster.direction(), self.level(thruster));
        }
        net
    }

    /// The roll thrusters' net turn, positive lifting the right shoulder.
    pub fn roll(&self) -> f64 {
        self.level(Thruster::RollLeft) - self.level(Thruster::RollRight)
    }
}

/// What the pilot asks of the body: how hard to fire each thruster, and in which frame. Opposed
/// thrusters asked for together both fire and cancel, they do not cancel the asking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvatarInput {
    /// The head's level frame in world space: unit right, up and back.
    pub frame: [Vec3d; 3],
    /// Wanted level of each thruster, 0 to 1, in the order of `Thruster::ALL`.
    pub levels: [f64; 8],
}

impl Default for AvatarInput {
    fn default() -> Self {
        AvatarInput {
            frame: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            levels: [0.0; 8],
        }
    }
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

/// One substep: spool the thrusters, let them and the legs act, then the gyros.
pub fn drive(
    body: &mut Body,
    thrusters: &mut Thrusters,
    input: &AvatarInput,
    vessel: &impl Vessel,
    dt: f64,
) {
    thrusters.spool(input, dt);
    let net = thrusters.net();
    let [right, up, back] = input.frame;
    let mut thrust = [0.0; 3];
    add_scaled(&mut thrust, &right, net[0] * THRUST.0);
    add_scaled(&mut thrust, &up, net[1] * THRUST.0);
    add_scaled(&mut thrust, &back, net[2] * THRUST.0);
    match (body.solid, body.ground) {
        (true, Some(ground)) => walk(body, &thrust, &ground, vessel, dt),
        (true, None) => add_scaled(&mut body.v, &thrust, dt),
        (false, _) => fly(body, &thrust, vessel, dt),
    }
    let air_spin = if vessel.air_velocity(body.p).is_some() {
        vessel.angular_velocity()
    } else {
        [0.0; 3]
    };
    let roll = thrusters.roll() * ROLL_RATE;
    let k = (dt / GYRO_TAU).min(1.0);
    for ((w, air), axis) in body.w.iter_mut().zip(air_spin).zip(back) {
        *w += (air + axis * roll - *w) * k;
    }
}

/// Legs: push along the ground until the feet move over it at walking speed in the direction
/// thrust, within what friction allows; thrust along the ground's normal acts as it is.
fn walk(body: &mut Body, thrust: &Vec3d, ground: &Ground, vessel: &impl Vessel, dt: f64) {
    let n = ground.normal;
    let ground_velocity = vessel.wall_velocity(ground.point);
    let along = limited(flatten(thrust, &n).map(|t| t / THRUST.0), 1.0);
    let speed = WALK_SPEED.0 as f64;
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
    add_scaled(&mut body.v, &n, dot(thrust, &n) * dt);
}

/// Thrusters and the flight assist, which brakes toward the surrounding air, or toward the stars
/// where there is none.
fn fly(body: &mut Body, thrust: &Vec3d, vessel: &impl Vessel, dt: f64) {
    let rest = vessel.air_velocity(body.p).unwrap_or([0.0; 3]);
    add_scaled(&mut body.v, thrust, dt);
    let slip = [
        rest[0] - body.v[0],
        rest[1] - body.v[1],
        rest[2] - body.v[2],
    ];
    let brake = limited(slip.map(|s| s / ASSIST_TAU), THRUST.0);
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
