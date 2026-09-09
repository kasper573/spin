//! The viewer's body: a ballasted sphere the height of a person with twelve thrusters, one along
//! each axis of the hull and a pair for each way of turning it, legs that walk it over whatever
//! ground it stands on, and gyros that hold its attitude wherever the turning thrusters last
//! left it. The eye is fixed in the hull, so the body turns to look.
//!
//! The thrusters spool up and down rather than switch, so their levels are state the hull carries.
//! Standing on the ground, the horizontal thrust is the legs' orders: they push along the ground,
//! within what friction allows, until the feet move over it at walking speed in the direction
//! thrust. Thrust along the ground's normal acts as it is, so thrusting up harder than the ground
//! pulls lifts the body off it, and in the air every thruster acts as it is. A ghost adds a flight
//! assist that brakes it against the surrounding air. Ground is whatever wall of the vessel the
//! body touched last, so a ring's floor works as well as a flat one.
//!
//! The turning thrusters ask the gyros for a rate of turn, and the gyros, reaction wheels with
//! limited authority, hold the hull to the attitude those rates carry along, relative to the air
//! it is in. Nothing else steers it: the ballast rights the hull only through what pushes on the
//! hull, the ground, the water or the air, and the gyros then bring it back to where it was held.
//! Where there is no air and nothing to push on, only the pilot turns it.
//!
//! All of this happens in the vessel's own frame, where its walls and its air stand still.
use serde::{Deserialize, Serialize};

use crate::core::math::{
    Quatd, Vec3d, add_scaled, dot, norm, quat_between, quat_conjugate, quat_integrate, quat_mul,
    quat_rotate, rotation_vector,
};
use crate::core::rigid::{Body, BodyShape, Ground};
use crate::core::units::{EARTH_GRAVITY, Metres, MetresPerSecond, MetresPerSecondSquared, Seconds};
use crate::core::vessel::Vessel;

pub const RADIUS: f64 = 0.5;
pub const MASS: f64 = 80.0;
/// How far the centre of mass sits below the hull's centre; the lever that rights the body.
const BALLAST: f64 = 0.2;
/// Water the hull displaces (m³): little more than the person inside it, so it floats, but only
/// just, and the thrusters can push it under.
const DISPLACEMENT: f64 = 0.1;
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
/// A thruster equalized to a gravity pulls this many times it, so thrusting straight up from
/// the floor nets the rest upward.
pub const THRUST_OVER_GRAVITY: f64 = 1.78;
/// A thruster takes this long to go from idle to full, and as long to die down.
pub const SPOOL_TIME: Seconds = Seconds(0.3);
/// The flight assist brakes a ghost's motion relative to the air on this time scale, which also
/// caps its cruising speed at the thrusters' power times it.
const ASSIST_TAU: f64 = 0.35;
/// The gyros bring the hull back onto its held attitude on this time scale, and can turn it no
/// harder than this (rad/s²), so a hard knock still moves it before they answer.
const HOLD_TAU: f64 = 0.15;
const GYRO_AUTHORITY: f64 = 30.0;
/// Rate of turn (rad/s) a turning thruster asks of the gyros at full level.
pub const TURN_RATE: f64 = 1.6;
/// The mount of a turning thruster sits this far round the hull from straight ahead.
const TURNING_MOUNT: f64 = 0.7;

/// The twelve thrusters. The linear ones point along the axes of the hull; the roll pair sit at
/// the shoulders, each lifting its own side, the pitch pair at the forehead and chin and the
/// yaw pair at the cheeks, each pushing the face its way.
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
    PitchUp,
    PitchDown,
    YawLeft,
    YawRight,
}

impl Thruster {
    pub const ALL: [Thruster; 12] = [
        Thruster::Forward,
        Thruster::Back,
        Thruster::Left,
        Thruster::Right,
        Thruster::Up,
        Thruster::Down,
        Thruster::RollLeft,
        Thruster::RollRight,
        Thruster::PitchUp,
        Thruster::PitchDown,
        Thruster::YawLeft,
        Thruster::YawRight,
    ];

    /// Where the thruster sits on the hull in its frame (x right, y up, z back): a linear one on
    /// the side it pushes away from, a roll one at the shoulder it lifts, a pitch one above or
    /// below the face and a yaw one beside it.
    pub fn mount(self) -> Vec3d {
        let face = -RADIUS * TURNING_MOUNT;
        let beside = RADIUS * TURNING_MOUNT;
        match self {
            Thruster::RollLeft => [RADIUS, 0.0, 0.0],
            Thruster::RollRight => [-RADIUS, 0.0, 0.0],
            Thruster::PitchUp => [0.0, -beside, face],
            Thruster::PitchDown => [0.0, beside, face],
            Thruster::YawLeft => [beside, 0.0, face],
            Thruster::YawRight => [-beside, 0.0, face],
            linear => linear.direction().map(|d| -d * RADIUS),
        }
    }

    /// The direction a linear thruster pushes in the hull's frame (x right, y up, z back); the
    /// turning ones push nothing along an axis.
    pub fn direction(self) -> Vec3d {
        match self {
            Thruster::Forward => [0.0, 0.0, -1.0],
            Thruster::Back => [0.0, 0.0, 1.0],
            Thruster::Left => [-1.0, 0.0, 0.0],
            Thruster::Right => [1.0, 0.0, 0.0],
            Thruster::Up => [0.0, 1.0, 0.0],
            Thruster::Down => [0.0, -1.0, 0.0],
            _ => [0.0; 3],
        }
    }

    /// Whether the thruster turns the hull rather than pushing it along.
    pub fn turns(self) -> bool {
        self.direction() == [0.0; 3]
    }
}

/// The peak acceleration of a thruster equalized to this gravity.
pub fn equalized_thrust(gravity: MetresPerSecondSquared) -> MetresPerSecondSquared {
    MetresPerSecondSquared(gravity.0 * THRUST_OVER_GRAVITY)
}

/// How hard each thruster is firing, 0 to 1, in the order of `Thruster::ALL`, and how hard one
/// pushes at full.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Thrusters {
    levels: [f64; 12],
    pub power: MetresPerSecondSquared,
}

impl Default for Thrusters {
    fn default() -> Self {
        Thrusters {
            levels: [0.0; 12],
            power: equalized_thrust(EARTH_GRAVITY),
        }
    }
}

impl Thrusters {
    /// Idle thrusters of this power.
    pub fn with_power(power: MetresPerSecondSquared) -> Thrusters {
        Thrusters {
            levels: [0.0; 12],
            power,
        }
    }

    pub fn level(&self, thruster: Thruster) -> f64 {
        self.levels[thruster as usize]
    }

    pub fn levels(&self) -> [f64; 12] {
        self.levels
    }

    /// Move every thruster toward what the input asks of it, as fast as it can spool.
    pub fn spool(&mut self, input: &AvatarInput, dt: f64) {
        let rate = dt / SPOOL_TIME.0 as f64;
        for (level, wanted) in self.levels.iter_mut().zip(input.levels) {
            *level += (wanted.clamp(0.0, 1.0) - *level).clamp(-rate, rate);
        }
    }

    /// The linear thrusters' net push as a multiple of one thruster's, in the hull's frame.
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

    /// The pitch thrusters' net turn, positive lifting the face.
    pub fn pitch(&self) -> f64 {
        self.level(Thruster::PitchUp) - self.level(Thruster::PitchDown)
    }

    /// The yaw thrusters' net turn, positive turning the face to the left.
    pub fn yaw(&self) -> f64 {
        self.level(Thruster::YawLeft) - self.level(Thruster::YawRight)
    }
}

/// What the pilot asks of the body: how hard to fire each thruster, in the order of
/// `Thruster::ALL`. Opposed thrusters asked for together both fire and cancel, they do not
/// cancel the asking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvatarInput {
    /// Wanted level of each thruster, 0 to 1.
    pub levels: [f64; 12],
}

impl Default for AvatarInput {
    fn default() -> Self {
        AvatarInput { levels: [0.0; 12] }
    }
}

/// The gyros' reference: the attitude they hold the hull to, carried along by the turning
/// thrusters, by the air the hull is in and, standing, by the ground turning under its feet.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Gyros {
    pub held: Quatd,
    /// The ground's normal under the hull during the last substep it stood on one.
    pub footing: Option<Vec3d>,
}

impl Default for Gyros {
    fn default() -> Self {
        Gyros {
            held: [0.0, 0.0, 0.0, 1.0],
            footing: None,
        }
    }
}

impl Gyros {
    /// Gyros holding the hull as it is.
    pub fn holding(body: &Body) -> Gyros {
        Gyros {
            held: body.q,
            footing: None,
        }
    }
}

pub fn shape() -> BodyShape {
    let mut shape = BodyShape::weighted_sphere(RADIUS, MASS, BALLAST, BOUNDARY_SAMPLES)
        .displacing(DISPLACEMENT);
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
    gyros: &mut Gyros,
    input: &AvatarInput,
    vessel: &impl Vessel,
    dt: f64,
) {
    thrusters.spool(input, dt);
    let net = thrusters.net();
    let power = thrusters.power.0;
    let mut thrust = [0.0; 3];
    add_scaled(&mut thrust, &body.rotate(&[1.0, 0.0, 0.0]), net[0] * power);
    add_scaled(&mut thrust, &body.rotate(&[0.0, 1.0, 0.0]), net[1] * power);
    add_scaled(&mut thrust, &body.rotate(&[0.0, 0.0, 1.0]), net[2] * power);
    match (body.solid, body.ground) {
        (true, Some(ground)) => walk(body, &thrust, power, &ground, dt),
        (true, None) => add_scaled(&mut body.v, &thrust, dt),
        (false, _) => fly(body, &thrust, power, vessel, dt),
    }
    hold(body, thrusters, gyros, vessel, dt);
}

/// Legs: push along the ground until the feet move over it at walking speed in the direction
/// thrust, within what friction allows; thrust along the ground's normal acts as it is.
fn walk(body: &mut Body, thrust: &Vec3d, power: f64, ground: &Ground, dt: f64) {
    let n = ground.normal;
    let along = limited(flatten(thrust, &n).map(|t| t / power.max(1e-9)), 1.0);
    let speed = WALK_SPEED.0 as f64;
    let wanted = along.map(|a| a * speed);
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
fn fly(body: &mut Body, thrust: &Vec3d, power: f64, vessel: &impl Vessel, dt: f64) {
    let rest = if vessel.has_air(body.p) {
        [0.0; 3]
    } else {
        vessel.star_velocity(body.p)
    };
    add_scaled(&mut body.v, thrust, dt);
    let slip = [
        rest[0] - body.v[0],
        rest[1] - body.v[1],
        rest[2] - body.v[2],
    ];
    let brake = limited(slip.map(|s| s / ASSIST_TAU), power);
    add_scaled(&mut body.v, &brake, dt);
}

/// Gyros: carry the held attitude along with the ground turning under the feet, or with the air
/// when there is no ground, and as the turning thrusters ask; then turn the hull toward it, as
/// hard as they may. Standing, the local vertical turns as the hull walks round the vessel, and
/// the reference turns with it, so walking keeps the hull as upright as it stood.
fn hold(body: &mut Body, thrusters: &Thrusters, gyros: &mut Gyros, vessel: &impl Vessel, dt: f64) {
    let footing = match (body.solid, body.ground) {
        (true, Some(ground)) => Some(ground.normal),
        _ => None,
    };
    let mut rate = match (footing, gyros.footing) {
        (Some(now), Some(last)) => rotation_vector(&quat_between(&last, &now)).map(|c| c / dt),
        _ if vessel.has_air(body.p) => [0.0; 3],
        _ => vessel.angular_velocity().map(|w| -w),
    };
    gyros.footing = footing;
    let held = gyros.held;
    let right = quat_rotate(&held, &[1.0, 0.0, 0.0]);
    let up = quat_rotate(&held, &[0.0, 1.0, 0.0]);
    let back = quat_rotate(&held, &[0.0, 0.0, 1.0]);
    add_scaled(&mut rate, &right, thrusters.pitch() * TURN_RATE);
    add_scaled(&mut rate, &up, thrusters.yaw() * TURN_RATE);
    add_scaled(&mut rate, &back, thrusters.roll() * TURN_RATE);
    gyros.held = quat_integrate(&held, &rate, dt);
    let lean = rotation_vector(&quat_mul(&gyros.held, &quat_conjugate(&body.q)));
    let mut wanted = rate;
    add_scaled(&mut wanted, &lean, 1.0 / HOLD_TAU);
    let change = limited(
        [
            wanted[0] - body.w[0],
            wanted[1] - body.w[1],
            wanted[2] - body.w[2],
        ],
        GYRO_AUTHORITY * dt,
    );
    add_scaled(&mut body.w, &change, 1.0);
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
