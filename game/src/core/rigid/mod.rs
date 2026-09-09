//! Rigid bodies: shared shapes, per-body state, impulse-based contacts with the vessel, and the
//! step that applies what the water did, drags them through the air and resolves their contacts.
mod body;
mod contacts;

pub use body::{Body, BodyShape, Ground, Hull, HullSphere, WaterCoupling};
pub use contacts::collide_vessel;

use crate::core::math::{add_scaled, cross, quat_from_rotation_vector, quat_rotate};
use crate::core::units::{MetresPerSecond, RadiansPerSecond, Seconds};
use crate::core::vessel::Vessel;

#[derive(Clone, Debug, PartialEq)]
pub struct BodyParams {
    pub air: bool,
    /// Time constant for the vessel's air to drag free bodies along with its walls.
    pub air_tau: Seconds,
    /// Safety clamps on speed and spin, well above anything the vessel's walls reach.
    pub max_speed: MetresPerSecond,
    pub max_spin: RadiansPerSecond,
}

impl Default for BodyParams {
    fn default() -> Self {
        BodyParams {
            air: true,
            air_tau: Seconds(12.0),
            max_speed: MetresPerSecond(40.0),
            max_spin: RadiansPerSecond(25.0),
        }
    }
}

/// One substep: apply the water's impulses (if any arrived), carry every body through the
/// frame's turning, apply the air's drag, integrate, and resolve contacts with the vessel.
///
/// The frame's turning is felt exactly: a body's motion among the stars over the substep is
/// straight and even, and it is put back into the frame where the frame has turned to by the
/// end of the substep, so a body flying freely keeps its motion among the stars to rounding
/// error whatever the frame does, and what a body at rest on a wall gains is only the wall's
/// own acceleration, which the contact takes back. A body's spin, measured in the frame,
/// turns with the frame and falls as the frame spins up, likewise exactly. The air, at rest in
/// the frame, pushes on the hull's centre of pressure and stills its spin, so a ballasted hull
/// that drifts through it turns ballast-first, and where there is no air nothing turns it at
/// all. The safety clamps on speed and spin are on the body's motion among the stars, never on
/// its motion in the frame, so nothing the frame does can bring them down on a body that is
/// only at rest.
pub fn step(
    dt: f64,
    vessel: &impl Vessel,
    shapes: &[BodyShape],
    bodies: &mut [Body],
    water: Option<&[WaterCoupling]>,
    params: &BodyParams,
) {
    let air_k = if params.air {
        dt / params.air_tau.0 as f64
    } else {
        0.0
    };
    let spin = vessel.angular_velocity();
    let spin_mag = (spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]).sqrt();
    let spin_up = vessel.angular_acceleration();
    let pivot = vessel.pivot();
    // the frame's spin at the start of the substep, and its turn over it
    let mut spin_before = spin;
    add_scaled(&mut spin_before, &spin_up, -dt);
    let mut turn = spin.map(|s| s * dt);
    add_scaled(&mut turn, &spin_up, -0.5 * dt * dt);
    let carried = quat_from_rotation_vector(&turn.map(|t| -t));
    for (i, b) in bodies.iter_mut().enumerate() {
        if let Some(impulse) = water.and_then(|w| w.get(i)) {
            // water may push a body with a few times the vessel's artificial gravity, no more
            let shape = &shapes[b.shape];
            let max_accel = 20.0 + 4.0 * spin_mag * spin_mag * shape.reach();
            b.couple(impulse, max_accel);
        }
        // straight flight among the stars, put back into the frame where it has turned to
        let mut from_pivot = b.p;
        add_scaled(&mut from_pivot, &pivot, -1.0);
        let mut among_stars = cross(&spin_before, &from_pivot);
        add_scaled(&mut among_stars, &b.v, 1.0);
        add_scaled(&mut from_pivot, &among_stars, dt);
        let landed = quat_rotate(&carried, &from_pivot);
        b.p = pivot;
        add_scaled(&mut b.p, &landed, 1.0);
        b.v = quat_rotate(&carried, &among_stars);
        add_scaled(&mut b.v, &cross(&spin, &landed), -1.0);
        b.w = quat_rotate(&carried, &b.w);
        add_scaled(&mut b.w, &spin_up, -dt);
        if air_k > 0.0 && vessel.has_air(b.p) {
            let at = b.to_world(&shapes[b.shape].hull.centre_of_pressure());
            let hull = b.point_velocity(&at);
            let impulse = hull.map(|h| -h * air_k / b.inv_m);
            b.apply_impulse(&impulse, &at);
            relax(&mut b.w, &[0.0; 3], air_k);
        }
        let rest = vessel.star_velocity(b.p);
        b.clamp(params.max_speed, params.max_spin, &rest, &spin);
        b.turn(dt, &spin_up);
    }
    for b in bodies.iter_mut() {
        b.ground = None;
        if b.solid {
            collide_vessel(b, &shapes[b.shape], vessel, dt);
        }
    }
}

/// Move `v` a fraction `k` of the way toward `target`.
fn relax(v: &mut [f64; 3], target: &[f64; 3], k: f64) {
    for (x, t) in v.iter_mut().zip(target) {
        *x += (t - *x) * k;
    }
}
