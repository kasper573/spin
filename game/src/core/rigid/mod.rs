//! Rigid bodies: shared shapes, per-body state, impulse-based contacts with the vessel, and the
//! step that applies what the water did, drags them through the air and resolves their contacts.
mod body;
mod contacts;

pub use body::{Body, BodyShape, Ground, Hull, HullSphere, WaterCoupling};
pub use contacts::collide_vessel;

use crate::core::math::{add_scaled, quat_from_rotation_vector, quat_rotate};
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

/// One substep: apply the water's impulses (if any arrived), the vessel's frame and the air's
/// drag, integrate, and resolve contacts with the vessel. The frame's turning is felt as the
/// Coriolis turn of the velocity, exactly, and then as its rest acceleration; in that order,
/// so that what the walls take back of the rest acceleration was never turned, and a body at
/// rest on them stays at rest. The air, at rest in the frame, pushes on the hull's centre of
/// pressure and stills its spin, so a ballasted hull that drifts through it turns
/// ballast-first, and where there is no air nothing turns it at all.
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
    let coriolis = quat_from_rotation_vector(&spin.map(|s| -2.0 * s * dt));
    for (i, b) in bodies.iter_mut().enumerate() {
        if let Some(impulse) = water.and_then(|w| w.get(i)) {
            // water may push a body with a few times the vessel's artificial gravity, no more
            let shape = &shapes[b.shape];
            let max_accel = 20.0 + 4.0 * spin_mag * spin_mag * shape.reach();
            b.couple(impulse, max_accel);
        }
        b.v = quat_rotate(&coriolis, &b.v);
        add_scaled(&mut b.v, &vessel.rest_acceleration(b.p), dt);
        if air_k > 0.0 && vessel.has_air(b.p) {
            let at = b.to_world(&shapes[b.shape].hull.centre_of_pressure());
            let hull = b.point_velocity(&at);
            let impulse = hull.map(|h| -h * air_k / b.inv_m);
            b.apply_impulse(&impulse, &at);
            relax(&mut b.w, &[0.0; 3], air_k);
        }
        b.integrate(dt, params.max_speed, params.max_spin);
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
