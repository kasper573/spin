//! Sequential impulses with Coulomb friction against the moving vessel walls.
use super::{Body, BodyShape, Ground, Hull};
use crate::core::math::{Vec3d, add_scaled, cross, dot, mat3mul, norm};
use crate::core::units::Newtons;
use crate::core::vessel::Vessel;

/// Resolve the body's contacts with the vessel over a substep of `dt`, and remember the wall it
/// stood on as its ground.
pub fn collide_vessel(body: &mut Body, shape: &BodyShape, vessel: &impl Vessel, dt: f64) {
    let (restitution, friction) = (shape.restitution, shape.friction);
    let mut support = 0.0;
    let mut ground: Option<Ground> = None;
    let mut stand = |ground: &mut Option<Ground>, point: Vec3d, normal: Vec3d, impulse: f64| {
        support += impulse;
        if ground.is_none_or(|g| impulse > g.support.0) {
            *ground = Some(Ground {
                point,
                normal,
                support: Newtons(impulse),
            });
        }
    };
    let Hull { radius, centre } = shape.hull;
    for _pass in 0..2 {
        let c = body.to_world(&centre);
        for pen in vessel.sphere_penetrations(c, radius).iter() {
            let at = [
                c[0] - pen.normal[0] * radius,
                c[1] - pen.normal[1] * radius,
                c[2] - pen.normal[2] * radius,
            ];
            let wall = vessel.wall_velocity(at);
            let j = resolve_wall_contact(body, &at, &pen, &wall, restitution, friction);
            stand(&mut ground, at, pen.normal, j);
        }
    }
    body.ground = ground.map(|g| Ground {
        support: Newtons(support / dt),
        ..g
    });
}

/// Inverse effective mass of the body at lever arm r along direction n.
#[inline]
fn eff_mass(body: &Body, r: &Vec3d, n: &Vec3d) -> f64 {
    let t = cross(r, n);
    let t2 = mat3mul(&body.iw, &t);
    let t3 = cross(&t2, r);
    body.inv_m + dot(n, &t3)
}

/// The tangential impulse goes through the centre of mass, so the body slides rather than
/// rolls. Returns the normal impulse.
fn resolve_wall_contact(
    body: &mut Body,
    wp: &Vec3d,
    pen: &crate::core::vessel::Penetration,
    wall: &Vec3d,
    restitution: f64,
    friction: f64,
) -> f64 {
    let mut normal_impulse = 0.0;
    let n = &pen.normal;
    let r = [wp[0] - body.p[0], wp[1] - body.p[1], wp[2] - body.p[2]];
    let relative = |body: &Body| {
        let vp = body.point_velocity(wp);
        [vp[0] - wall[0], vp[1] - wall[1], vp[2] - wall[2]]
    };
    let rv = relative(body);
    let vn = dot(&rv, n);
    if vn < 0.0 {
        let kn = eff_mass(body, &r, n);
        let rest = if vn < -0.5 { restitution } else { 0.0 };
        let j = -(1.0 + rest) * vn / kn;
        normal_impulse = j;
        body.apply_impulse(&[j * n[0], j * n[1], j * n[2]], wp);
        let rv = relative(body);
        let vn2 = dot(&rv, n);
        let t = [rv[0] - vn2 * n[0], rv[1] - vn2 * n[1], rv[2] - vn2 * n[2]];
        let vt = norm(&t);
        if vt > 1e-6 {
            let t = [t[0] / vt, t[1] / vt, t[2] / vt];
            let jt = (friction * j).min(vt / body.inv_m);
            body.apply_central_impulse(&[-jt * t[0], -jt * t[1], -jt * t[2]]);
        }
    }
    add_scaled(&mut body.p, n, pen.depth.min(0.05) * 0.4);
    normal_impulse
}
