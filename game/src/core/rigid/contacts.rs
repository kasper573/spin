//! Sequential impulses with Coulomb friction against the moving vessel walls and other bodies.
use super::{Body, BodyShape, Collider};
use crate::core::math::{Vec3d, add_scaled, cross, dot, mat3mul, norm};
use crate::core::vessel::Vessel;

pub fn collide_vessel(
    body: &mut Body,
    shape: &BodyShape,
    vessel: &impl Vessel,
    restitution: f64,
    friction: f64,
) {
    match shape.collider {
        Collider::Box { .. } => {
            for _pass in 0..2 {
                for lp in &shape.points {
                    let mut wp = body.to_world(lp);
                    for pen in vessel.penetrations(wp).iter() {
                        let wall = vessel.wall_velocity(wp);
                        resolve_wall_contact(body, &wp, &pen, &wall, restitution, friction, true);
                        wp = body.to_world(lp);
                    }
                }
            }
        }
        Collider::Sphere { radius, centre } => {
            for _pass in 0..2 {
                let c = body.to_world(&centre);
                for pen in vessel.sphere_penetrations(c, radius).iter() {
                    let at = [
                        c[0] - pen.normal[0] * radius,
                        c[1] - pen.normal[1] * radius,
                        c[2] - pen.normal[2] * radius,
                    ];
                    let wall = vessel.wall_velocity(at);
                    resolve_wall_contact(body, &at, &pen, &wall, restitution, friction, false);
                }
            }
        }
    }
}

/// Body `a` against body `b`. Called twice with the roles swapped.
pub fn collide_pair(
    bodies: &mut [Body],
    a: usize,
    b: usize,
    shapes: &[BodyShape],
    restitution: f64,
    friction: f64,
) {
    let (ca, cb) = (
        shapes[bodies[a].shape].collider,
        shapes[bodies[b].shape].collider,
    );
    match (ca, cb) {
        (Collider::Box { .. }, Collider::Box { half }) => {
            for lp in &shapes[bodies[a].shape].points {
                let wp = bodies[a].to_world(lp);
                if let Some((n, pen)) = box_surface(&bodies[b], half, &wp) {
                    resolve_pair_contact(bodies, a, b, &wp, &n, pen, restitution, friction);
                }
            }
        }
        (Collider::Sphere { radius, centre }, Collider::Box { half }) => {
            let c = bodies[a].to_world(&centre);
            let l = bodies[b].to_local(&c);
            let clamped = [
                l[0].clamp(-half[0], half[0]),
                l[1].clamp(-half[1], half[1]),
                l[2].clamp(-half[2], half[2]),
            ];
            let closest = bodies[b].to_world(&clamped);
            let d = [c[0] - closest[0], c[1] - closest[1], c[2] - closest[2]];
            let dist = norm(&d);
            if dist < radius && dist > 1e-9 {
                let n = [-d[0] / dist, -d[1] / dist, -d[2] / dist];
                resolve_pair_contact(
                    bodies,
                    a,
                    b,
                    &closest,
                    &n,
                    radius - dist,
                    restitution,
                    friction,
                );
            }
        }
        (Collider::Box { .. }, Collider::Sphere { .. })
        | (Collider::Sphere { .. }, Collider::Sphere { .. }) => {}
    }
}

/// Inverse effective mass of the body at lever arm r along direction n.
#[inline]
fn eff_mass(body: &Body, r: &Vec3d, n: &Vec3d) -> f64 {
    let t = cross(r, n);
    let t2 = mat3mul(&body.iw, &t);
    let t3 = cross(&t2, r);
    body.inv_m + dot(n, &t3)
}

/// Where a world point sits inside a box body: the nearest face's outward normal and the depth.
fn box_surface(body: &Body, half: Vec3d, wp: &Vec3d) -> Option<(Vec3d, f64)> {
    let l = body.to_local(wp);
    let (ax, ay, az) = (l[0].abs(), l[1].abs(), l[2].abs());
    let [hx, hy, hz] = half;
    if ax >= hx || ay >= hy || az >= hz {
        return None;
    }
    let (px, py, pz) = (hx - ax, hy - ay, hz - az);
    let (k, pen, sgn) = if py <= px && py <= pz {
        (1, py, l[1].signum())
    } else if px <= pz {
        (0, px, l[0].signum())
    } else {
        (2, pz, l[2].signum())
    };
    let m = body.m;
    Some(([m[k] * sgn, m[3 + k] * sgn, m[6 + k] * sgn], pen))
}

/// Push `a` out of `b` along `n` (pointing from b toward a) at world point `wp`.
#[allow(clippy::too_many_arguments)]
fn resolve_pair_contact(
    bodies: &mut [Body],
    a: usize,
    b: usize,
    wp: &Vec3d,
    n: &Vec3d,
    pen: f64,
    restitution: f64,
    friction: f64,
) {
    let ra = [
        wp[0] - bodies[a].p[0],
        wp[1] - bodies[a].p[1],
        wp[2] - bodies[a].p[2],
    ];
    let rb = [
        wp[0] - bodies[b].p[0],
        wp[1] - bodies[b].p[1],
        wp[2] - bodies[b].p[2],
    ];
    let relative = |bodies: &[Body]| {
        let va = bodies[a].point_velocity(wp);
        let vb = bodies[b].point_velocity(wp);
        [va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]]
    };
    let rv = relative(bodies);
    let vn = dot(&rv, n);
    if vn < 0.0 {
        let kk = eff_mass(&bodies[a], &ra, n) + eff_mass(&bodies[b], &rb, n);
        let rest = if vn < -0.5 { restitution } else { 0.0 };
        let j = -(1.0 + rest) * vn / kk;
        let jv = [j * n[0], j * n[1], j * n[2]];
        bodies[a].apply_impulse(&jv, wp);
        bodies[b].apply_impulse(&[-jv[0], -jv[1], -jv[2]], wp);
        let rv = relative(bodies);
        let vn2 = dot(&rv, n);
        let t = [rv[0] - vn2 * n[0], rv[1] - vn2 * n[1], rv[2] - vn2 * n[2]];
        let vt = norm(&t);
        if vt > 1e-6 {
            let t = [t[0] / vt, t[1] / vt, t[2] / vt];
            let kt = eff_mass(&bodies[a], &ra, &t) + eff_mass(&bodies[b], &rb, &t);
            let jt = (friction * j).min(vt / kt);
            let jtv = [-jt * t[0], -jt * t[1], -jt * t[2]];
            bodies[a].apply_impulse(&jtv, wp);
            bodies[b].apply_impulse(&[-jtv[0], -jtv[1], -jtv[2]], wp);
        }
    }
    let c = pen.min(0.05) * 0.25;
    add_scaled(&mut bodies[a].p, n, c);
    add_scaled(&mut bodies[b].p, n, -c);
}

/// `friction_at_hull` applies the tangential impulse where the wall touches (a box can roll over);
/// otherwise it goes through the centre of mass and the body slides.
fn resolve_wall_contact(
    body: &mut Body,
    wp: &Vec3d,
    pen: &crate::core::vessel::Penetration,
    wall: &Vec3d,
    restitution: f64,
    friction: f64,
    friction_at_hull: bool,
) {
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
        body.apply_impulse(&[j * n[0], j * n[1], j * n[2]], wp);
        let rv = relative(body);
        let vn2 = dot(&rv, n);
        let t = [rv[0] - vn2 * n[0], rv[1] - vn2 * n[1], rv[2] - vn2 * n[2]];
        let vt = norm(&t);
        if vt > 1e-6 {
            let t = [t[0] / vt, t[1] / vt, t[2] / vt];
            if friction_at_hull {
                let kt = eff_mass(body, &r, &t);
                let jt = (friction * j).min(vt / kt);
                body.apply_impulse(&[-jt * t[0], -jt * t[1], -jt * t[2]], wp);
            } else {
                let jt = (friction * j).min(vt / body.inv_m);
                body.apply_central_impulse(&[-jt * t[0], -jt * t[1], -jt * t[2]]);
            }
        }
    }
    add_scaled(&mut body.p, n, pen.depth.min(0.05) * 0.4);
}
