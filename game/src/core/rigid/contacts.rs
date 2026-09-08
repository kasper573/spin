//! Sequential impulses with Coulomb friction against the moving vessel walls and other bodies.
use super::{Body, BoxShape};
use crate::core::math::{Vec3d, add_scaled, cross, dot, mat3mul};
use crate::core::vessel::Vessel;

pub fn collide_vessel(
    body: &mut Body,
    shape: &BoxShape,
    vessel: &impl Vessel,
    restitution: f64,
    friction: f64,
) {
    for _pass in 0..2 {
        for lp in &shape.points {
            let mut wp = body.to_world(lp);
            for pen in vessel.penetrations(wp).iter() {
                let wall = vessel.wall_velocity(wp);
                resolve_wall_contact(
                    body,
                    &wp,
                    &pen.normal,
                    pen.depth,
                    &wall,
                    restitution,
                    friction,
                );
                wp = body.to_world(lp);
            }
        }
    }
}

/// Sample points of body `a` against the box of body `b`. Called twice with the roles swapped.
pub fn collide_pair(
    bodies: &mut [Body],
    a: usize,
    b: usize,
    shape: &BoxShape,
    restitution: f64,
    friction: f64,
) {
    for lp in &shape.points {
        let wp = bodies[a].to_world(lp);
        let l = bodies[b].to_local(&wp);
        let (ax, ay, az) = (l[0].abs(), l[1].abs(), l[2].abs());
        let [hx, hy, hz] = bodies[b].half;
        if ax >= hx || ay >= hy || az >= hz {
            continue;
        }
        let (px, py, pz) = (hx - ax, hy - ay, hz - az);
        let (k, pen, sgn) = if py <= px && py <= pz {
            (1, py, if l[1] >= 0.0 { 1.0 } else { -1.0 })
        } else if px <= pz {
            (0, px, if l[0] >= 0.0 { 1.0 } else { -1.0 })
        } else {
            (2, pz, if l[2] >= 0.0 { 1.0 } else { -1.0 })
        };
        let m = bodies[b].m;
        let n = [m[k] * sgn, m[3 + k] * sgn, m[6 + k] * sgn];
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
        let va = bodies[a].point_velocity(&wp);
        let vb = bodies[b].point_velocity(&wp);
        let rv = [va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]];
        let vn = dot(&rv, &n);
        if vn < 0.0 {
            let kk = eff_mass(&bodies[a], &ra, &n) + eff_mass(&bodies[b], &rb, &n);
            let rest = if vn < -0.5 { restitution } else { 0.0 };
            let j = -(1.0 + rest) * vn / kk;
            let jv = [j * n[0], j * n[1], j * n[2]];
            bodies[a].apply_impulse(&jv, &wp);
            bodies[b].apply_impulse(&[-jv[0], -jv[1], -jv[2]], &wp);
            let va = bodies[a].point_velocity(&wp);
            let vb = bodies[b].point_velocity(&wp);
            let rv = [va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]];
            let vn2 = dot(&rv, &n);
            let t = [rv[0] - vn2 * n[0], rv[1] - vn2 * n[1], rv[2] - vn2 * n[2]];
            let vt = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if vt > 1e-6 {
                let t = [t[0] / vt, t[1] / vt, t[2] / vt];
                let kt = eff_mass(&bodies[a], &ra, &t) + eff_mass(&bodies[b], &rb, &t);
                let jt = (friction * j).min(vt / kt);
                let jtv = [-jt * t[0], -jt * t[1], -jt * t[2]];
                bodies[a].apply_impulse(&jtv, &wp);
                bodies[b].apply_impulse(&[-jtv[0], -jtv[1], -jtv[2]], &wp);
            }
        }
        let c = pen.min(0.05) * 0.25;
        add_scaled(&mut bodies[a].p, &n, c);
        add_scaled(&mut bodies[b].p, &n, -c);
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

fn resolve_wall_contact(
    body: &mut Body,
    wp: &Vec3d,
    n: &Vec3d,
    pen: f64,
    wall: &Vec3d,
    restitution: f64,
    friction: f64,
) {
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
        let vt = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if vt > 1e-6 {
            let t = [t[0] / vt, t[1] / vt, t[2] / vt];
            let kt = eff_mass(body, &r, &t);
            let jt = (friction * j).min(vt / kt);
            body.apply_impulse(&[-jt * t[0], -jt * t[1], -jt * t[2]], wp);
        }
    }
    add_scaled(&mut body.p, n, pen.min(0.05) * 0.4);
}
