//! Where the crosshair points: the first terrain the view ray meets, or otherwise the far wall or
//! cap of the drum where the ray leaves the glass interior. Cast in the drum's frame from the
//! avatar's eye, in double precision and without ever forming the ring's radius squared, so it
//! holds at any size of ring.
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::core::avatar;
use crate::core::math::mat3mul;
use crate::core::vessel::Vessel;
use crate::systems::drum::Drum;
use crate::systems::sim::{SimSet, Simulation};

const EPS: f64 = 1e-9;
/// The finest the terrain is marched at; a big ring's landscape is coarser, and is marched at a
/// quarter of its own segments.
const MARCH_STEP: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimPoint {
    pub point: DVec3,
    /// Inward-facing surface normal at the point.
    pub normal: DVec3,
}

/// The current crosshair target, refreshed every frame before commands run.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq)]
pub struct Aim(pub Option<AimPoint>);

pub struct AimPlugin;

impl Plugin for AimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Aim>()
            .add_systems(Update, update.before(SimSet::Command));
    }
}

pub fn cast(origin: DVec3, dir: DVec3, drum: &Drum) -> Option<AimPoint> {
    let (radius, half_width) = (drum.ring.radius.0 as f64, drum.ring.half_width.0 as f64);
    let mut t_in = f64::NEG_INFINITY;
    let mut t_out = f64::INFINITY;

    // the ray against the glass cylinder, with the origin's distance from it kept small
    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * ((radius + origin.x) * dir.x + origin.z * dir.z);
    let c = 2.0 * radius * origin.x + origin.x * origin.x + origin.z * origin.z;
    if a > EPS {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let q = -0.5 * (b + b.signum() * disc.sqrt());
        let (r0, r1) = if q != 0.0 { (q / a, c / q) } else { (0.0, 0.0) };
        t_in = r0.min(r1);
        t_out = r0.max(r1);
    } else if c > 0.0 {
        return None;
    }

    let axial = drum.axial(origin.to_array());
    if dir.y.abs() > EPS {
        let t0 = (-half_width - axial) / dir.y;
        let t1 = (half_width - axial) / dir.y;
        t_in = t_in.max(t0.min(t1));
        t_out = t_out.min(t0.max(t1));
    } else if axial.abs() > half_width {
        return None;
    }

    let t_start = t_in.max(0.0);
    if t_out <= t_start {
        return None;
    }

    if !drum.landscape.is_empty()
        && let Some(hit) = march_terrain(origin, dir, t_start, t_out, drum)
    {
        return Some(hit);
    }

    let point = origin + dir * t_out;
    let at_cap = (drum.axial(point.to_array()).abs() - half_width).abs() < 1e-4 * half_width;
    let normal = if at_cap {
        DVec3::new(0.0, if point.y + axial > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        let (_, outward) = drum.depth_and_outward(point.to_array());
        -DVec3::from_array(outward)
    };
    Some(AimPoint { point, normal })
}

fn update(sim: Res<Simulation>, mut aim: ResMut<Aim>) {
    let hull = sim.avatar();
    let eye = DVec3::from_array(avatar::eye(hull));
    let forward = DVec3::from_array(mat3mul(&hull.m, &[0.0, 0.0, -1.0]));
    aim.0 = cast(eye, forward, &sim.drum);
}

fn penetration_at(origin: DVec3, dir: DVec3, t: f64, drum: &Drum) -> (f64, [f64; 3]) {
    let p = origin + dir * t;
    drum.penetrations(p.to_array())
        .iter()
        .next()
        .map_or((-1.0, [0.0; 3]), |pen| (pen.depth, pen.normal))
}

/// Fixed-step march along the ray, refined by bisection at the first terrain crossing.
fn march_terrain(origin: DVec3, dir: DVec3, t0: f64, t1: f64, drum: &Drum) -> Option<AimPoint> {
    if penetration_at(origin, dir, t0, drum).0 > 0.0 {
        return None;
    }
    let step = MARCH_STEP.max(drum.landscape.segment_arc() / 4.0);
    let mut t_prev = t0;
    let mut t = t0 + step;
    while t < t1 {
        if penetration_at(origin, dir, t, drum).0 > 0.0 {
            let (mut lo, mut hi) = (t_prev, t);
            for _ in 0..8 {
                let mid = (lo + hi) / 2.0;
                if penetration_at(origin, dir, mid, drum).0 > 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let (_, n) = penetration_at(origin, dir, lo, drum);
            return Some(AimPoint {
                point: origin + dir * lo,
                normal: DVec3::from_array(n),
            });
        }
        t_prev = t;
        t += step;
    }
    None
}
