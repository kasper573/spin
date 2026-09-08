//! Where the crosshair points: the first terrain the view ray meets, or otherwise the far wall or
//! cap of the drum where the ray leaves the glass interior.
use bevy::prelude::*;

use crate::systems::drum::{Drum, HALF_WIDTH, RADIUS};
use crate::systems::sim::{SimSet, Simulation};

const EPS: f32 = 1e-6;
const MARCH_STEP: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimPoint {
    pub point: Vec3,
    /// Inward-facing surface normal at the point.
    pub normal: Vec3,
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

pub fn cast(origin: Vec3, dir: Vec3, drum: &Drum) -> Option<AimPoint> {
    let mut t_in = f32::NEG_INFINITY;
    let mut t_out = f32::INFINITY;

    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * (origin.x * dir.x + origin.z * dir.z);
    let c = origin.x * origin.x + origin.z * origin.z - RADIUS * RADIUS;
    if a > EPS {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        t_in = (-b - s) / (2.0 * a);
        t_out = (-b + s) / (2.0 * a);
    } else if c > 0.0 {
        return None;
    }

    if dir.y.abs() > EPS {
        let t0 = (-HALF_WIDTH - origin.y) / dir.y;
        let t1 = (HALF_WIDTH - origin.y) / dir.y;
        t_in = t_in.max(t0.min(t1));
        t_out = t_out.min(t0.max(t1));
    } else if origin.y.abs() > HALF_WIDTH {
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
    let normal = if (point.y.abs() - HALF_WIDTH).abs() < 1e-4 * HALF_WIDTH {
        Vec3::new(0.0, if point.y > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        let r = point.xz().length().max(EPS);
        Vec3::new(-point.x / r, 0.0, -point.z / r)
    };
    Some(AimPoint { point, normal })
}

fn update(sim: Res<Simulation>, cameras: Query<&Transform, With<Camera3d>>, mut aim: ResMut<Aim>) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    aim.0 = cast(camera.translation, camera.forward().as_vec3(), &sim.drum);
}

fn penetration_at(origin: Vec3, dir: Vec3, t: f32, drum: &Drum) -> (f64, [f64; 3]) {
    let p = origin + dir * t;
    drum.landscape
        .penetration(p.x as f64, p.y as f64, p.z as f64, drum.angle.0, 0.0)
}

/// Fixed-step march along the ray, refined by bisection at the first terrain crossing.
fn march_terrain(origin: Vec3, dir: Vec3, t0: f32, t1: f32, drum: &Drum) -> Option<AimPoint> {
    if penetration_at(origin, dir, t0, drum).0 > 0.0 {
        return None;
    }
    let mut t_prev = t0;
    let mut t = t0 + MARCH_STEP;
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
                normal: Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32),
            });
        }
        t_prev = t;
        t += MARCH_STEP;
    }
    None
}
