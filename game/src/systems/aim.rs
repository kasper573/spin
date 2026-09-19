//! Where the crosshair points: the first terrain the view ray meets, or otherwise the far wall or
//! cap of the drum where the ray leaves the glass interior, and on from there wherever that is
//! the open mouth of a portal. Cast in the drum's frame from the avatar's eye, in double
//! precision and without ever forming the ring's radius squared, so it holds at any size of
//! ring.
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::core::math::{Quatd, quat_mul, quat_rotate};
use crate::core::units::{Metres, Radians};
use crate::systems::drum::{
    CapSide, Drum, DrumSurface, Mouth, MouthColour, MouthCoords, PATCH, Round,
};
use crate::systems::scene::{
    LinesBeyondBlue, LinesBeyondOrange, SettleVantages, Vantage, Vantages,
};
use crate::systems::sim::{SimSet, Simulation};

const EPS: f64 = 1e-9;
/// The shortest step the ray is marched in, as a share of the ground's cells, and how many
/// times the step it crossed the ground in is halved to find where.
const FINEST_STEP: f64 = 0.125;
const HALVINGS: usize = 12;
/// The widest turn round the axis a stretch of sculpted ground is found over at once: the ray
/// is between two turns where it is past the half plane of the one and short of the other's,
/// which holds only while they are less than half a turn apart.
const QUARTER_TURN: f64 = std::f64::consts::FRAC_PI_2;

/// How many portals a ray is followed through: two mouths that face each other would
/// otherwise be gone through for ever.
const MOST_PASSES: usize = 4;
/// How far either side of an open mouth the ray is taken up again, as a share of how far it
/// has come.
const THROUGH: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimPoint {
    pub point: DVec3,
    /// Inward-facing surface normal at the point.
    pub normal: DVec3,
    pub surface: DrumSurface,
    /// How far the ray ran to get there, and the turn the portals it went through on the way
    /// put it through: what turns a direction at the eye into that direction at the point.
    pub range: f64,
    pub turn: Quatd,
    /// The first portal mouth the ray met, on its way or at its end.
    pub mouth: Option<MouthColour>,
}

/// What the crosshair's marker outlines: a disc this big on this mouth's place, which is
/// wherever a tool has it, with the mouth's top marked where that matters, and marked as
/// refused where what the tool would put there cannot go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimMarker {
    pub mouth: Mouth,
    pub radius: Metres,
    pub top: bool,
    pub refused: bool,
}

/// The current crosshair target, refreshed every frame before commands run, and what is
/// drawn there: the marker of the tool at work, if one is, and whether the crosshair is live.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub target: Option<AimPoint>,
    /// The marker of the tool at work on the target, when one is.
    pub marker: Option<AimMarker>,
    /// Whether the viewer is at the controls: the crosshair is being steered, so its marker
    /// is drawn bright, and the tool that is out answers to the mouse.
    pub engaged: bool,
}

/// The marker at rest is this big, the same on a ring of any size, and every marker floats
/// this share of its own radius above the surface, clear of it at any size of ring.
const MARKER_AT_REST: Metres = Metres(0.3);
const MARKER_LIFT: f64 = 0.01;
/// Where the tick that marks a marker's top starts, as a share of the way out to its rim.
const TICK_FROM: f64 = 0.6;

pub struct AimPlugin;

impl Plugin for AimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Aim>()
            .add_systems(Update, update.before(SimSet::Command))
            .add_systems(Update, marker.in_set(SimSet::Observe).after(SettleVantages));
    }
}

/// What a ray from a point of the drum's frame meets, followed through whatever open portals
/// it goes into.
pub fn cast(origin: DVec3, dir: DVec3, drum: &Drum) -> Option<AimPoint> {
    let (mut origin, mut dir) = (origin, dir);
    let mut met: Option<AimPoint> = None;
    for _ in 0..=MOST_PASSES {
        let Some(hit) = cast_once(origin, dir, drum) else {
            break;
        };
        let so_far = met.unwrap_or(AimPoint {
            range: 0.0,
            turn: [0.0, 0.0, 0.0, 1.0],
            mouth: None,
            ..hit
        });
        let here = AimPoint {
            range: so_far.range + hit.range,
            turn: so_far.turn,
            mouth: so_far.mouth.or(hit.mouth),
            ..hit
        };
        met = Some(here);
        let step = THROUGH * here.range.max(1.0);
        let Some(passage) = drum.passage_through_mouths(
            (hit.point - dir * step).to_array(),
            (hit.point + dir * step).to_array(),
        ) else {
            break;
        };
        origin = DVec3::from_array(passage.point);
        dir = DVec3::from_array(passage.turned(dir.to_array()));
        met = Some(AimPoint {
            turn: quat_mul(&passage.turn, &here.turn),
            ..here
        });
    }
    met
}

/// A disc about an aim point, the way the surface lies there.
pub fn disc_at(drum: &Drum, at: AimPoint) -> Mouth {
    Mouth {
        anchor: drum.mouth_anchor(at.point.to_array(), at.surface),
        roll: Radians(0.0),
    }
}

/// The outline of a disc of `radius` about an aim point, wrapped onto the surface it rests on
/// and held `lift` above it: over the ground it runs round the wall and along the axis at that
/// distance, rising and falling with whatever the ground does there, so that it outlines just
/// what a brush of that size would touch; on a cap it lies flat on the cap.
pub fn outline(drum: &Drum, at: AimPoint, radius: f64, lift: f64) -> Vec<DVec3> {
    drum.mouth_rim(&disc_at(drum, at), radius, lift)
        .into_iter()
        .map(DVec3::from_array)
        .collect()
}

fn cast_once(origin: DVec3, dir: DVec3, drum: &Drum) -> Option<AimPoint> {
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
    let (normal, surface) = if at_cap {
        let side = if point.y + axial > 0.0 {
            CapSide::High
        } else {
            CapSide::Low
        };
        (DVec3::new(0.0, -side.sign(), 0.0), DrumSurface::Cap(side))
    } else {
        let (_, outward) = drum.depth_and_outward(point.to_array());
        (-DVec3::from_array(outward), DrumSurface::Wall)
    };
    Some(met_at(point, normal, surface, t_out, drum))
}

fn met_at(point: DVec3, normal: DVec3, surface: DrumSurface, range: f64, drum: &Drum) -> AimPoint {
    AimPoint {
        point,
        normal,
        surface,
        range,
        turn: [0.0, 0.0, 0.0, 1.0],
        mouth: drum.mouth_under(point.to_array(), surface),
    }
}

fn update(sim: Res<Simulation>, mut aim: ResMut<Aim>) {
    let (eye, attitude) = sim.eye();
    let forward = DVec3::from_array(quat_rotate(&attitude, &[0.0, 0.0, -1.0]));
    aim.target = cast(DVec3::from_array(eye), forward, &sim.drum);
}

/// The marker on the target: the outline of what the tool at work marks, or of the marker at
/// rest, wrapped onto the surface there, and drawn for every vantage it may be seen from.
fn marker(
    aim: Res<Aim>,
    sim: Res<Simulation>,
    vantages: Res<Vantages>,
    mut own: Gizmos,
    mut beyond_blue: Gizmos<LinesBeyondBlue>,
    mut beyond_orange: Gizmos<LinesBeyondOrange>,
) {
    let Some(target) = aim.target else {
        return;
    };
    let marked = aim.marker.unwrap_or(AimMarker {
        mouth: disc_at(&sim.drum, target),
        radius: MARKER_AT_REST,
        top: false,
        refused: false,
    });
    let radius = marked.radius.0 as f64;
    let alpha = if aim.engaged { 0.9 } else { 0.35 };
    let colour = if marked.refused {
        Color::srgba(1.0, 0.15, 0.1, alpha)
    } else {
        Color::srgba(1.0, 1.0, 1.0, alpha)
    };
    let lift = radius * MARKER_LIFT;
    let mut outline = sim.drum.mouth_rim(&marked.mouth, radius, lift);
    outline.extend(outline.first().copied());
    let up = |v: f64| MouthCoords { u: 0.0, v, h: lift };
    let tick = marked.top.then(|| {
        [TICK_FROM, 1.0].map(|share| sim.drum.mouth_point(&marked.mouth, up(share * radius)))
    });
    let strips = [
        Some(outline.as_slice()),
        tick.as_ref().map(|t| t.as_slice()),
    ];
    let [seen, blue, orange] = vantages.0;
    for strip in strips.into_iter().flatten() {
        draw(&mut own, seen, strip, colour);
        draw(&mut beyond_blue, blue, strip, colour);
        draw(&mut beyond_orange, orange, strip, colour);
    }
}

fn draw<Lines: GizmoConfigGroup>(
    gizmos: &mut Gizmos<Lines>,
    vantage: Option<Vantage>,
    strip: &[[f64; 3]],
    colour: Color,
) {
    if let Some(vantage) = vantage {
        gizmos.linestrip(strip.iter().map(|p| vantage.local(*p)), colour);
    }
}

/// The first crossing of the ground along the ray between `t0` and `t1`. Bare ground lies at
/// the depth it was laid with, a cylinder the ray is crossed with outright; sculpted ground
/// differs from that only over its patches, so only the stretches of the ray over them are
/// traced, however far the ray runs over bare ground on its way.
fn march_terrain(origin: DVec3, dir: DVec3, t0: f64, t1: f64, drum: &Drum) -> Option<AimPoint> {
    if clearance(origin, dir, t0, drum).0 < 0.0 {
        return None;
    }
    let hit = |t: f64| {
        let point = origin + dir * t;
        let normal = DVec3::from_array(drum.ground_normal(point.to_array()));
        met_at(point, normal, DrumSurface::Wall, t, drum)
    };
    let bare = below(origin, dir, t0, drum.landscape.base() as f64, drum).filter(|t| *t <= t1);
    let stretches = over_sculpted(origin, dir, t0, t1, drum);
    for &(start, end) in &stretches {
        if let Some(t) = bare
            && t < start
        {
            return Some(hit(t));
        }
        if let Some(t) = trace(origin, dir, start, end, drum) {
            return Some(hit(t));
        }
    }
    bare.filter(|t| stretches.last().is_none_or(|(_, end)| t > end))
        .map(hit)
}

/// How high the ray stands over the ground under it at `t`, and over the glass.
fn clearance(origin: DVec3, dir: DVec3, t: f64, drum: &Drum) -> (f64, f64) {
    let p = (origin + dir * t).to_array();
    let height = drum.height_above_glass(p);
    (height - drum.ground(p), height)
}

/// Where from `t0` on the ray first stands no higher than `depth` over the glass: where it
/// leaves the cylinder that depth in from the glass, worked out without forming the ring's
/// radius squared.
fn below(origin: DVec3, dir: DVec3, t0: f64, depth: f64, drum: &Drum) -> Option<f64> {
    let radius = drum.ring.radius.0 as f64;
    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * ((radius + origin.x) * dir.x + origin.z * dir.z);
    let c = 2.0 * radius * (origin.x + depth) + origin.x * origin.x + origin.z * origin.z
        - depth * depth;
    let disc = b * b - 4.0 * a * c;
    if a <= EPS || disc < 0.0 {
        return (c >= 0.0).then_some(t0);
    }
    let q = -0.5 * (b + b.signum() * disc.sqrt());
    let (r0, r1) = if q != 0.0 { (q / a, c / q) } else { (0.0, 0.0) };
    let (inside, leaves) = (r0.min(r1), r0.max(r1));
    Some(if t0 > inside && t0 < leaves {
        leaves
    } else {
        t0
    })
}

/// The stretches of the ray between `t0` and `t1` over sculpted ground, and the cell round it
/// the ground is drawn up or down to it over, in order and not overlapping.
fn over_sculpted(origin: DVec3, dir: DVec3, t0: f64, t1: f64, drum: &Drum) -> Vec<(f64, f64)> {
    let grid = drum.landscape.grid();
    let radius = drum.ring.radius.0 as f64;
    let site = drum.site;
    let patch = PATCH as f64;
    let span = (patch + 1.0) * grid.arc / radius;
    let pieces = (span / QUARTER_TURN).ceil().max(1.0);
    let mut stretches: Vec<(f64, f64)> = Vec::new();
    for (round, along) in drum.landscape.patches() {
        let first = (along as f64 * patch - 1.0) * grid.along - site.y;
        let last = (along as f64 * patch + patch) * grid.along - site.y;
        let Some((lo, hi)) = between(origin.y, dir.y, first, last) else {
            continue;
        };
        let (lo, hi) = (lo.max(t0), hi.min(t1));
        if lo > hi {
            continue;
        }
        let start = Round {
            cell: grid.wrap(round as i128 * PATCH as i128 - 1),
            across: 0.0,
        };
        let from = site.round.arc_to(start, grid) / radius;
        for k in 0..pieces as usize {
            let a = from + span * k as f64 / pieces;
            let b = from + span * (k + 1) as f64 / pieces;
            if let Some((ta, tb)) = within_turns(origin, dir, radius, a, b)
                && ta.max(lo) <= tb.min(hi)
            {
                stretches.push((ta.max(lo), tb.min(hi)));
            }
        }
    }
    stretches.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(stretches.len());
    for (start, end) in stretches {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Where along the ray a coordinate that starts at `at` and changes by `rate` per unit of the
/// ray lies between `first` and `last`.
fn between(at: f64, rate: f64, first: f64, last: f64) -> Option<(f64, f64)> {
    if rate.abs() <= EPS {
        return (first..=last)
            .contains(&at)
            .then_some((f64::NEG_INFINITY, f64::INFINITY));
    }
    let (a, b) = ((first - at) / rate, (last - at) / rate);
    Some((a.min(b), a.max(b)))
}

/// Where along the ray it is between two turns round the axis from the site, less than half a
/// turn apart: on the spinward side of the half plane at the one and the far side of the other.
fn within_turns(origin: DVec3, dir: DVec3, radius: f64, a: f64, b: f64) -> Option<(f64, f64)> {
    let (x, z) = (radius + origin.x, origin.z);
    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let past_a = ahead(ca * z - sa * x, ca * dir.z - sa * dir.x)?;
    let short_of_b = ahead(sb * x - cb * z, sb * dir.x - cb * dir.z)?;
    let (lo, hi) = (past_a.0.max(short_of_b.0), past_a.1.min(short_of_b.1));
    (lo <= hi).then_some((lo, hi))
}

/// Where along the ray a quantity that starts at `at` and changes by `rate` per unit of the ray
/// is not negative.
fn ahead(at: f64, rate: f64) -> Option<(f64, f64)> {
    if rate.abs() <= EPS {
        return (at >= 0.0).then_some((f64::NEG_INFINITY, f64::INFINITY));
    }
    let root = -at / rate;
    Some(if rate > 0.0 {
        (root, f64::INFINITY)
    } else {
        (f64::NEG_INFINITY, root)
    })
}

/// The first crossing of the ground between `t0` and `t1`, traced in steps no longer than the
/// ground could rise to meet the ray in: how high the ray stands over the ground under it, over
/// how steeply the ground rises anywhere, and the step it crossed in halved down to where it
/// crossed.
fn trace(origin: DVec3, dir: DVec3, t0: f64, t1: f64, drum: &Drum) -> Option<f64> {
    let grid = drum.landscape.grid();
    let finest = FINEST_STEP * grid.arc.min(grid.along);
    let steepest = drum.landscape.steepest();
    let radius = drum.ring.radius.0 as f64;
    let (mut c, mut height) = clearance(origin, dir, t0, drum);
    if c < 0.0 {
        return Some(t0);
    }
    let mut t = t0;
    while t < t1 {
        // the ground under a point off the glass passes under it faster than the glass does
        let under = radius / (radius - height).max(EPS * radius);
        let next = (t + (c / (1.0 + steepest * under)).max(finest)).min(t1);
        let (after, rise) = clearance(origin, dir, next, drum);
        if after < 0.0 {
            let (mut lo, mut hi) = (t, next);
            for _ in 0..HALVINGS {
                let mid = 0.5 * (lo + hi);
                if clearance(origin, dir, mid, drum).0 < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(lo);
        }
        (t, c, height) = (next, after, rise);
    }
    None
}
