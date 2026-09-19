// The viewer's own figure as a mirror sees it: a few limbs with rounded ends, which a mirrored
// ray is cast against before anything else is looked for along it, since none of the figure
// is on screen to be found there. See `systems/figure.rs`.
#define_import_path figure

#import bevy_pbr::mesh_view_bindings::{view, lights}
#import optics::{bounce, sunlight, met_on_screen, scene_at, behind}
#import ring::sun_reaches

struct Limb {
    // one end and the radius, then the other end
    near: vec4<f32>,
    far: vec4<f32>,
    colour: vec4<f32>,
    glow: vec4<f32>,
}

struct Figure {
    limbs: array<Limb, 16>,
    // a sphere round all of them
    within: vec4<f32>,
    count: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(10) var<uniform> figure: Figure;

const PI: f32 = 3.14159265;
const NOWHERE: f32 = 1e9;
// the limbs are the figure's bulk, and what is drawn of it stands no further off them than this
const SKIN: f32 = 0.05;

/// What a mirror shows along a ray from a point of the world, which lies at `at` in the ring's
/// frame: the figure where the ray meets it, else the scene where the ray meets that on
/// screen, if the mirror is one that can be `marched` across the screen, else what lies
/// `beyond`. The figure is on screen only as the eye sees it, never as a mirror would, so what
/// the ray meets on screen is passed over where it is the figure.
fn mirrored(origin: vec3<f32>, dir: vec3<f32>, beyond: vec3<f32>, at: vec3<f32>, ring: vec2<f32>, marched: bool) -> vec3<f32> {
    let own = figure_seen(origin, dir, beyond, at, ring);
    if (!marched) {
        return own.colour;
    }
    let met = met_on_screen(origin, dir, own.distance);
    if (met.share <= 0.0 || on_figure(scene_at(met.uv))) {
        return own.colour;
    }
    return mix(own.colour, behind(met.uv), met.share);
}

/// What a ray sees of the figure: its colour where the ray meets it and how far along that
/// is, or what lies `beyond` and nowhere.
struct FigureSeen {
    colour: vec3<f32>,
    distance: f32,
}

/// Where a ray first meets a sphere from outside it, or nowhere.
fn sphere_met(origin: vec3<f32>, dir: vec3<f32>, centre: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - centre;
    let b = dot(oc, dir);
    let h = b * b - (dot(oc, oc) - radius * radius);
    if (h < 0.0) {
        return NOWHERE;
    }
    let t = -b - sqrt(h);
    return select(NOWHERE, t, t > 0.0);
}

/// Where a ray first meets a limb from outside it: its shaft, or the round of either end.
fn limb_met(origin: vec3<f32>, dir: vec3<f32>, limb: Limb) -> f32 {
    let radius = limb.near.w;
    let shaft = limb.far.xyz - limb.near.xyz;
    let oa = origin - limb.near.xyz;
    let length2 = dot(shaft, shaft);
    let along = dot(shaft, dir);
    let a = length2 - along * along;
    var met = min(sphere_met(origin, dir, limb.near.xyz, radius), sphere_met(origin, dir, limb.far.xyz, radius));
    if (a > 1e-8) {
        let begun = dot(shaft, oa);
        let b = length2 * dot(dir, oa) - begun * along;
        let c = length2 * dot(oa, oa) - begun * begun - radius * radius * length2;
        let h = b * b - a * c;
        if (h >= 0.0) {
            let t = (-b - sqrt(h)) / a;
            let y = begun + t * along;
            if (t > 0.0 && y > 0.0 && y < length2) {
                met = min(met, t);
            }
        }
    }
    return met;
}

/// Whether a point of the world is on the figure.
fn on_figure(p: vec3<f32>) -> bool {
    let reach = figure.within.w + SKIN;
    let off = p - figure.within.xyz;
    if (figure.count == 0u || dot(off, off) > reach * reach) {
        return false;
    }
    var nearest = NOWHERE;
    for (var i = 0u; i < figure.count; i++) {
        let limb = figure.limbs[i];
        let shaft = limb.far.xyz - limb.near.xyz;
        let share = clamp(dot(p - limb.near.xyz, shaft) / max(dot(shaft, shaft), 1e-8), 0.0, 1.0);
        nearest = min(nearest, distance(p, limb.near.xyz + shaft * share) - limb.near.w);
    }
    return nearest < SKIN;
}

/// The figure along a ray from a point of the world, which lies at `at` in the ring's frame.
/// The figure is lit as the ring would light anything matte standing where it stands.
fn figure_seen(origin: vec3<f32>, dir: vec3<f32>, beyond: vec3<f32>, at: vec3<f32>, ring: vec2<f32>) -> FigureSeen {
    var seen = FigureSeen(beyond, NOWHERE);
    // a ray that sets out from outside the sphere round them all, and away from it or wide of
    // it, meets none of them
    let off = origin - figure.within.xyz;
    let outside = dot(off, off) - figure.within.w * figure.within.w;
    let toward = -dot(off, dir);
    if (figure.count == 0u || (outside > 0.0 && (toward < 0.0 || toward * toward < outside))) {
        return seen;
    }
    var nearest = 0u;
    for (var i = 0u; i < figure.count; i++) {
        let t = limb_met(origin, dir, figure.limbs[i]);
        if (t < seen.distance) {
            seen.distance = t;
            nearest = i;
        }
    }
    if (seen.distance == NOWHERE) {
        return seen;
    }
    let limb = figure.limbs[nearest];
    let p = origin + dir * seen.distance;
    let shaft = limb.far.xyz - limb.near.xyz;
    let share = clamp(dot(p - limb.near.xyz, shaft) / max(dot(shaft, shaft), 1e-8), 0.0, 1.0);
    let n = normalize(p - limb.near.xyz - shaft * share);
    var light = bounce();
    let exposure = view.exposure;
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let ndl = dot(n, l);
        if (ndl > 0.0 && sun_reaches(at + p - origin, l, ring)) {
            light += sunlight(i) * ndl / PI;
        }
    }
    seen.colour = limb.colour.rgb * light + limb.glow.rgb * exposure;
    return seen;
}
