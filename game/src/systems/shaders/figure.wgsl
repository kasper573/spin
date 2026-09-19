// The viewer's own figure as a mirror sees it: the solids it is made of, which a mirrored ray
// is cast against before anything else is looked for along it, since the figure is not on
// screen as a mirror would show it. See `systems/figure.rs`.
#define_import_path figure

#import bevy_pbr::mesh_view_bindings::{view, lights}
#import optics::{bounce, sunlight, glint, sunbeams_at, sunbeam_glints_at, met_on_screen, scene_at, behind}
#import portals::mouths
#import ring::sun_reaches

struct Part {
    // the part's own axes in the world and where it stands, and beside them: how many corners
    // its outline has, or none if it is round; how far it runs either way along its z, or its
    // y if it is round; its radius; and the bore through its flat ends, or less than none if
    // its ends are round
    x: vec4<f32>,
    y: vec4<f32>,
    z: vec4<f32>,
    at: vec4<f32>,
    // its colour, and how much of a metal it is
    colour: vec4<f32>,
    // what it gives off of itself, and how rough it is
    glow: vec4<f32>,
    // the corners of its outline, two to a row
    outline: array<vec4<f32>, 4>,
}

struct Figure {
    parts: array<Part, 32>,
    // a sphere round all of them
    within: vec4<f32>,
    count: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(10) var<uniform> figure: Figure;

const PI: f32 = 3.14159265;
const NOWHERE: f32 = 1e9;
// how far off a part what is drawn of it may be taken to stand, the depth it was drawn at
// being only so fine
const SKIN: f32 = 0.02;

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

/// Where a ray meets a part from outside it, and which way the part faces there, in the
/// part's own frame: nowhere if it misses, and facing no way where it runs down a bore.
struct Met {
    distance: f32,
    facing: vec3<f32>,
}

/// The figure along a ray from a point of the world, which lies at `at` in the ring's frame.
/// The figure is lit as the ring lights anything standing where it stands.
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
    var facing = vec3(0.0);
    for (var i = 0u; i < figure.count; i++) {
        let met = part_met(origin, dir, i);
        if (met.distance < seen.distance) {
            seen.distance = met.distance;
            facing = met.facing;
            nearest = i;
        }
    }
    if (seen.distance == NOWHERE) {
        return seen;
    }
    // down a bore there is nothing to see
    seen.colour = vec3(0.0);
    if (all(facing == vec3(0.0))) {
        return seen;
    }
    let x = figure.parts[nearest].x.xyz;
    let y = figure.parts[nearest].y.xyz;
    let z = figure.parts[nearest].z.xyz;
    let n = normalize(x * facing.x + y * facing.y + z * facing.z);
    let colour = figure.parts[nearest].colour;
    let glow = figure.parts[nearest].glow;
    let albedo = colour.rgb * (1.0 - colour.a);
    let sheen = mix(vec3(1.0), colour.rgb, colour.a);
    let f0 = mix(0.04, 0.8, colour.a);
    let roughness = max(glow.a, 0.05);
    var matte = bounce();
    var shine = vec3(0.0);
    let here = at + dir * seen.distance;
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        if (dot(n, l) > 0.0 && sun_reaches(here, l, ring)) {
            matte += sunlight(i) * dot(n, l) / PI;
            shine += sunlight(i) * glint(n, -dir, l, roughness, f0);
        }
    }
    let drawn = here - mouths.about.xyz;
    matte += sunbeams_at(drawn, n) / PI;
    shine += sunbeam_glints_at(drawn, n, -dir, roughness, f0);
    seen.colour = albedo * matte + sheen * shine + glow.rgb * view.exposure;
    return seen;
}

/// Whether a point of the world is on the figure.
fn on_figure(p: vec3<f32>) -> bool {
    let reach = figure.within.w + SKIN;
    let off = p - figure.within.xyz;
    if (figure.count == 0u || dot(off, off) > reach * reach) {
        return false;
    }
    for (var i = 0u; i < figure.count; i++) {
        if (within_part(in_frame_of(i, p - figure.parts[i].at.xyz), i)) {
            return true;
        }
    }
    return false;
}

/// A direction of the world, or a place told from where the part stands, in the part's frame.
fn in_frame_of(i: u32, v: vec3<f32>) -> vec3<f32> {
    return vec3(dot(v, figure.parts[i].x.xyz), dot(v, figure.parts[i].y.xyz), dot(v, figure.parts[i].z.xyz));
}

fn corner(i: u32, k: u32) -> vec2<f32> {
    let pair = figure.parts[i].outline[k / 2u];
    return select(pair.xy, pair.zw, (k & 1u) == 1u);
}

/// Whether a point of a part's frame lies within a skin's breadth of the part.
fn within_part(p: vec3<f32>, i: u32) -> bool {
    let corners = u32(figure.parts[i].x.w);
    let half = figure.parts[i].y.w;
    if (corners == 0u) {
        let radius = figure.parts[i].z.w;
        let ends_round = figure.parts[i].at.w < 0.0;
        let along = max(abs(p.y) - half, 0.0);
        let across = length(p.xz);
        if (ends_round) {
            return length(vec2(across, along)) < radius + SKIN;
        }
        return across < radius + SKIN && along < SKIN;
    }
    if (abs(p.z) > half + SKIN) {
        return false;
    }
    for (var k = 0u; k < corners; k++) {
        let a = corner(i, k);
        let edge = corner(i, (k + 1u) % corners) - a;
        if (dot(normalize(vec2(edge.y, -edge.x)), p.xy - a) > SKIN) {
            return false;
        }
    }
    return true;
}

/// Where a ray of the world first meets a part from outside it.
fn part_met(origin: vec3<f32>, dir: vec3<f32>, i: u32) -> Met {
    let o = in_frame_of(i, origin - figure.parts[i].at.xyz);
    let d = in_frame_of(i, dir);
    if (figure.parts[i].x.w == 0.0) {
        return round_met(o, d, i);
    }
    return prism_met(o, d, i);
}

/// A prism is what lies inside all its faces, so a ray is in it from the last face it comes
/// in through to the first it goes out through, if it comes in through the last before then.
fn prism_met(o: vec3<f32>, d: vec3<f32>, i: u32) -> Met {
    let nothing = Met(NOWHERE, vec3(0.0));
    let half_depth = figure.parts[i].y.w;
    var enters = -NOWHERE;
    var leaves = NOWHERE;
    var facing = vec3(0.0);
    if (abs(d.z) < 1e-8) {
        if (abs(o.z) > half_depth) {
            return nothing;
        }
    } else {
        let one = (-half_depth - o.z) / d.z;
        let other = (half_depth - o.z) / d.z;
        enters = min(one, other);
        leaves = max(one, other);
        facing = vec3(0.0, 0.0, -sign(d.z));
    }
    let corners = u32(figure.parts[i].x.w);
    for (var k = 0u; k < corners; k++) {
        let a = corner(i, k);
        let edge = corner(i, (k + 1u) % corners) - a;
        let outward = vec2(edge.y, -edge.x);
        let out_by = dot(outward, o.xy - a);
        let rate = dot(outward, d.xy);
        if (abs(rate) < 1e-9) {
            if (out_by > 0.0) {
                return nothing;
            }
        } else if (rate < 0.0) {
            if (-out_by / rate > enters) {
                enters = -out_by / rate;
                facing = vec3(normalize(outward), 0.0);
            }
        } else {
            leaves = min(leaves, -out_by / rate);
        }
    }
    if (enters > leaves || enters <= 0.0) {
        return nothing;
    }
    return Met(enters, facing);
}

/// Where a ray first meets a sphere from outside it, or nowhere.
fn sphere_met(o: vec3<f32>, d: vec3<f32>, centre: vec3<f32>, radius: f32) -> f32 {
    let oc = o - centre;
    let b = dot(oc, d);
    let h = b * b - (dot(oc, oc) - radius * radius);
    if (h < 0.0) {
        return NOWHERE;
    }
    let t = -b - sqrt(h);
    return select(NOWHERE, t, t > 0.0);
}

/// Something round about its y axis: its side, and its two ends, flat or round. A ray that
/// comes in through the bore of a flat end runs down a tube it sees nothing in, unless the
/// tube is no longer than a ring, which it goes clean through.
fn round_met(o: vec3<f32>, d: vec3<f32>, i: u32) -> Met {
    let half_length = figure.parts[i].y.w;
    let radius = figure.parts[i].z.w;
    let bore = figure.parts[i].at.w;
    var met = Met(NOWHERE, vec3(0.0));
    let a = dot(d.xz, d.xz);
    if (a > 1e-10) {
        let b = dot(o.xz, d.xz);
        let h = b * b - a * (dot(o.xz, o.xz) - radius * radius);
        if (h >= 0.0) {
            let t = (-b - sqrt(h)) / a;
            let p = o + d * t;
            if (t > 0.0 && abs(p.y) <= half_length) {
                met = Met(t, vec3(p.x, 0.0, p.z) / radius);
            }
        }
    }
    if (bore < 0.0) {
        for (var end = -1.0; end <= 1.0; end += 2.0) {
            let centre = vec3(0.0, end * half_length, 0.0);
            let t = sphere_met(o, d, centre, radius);
            if (t < met.distance) {
                met = Met(t, (o + d * t - centre) / radius);
            }
        }
        return met;
    }
    if (abs(d.y) > 1e-8) {
        let end = -sign(d.y);
        let t = (end * half_length - o.y) / d.y;
        let p = o + d * t;
        let across = dot(p.xz, p.xz);
        if (t > 0.0 && t < met.distance && across <= radius * radius) {
            if (across >= bore * bore) {
                met = Met(t, vec3(0.0, end, 0.0));
            } else if (half_length > bore) {
                met = Met(t, vec3(0.0));
            }
        }
    }
    return met;
}
