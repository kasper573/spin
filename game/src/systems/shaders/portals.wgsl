// The portals, as the surfaces they are let into show them. A portal is not a thing standing on
// its surface but a part of it, so the ground and the glass draw it themselves where they find
// a mouth: a ring of flame in the mouth's colour round its rim, licking out over the surface
// and carried round it; the mouth filled in with a vortex of that colour from the rim inward
// as far as it is shut, which is all of it while it is alone and none of it once its pair has
// opened; and what is left open showing what lies beyond the other mouth. The colour is brighter toward the mouth's top and
// darker toward its bottom, so that a mouth shows which way up it is. From behind, a mouth is
// filled in whatever it is from the front.
#define_import_path portals

#import bevy_pbr::mesh_view_bindings::view
#import ripples::{carried, noise3}
#import ring::sun_reaches

struct Mouth {
    // where the mouth's chart is measured from, about the point everything is drawn about; w
    // is what the mouth is on: 0 for no mouth, 1 for the wall, 2 for a cap
    origin: vec4<f32>,
    // how its surface runs there: spinward, with the sine of the mouth's roll; upward, with the
    // cosine; and off the surface into the room
    spinward: vec4<f32>,
    upward: vec4<f32>,
    inward: vec4<f32>,
    colour: vec4<f32>,
    // the middle of the mouth, on its surface, and the rows of the turn that takes what goes in
    // at it out of the other mouth, as near as one turn for the whole mouth can
    centre: vec4<f32>,
    through: array<vec4<f32>, 3>,
    // what takes a point drawn here to where the picture of what is seen through the mouth
    // shows what that point of the mouth does, as the picture's camera clips it
    found: mat4x4<f32>,
}

struct Mouths {
    mouths: array<Mouth, 2>,
    // the ring's radius, a mouth's radius, how much of a mouth is filled in, and the time
    shape: vec4<f32>,
    // how far past the rim the flames reach, as a share of the mouth's radius, how much
    // darker a mouth's bottom is than its top, and how far the ring's caps stand from its middle
    look: vec4<f32>,
    // where the point everything is drawn about lies in the ring's frame, whose middle is the
    // middle of the drum's axis
    about: vec4<f32>,
}

#ifdef MOUTHS_ON_A_SOLID
// the standard material has these bindings' usual places for its own
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> mouths: Mouths;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var through_blue: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var through_orange: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var through_sampler: sampler;
#else
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var<uniform> mouths: Mouths;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var through_blue: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var through_orange: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var through_sampler: sampler;
#endif

const WALL: f32 = 1.0;
const CAP: f32 = 2.0;
// how far inside the rim, and inside the edge of what is filled in, the fire glows
const EMBERS: f32 = 0.1;
// the tongues of flame: how wide they are, as a share of the mouth's radius, how fast they
// run out from the rim and drift toward the mouth's top, in radii a second, and how fast they
// flicker where they stand
const TONGUE: f32 = 0.16;
const RISE: f32 = 0.55;
const DRAFT: f32 = 0.15;
const FLICKER: f32 = 1.3;
// the vortex a mouth is filled in with: how fast it turns at the rim, in radians a second, and
// it turns faster toward its middle, as whatever keeps its spin while it is drawn inward does;
// how wide its streaks are, as a share of the mouth's radius; how long a run of it is wound
// before another has taken over from it, in seconds; and how much darker it is between its
// arms and in its eye than along them
const SWIRL: f32 = 0.9;
const STREAK: f32 = 0.22;
const WINDING: f32 = 3.0;
const TROUGH: f32 = 0.4;
const EYE: f32 = 0.2;
// what the fire gives off at its hottest, in nits, which is enough to stand out against ground
// the sun is on, and what is filled in gives off, which is only enough to be seen by in the
// dark: it shows by the light that falls on it, as the ground does, so that an eye adapted to
// whatever light there is sees its colour and which way up it is, rather than a glare
const FIRE: f32 = 50000.0;
const FILLED: f32 = 250.0;
// how much of the light falling on what is filled in it gives back in its colour
const PAINT: f32 = 0.85;
// the brightest the eye tells from white, as in `optics.wgsl`
const SATURATION: f32 = 4.0;
// how wide a sun stands in the sky, as the tangent of half of it: the edge of its light is as
// much softer as the light has come far from the rim that cut it
const SUN_WIDTH: f32 = 0.00465;

/// What a point of a surface shows of a portal there.
struct Painted {
    // how much of the point is filled in, whose colour this is as a matte surface has it
    paint: f32,
    albedo: vec3<f32>,
    // how much of the point is open, which mouth it is an opening of, and what is seen through
    // it there, as the eye is exposed to it
    open: f32,
    mouth: u32,
    beyond: vec3<f32>,
    // the light the fire gives off there, as the eye is exposed to it
    glow: vec3<f32>,
}

/// A point in a mouth's chart: across it and up it, in metres.
fn chart(mouth: Mouth, p: vec3<f32>) -> vec2<f32> {
    let q = p - mouth.origin.xyz;
    var flat = vec2(dot(q, mouth.spinward.xyz), dot(q, mouth.upward.xyz));
    if (mouth.origin.w == WALL) {
        // walked round the wall rather than measured straight across it
        let radius = mouths.shape.x;
        flat.x = radius * atan2(flat.x, radius - dot(q, mouth.inward.xyz));
    }
    let roll = vec2(mouth.spinward.w, mouth.upward.w);
    return vec2(flat.x * roll.y + flat.y * roll.x, flat.y * roll.y - flat.x * roll.x);
}

/// The fire round a rim `radius` out, at a point of the chart `edge` metres outside it: how
/// hot, from nothing to one. Its detail is left at its mean where a pixel is too wide to draw
/// it.
fn fire(at: vec2<f32>, edge: f32, radius: f32, reach: f32, footprint: f32) -> f32 {
    let t = mouths.shape.w;
    let scale = TONGUE * radius;
    let outward = at / max(length(at), 1e-6);
    let round = vec2(-outward.y, outward.x);
    let flow = (outward * RISE + round * SWIRL + vec2(0.0, DRAFT)) * radius;
    let run = carried(vec3(at, 0.0), vec3(flow, 0.0), t);
    let drawn = 1.0 - smoothstep(0.25 * scale, 0.5 * scale, footprint);
    let a = noise3(vec3(run.a.xy / scale, t * FLICKER));
    let b = noise3(vec3(run.b.xy / scale + 17.0, t * FLICKER));
    let fine_a = noise3(vec3(run.a.xy / scale * 2.7, t * FLICKER * 1.7 + 5.0));
    let fine_b = noise3(vec3(run.b.xy / scale * 2.7 + 9.0, t * FLICKER * 1.7 + 5.0));
    let coarse = mix(b, a, run.weight_a);
    let fine = mix(fine_b, fine_a, run.weight_a);
    let lick = mix(0.5, 0.65 * coarse + 0.35 * fine, drawn);
    if (edge <= 0.0) {
        let inside = edge / (EMBERS * radius);
        return exp(-inside * inside);
    }
    // a tongue stands as far out as the fire is strong under it, and thins toward its tip
    let height = reach * smoothstep(0.15, 0.85, lick);
    let up = saturate(1.0 - edge / max(height, 1e-6));
    return sqrt(up) * up * (0.4 + 0.6 * lick);
}

/// A ray that goes in at a mouth as it comes out of the other: where from, and which way.
struct Emerged {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

fn emerged(i: u32, p: vec3<f32>, dir: vec3<f32>) -> Emerged {
    let mouth = mouths.mouths[i];
    let far = mouths.mouths[1u - i];
    let rows = mouth.through;
    let off = p - mouth.centre.xyz;
    let turned = vec3(dot(rows[0].xyz, off), dot(rows[1].xyz, off), dot(rows[2].xyz, off));
    let heading = vec3(dot(rows[0].xyz, dir), dot(rows[1].xyz, dir), dot(rows[2].xyz, dir));
    return Emerged(far.centre.xyz + turned, heading);
}

fn turned_through(rows: array<vec4<f32>, 3>, v: vec3<f32>) -> vec3<f32> {
    return vec3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

/// A sun's light as it comes out of a mouth, having gone in at the other where the sun reaches
/// that: the way toward it, which is into the mouth, and how wide what is open of the mouth
/// lets it out. There is none while the pair is shut, or where the sun is behind the other
/// mouth's surface.
struct Sunbeam {
    toward: vec3<f32>,
    opening: f32,
    // how steeply it comes out of the mouth's surface, which is less than nothing if it does
    rise: f32,
}

fn sunbeam(pair: Mouths, k: u32, to_sun: vec3<f32>) -> Sunbeam {
    let near = pair.mouths[k];
    let far = pair.mouths[1u - k];
    let opening = pair.shape.y * (1.0 - pair.shape.z);
    if (near.origin.w == 0.0 || far.origin.w == 0.0 || dot(to_sun, far.inward.xyz) <= 0.0) {
        return Sunbeam(vec3(0.0), 0.0, 0.0);
    }
    let toward = turned_through(far.through, to_sun);
    return Sunbeam(toward, opening, -dot(toward, near.inward.xyz));
}

/// Whether a sun reaches the point of the other mouth that light leaving mouth `k` at `off`
/// from its middle went in at.
fn sun_enters(pair: Mouths, k: u32, off: vec3<f32>, to_sun: vec3<f32>) -> bool {
    let entered = pair.mouths[1u - k].centre.xyz + turned_through(pair.mouths[k].through, off);
    return sun_reaches(entered + pair.about.xyz, to_sun, vec2(pair.shape.x, pair.look.z));
}

/// How much of a sun's light a point gets through mouth `k`, and the way toward it there.
fn sunbeam_at(pair: Mouths, k: u32, p: vec3<f32>, to_sun: vec3<f32>) -> vec4<f32> {
    let beam = sunbeam(pair, k, to_sun);
    let near = pair.mouths[k];
    let height = dot(p - near.centre.xyz, near.inward.xyz);
    if (beam.opening <= 0.0 || beam.rise <= 1e-4 || height <= 0.0) {
        return vec4(0.0);
    }
    let run = height / beam.rise;
    let off = p + beam.toward * run - near.centre.xyz;
    let soft = max(run * SUN_WIDTH, 1e-3 * beam.opening);
    let share = 1.0 - smoothstep(beam.opening - soft, beam.opening + soft, length(off));
    if (share <= 0.0 || !sun_enters(pair, k, off, to_sun)) {
        return vec4(0.0);
    }
    return vec4(beam.toward, share);
}

/// The stretch of a ray that a sun's light through mouth `k` falls across, as distances along
/// it from `at`, and the way toward that light: the light fills whatever slides along it onto
/// what is open of the mouth, which is a slanted tube, and a straight ray crosses that once.
struct SunbeamRun {
    entering: f32,
    leaving: f32,
    toward: vec3<f32>,
}

fn sunbeam_run(pair: Mouths, k: u32, at: vec3<f32>, dir: vec3<f32>, distance: f32, to_sun: vec3<f32>) -> SunbeamRun {
    let beam = sunbeam(pair, k, to_sun);
    var none = SunbeamRun(0.0, 0.0, beam.toward);
    if (beam.opening <= 0.0 || beam.rise <= 1e-4) {
        return none;
    }
    let near = pair.mouths[k];
    let n = near.inward.xyz;
    let w = at - near.centre.xyz;
    let height = dot(w, n);
    let climb = dot(dir, n);
    let slid = w + beam.toward * (height / beam.rise);
    let rate = dir + beam.toward * (climb / beam.rise);
    let a = dot(rate, rate);
    let b = 2.0 * dot(slid, rate);
    let c = dot(slid, slid) - beam.opening * beam.opening;
    var entering = 0.0;
    var leaving = distance;
    if (a < 1e-12) {
        if (c > 0.0) {
            return none;
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if (disc <= 0.0) {
            return none;
        }
        let root = sqrt(disc);
        entering = max((-b - root) / (2.0 * a), 0.0);
        leaving = min((-b + root) / (2.0 * a), distance);
    }
    // the light is only before the mouth, in the room
    if (climb > 1e-9) {
        entering = max(entering, -height / climb);
    } else if (climb < -1e-9) {
        leaving = min(leaving, -height / climb);
    } else if (height <= 0.0) {
        return none;
    }
    if (leaving <= entering) {
        return none;
    }
    let middle = 0.5 * (entering + leaving);
    if (!sun_enters(pair, k, slid + rate * middle, to_sun)) {
        return none;
    }
    return SunbeamRun(entering, leaving, beam.toward);
}

/// What is seen through a mouth at a point of it, as the eye is exposed to it.
fn beyond(i: u32, p: vec3<f32>) -> vec3<f32> {
    let clip = mouths.mouths[i].found * vec4(p, 1.0);
    let uv = clip.xy / max(clip.w, 1e-6) * vec2(0.5, -0.5) + 0.5;
    let blue = textureSampleLevel(through_blue, through_sampler, uv, 0.0).rgb;
    let orange = textureSampleLevel(through_orange, through_sampler, uv, 0.0).rgb;
    return select(orange, blue, i == 0u);
}

/// One run of the vortex at a point of the chart, wound for `wound` seconds: the point turned
/// back round the middle by as far as the vortex has turned there, which is further the nearer
/// the middle, so that what was a blot is drawn out into an arm.
fn wound_back(at: vec2<f32>, radius: f32, wound: f32) -> vec2<f32> {
    let across = length(at) / radius;
    let turned = SWIRL * wound / (across + EYE);
    let c = cos(turned);
    let s = sin(turned);
    return vec2(at.x * c + at.y * s, at.y * c - at.x * s);
}

/// How bright the vortex is at a point of the chart, from its troughs to one along its arms.
/// It is wound in two runs that overlap, each faded in and out, so that neither is ever seen
/// to start over, and each is already wound when it fades in, so that it is never seen slack.
fn vortex(at: vec2<f32>, radius: f32, footprint: f32) -> f32 {
    let t = mouths.shape.w;
    let scale = STREAK * radius;
    let phase_a = fract(t / WINDING);
    let phase_b = fract(t / WINDING + 0.5);
    let a = noise3(vec3(wound_back(at, radius, (1.0 + phase_a) * WINDING) / scale, 3.0));
    let b = noise3(vec3(wound_back(at, radius, (1.0 + phase_b) * WINDING) / scale, 11.0));
    let weight_a = 1.0 - abs(2.0 * phase_a - 1.0);
    let arms = smoothstep(0.3, 0.7, mix(b, a, weight_a));
    let drawn = 1.0 - smoothstep(0.25 * scale, 0.5 * scale, footprint);
    let eye = smoothstep(0.0, EYE, length(at) / radius);
    return mix(TROUGH, 1.0, mix(0.5, arms, drawn)) * mix(TROUGH, 1.0, eye);
}

/// What the surface at `p` shows of the portals, where it is of the kind `on` and a pixel is
/// `footprint` metres wide on it, seen from the room or from behind.
fn painted(p: vec3<f32>, on: f32, from_the_room: bool, footprint: f32) -> Painted {
    var shown: Painted;
    shown.paint = 0.0;
    shown.albedo = vec3(0.0);
    shown.open = 0.0;
    shown.mouth = 0u;
    shown.beyond = vec3(0.0);
    shown.glow = vec3(0.0);
    let radius = mouths.shape.y;
    let reach = mouths.look.x * radius;
    for (var i = 0u; i < 2u; i++) {
        let mouth = mouths.mouths[i];
        if (mouth.origin.w != on) {
            continue;
        }
        if (on == CAP && abs(dot(p - mouth.origin.xyz, mouth.inward.xyz)) > 0.5 * radius) {
            continue;
        }
        let at = chart(mouth, p);
        let across = length(at);
        if (across > radius + reach + footprint) {
            continue;
        }
        let grade = mix(mouths.look.y, 1.0, saturate(0.5 + 0.5 * at.y / (radius + reach)));
        let colour = mouth.colour.rgb * grade;
        let within = 1.0 - smoothstep(-0.5 * footprint, 0.5 * footprint, across - radius);
        let shut = select(1.0, mouths.shape.z, from_the_room);
        let opening = radius * (1.0 - shut);
        let past_opening = smoothstep(-0.5 * footprint, 0.5 * footprint, across - opening);
        let filled = within * select(past_opening, 1.0, shut >= 1.0);
        var heat = fire(at, across - radius, radius, reach, footprint);
        if (shut < 1.0 && shut > 0.0) {
            // what is filled in burns at its inner edge too while it draws back
            let inner = (across - opening) / (EMBERS * radius);
            heat = max(heat, exp(-inner * inner) * within);
        }
        let swirl = vortex(at, radius, footprint);
        shown.paint = filled;
        shown.albedo = colour * (PAINT * swirl);
        shown.open = within - filled;
        shown.mouth = i;
        if (shown.open > 0.0) {
            shown.beyond = beyond(i, p);
        }
        // the hottest of the fire burns toward white
        let white = heat * heat * heat * heat * heat * heat;
        let given_off = (mouth.colour.rgb * heat + vec3(white) * 0.35) * FIRE
            + mouth.colour.rgb * (filled * FILLED * swirl);
        // the eye saturates at the brightest it can tell apart, but keeps the colour it saw it
        // in, and what is darker toward the mouth's bottom stays so however bright the fire
        let exposed = given_off * view.exposure;
        let brightest = max(exposed.r, max(exposed.g, exposed.b));
        // fire is told apart by how bright it is over many times its brightness, so it dies
        // down toward the mouth's bottom by more than what is filled in darkens
        shown.glow = exposed * min(1.0, SATURATION / max(brightest, 1e-6)) * grade * grade;
    }
    return shown;
}
