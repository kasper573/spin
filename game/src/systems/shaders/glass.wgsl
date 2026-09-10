// The glass: a thick pane that passes nearly all the light through it, tinted by a whisker,
// and mirrors the rest, more the more obliquely it is seen, with the sun glinting off it. It
// is gridded into panes fixed to the wheel by grooves bevelled into its inner face, whose
// slanted sides tilt what they mirror and bend what is seen through them.
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::{view, lights}
#ifdef DISTANCE_FOG
#import bevy_pbr::mesh_view_bindings::fog
#import bevy_pbr::pbr_functions::apply_fog
#endif
#import optics::{mirrored, ring_seen, seen_through, fresnel, glint, sunlight, depth_of, saturated}

struct Glass {
    // how much of each colour a pane lets through
    tint: vec4<f32>,
    // turns a direction of the drum's frame into one among the stars
    to_stars: vec4<f32>,
    background: vec4<f32>,
    // the pane size round the wall and along it, the groove width and the glass thickness
    panes: vec4<f32>,
    origin: vec4<f32>,
    ring: vec4<f32>,
    ground: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> glass: Glass;

const IOR: f32 = 1.52;
// glass mirrors this much face on
const F0: f32 = 0.04;
const ROUGHNESS: f32 = 0.04;
// the grooves' sides fall this steeply: depth over half their width
const SLANT: f32 = 1.0;
// how far behind the glass the view bent by a groove is followed before it is looked up
const REACH: f32 = 3.0;

/// How much of a pixel `w` wide about `x` lies on each side of a groove cut `half` wide about
/// `x = 0`: the side before it and the side beyond it.
fn groove(x: f32, half: f32, w: f32) -> vec2<f32> {
    let before = max(min(x + 0.5 * w, 0.0) - max(x - 0.5 * w, -half), 0.0);
    let beyond = max(min(x + 0.5 * w, half) - max(x - 0.5 * w, 0.0), 0.0);
    return vec2(before, beyond) / w;
}

/// A facet of the glass a pixel shows: how its normal slants from the pane's, and how much
/// of the pixel it covers.
struct Facet {
    slant: vec3<f32>,
    share: f32,
}

/// The groove facet that covers most of this pixel: the glass is gridded into square panes
/// fixed to the wheel, round the wall and across the caps, and the mesh carries each point's
/// place in that grid, in panes, so the grid is exact whatever size the wheel is. The grooves
/// run along the pane edges, and each side of one slants toward its middle.
fn bevel(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> Facet {
    var pane = glass.panes.xy;
    if (abs(n.y) > 0.5) {
        pane = vec2(glass.panes.y);
    }
    let half = glass.panes.z * 0.5 / pane;
    let x = fract(uv + 0.5) - 0.5;
    let w = max(fwidth(uv), vec2(1e-6));
    let u = groove(x.x, half.x, w.x);
    let v = groove(x.y, half.y, w.y);
    // the directions the grid runs in, from how the grid and the world change across the pixel
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    var out = Facet(vec3(0.0), 0.0);
    if (abs(det) < 1e-12) {
        return out;
    }
    let along_u = normalize((dp1 * duv2.y - dp2 * duv1.y) / det);
    let along_v = normalize((dp2 * duv1.x - dp1 * duv2.x) / det);
    let shares = vec4(u.x, u.y, v.x, v.y);
    let most = max(max(shares.x, shares.y), max(shares.z, shares.w));
    if (most <= 0.0) {
        return out;
    }
    var slant = along_u * SLANT;
    if (most == shares.y) {
        slant = -along_u * SLANT;
    } else if (most == shares.z) {
        slant = along_v * SLANT;
    } else if (most == shares.w) {
        slant = -along_v * SLANT;
    }
    out.slant = slant;
    out.share = min(u.x + u.y + v.x + v.y, 1.0);
    return out;
}

/// What a facet of the glass with normal `face` shows at `p`, seen along `v`: what passes
/// through the thick glass, bent in at this face and out through the flat one behind it, and
/// what the face mirrors, with the sun glinting off it.
fn shade(face: vec3<f32>, n: vec3<f32>, p: vec3<f32>, v: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let cos_theta = max(dot(face, v), 0.0);
    // a pane has two faces: what the first passes, the second mirrors back in part
    let f = fresnel(cos_theta, F0);
    let mirrored_share = f + (1.0 - f) * (1.0 - f) * f / (1.0 - f * f);
    var inside = refract(-v, face, 1.0 / IOR);
    if (all(inside == vec3(0.0))) {
        inside = -v;
    }
    let exit = p + inside * (glass.panes.w / max(dot(inside, -n), 0.2));
    var beyond = refract(inside, n, IOR);
    if (all(beyond == vec3(0.0))) {
        beyond = inside;
    }
    let passed = seen_through(uv, depth_of(p), exit + beyond * REACH) * glass.tint.rgb;
    let reflected = reflect(-v, face);
    let mirror = mirrored(p, reflected, ring_seen(p + glass.origin.xyz, reflected, glass.ring.xy, glass.ground.rgb, glass.to_stars, glass.background.rgb));
    var colour = mix(passed, mirror, mirrored_share);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        colour += sunlight(i) * glint(face, v, l, ROUGHNESS, F0);
    }
    return colour;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.world_position.xyz;
    let v = normalize(view.world_position - p);
    // a pane has a face toward the eye whichever way it was wound
    var n = normalize(in.world_normal);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let uv = (in.position.xy - view.viewport.xy) / view.viewport.zw;
    var colour = shade(n, n, p, v, uv);
#ifdef VERTEX_UVS_A
    // where a groove crosses the pixel, its slanted side shows in its share of it: the side
    // that faces the eye, since the other is hidden behind it
    let facet = bevel(n, p, in.uv);
    if (facet.share > 0.0) {
        var face = normalize(n + facet.slant);
        if (dot(face, v) < 0.1) {
            face = normalize(n - facet.slant);
        }
        colour = mix(colour, shade(face, n, p, v, uv), facet.share);
    }
#endif
    let out = vec4(saturated(colour), 1.0);
#ifdef DISTANCE_FOG
    // with the eye under water, the glass is seen through it
    return apply_fog(fog, out, p, view.world_position, in.position.xy);
#else
    return out;
#endif
}
