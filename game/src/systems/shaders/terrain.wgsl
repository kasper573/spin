// The ground: grass over dirt where it has been dug toward the glass, tiled in metre squares
// of two tints so that walking over it reads as motion and distance, lit by the sun where the
// ring does not shade it and by the sun's light bounced round the ring elsewhere. Where water
// lies over it, the sunlight reaching it has come down through the water: dimmed by the depth
// it crossed, and gathered into caustics by the ripples on the surface it came through.
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::shadows::fetch_directional_shadow
#import optics::{bounce, sunlight}
#ifdef DISTANCE_FOG
#import bevy_pbr::mesh_view_bindings::fog
#import bevy_pbr::pbr_functions::apply_fog
#endif
#import ripples::{carried, crossing_length, waves_carried}

struct Terrain {
    dirt: vec4<f32>,
    grass: vec4<f32>,
    grass_dark: vec4<f32>,
    // the site everything is drawn about: its place round the ring in segments of the grid,
    // its place along the axis and the glass radius, in metres
    site: vec4<f32>,
    // where the point everything is drawn about lies in the site's frame, in metres
    origin: vec4<f32>,
    // the landscape grid: its angle per segment, its row spacing, the drum's half width, in
    // metres, and the water each particle of a column adds over its footprint
    grid: vec4<f32>,
    // x: seconds; y: metres per unit of a column's height; z: metres per second per unit of
    // a column's flow; w: how many rows and segments the grid has, packed
    clock: vec4<f32>,
    absorption: vec4<f32>,
    scatter: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> terrain: Terrain;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> columns: array<vec4<u32>>;

const PI: f32 = 3.14159265;
const IOR: f32 = 1.333;
const F0: f32 = 0.02;
// the tightest the sun's disc can be focused, as a share of the light it started with
const SHARPEST: f32 = 0.4;
// wet ground is darker, once this many particles' worth of water lies over a column of it
const WET: f32 = 0.35;
const WET_BY: f32 = 4.0;

/// 1 on a light tile, 0 on a dark one, blended over the width of a pixel so the edges stay
/// crisp at any distance without shimmering.
fn checker(u: f32, v: f32) -> f32 {
    let fu = fwidth(u);
    let fv = fwidth(v);
    // the checker as a product of two square waves, each filtered over the pixel footprint
    let su = 1.0 - 2.0 * abs(fract(u) - 0.5);
    let sv = 1.0 - 2.0 * abs(fract(v) - 0.5);
    let a = smoothstep(0.5 - fu, 0.5 + fu, su);
    let b = smoothstep(0.5 - fv, 0.5 + fv, sv);
    return a * (1.0 - b) + b * (1.0 - a);
}

fn fresnel(cos_theta: f32) -> f32 {
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    return F0 + (1.0 - F0) * m * m * m * m * m;
}

/// The water over a point of the ground: how deep it is there, how high above the glass its
/// top reaches, and the flow in it.
struct Column {
    depth: f32,
    top: f32,
    flow: vec3<f32>,
}

/// One column of the survey: the water's thickness over it in metres, the flow in it, and
/// how high above the glass its water reaches.
fn column_at(segment: i32, row: i32) -> vec4<f32> {
    let segments = i32(terrain.clock.w % 4096.0);
    let rows = i32(floor(terrain.clock.w / 4096.0));
    let i = ((segment + segments) % segments) * rows + clamp(row, 0, rows - 1);
    let c = columns[i];
    let count = f32(c.y);
    // the water over the column is its particles' volume over the column's footprint
    let thickness = count * terrain.grid.w;
    var flow = vec2(0.0);
    if (c.y > 0u) {
        flow = vec2(f32(bitcast<i32>(c.z)), f32(bitcast<i32>(c.w))) * terrain.clock.z / count;
    }
    return vec4(thickness, flow.x, flow.y, f32(c.x) * terrain.clock.y);
}

/// The survey read at a point of the site's frame, blended over the four columns round it.
/// The point's place round the ring is its turn from the site, which is small however big
/// the ring is, on top of the site's own.
fn column_over(p: vec3<f32>) -> Column {
    let segments = terrain.clock.w % 4096.0;
    let turn = atan2(p.z, terrain.site.z + p.x);
    let u = (terrain.site.x + turn / terrain.grid.x + segments) % segments - 0.5;
    let v = (p.y + terrain.site.y + terrain.grid.z) / terrain.grid.y - 0.5;
    let i = floor(u);
    let j = floor(v);
    let fu = u - i;
    let fv = v - j;
    let aa = column_at(i32(i), i32(j));
    let ba = column_at(i32(i) + 1, i32(j));
    let ab = column_at(i32(i), i32(j) + 1);
    let bb = column_at(i32(i) + 1, i32(j) + 1);
    let c = mix(mix(aa, ba, fu), mix(ab, bb, fu), fv);
    var out: Column;
    out.depth = c.x;
    out.top = c.w;
    let spinward = normalize(vec3(-p.z, 0.0, terrain.site.z + p.x));
    out.flow = spinward * c.y + vec3(0.0, c.z, 0.0);
    return out;
}

/// How much of the sun's light reaches the bed through this much water, by way of the surface
/// above: what crossing the water leaves of it, and how the ripples it came through gather it.
/// Ripples too small to resolve at a pixel `footprint` wide are left out, since their caustics
/// even out over the pixel rather than showing as a pattern in it.
fn sunlight_through(p: vec3<f32>, up: vec3<f32>, l: vec3<f32>, water: Column, footprint: f32) -> f32 {
    let cos_in = dot(l, up);
    if (cos_in <= 0.02) {
        return 0.0;
    }
    let sin_r2 = (1.0 - cos_in * cos_in) / (IOR * IOR);
    let cos_r = sqrt(1.0 - sin_r2);
    let path = water.depth / cos_r;
    // where the light came through the surface: back up its refracted way
    let down = refract(-l, up, 1.0 / IOR);
    let entry = p - down * path;
    let run = carried(entry, water.flow, terrain.clock.x);
    // waves whose caustics have crossed before the light reaches the bed draw no pattern on
    // it, and neither do those too small to resolve: a wave is left out once the footprint
    // reaches half its length
    let crossed = crossing_length(path, 1.0 - 1.0 / IOR);
    let w = waves_carried(run, terrain.clock.x, max(footprint, 0.5 * crossed));
    // rays bent by the ripples' slopes converge or spread by the time they reach the bed
    let spread = (1.0 - 1.0 / IOR) * path;
    let e1 = vec3(0.0, 1.0, 0.0);
    let e2 = cross(up, e1);
    let m = mat2x2<f32>(
        1.0 + spread * dot(e1, w.curve * e1), spread * dot(e2, w.curve * e1),
        spread * dot(e1, w.curve * e2), 1.0 + spread * dot(e2, w.curve * e2),
    );
    // the sun is a disc rather than a point, so where several waves' bending adds up to cross
    // the rays its focus is never a spike: the brightness rounds off instead of running away,
    // and rounding it off smoothly leaves no edge for the eye to read as a line
    let focus = 1.0 / sqrt(determinant(m) * determinant(m) + SHARPEST * SHARPEST);
    let caustic = mix(1.0, focus, smoothstep(0.0, 0.15, water.depth));
    return (1.0 - fresnel(cos_in)) * caustic;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.world_position.xyz;
    // the ground has a face toward the eye whichever way it was wound; seen from below,
    // through the glass, it is the dirt pressed against the glass
    var n = normalize(in.world_normal);
    let to_eye = view.world_position - p;
    let underside = dot(n, to_eye) < 0.0;
    if (underside) {
        n = -n;
    }
    // how wide a pixel is on the ground here, for a 60 degree view: at a grazing angle a pixel
    // covers a long stretch of it
    let range = length(to_eye);
    let footprint = range * 1.15 / view.viewport.w / max(dot(n, to_eye / range), 0.02);
    var light = 0.0;
    var height = 0.0;
#ifdef VERTEX_UVS_A
    light = checker(in.uv.x, in.uv.y);
#endif
#ifdef VERTEX_UVS_B
    height = in.uv_b.x;
#endif
    let grass = mix(terrain.grass_dark.rgb, terrain.grass.rgb, light);
    var albedo = mix(terrain.dirt.rgb, grass, smoothstep(0.15, 0.45, height));

    // the water lying over this ground, looked up about the site
    let at = p + terrain.origin.xyz;
    let up = -normalize(vec3(terrain.site.z + at.x, 0.0, at.z));
    var water = column_over(at);
    // water flying over the ground, not lying on it, neither wets nor dims it
    let lying = 1.0 - smoothstep(0.3, 0.8, water.top - height - water.depth);
    water.depth *= lying;
    if (underside) {
        albedo = terrain.dirt.rgb;
        water.depth = 0.0;
    }
    // a stray drop dampens a patch, a body of water soaks it
    let wet = smoothstep(0.0, WET_BY * terrain.grid.w, water.depth);
    albedo *= 1.0 - WET * wet;
    let dimmed = exp(-terrain.absorption.rgb * water.depth);

    let view_z = dot(vec4(view.view_from_world[0].z, view.view_from_world[1].z, view.view_from_world[2].z, view.view_from_world[3].z), in.world_position);
    var colour = vec3(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let sun = lights.directional_lights[i];
        let l = sun.direction_to_light;
        let ndl = max(dot(n, l), 0.0);
        if (ndl <= 0.0) {
            continue;
        }
        let shadow = fetch_directional_shadow(i, in.world_position, n, view_z, in.position.xy);
        var through = vec3(1.0);
        if (water.depth > 0.0) {
            let cos_in = dot(l, up);
            let sin_r2 = (1.0 - cos_in * cos_in) / (IOR * IOR);
            let path = water.depth / sqrt(max(1.0 - sin_r2, 1e-3));
            through = exp(-terrain.absorption.rgb * path) * sunlight_through(at, up, l, water, footprint);
        }
        colour += sunlight(i) * albedo / PI * ndl * shadow * through;
    }
    // the light bounced round the ring comes down through the water too
    colour += bounce() * albedo * dimmed;
#ifdef DISTANCE_FOG
    // with the eye under water, everything is seen through it
    return apply_fog(fog, vec4(colour, 1.0), p, view.world_position, in.position.xy);
#else
    return vec4(colour, 1.0);
#endif
}
