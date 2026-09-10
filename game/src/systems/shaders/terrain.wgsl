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
#import ripples::{carried, crossing_length, noise3, waves_carried}

struct Terrain {
    dirt: vec4<f32>,
    grass: vec4<f32>,
    grass_dark: vec4<f32>,
    bed: vec4<f32>,
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
// the widest the survey is ever gathered, in cells: water coarser than this is read as if its
// grain were this wide, which costs it a little smoothing rather than an unbounded gather
const GATHER: f32 = 4.0;
// water standing this deep has laid its bed down over the ground and drowned what grew there,
// over a stretch of shore rather than at a line, since a shore is never a line
const BED_BY: f32 = 0.9;

// the ripple marks a current leaves in a sandy bed: their spacing, their height, how far
// their crests meander and turn out of true, and the spacing and depth of the grain
// speckling the sand between them
const SAND_RIPPLE: f32 = 0.13;
const SAND_RELIEF: f32 = 0.009;
const SAND_MEANDER: f32 = 3.0;
const SAND_TURN: f32 = 0.35;
const SAND_GRAIN: f32 = 0.05;
const SAND_MOTTLE: f32 = 0.16;
// the patchiness of the bed itself, which is what is left of it once the ripples and the grain
// are finer than a pixel: coarse sand against fine, and the looser sand of a ripple field
const SAND_PATCH: f32 = 1.1;
const SAND_PATCHY: f32 = 0.20;
const SAND_WORKED: f32 = 0.13;

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

/// The survey read at a point of the site's frame. The point's place round the ring is its turn
/// from the site, which is small however big the ring is, on top of the site's own.
///
/// A column holds the particles standing over one cell of the grid, and the cells are finer
/// than the water is grained: over shallow water a cell catches one particle or none, so its
/// count steps between whole particles and its top jumps by a particle's width. Reading a
/// single cell puts those steps into the ground, as a pattern in the grid's own shape, which
/// is the grid showing rather than the water. The field is gathered over the water's grain
/// instead, since nothing about the water's surface is known finer than the particles it is
/// made of, and what is gathered is smooth to the width of one of them.
fn column_over(p: vec3<f32>) -> Column {
    let segments = terrain.clock.w % 4096.0;
    let turn = atan2(p.z, terrain.site.z + p.x);
    let u = (terrain.site.x + turn / terrain.grid.x + segments) % segments - 0.5;
    let v = (p.y + terrain.site.y + terrain.grid.z) / terrain.grid.y - 0.5;
    let i = floor(u);
    let j = floor(v);
    let fu = u - i;
    let fv = v - j;
    // how wide a cell is on the ground, and how far apart the water's particles stand: a
    // particle's water spread over a cell is the depth one of them adds there, so the cube
    // root of that depth times the cell's area is the spacing they sit at
    let across = terrain.site.z * terrain.grid.x;
    let along = terrain.grid.y;
    let grain = pow(terrain.grid.w * across * along, 1.0 / 3.0);
    let reach = clamp(vec2(grain / across, grain / along), vec2(1.0), vec2(GATHER));
    // only the cells within a reach of the point carry any weight, so those are the ones read:
    // where the water is no coarser than the grid, that is the four cells round it and no more
    let first = vec2<i32>(ceil(vec2(fu, fv) - reach));
    let last = vec2<i32>(floor(vec2(fu, fv) + reach));
    // how much water stands over a cell is a density, and gathers by area. How high it reaches
    // and which way it flows belong to the water that is there rather than to the cell, so they
    // gather weighted by it: a cell with no water in it has no height to lend its neighbours.
    var thickness = 0.0;
    var area = 0.0;
    var carried = vec2(0.0);
    var top = 0.0;
    var held = 0.0;
    for (var dj = first.y; dj <= last.y; dj++) {
        let wv = max(1.0 - abs(f32(dj) - fv) / reach.y, 0.0);
        for (var di = first.x; di <= last.x; di++) {
            let w = max(1.0 - abs(f32(di) - fu) / reach.x, 0.0) * wv;
            let cell = column_at(i32(i) + di, i32(j) + dj);
            thickness += cell.x * w;
            area += w;
            carried += cell.yz * (cell.x * w);
            top += cell.w * (cell.x * w);
            held += cell.x * w;
        }
    }
    let c = vec4(
        thickness / max(area, 1e-6),
        carried / max(held, 1e-6),
        top / max(held, 1e-6),
    );
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
    // How deep the water lies over this ground: the height its top reaches above the ground,
    // which is what light crossing it has to cross. The particles counted over a column say how
    // much water is there, but only in whole particles, which over shallow water is a handful:
    // taking the depth from that count would put the count's own steps into the ground's colour
    // and its light. The count is still what tells water lying on the ground from spray flying
    // over it, which reaches no deeper than the water it is made of.
    let span = max(water.top - height, 0.0);
    let lying = 1.0 - smoothstep(0.3, 0.8, span - water.depth);
    water.depth = span * lying;
    if (underside) {
        albedo = terrain.dirt.rgb;
        water.depth = 0.0;
    }
    // where water stands, the ground is the water's bed: the carbonate settled out of it, with
    // nothing growing under it, shading back into the bank over the shallows at the shore
    albedo = mix(albedo, terrain.bed.rgb, smoothstep(0.0, BED_BY, span * lying));
    // a stray drop dampens a patch, a body of water soaks it
    let wet = smoothstep(0.0, WET_BY * terrain.grid.w, water.depth);
    albedo *= 1.0 - WET * wet;
    // the bed's ripples and grain, which are what the light coming down through the water has
    // to break over: without them the sand takes the light evenly and reads as a flat sheet
    let bedded = smoothstep(0.0, BED_BY, span * lying);
    if (bedded > 0.0) {
        // the ripples lie across the water's run, or across the ring where it barely moves
        let spinward = normalize(vec3(-at.z, 0.0, terrain.site.z + at.x));
        let run = water.flow - up * dot(water.flow, up);
        let side = cross(up, run);
        let along = select(spinward, side / max(length(side), 1e-6), length(side) > 1e-3);
        // no bed is a corrugation: the crests turn slowly out of true, meander over a few
        // wavelengths, and give out over stretches the water has left alone
        let turn = (noise3(at / (SAND_RIPPLE * 40.0)) - 0.5) * SAND_TURN;
        let across = normalize(cross(up, along) + along * turn);
        let meander = (noise3(at / (SAND_RIPPLE * 4.0)) - 0.5)
            + (noise3(at / (SAND_RIPPLE * 15.0)) - 0.5) * 2.0;
        let phase = dot(at, across) * 2.0 * PI / SAND_RIPPLE + meander * SAND_MEANDER;
        let worked = smoothstep(0.3, 0.7, noise3(at / (SAND_RIPPLE * 25.0)));
        // each of the three scales is left at its mean once a pixel is too wide to draw it,
        // so what a bed loses with distance is its grain first and its patchiness last
        let crisp = 1.0 - smoothstep(0.25 * SAND_RIPPLE, 0.5 * SAND_RIPPLE, footprint);
        let relief = SAND_RELIEF * bedded * crisp * worked;
        let slope = relief * cos(phase) * 2.0 * PI / SAND_RIPPLE;
        n = normalize(n - across * slope);
        let grained = 1.0 - smoothstep(0.25 * SAND_GRAIN, 0.5 * SAND_GRAIN, footprint);
        let grain = (noise3(at / SAND_GRAIN) - 0.5) * SAND_MOTTLE * grained;
        let broad = 1.0 - smoothstep(0.25 * SAND_PATCH, 0.5 * SAND_PATCH, footprint);
        // sand the water has worked lies looser and darker than sand it has left flat
        let coarse = ((noise3(at / SAND_PATCH) - 0.5) * SAND_PATCHY - worked * SAND_WORKED) * broad;
        albedo *= 1.0 + (grain + coarse) * bedded;
    }
    let extinction = terrain.absorption.rgb + terrain.scatter.rgb;
    let dimmed = exp(-extinction * water.depth);

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
            through = exp(-extinction * path) * sunlight_through(at, up, l, water, footprint);
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
