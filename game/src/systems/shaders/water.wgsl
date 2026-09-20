// Water, drawn from the surface the GPU extracted. A pixel of water shows what is behind it,
// refracted and dimmed by the water it was seen through; what is around it, mirrored, the ring
// found by marching the reflected ray across the depth of the scene and space beyond that; the
// sun glinting off it; and foam where it churns. Ripples ride on the flow, so still water lies
// like glass and moving water sparkles. Seen from under the surface the same rules give the
// mirror of the bed beyond the critical angle and the world above within it. After the
// surface come the parcels of water too small for the extraction's grid to draw, each a square
// facing the eye whose every pixel casts a ray at the surface the grid would have found had it
// been fine enough: a ball of the parcel's water, or the puddle it lies in once it has come
// down on a wall. What misses is discarded, and what hits takes that surface's depth and is
// shaded as all the water is.
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world}
#import optics::{rotate, ring_seen, seen_through, scene_depth, fresnel, glint, lamp_glint_at, sunbeam_glints_at, diffuse_light_at, sunlight, sun_shadow, depth_of, saturated, through_water, through_ring_air}
#import figure::mirrored
#import ring::{ring_run, ring_up}
#import ripples::{Carried, carried, waves_carried, noise3}
#import air::Air
#import fluid_common::spray_of

struct Water {
    to_stars: vec4<f32>,
    from_water: vec4<f32>,
    origin: vec4<f32>,
    ring: vec4<f32>,
    ground: vec4<f32>,
    background: vec4<f32>,
    units: vec4<f32>,
    clock: vec4<f32>,
    absorption: vec4<f32>,
    scatter: vec4<f32>,
    air: Air,
}

struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    velocity: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> water: Water;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> vertices: array<SurfaceVertex>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<storage, read> indices: array<u32>;
// vertex count, index count, block count, then droplet count
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<storage, read> counters: array<u32>;
// xyz: position like a surface vertex's; w: foam
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage, read> droplets: array<vec4<f32>>;
// how much water each line of sight crosses, counted in steps and what is left over of one,
// and how many more faces it leaves the water by than enters it by: see `water_column.wgsl`,
// whose step this is
@group(#{MATERIAL_BIND_GROUP}) @binding(15) var columns: texture_2d<f32>;
const COLUMN_STEP: f32 = 4.0;

const PI: f32 = 3.14159265;
// water's index against the air it is seen through rather than against vacuum
const IOR: f32 = 1.333 / 1.000293;
// air to water, so the mirror is faint face on
const F0: f32 = 0.02;
// a parcel come down on the ground or the glass lies on it as a puddle: how wide it spreads,
// in radii of a ball of its water, which with its volume says how thick it lies, and how far
// its middle stands off the wall it lies on
const PUDDLE_WIDE: f32 = 2.5;
const LAIN_OFF: f32 = 0.806;
// and how far from its middle a parcel may be seen, in the same: to the rim of its puddle
const PUDDLE_REACH: f32 = 3.2;
// how white a drop of water that was all froth when it was thrown clear is
const FROTHY: f32 = 0.6;
const SPRAY_WHITE: f32 = 0.9;
// water thinner than this many cells of the extraction grid is a film whose edge the grid
// cannot make out: it is drawn the fainter the thinner it is, rather than ending at a rim the
// grid's own shape gave it
const FILM: f32 = 0.4;
// a sheet of water thinner than this, in metres, is drawn the fainter the thinner it is
const SHEET_FILM: f32 = 0.001;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) foam: f32,
    // in the water's frame about the axis, in metres, so surface detail turns with the water
    @location(3) wheel_position: vec3<f32>,
    // the flow in the water's frame, metres per second
    @location(4) wheel_velocity: vec3<f32>,
    // on a parcel's square: the parcel's centre, and the radius of a ball of what is left of
    // its water, otherwise zero
    @location(5) drop: vec4<f32>,
    // and on a parcel's square, the wall it has come down on: which way the wall faces, and
    // how much of the way the parcel has come from a particle's width off it to lying against it
    @location(6) @interpolate(flat) lain: vec4<f32>,
    // on the surface of a sheet of water lying on the floor, how deep it stands there;
    // otherwise negative
    @location(7) standing: f32,
}

/// A point of the water's surface as a pixel sees it, however the surface was come by.
struct Surfaced {
    pixel: vec2<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    foam: f32,
    wheel_position: vec3<f32>,
    wheel_velocity: vec3<f32>,
    // how much water lies behind the point along the line of sight, in metres
    behind: f32,
    // how much of a film too thin to draw the water is here: nothing of one at 1
    film: f32,
    // how much of the foam's lace shows: froth floats on a surface in patches the flow
    // carries, but is mixed all through a parcel thrown clear of it, which is evenly white
    laced: f32,
}

struct Shaded {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vertex(@location(0) numbered: vec3<f32>, @builtin(instance_index) instance: u32) -> Fragment {
    var out: Fragment;
    let i = u32(numbered.x);
    out.drop = vec4(0.0);
    out.lain = vec4(0.0);
    out.standing = -1.0;
    let world_from_local = get_world_from_local(instance);
    if (i < counters[1]) {
        let v = vertices[indices[i]];
        let world = mesh_position_local_to_world(world_from_local, vec4(v.position.xyz, 1.0));
        out.clip = view.clip_from_world * world;
        out.world_position = world.xyz;
        out.world_normal = mesh_normal_local_to_world(v.normal.xyz, instance);
        out.foam = v.position.w;
        out.wheel_position = v.position.xyz * water.units.x;
        out.wheel_velocity = v.velocity.xyz * water.clock.y;
        if (water.units.z > 0.5) {
            out.standing = v.normal.w;
        }
        return out;
    }
    let square = i - counters[1];
    let drop = square / 6u;
    if (drop >= counters[3]) {
        // past the parcels: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let parcel = droplets[3u * drop];
    let motion = droplets[3u * drop + 1u];
    let lain = droplets[3u * drop + 2u];
    let centre = mesh_position_local_to_world(world_from_local, vec4(parcel.xyz, 1.0)).xyz;
    let to_eye = view.world_position - centre;
    let distance = length(to_eye);
    let facing = to_eye / distance;
    var right = cross(vec3(0.0, 1.0, 0.0), facing);
    if (dot(right, right) < 1e-6) {
        right = cross(vec3(1.0, 0.0, 0.0), facing);
    }
    right = normalize(right);
    let up = cross(facing, right);
    // what the air has torn off the parcel is no longer in it
    let radius = water.clock.z * pow(1.0 - spray_of(motion.w), 1.0 / 3.0);
    // the outline of all the parcel may reach to, on a square through its centre, widens as
    // the eye comes close
    let reach = radius * select(1.0, PUDDLE_REACH, lain.w > 0.0) + distance * 1.15 / view.viewport.w;
    let outline = reach / sqrt(max(1.0 - reach * reach / (distance * distance), 0.05));
    let corner = square % 6u;
    let x = select(-1.0, 1.0, corner == 1u || corner == 2u || corner == 4u);
    let y = select(-1.0, 1.0, corner == 2u || corner == 4u || corner == 5u);
    let world = centre + (right * x + up * y) * outline;
    out.clip = view.clip_from_world * vec4(world, 1.0);
    out.world_position = world;
    out.drop = vec4(centre, radius);
    out.foam = parcel.w;
    out.wheel_position = parcel.xyz * water.units.x;
    out.wheel_velocity = motion.xyz * water.clock.y;
    let faces = (world_from_local * vec4(lain.xyz, 0.0)).xyz;
    out.lain = vec4(faces / max(length(faces), 1e-9), lain.w);
    return out;
}

/// What a ray loses to a metre of water, and the colour the water shows of its own where light
/// `light` falls on it, being the share of a ray that scattering rather than absorption takes.
fn extinction() -> vec3<f32> {
    return water.absorption.rgb + water.scatter.rgb;
}

fn glow(light: vec3<f32>) -> vec3<f32> {
    return water.scatter.rgb / extinction() * light;
}

/// Where a ray meets the puddle a parcel lies in on a wall: the cap of a ball, as wide as the
/// parcel has spread and as thick at its middle as its water then lies, flat on the wall
/// facing `up` at `foot`, which it meets at the shallow angle water meets what it wets at.
/// Gives how far along the ray, and the face's normal there; nothing is met at a negative
/// distance.
struct PuddleMet {
    t: f32,
    n: vec3<f32>,
    // how far the ray runs through the puddle
    through: f32,
}

fn puddle_met(eye: vec3<f32>, dir: vec3<f32>, foot: vec3<f32>, up: vec3<f32>, wide: f32, thick: f32) -> PuddleMet {
    let round = (wide * wide + thick * thick) / (2.0 * thick);
    let middle = foot - up * (round - thick);
    let o = eye - middle;
    let b = dot(o, dir);
    let h = b * b - (dot(o, o) - round * round);
    if (h <= 0.0) {
        return PuddleMet(-1.0, up, 0.0);
    }
    let root = sqrt(h);
    var enters = -b - root;
    var leaves = -b + root;
    // only what stands on the wall's own side is puddle
    let sinking = dot(dir, up);
    let floor_at = -dot(eye - foot, up) / select(sinking, 1e-9, abs(sinking) < 1e-9);
    var flat = false;
    if (sinking < 0.0) {
        leaves = min(leaves, floor_at);
    } else if (floor_at > enters) {
        enters = floor_at;
        flat = true;
    }
    enters = max(enters, 0.0);
    if (leaves <= enters) {
        return PuddleMet(-1.0, up, 0.0);
    }
    if (flat) {
        return PuddleMet(enters, -up, leaves - enters);
    }
    return PuddleMet(enters, (o + dir * enters) / round, leaves - enters);
}

/// A pixel of a parcel's square: the ball of the parcel's water where the pixel's ray meets
/// it, or the puddle it lies in on the wall it has come down on, its water going from the one
/// to the other as it comes down.
fn parcel(in: Fragment) -> Shaded {
    let eye = view.world_position;
    let dir = normalize(in.world_position - eye);
    let centre = in.drop.xyz;
    let lain = in.lain.w;
    let up = in.lain.xyz;
    // a pixel is this wide at the parcel
    let pixel_wide = length(centre - eye) * 1.15 / view.viewport.w;

    var t = -1.0;
    var n = vec3(0.0);
    var through = 0.0;
    var covers = 1.0;
    var film = 1.0;
    let ball = in.drop.w * pow(max(1.0 - lain * lain, 0.0), 1.0 / 3.0);
    if (ball > 0.0) {
        let oc = eye - centre;
        let b = dot(oc, dir);
        let miss = sqrt(max(dot(oc, oc) - b * b, 0.0));
        // the ball's rim covers a share of the pixels it crosses
        let rim = (ball - miss) / pixel_wide + 0.5;
        if (rim > 0.0 && -b > 0.0) {
            let half = sqrt(max(ball * ball - miss * miss, 0.0));
            t = max(-b - half, 0.0);
            n = normalize(oc + dir * t);
            through = 2.0 * half;
            covers = min(rim, 1.0);
        }
    }
    if (lain > 0.0) {
        var wide = PUDDLE_WIDE * in.drop.w * lain;
        // as thick as so wide a cap of the parcel's water is when it has all come down
        let thick = 8.0 / 3.0 * in.drop.w / (PUDDLE_WIDE * PUDDLE_WIDE);
        let foot = centre - up * (LAIN_OFF * in.drop.w * (2.0 - lain));
        // no ground is so even that water spreads over it in a circle: it runs out further
        // one way than another, each puddle its own way
        let sinking = dot(dir, up);
        let floor_at = -dot(eye - foot, up) / select(sinking, 1e-9, abs(sinking) < 1e-9);
        let over = eye + dir * floor_at - foot;
        wide *= mix(0.65, 1.15, noise3(over * (1.6 / (PUDDLE_WIDE * in.drop.w)) + in.wheel_position * 7.0));
        let puddle = puddle_met(eye, dir, foot, up, wide, thick);
        if (puddle.t >= 0.0 && (t < 0.0 || puddle.t < t)) {
            t = puddle.t;
            n = puddle.n;
            through = puddle.through;
            covers = 1.0;
            // the rim of a puddle thins away to nothing
            film = smoothstep(0.0, 0.25 * thick, through);
        }
    }
    if (t < 0.0) {
        discard;
    }
    let hit = eye + dir * t;
    let from_water = vec4(-water.from_water.xyz, water.from_water.w);
    let shown = surface(Surfaced(in.clip.xy, hit, n, in.foam, in.wheel_position + rotate(from_water, hit - centre), in.wheel_velocity, through, film, 0.0));
    let clip = view.clip_from_world * vec4(hit, 1.0);
    var out: Shaded;
    out.depth = clip.z / clip.w;
    out.colour = vec4(shown.rgb, covers);
    return out;
}

@fragment
fn fragment(in: Fragment) -> Shaded {
    if (in.drop.w > 0.0) {
        return parcel(in);
    }
    let counted = textureLoad(columns, vec2<i32>(in.clip.xy), 0).xyz;
    var behind = max(counted.x * COLUMN_STEP + counted.y, 0.0);
    if (counted.z < -0.5) {
        // a line of sight that goes into more faces than it comes out of has gone into water
        // that does not end in the ring: the surface is left open where the water runs on
        // through a portal, and that water is as long as the ring holds any, to the mouth
        let away = in.world_position - view.world_position;
        let distance = length(away);
        let held = ring_run(in.world_position + water.origin.xyz, away / distance, water.ring.xy).distance;
        behind = max(counted.x * COLUMN_STEP + counted.y + distance + held, 0.0);
    }
    var film = smoothstep(0.0, FILM * water.units.y, behind);
    if (in.standing >= 0.0) {
        // a sheet lies on the floor, so the water behind its surface reaches whatever the scene
        // shows there, and how thin it is is known rather than made out
        behind = 1e9;
        film = smoothstep(0.0, SHEET_FILM, in.standing);
    }
    var out: Shaded;
    out.depth = in.clip.z;
    out.colour = surface(Surfaced(in.clip.xy, in.world_position, in.world_normal, in.foam, in.wheel_position, in.wheel_velocity, behind, film, 1.0));
    return out;
}

/// What is seen through the surface at a point, averaged over a patch `spread` wide across
/// the way `r` bends, so that the detail a pixel's own spread of slopes scrambles evens out
/// instead of showing as a pattern.
fn spread_over(uv: vec2<f32>, depth_here: f32, exit: vec3<f32>, r: vec3<f32>, spread: f32) -> vec3<f32> {
    if (spread < 1e-4) {
        return seen_through(uv, depth_here, exit);
    }
    var e1 = cross(r, vec3(0.0, 1.0, 0.0));
    if (dot(e1, e1) < 1e-6) {
        e1 = cross(r, vec3(1.0, 0.0, 0.0));
    }
    e1 = normalize(e1) * spread;
    let e2 = normalize(cross(r, e1)) * spread;
    return 0.25 * (seen_through(uv, depth_here, exit + e1)
        + seen_through(uv, depth_here, exit - e1)
        + seen_through(uv, depth_here, exit + e2)
        + seen_through(uv, depth_here, exit - e2));
}

/// The bubbles of the foam at a scale, carried on the flow: those too small to resolve at a
/// pixel this wide are left at their mean, so far foam is even rather than speckled.
fn bubble_grain(run: Carried, scale: f32, footprint: f32) -> f32 {
    let resolved = smoothstep(0.5 / scale, 0.1 / scale, footprint);
    if (resolved <= 0.0) {
        return 0.5;
    }
    let grain = noise3(run.a * scale) * run.weight_a + noise3(run.b * scale) * (1.0 - run.weight_a);
    return mix(0.5, grain, resolved);
}

/// A pixel of the water's surface.
fn surface(in: Surfaced) -> vec4<f32> {
    let eye = view.world_position;
    let to_eye = eye - in.world_position;
    let distance = length(to_eye);
    let v = to_eye / distance;
    var n = normalize(in.world_normal);
    // the surface is seen from within the water when its outward normal faces away
    let submerged = dot(n, v) < 0.0;
    if (submerged) {
        n = -n;
    }

    // ripples carried on the flow, growing with it and with the churn of foam
    let t = water.clock.x;
    let flow = in.wheel_velocity;
    let churn = clamp(length(flow) * 1.5 + in.foam * 2.0, 0.0, 1.0);
    let run = carried(in.wheel_position, flow, t);
    // how wide a pixel is on the water here, for a 60 degree view: at a grazing angle a pixel
    // covers a long stretch of it, so the detail that stretch would average out is left out
    let footprint = distance * 1.15 / view.viewport.w / max(abs(dot(n, v)), 0.02);
    let waves = waves_carried(run, t, footprint);
    let slope = waves.slope;
    let amplitude = 1.0 + churn;
    let g = rotate(water.from_water, slope) * amplitude;
    n = normalize(n + g - n * dot(n, g));
    if (dot(n, v) < 0.0) {
        n = normalize(n - 2.0 * v * dot(n, v) + v * 1e-3);
    }
    let roughness = 0.06 + 0.14 * churn;
    // how much sky this pixel's mirrored ray covers: twice how far the surface's own slope turns
    // across the pixel, and never less than the pixel's own width on the sky
    let bend = length(waves.curve[0]) + length(waves.curve[1]) + length(waves.curve[2]);
    let sky = 2.0 * bend * footprint * amplitude + 1.15 / view.viewport.w;

    let uv = (in.pixel - view.viewport.xy) / view.viewport.zw;
    let depth_here = depth_of(in.world_position);
    let scene = scene_depth(uv);
    // how much water lies behind this point of the surface, along the line of sight: as much
    // as was counted there, no further than the scene behind it, and no further than the ring
    // holds water, since the glass and space beyond it stand in no depth. What was counted
    // for an eye under water is the water before the surface, not behind it.
    let along = distance / max(depth_here, 1e-4);
    let site = in.world_position + water.origin.xyz;
    let held = ring_run(site, -v, water.ring.xy).distance;
    let column = min(min(max(scene - depth_here, 0.0) * along, held), select(in.behind, 1e9, submerged));

    // the light falling on the water here, which whatever it scatters is lit by
    let pixel = in.pixel;
    let light = diffuse_light_at(in.world_position, n, pixel) / PI;
    let up = ring_up(site, water.ring.x);
    let reflected = reflect(-v, n);
    // beyond the scene the mirror shows the ring, or from under water the bed, lit by the
    // light coming down through the surface and about as far off as the surface is
    var beyond = ring_seen(water.air, site, reflected, water.ring.xy, water.ground.rgb, water.to_stars, water.background.rgb, sky);
    if (submerged) {
        beyond = through_water(water.ground.rgb * light, glow(light), extinction(), dot(reflected, up), distance);
    }
    // from outside the drum the screen shows the far sides of everything the mirror would
    // show the near sides of, and from under water the bed the surface mirrors lies behind
    // the surface itself, so the mirror is only marched across the screen from inside the air
    var mirror = mirrored(in.world_position, reflected, beyond, site, water.ring.xy, water.ring.z > 0.5 && !submerged);
    var seen: vec3<f32>;
    var f: f32;
    if (!submerged) {
        let r = refract(-v, n, 1.0 / IOR);
        let reach = min(column, 6.0) * 0.5;
        // a pixel covers a stretch of the surface with a spread of slopes, which bends what it
        // shows over a patch of the scene rather than a point of it
        let spread = reach * (1.0 - 1.0 / IOR) * bend * footprint * amplitude;
        seen = through_water(spread_over(uv, depth_here, in.world_position + r * reach, r, spread), glow(light), extinction(), dot(r, up), column);
        f = fresnel(dot(n, v), F0);
    } else {
        let r = refract(-v, n, IOR);
        let total = all(r == vec3(0.0));
        if (total) {
            seen = vec3(0.0);
            f = 1.0;
        } else {
            seen = seen_through(uv, depth_here, in.world_position + r * 3.0);
            f = fresnel(dot(n, v), F0);
        }
        // the mirror of the bed is seen through water too
        mirror = through_water(mirror, glow(light), extinction(), dot(reflected, up), column);
    }

    var colour = mix(seen, mirror, f);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        colour += sunlight(i) * glint(n, v, l, roughness, F0) * sun_shadow(i, in.world_position, n, pixel);
    }
    colour += lamp_glint_at(in.world_position, n, v, roughness, F0, pixel) + sunbeam_glints_at(in.world_position, n, v, roughness, F0);

    // foam: bubbles ride on the flow like the ripples
    let grain = bubble_grain(run, 24.0, footprint);
    let grain2 = bubble_grain(run, 7.0, footprint);
    let lace = in.foam + ((grain - 0.5) * 0.5 + (grain2 - 0.5) * 0.35) * in.laced;
    let foam = smoothstep(0.42, 0.7, lace);
    // bubbles: brighter where the lace is thick, with dark water showing between them
    let bubbles = 0.55 + 0.45 * smoothstep(0.55, 0.95, lace + (grain - 0.5) * 0.6 * in.laced);
    let foam_colour = vec3(0.7, 0.74, 0.78) / PI * diffuse_light_at(in.world_position, n, pixel) * bubbles;
    colour = mix(colour, foam_colour, foam);

    if (submerged) {
        colour = through_water(colour, glow(light), extinction(), dot(-v, up), distance);
    }
    // at the shore the sheet thins away to nothing over a stretch the grid cannot resolve, so
    // it is given up rather than ended on the grid's own staircase
    colour = mix(seen_through(uv, depth_here, in.world_position), colour, select(in.film, 1.0, submerged));
    if (!submerged) {
        // with the eye out of the water, the air between it and the surface stands in the way
        colour = through_ring_air(colour, water.air, site, v, distance, water.ring.xy);
    }
    return vec4(saturated(colour), 1.0);
}
