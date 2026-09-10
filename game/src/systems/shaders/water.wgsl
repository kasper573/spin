// Water, drawn from the surface the GPU extracted. A pixel of water shows what is behind it,
// refracted and dimmed by the water it was seen through; what is around it, mirrored, the ring
// found by marching the reflected ray across the depth of the scene and space beyond that; the
// sun glinting off it; and foam where it churns. Ripples ride on the flow, so still water lies
// like glass and moving water sparkles. Seen from under the surface the same rules give the
// mirror of the bed beyond the critical angle and the world above within it. After the
// surface come the droplets the extraction left out of it, each a square facing the eye whose
// every pixel casts a ray at the sphere of the drop's volume: what misses is discarded, and
// what hits takes the sphere's depth and is shaded as the water is.
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world}
#import optics::{rotate, mirrored, ring_run, ring_seen, seen_through, scene_depth, fresnel, glint, diffuse_light_at, sunlight, sun_shadow, depth_of, saturated}
#import ripples::{Carried, carried, waves_carried, noise3}

struct Water {
    to_stars: vec4<f32>,
    from_water: vec4<f32>,
    origin: vec4<f32>,
    ring: vec4<f32>,
    ground: vec4<f32>,
    background: vec4<f32>,
    anchor: vec4<f32>,
    clock: vec4<f32>,
    absorption: vec4<f32>,
    scatter: vec4<f32>,
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

const PI: f32 = 3.14159265;
const IOR: f32 = 1.333;
// air to water, so the mirror is faint face on
const F0: f32 = 0.02;
// how far through water anything can be seen at all
const FARTHEST: f32 = 200.0;
const DROP_ROUGHNESS: f32 = 0.05;
// how far beyond a drop what is seen through it is taken from
const DROP_REACH: f32 = 1.0;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) foam: f32,
    // in the water's frame about the axis, in metres, so surface detail turns with the water
    @location(3) wheel_position: vec3<f32>,
    // the flow in the water's frame, metres per second
    @location(4) wheel_velocity: vec3<f32>,
    // on a droplet's square: the drop's centre and radius, otherwise zero
    @location(5) drop: vec4<f32>,
}

struct Shaded {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vertex(@builtin(vertex_index) i: u32, @builtin(instance_index) instance: u32) -> Fragment {
    var out: Fragment;
    out.drop = vec4(0.0);
    let world_from_local = get_world_from_local(instance);
    if (i < counters[1]) {
        let v = vertices[indices[i]];
        let world = mesh_position_local_to_world(world_from_local, vec4(v.position.xyz, 1.0));
        out.clip = view.clip_from_world * world;
        out.world_position = world.xyz;
        out.world_normal = mesh_normal_local_to_world(v.normal.xyz, instance);
        out.foam = v.position.w;
        out.wheel_position = (v.position.xyz + water.anchor.xyz) * water.anchor.w;
        out.wheel_velocity = v.velocity.xyz * water.clock.y;
        return out;
    }
    let square = i - counters[1];
    let drop = square / 6u;
    if (drop >= counters[3]) {
        // past the droplets: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let centre = mesh_position_local_to_world(world_from_local, vec4(droplets[drop].xyz, 1.0)).xyz;
    let to_eye = view.world_position - centre;
    let distance = length(to_eye);
    let facing = to_eye / distance;
    var right = cross(vec3(0.0, 1.0, 0.0), facing);
    if (dot(right, right) < 1e-6) {
        right = cross(vec3(1.0, 0.0, 0.0), facing);
    }
    right = normalize(right);
    let up = cross(facing, right);
    // the sphere's outline on a square through its centre widens as the eye comes close
    let radius = water.clock.z;
    let outline = radius / sqrt(max(1.0 - radius * radius / (distance * distance), 0.05));
    let corner = square % 6u;
    let x = select(-1.0, 1.0, corner == 1u || corner == 2u || corner == 4u);
    let y = select(-1.0, 1.0, corner == 2u || corner == 4u || corner == 5u);
    let world = centre + (right * x + up * y) * outline;
    out.clip = view.clip_from_world * vec4(world, 1.0);
    out.world_position = world;
    out.drop = vec4(centre, radius);
    return out;
}

/// Light dimmed by crossing this much water, and the water's own colour gathered over it.
/// Absorption and scattering both take light out of a ray, and the share scattering takes is
/// the share that comes back, so a stretch of water long enough to hide whatever lies beyond
/// it settles at that share of the light falling on it, and nothing deeper changes it.
fn through_water(colour: vec3<f32>, distance: f32, light: vec3<f32>) -> vec3<f32> {
    let d = min(distance, FARTHEST);
    let extinction = water.absorption.rgb + water.scatter.rgb;
    let left = exp(-extinction * d);
    return colour * left + water.scatter.rgb / extinction * light * (1.0 - left);
}

/// A pixel of a droplet's square: the drop's sphere where the pixel's ray hits it.
fn droplet(in: Fragment) -> Shaded {
    let eye = view.world_position;
    let dir = normalize(in.world_position - eye);
    let centre = in.drop.xyz;
    let radius = in.drop.w;
    let oc = eye - centre;
    let b = dot(oc, dir);
    let h = b * b - (dot(oc, oc) - radius * radius);
    if (h < 0.0) {
        discard;
    }
    let root = sqrt(h);
    var t = -b - root;
    if (t < 0.0) {
        // the eye is within the drop: see out through its far side
        t = -b + root;
    }
    if (t < 0.0) {
        discard;
    }
    let hit = eye + dir * t;
    let clip = view.clip_from_world * vec4(hit, 1.0);
    var out: Shaded;
    out.depth = clip.z / clip.w;

    let n = normalize(hit - centre);
    let v = -dir;
    let uv = (in.clip.xy - view.viewport.xy) / view.viewport.zw;
    let pixel = in.clip.xy;
    let depth_here = depth_of(hit);
    let light = diffuse_light_at(hit, n, pixel) / PI;
    if (dot(n, v) < 0.0) {
        // from within, the drop is water all round: only what is behind shows, through it
        let seen = seen_through(uv, depth_here, hit + dir * DROP_REACH);
        out.colour = vec4(saturated(through_water(seen, t, light)), 1.0);
        return out;
    }
    let reflected = reflect(dir, n);
    var mirror = ring_seen(hit + water.origin.xyz, reflected, water.ring.xy, water.ground.rgb, water.to_stars, water.background.rgb);
    if (water.ring.z > 0.5) {
        mirror = mirrored(hit, reflected, mirror);
    }
    // in through the near face, across the drop and out through the far one
    let entered = refract(dir, n, 1.0 / IOR);
    let chord = -2.0 * dot(entered, n) * radius;
    let exit = hit + entered * chord;
    let left = refract(entered, -normalize(exit - centre), IOR);
    let seen = through_water(seen_through(uv, depth_here, exit + left * DROP_REACH), chord, light);
    var colour = mix(seen, mirror, fresnel(dot(n, v), F0));
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        colour += sunlight(i) * glint(n, v, l, DROP_ROUGHNESS, F0) * sun_shadow(i, hit, n, pixel);
    }
    out.colour = vec4(saturated(colour), 1.0);
    return out;
}

@fragment
fn fragment(in: Fragment) -> Shaded {
    if (in.drop.w > 0.0) {
        return droplet(in);
    }
    var out: Shaded;
    out.depth = in.clip.z;
    out.colour = surface(in);
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
fn surface(in: Fragment) -> vec4<f32> {
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

    let uv = (in.clip.xy - view.viewport.xy) / view.viewport.zw;
    let depth_here = depth_of(in.world_position);
    let scene = scene_depth(uv);
    // how much water lies behind this point of the surface, along the line of sight: up to
    // the scene behind it, and no further than the ring holds water, since the glass and
    // space beyond it stand in no depth
    let along = distance / max(depth_here, 1e-4);
    let site = in.world_position + water.origin.xyz;
    let held = ring_run(site, -v, water.ring.xy).distance;
    let column = min(max(scene - depth_here, 0.0) * along, held);

    // the light falling on the water here, which whatever it scatters is lit by
    let pixel = in.clip.xy;
    let light = diffuse_light_at(in.world_position, n, pixel) / PI;
    let reflected = reflect(-v, n);
    // beyond the scene the mirror shows the ring, or from under water the bed, lit by the
    // light coming down through the surface and about as far off as the surface is
    var beyond = ring_seen(site, reflected, water.ring.xy, water.ground.rgb, water.to_stars, water.background.rgb);
    if (submerged) {
        beyond = through_water(water.ground.rgb * light, distance, light);
    }
    // from outside the drum the screen shows the far sides of everything the mirror would
    // show the near sides of, so only the ring itself is mirrored there
    // from outside the drum the screen shows the far sides of everything the mirror would
    // show the near sides of, and from under water the bed the surface mirrors lies behind
    // the surface itself, so the mirror is only marched across the screen from inside the air
    var mirror = beyond;
    if (water.ring.z > 0.5 && !submerged) {
        mirror = mirrored(in.world_position, reflected, beyond);
    }
    var seen: vec3<f32>;
    var f: f32;
    if (!submerged) {
        let r = refract(-v, n, 1.0 / IOR);
        let reach = min(column, 6.0) * 0.5;
        // a pixel covers a stretch of the surface with a spread of slopes, which bends what it
        // shows over a patch of the scene rather than a point of it
        let bend = length(waves.curve[0]) + length(waves.curve[1]) + length(waves.curve[2]);
        let spread = reach * (1.0 - 1.0 / IOR) * bend * footprint * amplitude;
        seen = through_water(spread_over(uv, depth_here, in.world_position + r * reach, r, spread), column, light);
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
        mirror = through_water(mirror, column, light);
    }

    var colour = mix(seen, mirror, f);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        colour += sunlight(i) * glint(n, v, l, roughness, F0) * sun_shadow(i, in.world_position, n, pixel);
    }

    // foam: bubbles ride on the flow like the ripples
    let grain = bubble_grain(run, 24.0, footprint);
    let grain2 = bubble_grain(run, 7.0, footprint);
    let lace = in.foam + (grain - 0.5) * 0.5 + (grain2 - 0.5) * 0.35;
    let foam = smoothstep(0.42, 0.7, lace);
    // bubbles: brighter where the lace is thick, with dark water showing between them
    let bubbles = 0.55 + 0.45 * smoothstep(0.55, 0.95, lace + (grain - 0.5) * 0.6);
    let foam_colour = vec3(0.7, 0.74, 0.78) / PI * diffuse_light_at(in.world_position, n, pixel) * bubbles;
    colour = mix(colour, foam_colour, foam);

    if (submerged) {
        colour = through_water(colour, distance, light);
    }
    return vec4(saturated(colour), 1.0);
}
