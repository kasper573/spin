// Water, drawn from the surface the GPU extracted. A pixel of water shows what is behind it,
// refracted and dimmed by the water it was seen through; what is around it, mirrored, the ring
// found by marching the reflected ray across the depth of the scene and space beyond that; the
// sun glinting off it; and foam where it churns. Ripples ride on the flow, so still water lies
// like glass and moving water sparkles. Seen from under the surface the same rules give the
// mirror of the bed beyond the critical angle and the world above within it.
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world}
#import optics::{rotate, mirrored, ring_seen, seen_through, scene_depth, fresnel, glint, diffuse_light, sunlight, depth_of, saturated}
#import ripples::{carried, waves_carried, noise3}

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
// vertex count, then index count
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<storage, read> counters: array<u32>;

const PI: f32 = 3.14159265;
const IOR: f32 = 1.333;
// air to water, so the mirror is faint face on
const F0: f32 = 0.02;
// how far through water anything can be seen at all
const FARTHEST: f32 = 200.0;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) foam: f32,
    // in the water's frame about the axis, in metres, so surface detail turns with the water
    @location(3) wheel_position: vec3<f32>,
    // the flow in the water's frame, metres per second
    @location(4) wheel_velocity: vec3<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) i: u32, @builtin(instance_index) instance: u32) -> Fragment {
    var out: Fragment;
    if (i >= counters[1]) {
        // past the extracted surface: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let v = vertices[indices[i]];
    let world_from_local = get_world_from_local(instance);
    let world = mesh_position_local_to_world(world_from_local, vec4(v.position.xyz, 1.0));
    out.clip = view.clip_from_world * world;
    out.world_position = world.xyz;
    out.world_normal = mesh_normal_local_to_world(v.normal.xyz, instance);
    out.foam = v.position.w;
    out.wheel_position = (v.position.xyz + water.anchor.xyz) * water.anchor.w;
    out.wheel_velocity = v.velocity.xyz * water.clock.y;
    return out;
}

/// Light dimmed by crossing this much water, and the water's own glow gathered over it: the
/// light falling on the water, scattered back out of it.
fn through_water(colour: vec3<f32>, distance: f32, light: vec3<f32>) -> vec3<f32> {
    let d = min(distance, FARTHEST);
    let glow = water.scatter.rgb * light * (1.0 - exp(-water.scatter.w * d));
    return colour * exp(-water.absorption.rgb * d) + glow;
}

@fragment
fn fragment(in: Fragment) -> @location(0) vec4<f32> {
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
    // the width of a pixel at this distance, for a 60 degree view
    let footprint = distance * 1.15 / view.viewport.w;
    let slope = waves_carried(run, t, footprint).slope;
    let amplitude = 1.0 + 4.0 * churn;
    let g = rotate(water.from_water, slope) * amplitude;
    n = normalize(n + g - n * dot(n, g));
    if (dot(n, v) < 0.0) {
        n = normalize(n - 2.0 * v * dot(n, v) + v * 1e-3);
    }
    let roughness = 0.06 + 0.14 * churn;

    let uv = (in.clip.xy - view.viewport.xy) / view.viewport.zw;
    let depth_here = depth_of(in.world_position);
    let scene = scene_depth(uv);
    // how much water lies behind this point of the surface, along the line of sight
    let along = distance / max(depth_here, 1e-4);
    let column = max(scene - depth_here, 0.0) * along;

    // the light falling on the water here, which whatever it scatters is lit by
    let light = diffuse_light(n) / PI;
    let reflected = reflect(-v, n);
    // beyond the scene the mirror shows the ring, or from under water the water's own glow
    var beyond = ring_seen(in.world_position + water.origin.xyz, reflected, water.ring.xy, water.ground.rgb, water.to_stars, water.background.rgb);
    if (submerged) {
        beyond = through_water(vec3(0.0), FARTHEST, light);
    }
    var mirror = mirrored(in.world_position, reflected, beyond);
    var seen: vec3<f32>;
    var f: f32;
    if (!submerged) {
        let r = refract(-v, n, 1.0 / IOR);
        let reach = min(column, 6.0) * 0.5;
        seen = through_water(seen_through(uv, depth_here, in.world_position + r * reach), column, light);
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
        colour += sunlight(i) * glint(n, v, l, roughness, F0);
    }

    // foam: bubbles ride on the flow like the ripples
    let grain = noise3(run.a * 24.0) * run.weight_a + noise3(run.b * 24.0) * (1.0 - run.weight_a);
    let grain2 = noise3(run.a * 7.0) * run.weight_a + noise3(run.b * 7.0) * (1.0 - run.weight_a);
    let lace = in.foam + (grain - 0.5) * 0.5 + (grain2 - 0.5) * 0.35;
    let foam = smoothstep(0.42, 0.7, lace);
    // bubbles: brighter where the lace is thick, with dark water showing between them
    let bubbles = 0.55 + 0.45 * smoothstep(0.55, 0.95, lace + (grain - 0.5) * 0.6);
    let foam_colour = vec3(0.7, 0.74, 0.78) / PI * diffuse_light(n) * bubbles;
    colour = mix(colour, foam_colour, foam);

    if (submerged) {
        colour = through_water(colour, distance, light);
    }
    return vec4(saturated(colour), 1.0);
}
