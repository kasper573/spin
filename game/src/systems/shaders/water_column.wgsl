// How much water lies along each line of sight: see `water_column.rs`. Every face of the
// water's surface adds how far along the line of sight it is where that leaves the water
// through it, and takes it away where it enters. A picture of half floats cannot add metres by
// the thousand and keep their hundredths, so a distance is added in two parts that each stay
// exact: how many whole steps it holds, and what is left over. The faces are counted too, one
// up for each the line of sight leaves by and one down for each it enters by, which tells a
// line of sight that ends inside water the surface does not close round.
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}

struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    velocity: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> seen_past: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> vertices: array<SurfaceVertex>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<storage, read> indices: array<u32>;
// vertex count, index count, block count, then droplet count
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<storage, read> counters: array<u32>;

// a step, in metres, and the furthest that is counted, in steps; `water.wgsl` reads the sum
const STEP: f32 = 4.0;
const FURTHEST: f32 = 2000.0;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) seen: vec3<f32>,
}

@vertex
fn vertex(@location(0) numbered: vec3<f32>, @builtin(instance_index) instance: u32) -> Fragment {
    var out: Fragment;
    let i = u32(numbered.x);
    if (i >= counters[1]) {
        // past the surface: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let v = vertices[indices[i]];
    let world = mesh_position_local_to_world(get_world_from_local(instance), vec4(v.position.xyz, 1.0));
    out.clip = view.clip_from_world * world;
    out.seen = (view.view_from_world * world).xyz;
    return out;
}

@fragment
fn fragment(in: Fragment, @builtin(front_facing) entering: bool) -> @location(0) vec4<f32> {
    var far = length(in.seen);
    let along = in.seen / far;
    // short of the plane the eye sees past, the face counts as lying in the plane
    let toward = dot(seen_past.xyz, along);
    if (dot(seen_past.xyz, in.seen) + seen_past.w < 0.0) {
        if (toward <= 0.0) {
            discard;
        }
        far = -seen_past.w / toward;
    }
    far = min(far, FURTHEST * STEP);
    let steps = floor(far / STEP);
    let counted = vec3(steps, far - steps * STEP, 1.0);
    return vec4(select(counted, -counted, entering), 0.0);
}
