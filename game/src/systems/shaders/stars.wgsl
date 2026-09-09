// The stars: a sphere about the viewer drawn at infinity, so that nothing is ever behind it
// however far away it is, and lit by a hash of the direction it is seen in.
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#import bevy_pbr::mesh_functions
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::view_transformations::position_world_to_clip

struct Sky {
    background: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sky: Sky;

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yxz + 33.33);
    return fract((r.x + r.y) * r.z);
}

/// Every point of the sphere lands on the far plane: as far as the depth buffer reaches.
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
    let clip = position_world_to_clip(out.world_position.xyz);
    out.position = vec4(clip.xy, 0.0, clip.w);
    out.instance_index = vertex.instance_index;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_position.xyz - view.world_position);
    let cell = floor(dir * 140.0);
    let h = hash3(cell);
    let centre = (cell + 0.5 + vec3(hash3(cell + 1.7), hash3(cell + 3.1), hash3(cell + 5.3)) - 0.5) / 140.0;
    let d = length(dir - normalize(centre)) * 140.0;
    let star = smoothstep(0.35, 0.0, d) * step(0.965, h);
    let bright = 0.5 + 0.5 * hash3(cell + 9.9);
    let tint = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.92, 0.8), hash3(cell + 2.2));
    return vec4(sky.background.rgb + tint * star * bright, 1.0);
}
