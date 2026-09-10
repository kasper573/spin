// The stars: a sphere about the viewer drawn at infinity, so that nothing is ever behind it
// however far away it is, turned with the sky, and lit by the direction each point of it is
// seen in from within it.
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#import bevy_pbr::mesh_functions
#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::mesh_view_bindings::view
#import space::space_colour
#ifdef DISTANCE_FOG
#import bevy_pbr::mesh_view_bindings::fog
#import bevy_pbr::pbr_functions::apply_fog
#endif

struct Sky {
    background: vec4<f32>,
    // the sun's direction among the stars
    sun: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sky: Sky;

/// Every point of the sphere lands on the far plane: as far as the depth buffer reaches. Its
/// direction among the stars, before the sphere is turned, rides along as the normal.
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
    let clip = position_world_to_clip(out.world_position.xyz);
    out.position = vec4(clip.xy, 0.0, clip.w);
    out.world_normal = normalize(vertex.position);
    out.instance_index = vertex.instance_index;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_normal);
    let out = vec4(space_colour(dir, normalize(sky.sun.xyz), sky.background.rgb), 1.0);
#ifdef DISTANCE_FOG
    // with the eye under water, space is seen through the water round the eye
    return apply_fog(fog, out, in.world_position.xyz, view.world_position, in.position.xy);
#else
    return out;
#endif
}
