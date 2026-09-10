// The star sphere's depth: every point of it on the far plane, as it is drawn, so that
// nothing is ever found to be behind it.
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#import bevy_pbr::mesh_functions
#import bevy_pbr::view_transformations::position_world_to_clip

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
    let clip = position_world_to_clip(out.world_position.xyz);
    out.position = vec4(clip.xy, 0.0, clip.w);
    return out;
}
