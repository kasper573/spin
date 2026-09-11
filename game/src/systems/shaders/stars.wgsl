// The stars: a sphere about the viewer drawn at infinity, so that nothing is ever behind it
// however far away it is, turned with the sky, and lit by the direction each point of it is
// seen in from within it.
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#import bevy_pbr::mesh_functions
#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::mesh_view_bindings::view
#import space::space_colour
#import air::{Air, air_bent}
#import optics::rotate
#import ring::ring_run

struct Sky {
    background: vec4<f32>,
    // the sun's direction among the stars
    sun: vec4<f32>,
    // turns a direction of the ring's frame into one among the stars
    to_stars: vec4<f32>,
    // where the viewpoint lies in the ring's frame, in metres
    origin: vec4<f32>,
    // the ring's radius and half width, in metres
    ring: vec4<f32>,
    air: Air,
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
    let to_sun = normalize(sky.sun.xyz);
    let dir = normalize(in.world_normal);
    // space stands in no water, whatever the eye stands in: from under water it is only ever
    // seen through the glass, which is what puts what it shows through the water in between.
    // It does stand behind the ring's air, which bends what crosses it, and bends each colour
    // by its own amount: what is seen through a long enough stretch of it at a grazing angle
    // is pulled out of shape and fringed with colour, the sun above all.
    let eye = view.world_position + sky.origin.xyz;
    let out = normalize(in.world_position.xyz - view.world_position);
    // how much sky one pixel covers, which is what the field is averaged over
    let spread = length(dpdx(out)) + length(dpdy(out));
    let held = ring_run(eye, out, sky.ring.xy).distance;
    if (held <= 0.0 || sky.air.slowing.w == 0.0) {
        return vec4(space_colour(dir, to_sun, sky.background.rgb, spread), 1.0);
    }
    let radius = sky.ring.x;
    let red = space_colour(rotate(sky.to_stars, air_bent(sky.air, eye, out, held, radius, sky.air.slowing.x)), to_sun, sky.background.rgb, spread);
    let green = space_colour(rotate(sky.to_stars, air_bent(sky.air, eye, out, held, radius, sky.air.slowing.y)), to_sun, sky.background.rgb, spread);
    let blue = space_colour(rotate(sky.to_stars, air_bent(sky.air, eye, out, held, radius, sky.air.slowing.z)), to_sun, sky.background.rgb, spread);
    return vec4(red.r, green.g, blue.b, 1.0);
}
