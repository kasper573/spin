#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct Glass {
    tint: vec4<f32>,
    sun: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> glass: Glass;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if (!front) {
        n = -n;
    }
    let v = normalize(view.world_position - in.world_position.xyz);
    let facing = max(dot(n, v), 0.0);
    let fresnel = pow(1.0 - facing, 4.0);
    let h = normalize(normalize(glass.sun.xyz) + v);
    let glint = pow(max(dot(n, h), 0.0), 160.0);
    let sheen = pow(max(dot(n, h), 0.0), 12.0) * 0.08;
    let colour = glass.tint.rgb * (0.35 + 0.65 * fresnel) + vec3(1.0) * (glint * 0.9 + sheen);
    let alpha = glass.tint.a + fresnel * 0.55 + glint * 0.8;
    return vec4(colour, clamp(alpha, 0.0, 1.0));
}
