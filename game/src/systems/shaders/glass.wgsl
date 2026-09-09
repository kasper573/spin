#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct Glass {
    tint: vec4<f32>,
    sun: vec4<f32>,
    // the pane size round the wall and along it, and the seam width
    panes: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> glass: Glass;

/// How much of a seam between panes this point of the glass is on: the glass is gridded into
/// square panes fixed to the wheel, round the wall and across the caps. The mesh carries each
/// point's place in that grid, in panes, so the grid is exact whatever size the wheel is.
fn seam(uv: vec2<f32>, n: vec3<f32>) -> f32 {
    var pane = glass.panes.xy;
    if (abs(n.y) > 0.5) {
        pane = vec2(glass.panes.y);
    }
    let edge = abs(fract(uv) - 0.5);
    let width = fwidth(uv) + glass.panes.z * 0.5 / pane;
    let line = 1.0 - smoothstep(vec2(0.0), width, 0.5 - edge);
    return max(line.x, line.y);
}

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
    var seams = 0.0;
#ifdef VERTEX_UVS_A
    seams = seam(in.uv, n) * (0.25 + 0.75 * facing);
#endif
    let colour = glass.tint.rgb * (0.35 + 0.65 * fresnel) + vec3(1.0) * (glint * 0.9 + sheen)
        + vec3(0.8, 0.9, 1.0) * seams * 0.3;
    let alpha = glass.tint.a + fresnel * 0.55 + glint * 0.8 + seams * 0.22;
    return vec4(colour, clamp(alpha, 0.0, 1.0));
}
