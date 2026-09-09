// The ground's colour, computed per fragment on top of the standard material: bare dirt where it
// has been dug toward the glass, grass at the initial depth and above, tiled in metre squares of
// two tints across the wheel's surface so that walking over it reads as motion and distance. The
// mesh carries each point's place in the tiling, in tiles round the ring and along its axis,
// and its height above the glass, so the tiles stay square and sharp whatever size the ring is.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct Terrain {
    dirt: vec4<f32>,
    grass: vec4<f32>,
    grass_dark: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> terrain: Terrain;

/// 1 on a light tile, 0 on a dark one, blended over the width of a pixel so the edges stay
/// crisp at any distance without shimmering.
fn checker(u: f32, v: f32) -> f32 {
    let fu = fwidth(u);
    let fv = fwidth(v);
    // the checker as a product of two square waves, each filtered over the pixel footprint
    let su = 1.0 - 2.0 * abs(fract(u) - 0.5);
    let sv = 1.0 - 2.0 * abs(fract(v) - 0.5);
    let a = smoothstep(0.5 - fu, 0.5 + fu, su);
    let b = smoothstep(0.5 - fv, 0.5 + fv, sv);
    return a * (1.0 - b) + b * (1.0 - a);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    var light = 0.0;
    var height = 0.0;
#ifdef VERTEX_UVS_A
    light = checker(in.uv.x, in.uv.y);
#endif
#ifdef VERTEX_UVS_B
    height = in.uv_b.x;
#endif
    let grass = mix(terrain.grass_dark.rgb, terrain.grass.rgb, light);
    let t = smoothstep(0.15, 0.45, height);
    let colour = mix(terrain.dirt.rgb, grass, t);
    pbr_input.material.base_color = vec4(colour, 1.0);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
    return out;
}
