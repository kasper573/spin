// A solid thing in the ring, lit as the standard material lights it, and by the light that
// reaches it through a pair of portals besides. See `systems/portal/solids.rs`.
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing}
#import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}
#import optics::{sunbeams_at, sunbeam_glints_at, rooms_beyond_at}

const PI: f32 = 3.14159265;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var lit = pbr_input_from_standard_material(in, is_front);
    lit.material.base_color = alpha_discard(lit.material, lit.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(lit);
    let colour = lit.material.base_color.rgb;
    let metal = lit.material.metallic;
    let p = in.world_position.xyz;
    // as the mirrors have the figure's finish: see `figure.wgsl`
    let matte = colour * (1.0 - metal) / PI * (sunbeams_at(p, lit.N) + rooms_beyond_at(p, lit.N));
    let sheen = mix(vec3(1.0), colour, metal);
    let shine = sheen * sunbeam_glints_at(p, lit.N, lit.V, max(lit.material.perceptual_roughness, 0.05), mix(0.04, 0.8, metal));
    out.color = vec4(out.color.rgb + matte + shine, out.color.a);
    out.color = main_pass_post_lighting_processing(lit, out.color);
    return out;
}
