#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct Sky {
    background: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sky: Sky;

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yxz + 33.33);
    return fract((r.x + r.y) * r.z);
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
