#import bevy_pbr::mesh_view_bindings::view

struct Water {
    sun: vec4<f32>,
    deep: vec4<f32>,
    shallow: vec4<f32>,
    // x: time (s), y: drum angle (rad)
    clock: vec4<f32>,
}

struct SurfaceVertex {
    // xyz: position, w: foam
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> water: Water;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> vertices: array<SurfaceVertex>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<storage, read> indices: array<u32>;
// vertex count, then index count
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<storage, read> counters: array<u32>;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) foam: f32,
}

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> Fragment {
    var out: Fragment;
    if (i >= counters[1]) {
        // past the extracted surface: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let v = vertices[indices[i]];
    out.clip = view.clip_from_world * vec4(v.position.xyz, 1.0);
    out.world_position = v.position.xyz;
    out.world_normal = v.normal.xyz;
    out.foam = v.position.w;
    return out;
}

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yxz + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(hash3(i), hash3(i + vec3(1.0, 0.0, 0.0)), u.x);
    let b = mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), u.x);
    let c = mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), u.x);
    let d = mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

@fragment
fn fragment(in: Fragment, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if (!front) {
        n = -n;
    }
    let v = normalize(view.world_position - in.world_position);
    let l = normalize(water.sun.xyz);
    let ndl = dot(n, l);
    var band = 0.25;
    if (ndl > 0.55) {
        band = 1.0;
    } else if (ndl > 0.05) {
        band = 0.62;
    }
    let facing = max(dot(n, v), 0.0);
    let fresnel = pow(1.0 - facing, 3.0);
    let h = normalize(l + v);
    let spec = step(0.985, max(dot(n, h), 0.0));

    // foam noise sits still in the drum's frame, so it turns with the water instead of sliding
    let angle = water.clock.y;
    let c = cos(angle);
    let s = sin(angle);
    let p = in.world_position;
    let wheel = vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
    let grain = noise3(wheel * 22.0 + vec3(0.0, water.clock.x * 0.6, 0.0));
    let grain2 = noise3(wheel * 9.0 - vec3(water.clock.x * 0.3, 0.0, 0.0));
    let foam = smoothstep(0.45, 0.75, in.foam + (grain - 0.5) * 0.45 + (grain2 - 0.5) * 0.25);

    var colour = mix(water.deep.rgb, water.shallow.rgb, band);
    colour = mix(colour, vec3(0.78, 0.9, 1.0), fresnel * 0.55);
    colour += vec3(1.0) * spec * 0.6;
    colour = mix(colour, vec3(0.96, 0.98, 1.0) * (0.75 + 0.25 * band), foam);
    return vec4(colour, 1.0);
}
