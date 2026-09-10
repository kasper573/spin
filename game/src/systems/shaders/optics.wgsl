// How light meets a smooth surface, shared by everything that reflects and lets light through:
// what a ray sees of the scene on screen, found by marching it against the depth of the scene,
// or of space beyond it; how much of the light a dielectric surface mirrors and how much it
// passes; and the sun's glint off it.
#define_import_path optics

#import bevy_pbr::mesh_view_bindings::{view, lights, view_transmission_texture, view_transmission_sampler}
#import bevy_pbr::prepass_utils::prepass_depth
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::depth_ndc_to_view_z
#import space::stars

const PI: f32 = 3.14159265;
const MARCH_STEPS: i32 = 28;
// the brightest the eye tells from white: the sun's mirror image is thousands of times
// brighter than that, and saturates the eye rather than flooding the view
const SATURATION: f32 = 4.0;

/// A colour as the eye takes it in, saturating at the brightest it can tell apart.
fn saturated(colour: vec3<f32>) -> vec3<f32> {
    return min(colour, vec3(SATURATION));
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

/// A point of the world on screen: its place in the viewport and its distance in front of
/// the camera, negative when behind it.
fn screen_uv(world: vec3<f32>) -> vec3<f32> {
    let clip = view.clip_from_world * vec4(world, 1.0);
    if (clip.w <= 0.0) {
        return vec3(-1.0, -1.0, 0.0);
    }
    let ndc = clip.xy / clip.w;
    return vec3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, clip.w);
}

fn on_screen(uv: vec2<f32>) -> bool {
    return all(uv >= vec2(0.0)) && all(uv <= vec2(1.0));
}

/// How far in front of the camera the opaque scene is at a point of the screen.
fn scene_depth(uv: vec2<f32>) -> f32 {
    let coord = view.viewport.xy + uv * view.viewport.zw;
    return -depth_ndc_to_view_z(prepass_depth(vec4(coord, 0.0, 0.0), 0u));
}

/// The scene as drawn so far, behind everything being drawn now.
fn behind(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0).rgb;
}

/// How far in front of the camera a point of the world is.
fn depth_of(world: vec3<f32>) -> f32 {
    return -(view.view_from_world * vec4(world, 1.0)).z;
}

/// What a ray from a point sees: the scene where the ray meets it on screen, found by marching
/// it against the depth of the scene, or else what lies `beyond` the scene.
fn mirrored(origin: vec3<f32>, dir: vec3<f32>, beyond: vec3<f32>) -> vec3<f32> {
    var t = 0.02;
    for (var i = 0; i < MARCH_STEPS; i++) {
        let q = origin + dir * t;
        let s = screen_uv(q);
        if (s.z <= 0.0 || !on_screen(s.xy)) {
            break;
        }
        let gap = s.z - scene_depth(s.xy);
        if (gap > 0.0) {
            if (gap < t * 0.6 + 0.05) {
                // fade out toward the edges of the screen, where the view runs out
                let edge = s.xy * (1.0 - s.xy);
                let fade = smoothstep(0.0, 0.02, min(edge.x, edge.y));
                return mix(beyond, behind(s.xy), fade);
            }
            break;
        }
        t *= 1.28;
    }
    return beyond;
}

/// Space in a direction of the world, given the turn that takes the world among the stars.
fn space_seen(dir: vec3<f32>, to_stars: vec4<f32>, background: vec3<f32>) -> vec3<f32> {
    return background + stars(rotate(to_stars, dir));
}

/// Whether the sun reaches a point of the ground: its light must come in through a cap, so
/// the way toward the sun from there must leave the ring's width before it crosses the ring.
fn sun_reaches(q: vec3<f32>, up: vec3<f32>, l: vec3<f32>, ring: vec2<f32>) -> bool {
    if (abs(l.y) < 1e-6) {
        return false;
    }
    let flat = l.xz;
    let across = 2.0 * ring.x * dot(up.xz, flat) / max(dot(flat, flat), 1e-12);
    let cap = select(-ring.y, ring.y, l.y > 0.0);
    return (cap - q.y) / l.y < across;
}

/// How far a ray from inside the ring runs before it meets the ring's wall, or leaves through
/// a cap, whichever comes first, and whether it was the wall. `at` and `dir` are in the site's
/// frame, whose origin lies on the wall with the axis `ring.x` in along -x, and whose caps lie
/// `ring.y` out along y. The far root is found without the near one's cancellation, so it
/// holds for any ring. A ray starting outside the ring runs nowhere in it.
struct RingRun {
    distance: f32,
    wall: bool,
}

fn ring_run(at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>) -> RingRun {
    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * ((ring.x + at.x) * dir.x + at.z * dir.z);
    let c = 2.0 * ring.x * at.x + at.x * at.x + at.z * at.z;
    if (c > 0.0) {
        return RingRun(0.0, false);
    }
    var t = 1e9;
    if (a >= 1e-12) {
        let root = sqrt(max(b * b - 4.0 * a * c, 0.0));
        var q = -0.5 * (b + root);
        if (b < 0.0) {
            q = -0.5 * (b - root);
        }
        t = q / a;
        if (q != 0.0) {
            t = max(t, c / q);
        }
    }
    if (abs(dir.y) > 1e-6) {
        let cap = (select(-ring.y, ring.y, dir.y > 0.0) - at.y) / dir.y;
        if (cap < t) {
            return RingRun(cap, false);
        }
    }
    return RingRun(t, true);
}

/// What a ray meets once it has left the screen, inside the ring: the ground where it strikes
/// the ring, lit by the sun where the sun reaches it and by the bounce light everywhere, or
/// space where it leaves through a cap.
fn ring_seen(at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>, ground: vec3<f32>, to_stars: vec4<f32>, background: vec3<f32>) -> vec3<f32> {
    let run = ring_run(at, dir, ring);
    if (!run.wall) {
        return space_seen(dir, to_stars, background);
    }
    let hit = at + dir * run.distance;
    let up = -normalize(vec3(ring.x + hit.x, 0.0, hit.z));
    var light = bounce();
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let ndl = dot(up, l);
        if (ndl > 0.0 && sun_reaches(hit, up, l, ring)) {
            light += sunlight(i) * ndl / PI;
        }
    }
    return ground * light;
}

/// What is seen through a surface at a point of the screen, refracted to `exit`: the scene
/// there unless something stands in front of the surface, or the refraction leaves the screen.
fn seen_through(uv: vec2<f32>, depth_here: f32, exit: vec3<f32>) -> vec3<f32> {
    var s = screen_uv(exit);
    if (s.z <= 0.0 || !on_screen(s.xy) || scene_depth(s.xy) < depth_here) {
        s = vec3(uv, depth_here);
    }
    return behind(s.xy);
}

/// Schlick's share of light a dielectric mirrors at this angle, given what it mirrors face on.
fn fresnel(cos_theta: f32, f0: f32) -> f32 {
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * m * m * m * m * m;
}

/// The sun's glint: a GGX lobe, tight on a still surface and spread by its roughness.
fn glint(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: f32) -> f32 {
    let nl = dot(n, l);
    let nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0) {
        return 0.0;
    }
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let a = roughness * roughness;
    let a2 = a * a;
    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-4);
    return d * vis * fresnel(max(dot(v, h), 0.0), f0) * nl;
}

/// The light bounced round the ring, as the eye is exposed to it.
fn bounce() -> vec3<f32> {
    return lights.ambient_color.rgb * view.exposure;
}

/// A sun's light, as the eye is exposed to it.
fn sunlight(i: u32) -> vec3<f32> {
    return lights.directional_lights[i].color.rgb * view.exposure;
}

/// How much of a sun's light reaches a point of the world, given the ring's shadow.
fn sun_shadow(i: u32, world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> f32 {
    let view_z = -(view.view_from_world * vec4(world, 1.0)).z;
    return fetch_directional_shadow(i, vec4(world, 1.0), n, view_z, pixel);
}

/// The light falling on a matte surface at a point of the world, facing `n`, where the ring
/// shades it from the suns.
fn diffuse_light_at(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    var light = bounce();
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let ndl = dot(n, l);
        if (ndl > 0.0) {
            light += sunlight(i) * ndl * sun_shadow(i, world, n, pixel);
        }
    }
    return light;
}
