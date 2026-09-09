// The drum as the fluid solver's vessel: a spinning cylinder with caps and a heightfield
// landscape on its inside wall, in its own turning frame and the water's units. Mirrors `Drum`
// on the CPU.
#define_import_path vessel

struct DrumUniform {
    // the drum's spin and its rate of change, per second of the water's clock
    spin: f32,
    spin_rate: f32,
    // the drum's size in the water's units, and how many of them a metre is
    radius: f32,
    half_width: f32,
    per_metre: f32,
    // 1 when any landscape is raised
    landscape: u32,
    segments: u32,
    rows: u32,
    pad: u32,
    dphi: f32,
    dy: f32,
}

@group(1) @binding(0) var<uniform> drum: DrumUniform;
@group(1) @binding(1) var heights: texture_2d<f32>;

struct Confined {
    p: vec3<f32>,
    // inward wall normals, w = 1 when present
    first: vec4<f32>,
    second: vec4<f32>,
}

/// Where free flight lands, and at what velocity in the frame.
struct Flight {
    p: vec3<f32>,
    v: vec3<f32>,
}

struct Penetration {
    depth: f32,
    normal: vec3<f32>,
}

fn height_at(segment: u32, row: u32) -> f32 {
    return textureLoad(heights, vec2<i32>(i32(row), i32(segment)), 0).r * drum.per_metre;
}

/// Bilinear height and its derivatives with respect to wheel angle and axial position.
fn landscape_sample(phi: f32, y: f32) -> vec3<f32> {
    let segments = f32(drum.segments);
    var u = (phi / drum.dphi) % segments;
    if (u < 0.0) {
        u += segments;
    }
    let i_a = u32(floor(u)) % drum.segments;
    let fu = u - floor(u);
    let i_b = (i_a + 1u) % drum.segments;
    let v = clamp((y + drum.half_width) / drum.dy, 0.0, f32(drum.rows) - 1.0 - 1e-6);
    let j_a = u32(floor(v));
    let fv = v - floor(v);
    let j_b = j_a + 1u;
    let h_aa = height_at(i_a, j_a);
    let h_ba = height_at(i_b, j_a);
    let h_ab = height_at(i_a, j_b);
    let h_bb = height_at(i_b, j_b);
    let h = (h_aa * (1.0 - fu) + h_ba * fu) * (1.0 - fv) + (h_ab * (1.0 - fu) + h_bb * fu) * fv;
    let dphi = ((h_ba - h_aa) * (1.0 - fv) + (h_bb - h_ab) * fv) / drum.dphi;
    let dy = ((h_ab - h_aa) * (1.0 - fu) + (h_bb - h_ba) * fu) / drum.dy;
    return vec3(h, dphi, dy);
}

/// Signed penetration of a point into the terrain (positive = inside) and the inward normal.
fn landscape_penetration(p: vec3<f32>, margin: f32) -> Penetration {
    let r = length(p.xz);
    if (r < 1e-6) {
        return Penetration(-drum.radius, vec3(0.0));
    }
    let s = landscape_sample(atan2(p.z, p.x), p.y);
    let f = r - (drum.radius - s.x - margin);
    let g_phi = s.y / r;
    let g = vec3((p.x - g_phi * p.z) / r, s.z, (p.z + g_phi * p.x) / r);
    let len = max(length(g), 1e-12);
    return Penetration(f / len, -g / len);
}

fn vessel_confine(p_in: vec3<f32>, margin: f32) -> Confined {
    var p = p_in;
    var out = Confined(p, vec4(0.0), vec4(0.0));
    var count = 0u;
    if (drum.landscape == 0u) {
        let limit = drum.radius - margin;
        let r = length(p.xz);
        if (r > limit) {
            p.x *= limit / r;
            p.z *= limit / r;
            out.first = vec4(-p.x / limit, 0.0, -p.z / limit, 1.0);
            count = 1u;
        }
    } else {
        for (var attempt = 0u; attempt < 2u; attempt++) {
            let pen = landscape_penetration(p, margin);
            if (pen.depth <= 0.0) {
                break;
            }
            p += pen.normal * pen.depth;
            if (count == 0u) {
                out.first = vec4(pen.normal, 1.0);
                count = 1u;
            }
        }
    }
    let cap = drum.half_width - margin;
    var cap_normal = vec4(0.0);
    if (p.y > cap) {
        p.y = cap;
        cap_normal = vec4(0.0, -1.0, 0.0, 1.0);
    } else if (p.y < -cap) {
        p.y = -cap;
        cap_normal = vec4(0.0, 1.0, 0.0, 1.0);
    }
    if (cap_normal.w > 0.0) {
        if (count == 0u) {
            out.first = cap_normal;
        } else {
            out.second = cap_normal;
        }
    }
    out.p = p;
    return out;
}

/// What a point at rest in the turning drum is accelerated by: flung outward by the spin, and
/// left behind as the spin changes.
fn vessel_gravity(p: vec3<f32>) -> vec3<f32> {
    let w = drum.spin;
    let a = drum.spin_rate;
    return vec3(w * w * p.x - a * p.z, 0.0, w * w * p.z + a * p.x);
}

/// A point turned about the axis by `phi`.
fn turned(p: vec3<f32>, phi: f32) -> vec3<f32> {
    let c = cos(phi);
    let s = sin(phi);
    return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

/// Free flight through the drum's frame over `dt`, which ends at the spin the frame is at:
/// straight and even among the stars, put back into the frame where it has turned to by the
/// end, so that free-flying water keeps its motion among the stars exactly.
fn vessel_flight(p: vec3<f32>, v: vec3<f32>, dt: f32) -> Flight {
    let before = drum.spin - drum.spin_rate * dt;
    let turn = drum.spin * dt - 0.5 * drum.spin_rate * dt * dt;
    let among_stars = v + vec3(before * p.z, 0.0, -before * p.x);
    let landed = turned(p + among_stars * dt, -turn);
    var out: Flight;
    out.p = landed;
    out.v = turned(among_stars, -turn) - vec3(drum.spin * landed.z, 0.0, -drum.spin * landed.x);
    return out;
}

/// The velocity, in the frame, of something at rest among the stars at `p`.
fn vessel_star_velocity(p: vec3<f32>) -> vec3<f32> {
    return vec3(-drum.spin * p.z, 0.0, drum.spin * p.x);
}

fn vessel_has_air(p: vec3<f32>) -> bool {
    return dot(p.xz, p.xz) < drum.radius * drum.radius && abs(p.y) < drum.half_width;
}
