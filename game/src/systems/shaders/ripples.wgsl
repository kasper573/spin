// The ripples on the water: a field of small waves of different lengths crossing at odd
// angles, each running at its own pace, over a drift of noise that breaks their regularity,
// carried along by the flow. The same field bends what is seen through the surface and focuses
// the sunlight that falls through it onto the bed, so both are drawn from here.
#define_import_path ripples

const PI: f32 = 3.14159265;
// the ripples are carried along by the flow in two overlapping runs of this length, each faded
// in and out, so that neither is ever seen to reset
const FLOW_PERIOD: f32 = 2.0;
// the height of each wave, as a fraction of its length
const STEEPNESS: f32 = 0.006;

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

/// A point of the water's frame carried back along the flow, in two runs that overlap, and
/// the weight the first run has now.
struct Carried {
    a: vec3<f32>,
    b: vec3<f32>,
    weight_a: f32,
}

fn carried(x: vec3<f32>, flow: vec3<f32>, t: f32) -> Carried {
    let phase_a = fract(t / FLOW_PERIOD);
    let phase_b = fract(t / FLOW_PERIOD + 0.5);
    var out: Carried;
    out.a = x - flow * phase_a * FLOW_PERIOD;
    out.b = x - flow * phase_b * FLOW_PERIOD;
    out.weight_a = 1.0 - abs(2.0 * phase_a - 1.0);
    return out;
}

/// The wave field at a point: the slope of the water's height and its curvature.
struct Waves {
    slope: vec3<f32>,
    curve: mat3x3<f32>,
}

/// The waves at a point, leaving out those too short to resolve at a pixel this wide, so that
/// the far water does not shimmer with aliasing.
fn waves(x: vec3<f32>, t: f32, footprint: f32) -> Waves {
    let kinds = array<vec4<f32>, 6>(
        vec4(0.71, 0.32, 0.63, 0.11),
        vec4(-0.44, 0.51, 0.74, 0.17),
        vec4(0.58, -0.62, 0.53, 0.26),
        vec4(-0.65, -0.43, -0.63, 0.39),
        vec4(0.22, 0.81, -0.54, 0.61),
        vec4(-0.79, 0.21, 0.58, 0.93),
    );
    var out: Waves;
    out.slope = vec3(0.0);
    out.curve = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));
    for (var i = 0; i < 6; i++) {
        let wave = kinds[i];
        let length = wave.w;
        let resolved = smoothstep(length * 0.5, length * 0.1, footprint);
        if (resolved <= 0.0) {
            continue;
        }
        let k = normalize(wave.xyz) * (2.0 * PI / length);
        let pace = 2.0 * PI * 0.25 / length;
        let wander = noise3(x * 1.7 + f32(i) * 7.3) - 0.5;
        let phase = dot(k, x) - pace * t * (1.0 + 0.3 * f32(i % 2)) + wander * 2.5;
        let height = STEEPNESS * length * resolved;
        out.slope += k * cos(phase) * height;
        let bend = -sin(phase) * height;
        out.curve += mat3x3<f32>(k * k.x * bend, k * k.y * bend, k * k.z * bend);
    }
    return out;
}

/// The two carried runs of the waves, blended.
fn waves_carried(run: Carried, t: f32, footprint: f32) -> Waves {
    let a = waves(run.a, t, footprint);
    let b = waves(run.b, t, footprint);
    var out: Waves;
    out.slope = a.slope * run.weight_a + b.slope * (1.0 - run.weight_a);
    out.curve = a.curve * run.weight_a + b.curve * (1.0 - run.weight_a);
    return out;
}
