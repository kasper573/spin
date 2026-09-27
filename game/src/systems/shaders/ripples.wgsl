// The ripples on the water: a field of small waves of different lengths crossing at odd
// angles, each running at its own pace and coming in patches, carried along by the flow. The
// waves' phases are left alone by the patches, so that the slope this gives back stays the
// gradient of the height it gives back, and the surface shades as the water it stands for.
// The same field bends what is seen through the surface and focuses the sunlight that falls
// through it onto the bed, so both are drawn from here.
#define_import_path ripples

const PI: f32 = 3.14159265;
// the ripples are carried along by the flow in two overlapping runs of this length, each faded
// in and out, so that neither is ever seen to reset; no run carries them further than this,
// since a run carried far is sheared into streaks wherever the flow varies, so in fast water
// the ripples lag the flow
const FLOW_PERIOD: f32 = 2.0;
const CARRY: f32 = 1.0;
// how fast the patches the waves come in drift, in cells of their noise a second
const SWELL_DRIFT: f32 = 0.094;
// the height of each wave, as a fraction of its length
const STEEPNESS: f32 = 0.006;
// each kind of wave comes in patches: how tall it stands where it is faintest and fullest,
// as a share of that height
const SWELL_LEAST: f32 = 0.25;
const SWELL_MOST: f32 = 1.75;

/// The world's time as shaders are told it: how far into a period it is, and the period, after
/// which it winds back to nought. Whatever moves steadily with it moves a whole number of times
/// in the period, so that it is never seen to wind back.
struct Clock {
    seconds: f32,
    period: f32,
}

/// How far through its cycle, as a share of it, something is that goes round about once every
/// `cycle` seconds: as many whole times in the clock's period as come nearest.
fn cycled(clock: Clock, cycle: f32) -> f32 {
    let times = max(round(clock.period / cycle), 1.0);
    return fract(clock.seconds / clock.period * times);
}

// noise that drifts with the clock repeats itself this many cells on along each axis, so that a
// drift of a whole number of repeats in the clock's period comes back where it started
const REPEAT: f32 = 64.0;

/// How far something drifting at about `rate` cells a second has drifted through noise that
/// repeats every `REPEAT` cells: as many whole repeats in the clock's period as come nearest.
fn drifted(clock: Clock, rate: f32) -> f32 {
    let repeats = round(rate * clock.period / REPEAT);
    return clock.seconds / clock.period * repeats * REPEAT;
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

/// Where noise drifting at `speed` cells a second has drifted to: round a circle so wide that it
/// turns once in the clock's period, so that it is back where it started when the clock winds
/// back, and turns too slowly to be seen to.
fn circled(clock: Clock, speed: f32) -> vec2<f32> {
    let turn = 2.0 * PI * clock.seconds / clock.period;
    return speed * clock.period / (2.0 * PI) * vec2(cos(turn), sin(turn));
}

/// Noise that repeats every `REPEAT` cells along each axis, for what drifts with the clock.
fn repeating_noise3(p: vec3<f32>) -> f32 {
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    // a power of two of cells, so that the cell and the next wrap by their low bits alone
    let wrap = vec3(i32(REPEAT) - 1);
    let i = vec3<f32>(vec3<i32>(floor(p)) & wrap);
    let j = vec3<f32>((vec3<i32>(floor(p)) + 1) & wrap);
    let a = mix(hash3(i), hash3(vec3(j.x, i.y, i.z)), u.x);
    let b = mix(hash3(vec3(i.x, j.y, i.z)), hash3(vec3(j.x, j.y, i.z)), u.x);
    let c = mix(hash3(vec3(i.x, i.y, j.z)), hash3(vec3(j.x, i.y, j.z)), u.x);
    let d = mix(hash3(vec3(i.x, j.y, j.z)), hash3(j), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

/// The shortest wave whose light has not yet crossed after `path` metres of water, whose
/// slopes were bent by `bend`. A wave of length L curves its rays by 4 pi^2 * STEEPNESS / L
/// where it stands fullest, so they cross after about L / (bend * 4 pi^2 * STEEPNESS * the
/// fullest swell); past that they have crossed and spread again, so the caustic it draws on
/// the bed has washed out into an even glow rather than the pattern a paraxial focus gives.
fn crossing_length(path: f32, bend: f32) -> f32 {
    return path * bend * 4.0 * PI * PI * STEEPNESS * SWELL_MOST;
}

/// A point of the water's frame carried back along the flow, in two runs that overlap, and
/// the weight the first run has now.
struct Carried {
    a: vec3<f32>,
    b: vec3<f32>,
    weight_a: f32,
}

// how fast the particles of water at rest still jostle, in metres per second
const JOSTLING: f32 = 0.12;

fn carried(x: vec3<f32>, flow: vec3<f32>, clock: Clock) -> Carried {
    let phase_a = cycled(clock, FLOW_PERIOD);
    let phase_b = fract(phase_a + 0.5);
    // ripples ride a current, not the jostling of the particles the flow is read from, which
    // differs from one to the next and would wring the ripples between them into rings: water
    // moving slower than its ripples run carries them nowhere they were not going
    let speed = length(flow);
    let current = flow * smoothstep(JOSTLING, 3.0 * JOSTLING, speed);
    let ride = current * min(1.0, CARRY / (FLOW_PERIOD * max(speed, 1e-3)));
    var out: Carried;
    out.a = x - ride * phase_a * FLOW_PERIOD;
    out.b = x - ride * phase_b * FLOW_PERIOD;
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
fn waves(x: vec3<f32>, clock: Clock, footprint: f32) -> Waves {
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
        let resolved = smoothstep(length * 0.25, length * 0.08, footprint);
        if (resolved <= 0.0) {
            continue;
        }
        let k = normalize(wave.xyz) * (2.0 * PI / length);
        let cycle = length / (0.25 * (1.0 + 0.3 * f32(i % 2)));
        let phase = dot(k, x) - 2.0 * PI * cycled(clock, cycle);
        // each kind of wave comes in drifting patches rather than everywhere at once
        let swell = noise3(x * (0.5 / length) + vec3(f32(i) * 3.1, circled(clock, SWELL_DRIFT)));
        let height = STEEPNESS * length * resolved * mix(SWELL_LEAST, SWELL_MOST, swell * swell);
        out.slope += k * cos(phase) * height;
        let bend = -sin(phase) * height;
        out.curve += mat3x3<f32>(k * k.x * bend, k * k.y * bend, k * k.z * bend);
    }
    return out;
}

/// The two carried runs of the waves, blended.
fn waves_carried(run: Carried, clock: Clock, footprint: f32) -> Waves {
    let a = waves(run.a, clock, footprint);
    let b = waves(run.b, clock, footprint);
    var out: Waves;
    out.slope = a.slope * run.weight_a + b.slope * (1.0 - run.weight_a);
    out.curve = a.curve * run.weight_a + b.curve * (1.0 - run.weight_a);
    return out;
}
