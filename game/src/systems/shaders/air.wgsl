// The air the ring holds, as light crosses it. What a metre of it does is worked out on the
// other side from what the air is — see `systems/air.rs` — and what arrives here is that,
// together with how the air thins away from the rim: a ring holds its air by spinning, so the
// air stands in the potential the spin makes and settles the way a planet's does with height.
//
// Nothing here is a fit to how the ring ought to look. The density along a ray says how much
// light each stretch of it takes out of the ray, how much it turns into it, and how sharply it
// bends it: what is taken and turned is integrated exactly along the ray, and the bending is
// walked in steps. What follows from that at the
// ring's own size is slight; what follows from it in air a hundred times thicker, or round a
// ring spun a hundred times harder, is a blue sky, a white haze and a sun pulled out of shape
// and fringed with colour, because those are what the same steps come to when the air is deep
// enough for them to show.
#define_import_path air

#import ring::sunlit_run

struct Air {
    // what a metre of the air at the rim scatters off its molecules, per channel
    rayleigh: vec4<f32>,
    // what a metre of it scatters off what it carries, per channel, and how far forward a
    // grain of that throws what it turns
    mie: vec4<f32>,
    // what a metre of it takes out of a ray altogether, per channel: what it turns aside and
    // what it swallows
    taken: vec4<f32>,
    // how much it slows light at the rim, as n - 1 per channel, and the coefficient of
    // r^2 - rim^2 in how its density falls away from the rim
    slowing: vec4<f32>,
}

/// How far a point of the ring's frame stands from the axis, that frame having its origin on
/// the wall with the axis `radius` in along -x.
fn from_axis(at: vec3<f32>, radius: f32) -> f32 {
    return length(vec2(radius + at.x, at.z));
}

const PI: f32 = 3.14159265;
// the most steps a ray is walked across the air in, and the fewest: how long a step may be is
// worked out where the step starts, so a ray that sets out in dense air and climbs out of it
// takes short steps while it is in it and one long one for everything after.
const FEWEST_STEPS: i32 = 6;
const MOST_STEPS: i32 = 96;
/// How much of the air's own e-folding a step may cross.
const EVENNESS: f32 = 0.25;
/// How far round the circle the air turns a ray on a step may carry it, in radians.
const TURNING: f32 = 0.02;

/// How dense the air is at a point against how dense it is at the rim.
fn air_density(air: Air, at: vec3<f32>, radius: f32) -> f32 {
    let r = from_axis(at, radius);
    return exp(air.slowing.w * (r * r - radius * radius));
}

/// How far a ray may run from a point in one step and still resolve the air it crosses. The
/// density goes as `exp(thinning * r^2)`, so what a step changes it by follows how far the step
/// carries the ray from the axis; a ray running level changes it only by the curve of its own
/// path, and one climbing away changes it at once. A stretch that holds almost nothing neither
/// dims a ray nor lights it however it is cut, so the thinner the air already is the farther a
/// step may reach: a ray leaving the air behind covers everything after it in one.
fn air_step(air: Air, at: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let thinning = abs(air.slowing.w);
    if (thinning <= 0.0) {
        return 1e30;
    }
    let axis = vec3(radius + at.x, 0.0, at.z);
    let r = max(length(axis), 1e-6);
    let climb = abs(dot(axis / r, dir)) * r;
    let even = sqrt(climb * climb + EVENNESS / thinning) - climb;
    return even / max(air_density(air, at, radius), 1e-6);
}

/// How readily a molecule turns light through this angle: as willingly backward as forward,
/// and half as willingly side on.
fn rayleigh_phase(cosine: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cosine * cosine);
}

/// How readily a grain turns it, as a lobe of the asymmetry Mie's solution gives the grain.
fn mie_phase(forward: f32, cosine: f32) -> f32 {
    let g = clamp(forward, -0.95, 0.95);
    let d = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(max(d, 1e-6)));
}

/// What a stretch of air did to the light that crossed it: the share of it that is left, and
/// the light the air turned into the ray along the way, already dimmed by the air in front of
/// where it was turned.
struct Crossing {
    left: vec3<f32>,
    turned: vec3<f32>,
}

/// Dawson's integral, `exp(-x^2)` times the integral of `exp(u^2)` from nothing to `x`, by
/// Rybicki's sum of Gaussians, which is good to the last digit a float keeps.
fn dawson(x: f32) -> f32 {
    let size = abs(x);
    if (size < 0.2) {
        let xx = x * x;
        return x * (1.0 - 2.0 / 3.0 * xx * (1.0 - 0.4 * xx * (1.0 - 2.0 / 7.0 * xx)));
    }
    let nearest = 2.0 * floor(0.5 * size / DAWSON_SPACING + 0.5);
    let off = size - nearest * DAWSON_SPACING;
    var grows = exp(2.0 * off * DAWSON_SPACING);
    let twice = grows * grows;
    var above = nearest + 1.0;
    var below = above - 2.0;
    var sum = 0.0;
    for (var i = 0; i < 6; i++) {
        let odd = f32(2 * i + 1) * DAWSON_SPACING;
        sum += exp(-odd * odd) * (grows / above + 1.0 / (below * grows));
        above += 2.0;
        below -= 2.0;
        grows *= twice;
    }
    return sign(x) * exp(-off * off) * sum / sqrt(PI);
}

const DAWSON_SPACING: f32 = 0.4;

/// A straight ray through the ring's air: the square of how far it stands from the axis grows
/// as `2 * climb * s + level * s^2` with the distance `s` run, so the density along it is the
/// density where it starts times the exponential of `thinning` times that.
struct AirRay {
    density: f32,
    thinning: f32,
    climb: f32,
    level: f32,
}

fn air_ray(air: Air, at: vec3<f32>, dir: vec3<f32>, radius: f32) -> AirRay {
    let climb = (radius + at.x) * dir.x + at.z * dir.z;
    return AirRay(air_density(air, at, radius), air.slowing.w, climb, dir.x * dir.x + dir.z * dir.z);
}

/// How much air the first `run` metres of a ray hold, in metres of the air at the rim: the
/// integral of the density along it, which is Dawson's. Where the density hardly changes over
/// the run that form is the small difference of two large terms, and the integral is summed
/// at Gauss's four points instead, which is as exact there.
fn air_held(ray: AirRay, run: f32) -> f32 {
    let changes = ray.thinning * (2.0 * abs(ray.climb) * run + ray.level * run * run);
    if (changes < 2.0) {
        let nodes = vec4(-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116);
        let weights = vec4(0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451);
        let s = 0.5 * run * (nodes + 1.0);
        let dense = exp(ray.thinning * (2.0 * ray.climb * s + ray.level * s * s));
        return ray.density * 0.5 * run * dot(weights, dense);
    }
    let scale = sqrt(ray.thinning * ray.level);
    let start = scale * ray.climb / ray.level;
    let there = ray.density * exp(ray.thinning * (2.0 * ray.climb * run + ray.level * run * run));
    return (there * dawson(start + scale * run) - ray.density * dawson(start)) / scale;
}

/// `(1 - exp(-x)) / x`, which a float loses to rounding where `x` is small.
fn share_taken(x: vec3<f32>) -> vec3<f32> {
    let series = 1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0));
    return select((1.0 - exp(-x)) / max(x, vec3(1e-30)), series, x < vec3(0.05));
}

/// What `distance` metres of air from `at` along `dir` do to the light crossing them, `dir`
/// running from the eye toward what it is looking at. `sun` is a sun's light and `to_sun` the
/// way to it; `ambient` is the light reaching the air from everywhere at once.
///
/// Nothing is walked: what a stretch of air takes out of a ray goes as the air it holds, and
/// what it turns into the ray, from a source that is as strong all along it, is that source
/// over what a metre takes times the share of the ray the stretch took. The sun is such a
/// source over the one stretch of the ray it reaches, and the light from everywhere over all
/// of it, so both are had from how much air lies before three points of the ray.
fn air_crossed(air: Air, at: vec3<f32>, dir: vec3<f32>, distance: f32, ring: vec2<f32>, sun: vec3<f32>, to_sun: vec3<f32>, ambient: vec3<f32>) -> Crossing {
    var out: Crossing;
    out.left = vec3(1.0);
    out.turned = vec3(0.0);
    if (distance <= 0.0) {
        return out;
    }
    // the light met on the way is on its way to the eye, so it is turned through the angle
    // between the way it was already going and the way the ray runs
    let cosine = dot(dir, to_sun);
    let molecules = air.rayleigh.rgb * rayleigh_phase(cosine);
    let grains = air.mie.rgb * mie_phase(air.mie.w, cosine);
    // the ring shades its own air, so the sun reaches one stretch of the ray
    let sunlit = sunlit_run(at, dir, distance, to_sun, ring);
    let entering = clamp(sunlit.entering, 0.0, distance);
    let leaving = clamp(sunlit.leaving, entering, distance);
    let ray = air_ray(air, at, dir, ring.x);
    let whole = air_held(ray, distance);
    let before = air_held(ray, entering);
    let within = air_held(ray, leaving) - before;
    out.left = exp(-air.taken.rgb * whole);
    // a phase function gathers to one over the whole sphere, so the light from everywhere is
    // turned into the ray as readily as the air scatters at all
    out.turned = (molecules + grains) * sun * exp(-air.taken.rgb * before) * within * share_taken(air.taken.rgb * within)
        + (air.rayleigh.rgb + air.mie.rgb) * ambient * whole * share_taken(air.taken.rgb * whole);
    return out;
}

/// The light a short stretch of air turns into a ray that crosses it between two distances from
/// `at`, where that stretch is lit from one way only, by something too narrow for
/// `air_crossed` to have found: a shaft of light. It reaches the ray's start dimmed by the air
/// before it.
fn air_shaft(air: Air, at: vec3<f32>, dir: vec3<f32>, begins: f32, ends: f32, radius: f32, light: vec3<f32>, to_light: vec3<f32>) -> vec3<f32> {
    let cosine = dot(dir, to_light);
    let turning = air.rayleigh.rgb * rayleigh_phase(cosine) + air.mie.rgb * mie_phase(air.mie.w, cosine);
    let middle = 0.5 * (begins + ends);
    let density = air_density(air, at + dir * middle, radius);
    let before = 0.5 * (air_density(air, at, radius) + density);
    return turning * density * light * (ends - begins) * exp(-air.taken.rgb * before * middle);
}

/// Where a ray ends up pointing after `distance` metres of air, having been bent by the air's
/// own gradient: a ray in a medium whose index varies obeys `d(n t)/ds = grad n`, so it turns
/// toward the denser air, which in a ring is toward the rim. `slowing` is `n - 1` at the rim
/// for the colour being traced, so tracing three colours through the same air gives three
/// directions, and a ray that crosses enough of it comes out spread into them.
fn air_bent(air: Air, at: vec3<f32>, dir: vec3<f32>, distance: f32, radius: f32, slowing: f32) -> vec3<f32> {
    if (distance <= 0.0 || air.slowing.w == 0.0) {
        return dir;
    }
    var p = at;
    var d = dir;
    let longest = distance / f32(FEWEST_STEPS);
    var run = 0.0;
    for (var k = 0; k < MOST_STEPS; k++) {
        if (run >= distance) {
            break;
        }
        let axis = vec3(radius + p.x, 0.0, p.z);
        let r = max(length(axis), 1e-6);
        let outward = axis / r;
        let thinner = exp(air.slowing.w * (r * r - radius * radius));
        let n = 1.0 + slowing * thinner;
        // the gradient of the index, which lies along the axis's own outward direction
        let gradient = outward * (slowing * thinner * air.slowing.w * 2.0 * r);
        // a step must resolve not only the air but the turn it puts in the ray, which is the
        // gradient across the index: air steep enough to turn a ray right round is walked in
        // steps short against the circle it turns it on. Air too steep to walk to the end in the
        // steps allowed is not walked further at all: what is left of the ray is carried
        // straight, which leaves it short of where the air would have taken it rather than
        // throwing it anywhere whatever.
        let turning = length(gradient) / n;
        var step = min(air_step(air, p, d, radius), longest);
        step = min(min(step, TURNING / max(turning, 1e-30)), distance - run);
        d = normalize(d + (gradient - d * dot(gradient, d)) / n * step);
        p += d * step;
        run += step;
    }
    return d;
}
