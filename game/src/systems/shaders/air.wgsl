// The air the ring holds, as light crosses it. What a metre of it does is worked out on the
// other side from what the air is — see `systems/air.rs` — and what arrives here is that,
// together with how the air thins away from the rim: a ring holds its air by spinning, so the
// air stands in the potential the spin makes and settles the way a planet's does with height.
//
// Nothing here is a fit to how the ring ought to look. A ray is walked across the air in steps,
// and at every step the density where it stands says how much light that stretch takes out of
// it, how much it turns into it, and how sharply it bends it. What follows from that at the
// ring's own size is slight; what follows from it in air a hundred times thicker, or round a
// ring spun a hundred times harder, is a blue sky, a white haze and a sun pulled out of shape
// and fringed with colour, because those are what the same steps come to when the air is deep
// enough for them to show.
#define_import_path air

#import ring::{sunlit_run, sunlit_share}

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

/// Walk `distance` metres of air from `at` along `dir`, which runs from the eye toward what it
/// is looking at. `sun` is a sun's light and `to_sun` the way to it; `ambient` is the light
/// reaching the air from everywhere at once.
fn air_crossed(air: Air, at: vec3<f32>, dir: vec3<f32>, distance: f32, ring: vec2<f32>, sun: vec3<f32>, to_sun: vec3<f32>, ambient: vec3<f32>) -> Crossing {
    let radius = ring.x;
    var out: Crossing;
    out.left = vec3(1.0);
    out.turned = vec3(0.0);
    if (distance <= 0.0) {
        return out;
    }
    // the light met at a step is on its way to the eye, so it is turned through the angle
    // between the way it was already going and the way the ray runs
    let cosine = dot(dir, to_sun);
    let molecules = air.rayleigh.rgb * rayleigh_phase(cosine);
    let grains = air.mie.rgb * mie_phase(air.mie.w, cosine);
    // the ring shades its own air, so a ray crosses a stretch the sun reaches and stretches it
    // does not; where that stretch begins and ends is worked out once for the whole ray
    let sunlit = sunlit_run(at, dir, distance, to_sun, ring);
    let longest = distance / f32(FEWEST_STEPS);
    var run = 0.0;
    for (var k = 0; k < MOST_STEPS; k++) {
        if (run >= distance) {
            break;
        }
        var step = min(air_step(air, at + dir * run, dir, radius), longest);
        if (k == MOST_STEPS - 1) {
            step = distance - run;
        }
        step = min(step, distance - run);
        let here = at + dir * (run + step * 0.5);
        let density = air_density(air, here, radius);
        let extinction = air.taken.rgb * density;
        let scattering = (air.rayleigh.rgb + air.mie.rgb) * density;
        // what this stretch turns into the ray, lit by the sun over as much of it as the sun
        // reaches, and by the light that reaches it from everywhere at once, which it turns
        // into the ray from every direction alike since a phase function gathers to one over
        // the whole sphere
        let lit = sun * sunlit_share(sunlit, run, run + step);
        let turned = (molecules + grains) * density * lit + scattering * ambient;
        let across = exp(-extinction * step);
        out.turned += out.left * turned * (vec3(1.0) - across) / max(extinction, vec3(1e-30));
        out.left *= across;
        run += step;
    }
    return out;
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
