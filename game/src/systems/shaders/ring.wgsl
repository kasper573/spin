// The ring itself, as a shape rays are cast against: a cylinder of glass closed by a disc at
// each end, with everything in it measured from a site on its wall. Its frame has the axis
// `ring.x` in along -x, so the axis runs through (-ring.x, y, 0), and its caps stand `ring.y`
// out along y from the middle of the drum.
#define_import_path ring

// how far outside the wall a point may stand and still count as on it, as a share of the
// ring's radius squared: what f32 leaves of a difference of numbers that size
const TOUCHING: f32 = 1e-5;

/// How far a ray from inside the ring runs before it meets the ring's wall, or leaves through
/// a cap, whichever comes first, and whether it was the wall. The far root is found without
/// the near one's cancellation, so it holds for any ring. A ray starting outside the ring runs
/// nowhere in it.
struct RingRun {
    distance: f32,
    wall: bool,
}

fn ring_run(at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>) -> RingRun {
    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * ((ring.x + at.x) * dir.x + at.z * dir.z);
    // how far the point stands outside the wall, squared: a difference of numbers as big as the
    // ring itself, so on the wall it is known only to a few parts in ten million of that, and
    // anything within that of the wall counts as setting out from the wall
    var c = 2.0 * ring.x * at.x + at.x * at.x + at.z * at.z;
    if (c > TOUCHING * ring.x * ring.x) {
        return RingRun(0.0, false);
    }
    c = min(c, 0.0);
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
            return RingRun(max(cap, 0.0), false);
        }
    }
    return RingRun(max(t, 0.0), true);
}

/// Which way is up at a point of the ring's frame, which is away from the axis the spin
/// presses everything from.
fn ring_up(at: vec3<f32>, radius: f32) -> vec3<f32> {
    return -normalize(vec3(radius + at.x, 0.0, at.z));
}

/// Whether a sun reaches a point inside the ring: its light comes in through a cap, so the way
/// toward it must leave the ring's width before it crosses the ring.
fn sun_reaches(at: vec3<f32>, to_sun: vec3<f32>, ring: vec2<f32>) -> bool {
    return !ring_run(at, to_sun, ring).wall;
}

/// The stretch of a ray the sun reaches, from `at` along `dir`, as distances along it.
struct Sunlit {
    entering: f32,
    leaving: f32,
}

/// Where a ray runs in the sun and where the ring's own wall shades it. A point is in the sun
/// when the way from it toward the sun crosses a cap's plane within that cap's rim, and that
/// place moves with the point along a straight line, so the points in the sun are those whose
/// image lies in a disc: one unbroken stretch of any ray, found by putting that image inside
/// the rim. Rays that cross a ring of any size are answered exactly, so a shaft of sunlight
/// has an edge rather than steps.
fn sunlit_run(at: vec3<f32>, dir: vec3<f32>, distance: f32, to_sun: vec3<f32>, ring: vec2<f32>) -> Sunlit {
    if (abs(to_sun.y) < 1e-6) {
        return Sunlit(0.0, 0.0);
    }
    let cap = select(-ring.y, ring.y, to_sun.y > 0.0);
    let over = 1.0 / to_sun.y;
    let base = at + to_sun * ((cap - at.y) * over);
    let rate = dir - to_sun * (dir.y * over);
    let x = ring.x + base.x;
    let a = rate.x * rate.x + rate.z * rate.z;
    let b = 2.0 * (x * rate.x + base.z * rate.z);
    let c = 2.0 * ring.x * base.x + base.x * base.x + base.z * base.z;
    if (a < 1e-12) {
        if (c <= 0.0) {
            return Sunlit(0.0, distance);
        }
        return Sunlit(0.0, 0.0);
    }
    let disc = b * b - 4.0 * a * c;
    if (disc <= 0.0) {
        return Sunlit(0.0, 0.0);
    }
    let root = sqrt(disc);
    let entering = clamp((-b - root) / (2.0 * a), 0.0, distance);
    let leaving = clamp((-b + root) / (2.0 * a), 0.0, distance);
    return Sunlit(entering, leaving);
}

/// How much of the stretch between two distances along a ray the sun reaches.
fn sunlit_share(lit: Sunlit, begins: f32, ends: f32) -> f32 {
    return max(min(ends, lit.leaving) - max(begins, lit.entering), 0.0) / max(ends - begins, 1e-12);
}
