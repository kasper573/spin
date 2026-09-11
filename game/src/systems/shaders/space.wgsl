// What is seen in a direction among the stars: the dark of space, the stars, and the sun. Shared
// by the star sphere and by everything that reflects it, so that a reflection of the sun lands
// where the sun is.
#define_import_path space

// how bright the stars are against a sunlit ring, which an eye adapted to the sun would not
// see at all; enough to show
const STARLIGHT: f32 = 0.2;

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yxz + 33.33);
    return fract((r.x + r.y) * r.z);
}

/// How finely the sky is cut into cells, one star to a cell at most, and how much of a cell a
/// star covers.
const CELLS: f32 = 140.0;
const STAR: f32 = 0.35;
/// What the whole field comes to averaged over the sky: the share of cells holding a star, by
/// what one of them covers of its cell, by how bright one is on average.
const FIELD: f32 = 0.035 * 0.19 * 0.75;

/// The stars in a direction, as something that covers `spread` radians of sky sees them: a hash
/// of the cell of the sky the direction falls in.
///
/// A star is a point, and a point sampled once is either hit or missed, so a mirror of the sky —
/// or a sky the eye is sweeping past — breaks into sparks unless what covers more sky than a
/// star does is told so. A star narrower than the spread is spread over it, keeping the light it
/// carries; and where the spread covers many cells at once, what is seen is the field's own
/// average rather than whichever cell was landed on.
fn stars(dir: vec3<f32>, spread: f32) -> vec3<f32> {
    let cells = spread * CELLS;
    let wide = STAR + cells;
    let cell = floor(dir * CELLS);
    let h = hash3(cell);
    let centre = (cell + 0.5 + vec3(hash3(cell + 1.7), hash3(cell + 3.1), hash3(cell + 5.3)) - 0.5) / CELLS;
    let d = length(dir - normalize(centre)) * CELLS;
    let star = smoothstep(wide, 0.0, d) * step(0.965, h) * (STAR * STAR) / (wide * wide);
    let bright = 0.5 + 0.5 * hash3(cell + 9.9);
    let tint = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.92, 0.8), hash3(cell + 2.2));
    let one = tint * star * bright;
    return mix(one, vec3(FIELD), smoothstep(0.5, 2.0, cells)) * STARLIGHT;
}

/// The sun's disc, and the glare around it that a lens and an eye both make of it.
fn sun(dir: vec3<f32>, to_sun: vec3<f32>) -> vec3<f32> {
    let c = dot(dir, to_sun);
    let disc = smoothstep(0.99985, 0.99995, c);
    let glare = pow(max(c, 0.0), 600.0) * 0.6 + pow(max(c, 0.0), 40.0) * 0.06;
    return vec3(1.0, 0.97, 0.9) * (disc * 4.0 + glare);
}

/// Everything seen in a direction among the stars, by something covering `spread` of sky.
fn space_colour(dir: vec3<f32>, to_sun: vec3<f32>, background: vec3<f32>, spread: f32) -> vec3<f32> {
    return background + stars(dir, spread) + sun(dir, to_sun);
}
