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

/// The stars in a direction: a hash of the cell of the sky the direction falls in.
fn stars(dir: vec3<f32>) -> vec3<f32> {
    let cell = floor(dir * 140.0);
    let h = hash3(cell);
    let centre = (cell + 0.5 + vec3(hash3(cell + 1.7), hash3(cell + 3.1), hash3(cell + 5.3)) - 0.5) / 140.0;
    let d = length(dir - normalize(centre)) * 140.0;
    let star = smoothstep(0.35, 0.0, d) * step(0.965, h);
    let bright = 0.5 + 0.5 * hash3(cell + 9.9);
    let tint = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.92, 0.8), hash3(cell + 2.2));
    return tint * star * bright * STARLIGHT;
}

/// The sun's disc, and the glare around it that a lens and an eye both make of it.
fn sun(dir: vec3<f32>, to_sun: vec3<f32>) -> vec3<f32> {
    let c = dot(dir, to_sun);
    let disc = smoothstep(0.99985, 0.99995, c);
    let glare = pow(max(c, 0.0), 600.0) * 0.6 + pow(max(c, 0.0), 40.0) * 0.06;
    return vec3(1.0, 0.97, 0.9) * (disc * 4.0 + glare);
}

/// Everything seen in a direction among the stars.
fn space_colour(dir: vec3<f32>, to_sun: vec3<f32>, background: vec3<f32>) -> vec3<f32> {
    return background + stars(dir) + sun(dir, to_sun);
}
