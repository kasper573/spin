// The drum's ground as the water lying on it has it (see `shallows.wgsl` for what a `ground`
// module must define): a chart of the glass in metres, round the ring from the water's site and
// along the axis from it, with heights measured inward from the glass. Water weighs what the
// spin makes it weigh at its height, less or more as it runs with the ground's turn or against
// it; a cell is shorter round the ring the higher it is taken.
#define_import_path ground
#import vessel::{drum, ground_under}

fn ground_bed(at: vec2<f32>) -> f32 {
    let turn = at.x * drum.per_metre / drum.radius;
    let half = sin(0.5 * turn);
    let on_the_glass = vec3(-2.0 * drum.radius * half * half, at.y * drum.per_metre, drum.radius * sin(turn));
    return ground_under(on_the_glass).x / drum.per_metre;
}

fn ground_weight(height: f32) -> f32 {
    let spin = drum.spin * drum.per_second;
    return spin * spin * from_the_axis(height);
}

/// A flow round the ring (toward +z of the water's frame, which the stars pass toward) along the
/// axis and inward, under the turn of the frame about the axis, the curve of the chart round it,
/// and the spin's quickening.
fn ground_turning(height: f32, flow: vec3<f32>) -> vec3<f32> {
    let spin = drum.spin * drum.per_second;
    let quickening = drum.spin_rate * drum.per_second * drum.per_second;
    let r = from_the_axis(height);
    return vec3(
        -2.0 * spin * flow.z + flow.x * flow.z / r + quickening * r,
        0.0,
        2.0 * spin * flow.x - flow.x * flow.x / r,
    );
}

fn ground_stretch(height: f32) -> vec2<f32> {
    return vec2(from_the_axis(height) * drum.per_metre / drum.radius, 1.0);
}

fn ground_held(height: f32) -> f32 {
    return height - 0.5 * height * height * drum.per_metre / drum.radius;
}

fn ground_raised(held: f32) -> f32 {
    return 2.0 * held / (1.0 + sqrt(max(1.0 - 2.0 * held * drum.per_metre / drum.radius, 0.0)));
}

fn from_the_axis(height: f32) -> f32 {
    return drum.radius / drum.per_metre - height;
}
