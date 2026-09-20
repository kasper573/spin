// The chart the water on the ground is kept by, as every kernel of it has it: what a frame asks,
// a cell as it is stored, and which slot a cell of the chart has.
#define_import_path shallows_chart
#import ground::{ground_held, ground_raised}

struct Shallows {
    size: vec2<u32>,
    wraps: vec2<u32>,
    // a cell's sides on the chart, and where on the chart the low corner of the first cell is
    cell: vec2<f32>,
    low: vec2<f32>,
    dt: f32,
    bed_friction: f32,
    // how many entries of `poured` are to be poured
    pouring: u32,
    // 1 when the cells have been put back as they were kept, for the ground to be laid under
    restoring: u32,
    // what `stand` stands the water to: a level, and how it tilts along the chart's two axes
    standing: vec4<f32>,
    // the water in flight: how many particles are counted, the cubic metres each holds, and how
    // near the ground one lies that lies on it
    particles: u32,
    particle_volume: f32,
    landing: f32,
    // the bodies' hulls: how many samples they have, the first of the frame's accumulators
    // they are summed into, and the water's drag on them and its density in those sums' units
    hull_samples: u32,
    accumulators: u32,
    body_drag: f32,
    rest_density: f32,
}

struct Cell {
    face: f32,
    // how fast the water at the face climbs, by the pressure beyond the weight
    climbing: f32,
    flow: vec2<f32>,
}

struct Pressing {
    at_bed: f32,
    // the right side and the diagonal of the pressure's system
    wanted: f32,
    own: f32,
    breaking: f32,
    // the climbing before the pressure has had its way
    climbed: f32,
    // the water the cell holds beyond what its face says, a face being a height in so many
    // bits and no higher than water can stand: what the face leaves out is kept for the next
    // step, or a film creeping over dry ground, let in by less a step than a face can tell,
    // would never come, and water heaped higher than it can stand would be lost
    owed: f32,
}

@group(0) @binding(0) var<uniform> shallows: Shallows;

const WALL: i32 = -1;

fn cell_count() -> u32 {
    return shallows.size.x * shallows.size.y;
}

fn cell_of(thread: u32) -> vec2<i32> {
    return vec2<i32>(i32(thread % shallows.size.x), i32(thread / shallows.size.x));
}

/// A cell that is to hold so much water over its bed: the face that says so as nearly as a
/// face can, and what it leaves out.
struct Stood {
    face: f32,
    owed: f32,
}

fn stood(held: f32, bed: f32) -> Stood {
    let under = ground_held(bed);
    let face = max(ground_raised(held + under), bed);
    return Stood(face, held - (ground_held(face) - under));
}

/// The slot of a cell, or WALL off the chart.
fn slot(c: vec2<i32>) -> i32 {
    let size = vec2<i32>(shallows.size);
    var at = c;
    if (shallows.wraps.x == 1u) {
        at.x = brought_round(at.x, size.x);
    }
    if (shallows.wraps.y == 1u) {
        at.y = brought_round(at.y, size.y);
    }
    if (at.x < 0 || at.x >= size.x || at.y < 0 || at.y >= size.y) {
        return WALL;
    }
    return at.y * size.x + at.x;
}

/// A place along an axis that closes on itself after `size` cells, brought onto the axis. A
/// negative one is brought up by whole turns before its remainder is taken: GPUs do not agree
/// on what is left over of a negative whole number divided, and some take it for the unsigned
/// number with the same bits; nor do they divide floats exactly enough to count turns by.
fn brought_round(at: i32, size: i32) -> i32 {
    var turned = at;
    if (turned < 0) {
        turned += (size - 1 - turned) / size * size;
    }
    return turned % size;
}
