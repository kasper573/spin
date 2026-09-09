// Types and grid helpers shared by every kernel of the fluid solver. Everything is in the
// canonical water's units (see `resolution.rs`) and the vessel's frame. Particles are binned into a
// box of cells that wraps round on itself, so the grid has no bounds: a cell and another one a
// whole box away share a slot, and any point anywhere lands in one. The box keeps neighbouring
// cells next to each other in memory, which the neighbour searches live on. Each particle
// carries a well-mixed key of its true cell, so a search over the 27 cells around a point takes
// only the particles of those cells: a shared slot costs a few wasted reads and never a double
// count.
#define_import_path fluid_common

struct Params {
    dt: f32,
    h: f32,
    h_sq: f32,
    poly: f32,
    spiky: f32,
    w_zero: f32,
    mass: f32,
    rest_density: f32,
    scorr_k: f32,
    scorr_wq: f32,
    eps_lambda: f32,
    max_delta: f32,
    max_speed: f32,
    margin: f32,
    air_k: f32,
    wall_keep: f32,
    viscosity: f32,
    body_drag: f32,
    wet_ref: f32,
    spacing: f32,
    count: u32,
    body_count: u32,
    sample_count: u32,
    pending: u32,
    // 1 / cell size
    inv_cell: f32,
    // slots in the cell table, a power of two
    cells: u32,
    // the first accumulator of this frame's slots
    accumulators: u32,
    // lattice sites on offer to the particles joining
    candidates: u32,
    // what thinning scales the kept particles' positions and velocities by
    thin_scale: f32,
    thin_scale_v: f32,
}

struct GpuBody {
    // xyz: centre of mass, w: 1 when solid
    position: vec4<f32>,
    // rows of the rotation matrix
    row_x: vec4<f32>,
    row_y: vec4<f32>,
    row_z: vec4<f32>,
    velocity: vec4<f32>,
    angular: vec4<f32>,
    // the hull's local centre, and its radius in w
    shape: vec4<f32>,
    // x: first sample of the shape, y: sample count, z: first boundary entry
    slots: vec4<u32>,
    // volume per sample, inverse mass, largest inverse inertia, reach
    extra: vec4<f32>,
}

struct Bodies {
    items: array<GpuBody, 16>,
}

struct Boundary {
    // xyz: world position, w: Ψ (volume weight)
    pos: vec4<f32>,
    // xyz: world velocity, w: body index
    vel: vec4<f32>,
}

struct SampleState {
    // xyz: kernel-weighted water velocity, w: water density at the sample
    field: vec4<f32>,
    // xyz: buoyancy force, w: wetness
    force: vec4<f32>,
}

// Accumulators are fixed-point integers with this many units per unit.
const FIXED: f32 = 65536.0;

@group(0) @binding(0) var<uniform> params: Params;

fn coords_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p * params.inv_cell));
}

/// A well-mixed 32-bit key for a cell, which the cell's particles carry.
fn cell_key(c: vec3<i32>) -> u32 {
    var h = (bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ (bitcast<u32>(c.z) * 83492791u);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

// the box of slots: cells this far apart along an axis share a slot
const TABLE_X: u32 = 64u;
const TABLE_Y: u32 = 32u;
const TABLE_Z: u32 = 64u;

/// The slot of a cell in the box.
fn cell_slot(c: vec3<i32>) -> u32 {
    let w = vec3<u32>(bitcast<u32>(c.x) & (TABLE_X - 1u), bitcast<u32>(c.y) & (TABLE_Y - 1u), bitcast<u32>(c.z) & (TABLE_Z - 1u));
    return (w.x * TABLE_Y + w.y) * TABLE_Z + w.z;
}

/// The cell at offset `n` of 27 around `c`.
fn neighbour_cell(c: vec3<i32>, n: u32) -> vec3<i32> {
    return c + vec3<i32>(i32(n / 9u), i32((n / 3u) % 3u), i32(n % 3u)) - vec3(1);
}
