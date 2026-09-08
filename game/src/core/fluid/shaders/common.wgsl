// Types and grid helpers shared by every kernel of the fluid solver.
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
    // xyz: grid origin, w: 1 / cell size
    grid_min: vec4<f32>,
    // xyz: cells per axis, w: total cells
    grid_dims: vec4<i32>,
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
    // box: half extents; sphere: local centre and radius in w
    shape: vec4<f32>,
    // x: first sample of the shape, y: sample count, z: first boundary entry, w: 1 for a sphere
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
    let c = vec3<i32>(floor((p - params.grid_min.xyz) * params.grid_min.w));
    return clamp(c, vec3(0), params.grid_dims.xyz - vec3(1));
}

fn cell_at(c: vec3<i32>) -> i32 {
    let d = params.grid_dims.xyz;
    if (any(c < vec3(0)) || any(c >= d)) {
        return -1;
    }
    return (c.x * d.y + c.y) * d.z + c.z;
}

fn cell_of(p: vec3<f32>) -> i32 {
    return cell_at(coords_of(p));
}
