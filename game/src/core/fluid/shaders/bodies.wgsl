// Two-way coupling between the water and the rigid bodies, one thread per boundary sample:
// samples are placed on the bodies' hulls, the water around each sample gives its wetness and
// the hydrostatic buoyancy, and the no-slip drag the particles felt is returned to the bodies.
// Everything a body receives is summed into fixed-point running totals the CPU reads back and
// differences, so a late or doubled readback still applies each substep exactly once.
#import vessel::vessel_air_velocity
#import fluid_common::{params, Bodies, GpuBody, Boundary, SampleState, coords_of, cell_key, cell_slot, neighbour_cell, FIXED}

@group(0) @binding(1) var<storage, read> position: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> velocity_next: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
@group(0) @binding(12) var<storage, read> key: array<u32>;

@group(2) @binding(0) var<uniform> bodies: Bodies;
@group(2) @binding(1) var<storage, read> samples: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> boundary: array<Boundary>;
@group(2) @binding(3) var<storage, read_write> sample_state: array<SampleState>;
@group(2) @binding(4) var<storage, read_write> accum: array<atomic<i32>>;

const ACC_BUOYANCY: u32 = 0u;
const ACC_BUOYANCY_TORQUE: u32 = 3u;
const ACC_FLOW: u32 = 6u;
const ACC_COUPLING: u32 = 9u;
const ACC_WET: u32 = 10u;
const ACC_STRIDE: u32 = 16u;

fn body_of_sample(k: u32) -> u32 {
    for (var b = 0u; b < params.body_count; b++) {
        let slots = bodies.items[b].slots;
        if (k >= slots.z && k < slots.z + slots.y) {
            return b;
        }
    }
    return 0u;
}

fn rotate(b: u32, l: vec3<f32>) -> vec3<f32> {
    let body = bodies.items[b];
    return vec3(dot(body.row_x.xyz, l), dot(body.row_y.xyz, l), dot(body.row_z.xyz, l));
}

fn add_fixed(index: u32, value: f32) {
    atomicAdd(&accum[index], i32(value * FIXED));
}

fn add_fixed3(index: u32, value: vec3<f32>) {
    add_fixed(index, value.x);
    add_fixed(index + 1u, value.y);
    add_fixed(index + 2u, value.z);
}

fn poly(r2: f32) -> f32 {
    let t = params.h_sq - r2;
    return params.poly * t * t * t;
}

@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.sample_count) {
        return;
    }
    let b = body_of_sample(k);
    let body = bodies.items[b];
    let local = samples[body.slots.x + (k - body.slots.z)];
    let r = rotate(b, local.xyz);
    let world = body.position.xyz + r;
    let vel = body.velocity.xyz + cross(body.angular.xyz, r);
    boundary[k] = Boundary(vec4(world, local.w), vec4(vel, f32(b)));
}

/// Hydrostatic buoyancy on bodies. PBF pressure is a per-step correction, not a depth-integrated
/// pressure, so Archimedes is added explicitly: local water density at each sample gives wetness,
/// and the water at rest in the vessel, turning with it, gives the pressure gradient
/// (rho * v_t^2 / r, pointing inward). The water right at the hull is not asked, since a moving
/// hull drags it along and would read its own motion back as pressure.
@compute @workgroup_size(64)
fn buoyancy(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.sample_count) {
        return;
    }
    let s = boundary[k];
    let x = s.pos.xyz;
    var rho = 0.0;
    var fv = vec3(0.0);
    let c = coords_of(x);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let pj = position[j];
            let vj = velocity[j];
            if (kj != k) {
                continue;
            }
            let r = x - pj.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let w = params.mass * poly(r2);
            rho += w;
            fv += w * vj.xyz;
        }
    }
    if (rho < 1.0) {
        sample_state[k] = SampleState(vec4(fv, rho), vec4(0.0));
        return;
    }
    let b = u32(s.vel.w);
    let body = bodies.items[b];
    let wet = min(rho / params.wet_ref, 1.0);
    let rr = max(length(x.xz), 1e-6);
    let tangent = vec2(x.z, -x.x) / rr;
    let vt = dot(vessel_air_velocity(x).xz, tangent);
    let ac = vt * vt / rr;
    let fmag = params.rest_density * body.extra.x * wet * ac;
    let force = vec3(-fmag * x.x / rr, 0.0, -fmag * x.z / rr);
    sample_state[k] = SampleState(vec4(fv, rho), vec4(force, wet));
    let impulse = force * params.dt;
    let base = b * ACC_STRIDE;
    add_fixed3(base + ACC_BUOYANCY, impulse);
    add_fixed3(base + ACC_BUOYANCY_TORQUE, cross(x - body.position.xyz, impulse));
    add_fixed(base + ACC_WET, wet / f32(body.slots.y));
}

/// The water's momentum around each sample, weighted by the same coupling the particles felt
/// toward it, so the CPU can relax the body toward the flow it is actually in. Reporting the
/// flow rather than a difference against the body's velocity keeps the coupling stable however
/// late the readback lands. The sums are scaled by the body's inverse mass to keep them within
/// the fixed-point range.
@compute @workgroup_size(64)
fn drag(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= params.sample_count) {
        return;
    }
    let s = boundary[k];
    let x = s.pos.xyz;
    let b = u32(s.vel.w);
    let body = bodies.items[b];
    var flow = vec3(0.0);
    var coupling = 0.0;
    let c = coords_of(x);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let pj = position[j];
            let vj = velocity[j];
            let scale = velocity_next[j].w;
            if (kj != k) {
                continue;
            }
            let r = x - pj.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let w = params.mass * params.body_drag * poly(r2) * (s.pos.w / params.rest_density) * scale;
            flow += w * vj.xyz;
            coupling += w;
        }
    }
    if (coupling <= 0.0) {
        return;
    }
    flow *= body.extra.y;
    coupling *= body.extra.y;
    let base = b * ACC_STRIDE;
    add_fixed3(base + ACC_FLOW, flow);
    add_fixed(base + ACC_COUPLING, coupling);
}
