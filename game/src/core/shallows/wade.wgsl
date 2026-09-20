// Bodies in the water lying on the ground: what it lifts them by and carries them with, sample
// by sample of their hulls, summed where the water in flight sums what it does to them (see
// the fluid's `bodies.wgsl`) and in its units, so that whoever reads the sums reads one water.
// A sample stands for a ball of its share of the hull, wet by as much of the ball as is under
// the water's face.
#import ground::{ground_charted, ground_carried}
#import vessel::{drum, Through, vessel_gravity, vessel_gone_through, vessel_turned}
#import shallows_chart::{shallows, Cell, WALL, slot}
#import fluid_common::{Bodies, GpuBody, Boundary, FIXED, ACC_BUOYANCY, ACC_BUOYANCY_TORQUE, ACC_FLOW, ACC_COUPLING, ACC_WET, ACC_STRIDE}

@group(0) @binding(1) var<storage, read> bed: array<f32>;
@group(0) @binding(2) var<storage, read> cells: array<Cell>;
@group(0) @binding(3) var<storage, read> boundary: array<Boundary>;
@group(0) @binding(4) var<storage, read> samples: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> accum: array<atomic<i32>>;
@group(0) @binding(6) var<uniform> bodies: Bodies;

const PI: f32 = 3.14159265;

fn add_fixed3(index: u32, value: vec3<f32>) {
    for (var axis = 0u; axis < 3u; axis++) {
        atomicAdd(&accum[shallows.accumulators + index + axis], i32(value[axis] * FIXED));
    }
}

/// The share of a ball under a level that stands `over` its middle.
fn ball_under(over: f32, radius: f32) -> f32 {
    let up = clamp(over, -radius, radius) + radius;
    return up * up * (3.0 * radius - up) / (4.0 * radius * radius * radius);
}

fn flow_at(c: vec2<i32>, axis: u32) -> f32 {
    let s = slot(c);
    if (s == WALL) {
        return 0.0;
    }
    return cells[s].flow[axis];
}

@compute @workgroup_size(64)
fn wade(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= shallows.hull_samples) {
        return;
    }
    let placed = boundary[k];
    let b = u32(placed.vel.w);
    let body = bodies.items[b];
    let x = placed.pos.xyz;
    let at = ground_charted(x);
    let c = vec2<i32>(floor((at.xy - shallows.low) / shallows.cell));
    let s = slot(c);
    if (s == WALL) {
        return;
    }
    let deep = cells[s].face - bed[s];
    if (deep <= 0.0) {
        return;
    }
    let radius = pow(0.75 * body.extra.x / PI, 1.0 / 3.0) / drum.per_metre;
    let wet = ball_under(cells[s].face - at.z, radius);
    if (wet <= 0.0) {
        return;
    }
    // the sample as its body has it: where on the body it is, and whether it is beyond an
    // opening, where what the water does to it is turned back through
    let local = samples[body.slots.x + (k - body.slots.z)];
    let arm = vec3(dot(body.row_x.xyz, local.xyz), dot(body.row_y.xyz, local.xyz), dot(body.row_z.xyz, local.xyz));
    let gone = vessel_gone_through(body.position.xyz + arm);
    let dt = shallows.dt * drum.per_second;
    var lift = -shallows.rest_density * body.extra.x * wet * vessel_gravity(x) * dt * body.extra.y;
    let rising = cells[s].climbing * clamp((at.z - bed[s]) / deep, 0.0, 1.0);
    let along = vec2(
        0.5 * (flow_at(c, 0u) + flow_at(c + vec2(1, 0), 0u)),
        0.5 * (flow_at(c, 1u) + flow_at(c + vec2(0, 1), 1u)),
    );
    let coupling = shallows.body_drag * placed.pos.w * wet * body.extra.y;
    var flow = coupling * ground_carried(at.xy, vec3(along, rising));
    if (gone.there) {
        lift = vessel_turned(lift, 1u - gone.opening);
        flow = vessel_turned(flow, 1u - gone.opening);
    }
    let base = b * ACC_STRIDE;
    add_fixed3(base + ACC_BUOYANCY, lift);
    add_fixed3(base + ACC_BUOYANCY_TORQUE, cross(arm, lift));
    add_fixed3(base + ACC_FLOW, flow);
    atomicAdd(&accum[shallows.accumulators + base + ACC_COUPLING], i32(coupling * FIXED));
    atomicAdd(&accum[shallows.accumulators + base + ACC_WET], i32(wet / f32(body.slots.y) * FIXED));
}
