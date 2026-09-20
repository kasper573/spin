// Water in flight that comes down on the ground, or into the water lying on it, becomes water
// lying on the ground: the particle is taken out of the water in flight, and what it held and
// how it moved are taken into the cell it came down in. A particle has come down when it is
// under the face of the water lying there, or lies on the ground and is not leaving it.
#import ground::{ground_charted, ground_flow, ground_held, ground_raised}
#import shallows_chart::{shallows, Cell, Pressing, WALL, cell_count, cell_of, slot, stood}

@group(0) @binding(1) var<storage, read> bed: array<f32>;
@group(0) @binding(2) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(3) var<storage, read_write> position: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> velocity: array<vec4<f32>>;
// four words a cell: the particles that came down in it since it last took them in, and their
// flow along the chart's axes and up, summed in fixed point
@group(0) @binding(5) var<storage, read_write> landed: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> pressing: array<Pressing>;

const FIXED: f32 = 256.0;

fn depth(s: i32) -> f32 {
    return max(cells[s].face - bed[s], 0.0);
}

fn fixed(v: f32) -> u32 {
    return bitcast<u32>(i32(round(v * FIXED)));
}

/// What came down in a cell: how deep it would stand spread over the cell, and its mean flow.
struct Landed {
    deep: f32,
    flow: vec3<f32>,
}

fn landed_in(s: i32) -> Landed {
    if (s == WALL) {
        return Landed(0.0, vec3(0.0));
    }
    let n = f32(atomicLoad(&landed[4 * s]));
    if (n == 0.0) {
        return Landed(0.0, vec3(0.0));
    }
    let sum = vec3(
        f32(bitcast<i32>(atomicLoad(&landed[4 * s + 1]))),
        f32(bitcast<i32>(atomicLoad(&landed[4 * s + 2]))),
        f32(bitcast<i32>(atomicLoad(&landed[4 * s + 3]))),
    );
    return Landed(n * shallows.particle_volume / (shallows.cell.x * shallows.cell.y), sum / (FIXED * n));
}

@compute @workgroup_size(64)
fn absorb(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= shallows.particles || position[i].w < 0.0) {
        return;
    }
    let at = ground_charted(position[i].xyz);
    let s = slot(vec2<i32>(floor((at.xy - shallows.low) / shallows.cell)));
    if (s == WALL) {
        return;
    }
    let flow = ground_flow(position[i].xyz, velocity[i].xyz);
    let under = cells[s].face > bed[s] && at.z < cells[s].face;
    let down = at.z - bed[s] < shallows.landing && flow.z <= 0.0;
    if (!under && !down) {
        return;
    }
    position[i].w = -1.0;
    atomicAdd(&landed[4 * s], 1u);
    atomicAdd(&landed[4 * s + 1], fixed(flow.x));
    atomicAdd(&landed[4 * s + 2], fixed(flow.y));
    atomicAdd(&landed[4 * s + 3], fixed(flow.z));
}

/// The flow through a cell's low faces once what came down either side of each has mixed its
/// motion into the water the face stands for, half of each of the two cells it lies between.
@compute @workgroup_size(64)
fn take_in_flow(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let c = cell_of(id.x);
    let here = landed_in(s);
    var flow = cells[s].flow;
    for (var axis = 0u; axis < 2u; axis++) {
        let low = slot(c - vec2<i32>(i32(axis == 0u), i32(axis == 1u)));
        if (low == WALL) {
            continue;
        }
        let there = landed_in(low);
        let came = here.deep + there.deep;
        if (came == 0.0) {
            continue;
        }
        let lay = depth(s) + depth(low);
        flow[axis] = (lay * flow[axis] + here.deep * here.flow[axis] + there.deep * there.flow[axis]) / (lay + came);
    }
    cells[s].flow = flow;
}

/// The water a cell takes in of what came down in it, and how it climbs for it.
@compute @workgroup_size(64)
fn take_in_water(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let came = landed_in(s);
    if (came.deep == 0.0) {
        return;
    }
    for (var word = 0; word < 4; word++) {
        atomicStore(&landed[4 * s + word], 0u);
    }
    let lay = depth(s);
    let held = max(ground_held(cells[s].face) - ground_held(bed[s]), 0.0) + pressing[s].owed + came.deep;
    let now = stood(held, bed[s]);
    cells[s].face = now.face;
    pressing[s].owed = now.owed;
    cells[s].climbing = (lay * cells[s].climbing + came.deep * came.flow.z) / (lay + came.deep);
}
