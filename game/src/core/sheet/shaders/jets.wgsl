// Water that leaves the sheet through its floor, and what becomes of it in the air. A drain is
// an opening in the floor that the water over it runs out of as fast as its head over what is
// on the far side drives it, which is Torricelli's speed; what runs out in a frame leaves the
// drain's outlet as a hoop of water, a ring of points round the outlet's rim that fly free
// from there, each straight and even among the stars as free water does. Hoop after hoop they
// are a jet, which is drawn as the one unbroken tube they are the ribs of. A hoop that comes
// down on the sheet, or on the floor under it, gives the sheet its water back and what of its
// run lies along the floor; what it came down with spreads it from where it landed.
#import lying_sheet::{SurfaceVertex, sheet, bed, after, clock, held, slot, corner, narrowing, water_of, depth_of, run_of}

const DRAINS: u32 = 2u;
// how many points round a hoop's rim, and how many hoops there may be in the air
const RIM: u32 = 16u;
// the speed (m/s) under which a jet goes into a pool without taking air with it
const SMOOTH_ENTRY: f32 = 1.0;
const HOOPS: u32 = 512u;
const PI: f32 = 3.14159265;

struct Drain {
    // the block of cells it lies among: the first of them, and how many round and along
    first: vec2<u32>,
    cells: vec2<u32>,
    // its middle, in cells from the block's first, and how far from its middle it is open, in
    // metres: nothing, where there is no drain
    middle: vec2<f32>,
    reach: f32,
    // the drain whose water is on the far side of this one and presses back, or none of them
    far: u32,
    // where its water comes out, in the frame the surface is drawn in: the outlet's middle,
    // the way out of it, and two ways across it
    out_middle: vec4<f32>,
    out_way: vec4<f32>,
    out_across: vec4<f32>,
    out_up: vec4<f32>,
}

struct Hoop {
    rim: array<vec4<f32>, RIM>,
    run: array<vec4<f32>, RIM>,
    // cubic metres
    water: f32,
    alive: u32,
    drain: u32,
    // which of all the hoops there have been it is: one and the next are joined in a tube
    born: u32,
    // how far it has flown, in metres
    flown: f32,
}

struct Flights {
    // how many hoops there have been
    born: u32,
    // the mean head over each drain, and the speed its water last ran out at
    head: array<f32, DRAINS>,
    speed: array<f32, DRAINS>,
}

@group(0) @binding(14) var<uniform> drains: array<Drain, DRAINS>;
// what each cell has lost down a drain since a hoop was last made of it: cubic metres, and
// that times its run round the ring and along it
@group(0) @binding(15) var<storage, read_write> drained: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> hoops: array<Hoop, HOOPS>;
@group(0) @binding(17) var<storage, read_write> flights: Flights;
// the jets' surface, as the water's shader draws one: a vertex for every point of every hoop,
// the corners of the triangles between one hoop and the next, and how many of those there are,
// second of four counts
@group(0) @binding(18) var<storage, read_write> jet_vertices: array<SurfaceVertex>;
@group(0) @binding(19) var<storage, read_write> jet_indices: array<u32>;
@group(0) @binding(20) var<storage, read_write> jet_counters: array<atomic<u32>>;

fn middle_floor(at: vec2<u32>) -> f32 {
    return 0.25 * (corner(at) + corner(at + vec2(1u, 0u)) + corner(at + vec2(0u, 1u))
        + corner(at + vec2(1u, 1u)));
}

/// The cell of a drain's block, wrapped round a ring whose cells close on themselves.
fn drain_cell(drain: Drain, within: vec2<u32>) -> vec2<u32> {
    var at = drain.first + within;
    if (sheet.closed == 1u) {
        at.x = at.x % sheet.used.x;
    }
    return at;
}

fn in_drain(drain: Drain, within: vec2<u32>) -> bool {
    let from_middle = (vec2<f32>(within) + 0.5 - drain.middle) * sheet.cell;
    return all(within < drain.cells) && length(from_middle) <= drain.reach;
}

/// What the water over a cell would run out through the floor with, as half the square of its
/// speed: the ring's potential from its surface down to the floor, turned as fast as the ring
/// and its own run round it turn it.
fn head_over(at: vec2<u32>) -> f32 {
    let state = after[slot(at)];
    let floor = middle_floor(at);
    let depth = depth_of(state.x, floor);
    let from_axis = sheet.radius - floor - 0.5 * depth;
    let turn = sheet.spin + run_of(state.x, state.y) / from_axis;
    return turn * turn * depth * from_axis;
}

/// The mean head over each drain.
@compute @workgroup_size(1)
fn press() {
    for (var d = 0u; d < DRAINS; d++) {
        let drain = drains[d];
        var head = 0.0;
        var cells = 0.0;
        for (var j = 0u; j < drain.cells.y; j++) {
            for (var i = 0u; i < drain.cells.x; i++) {
                if (drain.reach > 0.0 && in_drain(drain, vec2(i, j))) {
                    head += head_over(drain_cell(drain, vec2(i, j)));
                    cells += 1.0;
                }
            }
        }
        flights.head[d] = head / max(cells, 1.0);
    }
}

/// A step's water runs out of each drain, no more from a cell than it holds.
@compute @workgroup_size(8, 8)
fn sink(@builtin(global_invocation_id) id: vec3<u32>) {
    for (var d = 0u; d < DRAINS; d++) {
        let drain = drains[d];
        if (drain.reach <= 0.0 || !in_drain(drain, id.xy)) {
            continue;
        }
        var back = 0.0;
        if (drain.far < DRAINS) {
            back = flights.head[drain.far];
        }
        let at = drain_cell(drain, id.xy);
        let c = slot(at);
        let state = after[c];
        let floor = middle_floor(at);
        let depth = depth_of(state.x, floor);
        let speed = sqrt(2.0 * max(head_over(at) - back, 0.0));
        let left = max(depth - speed * clock.dt, 0.0);
        let taken = state.x - water_of(left, floor);
        if (taken <= 0.0) {
            continue;
        }
        let share = taken / state.x;
        after[c] = vec4(state.x - taken, state.yzw * (1.0 - share));
        let volume = taken * sheet.cell.x * sheet.cell.y;
        drained[c] += vec4(volume, volume * run_of(state.x, state.y), volume * run_of(state.x, state.z), 0.0);
    }
}

/// What ran out of each drain this frame leaves its outlet as a hoop, as fast as the head
/// there was drove it.
@compute @workgroup_size(1)
fn bear() {
    for (var d = 0u; d < DRAINS; d++) {
        let drain = drains[d];
        if (drain.reach <= 0.0) {
            continue;
        }
        var back = 0.0;
        if (drain.far < DRAINS) {
            back = flights.head[drain.far];
        }
        flights.speed[d] = sqrt(2.0 * max(flights.head[d] - back, 0.0));
        var water = 0.0;
        for (var j = 0u; j < drain.cells.y; j++) {
            for (var i = 0u; i < drain.cells.x; i++) {
                if (in_drain(drain, vec2(i, j))) {
                    let c = slot(drain_cell(drain, vec2(i, j)));
                    water += drained[c].x;
                    drained[c] = vec4(0.0);
                }
            }
        }
        if (water <= 0.0) {
            continue;
        }
        let h = flights.born % HOOPS;
        // a hoop still in the air where the next is to be made keeps its water: both are one
        if (hoops[h].alive == 1u) {
            water += hoops[h].water;
        }
        for (var k = 0u; k < RIM; k++) {
            let angle = 2.0 * PI * f32(k) / f32(RIM);
            let across = drain.out_across.xyz * cos(angle) + drain.out_up.xyz * sin(angle);
            hoops[h].rim[k] = vec4(drain.out_middle.xyz + across * drain.reach, 0.0);
            hoops[h].run[k] = vec4(drain.out_way.xyz * flights.speed[d], 0.0);
        }
        hoops[h].water = water;
        hoops[h].alive = 1u;
        hoops[h].drain = d;
        hoops[h].born = flights.born;
        hoops[h].flown = 0.0;
        flights.born += 1u;
    }
}

/// sin(t) and 1 - cos(t), and sin(t) - t cos(t): a small turn's are worked out as their series,
/// since the builtins are held only to an absolute error, which is all there is of them.
fn sines(t: f32) -> vec2<f32> {
    if (abs(t) < 0.1) {
        let t2 = t * t;
        return vec2(
            t * (1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0))),
            0.5 * t2 * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))),
        );
    }
    let half = sin(0.5 * t);
    return vec2(sin(t), 2.0 * half * half);
}

fn sine_lag(t: f32) -> f32 {
    if (abs(t) < 0.1) {
        let t2 = t * t;
        return t * t2 * (1.0 / 3.0 - t2 * (1.0 / 30.0 - t2 / 840.0));
    }
    return sin(t) - t * cos(t);
}

fn turned(p: vec3<f32>, s: f32, c: f32) -> vec3<f32> {
    return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

/// The frame's seconds of free flight for every point of every hoop: straight and even among
/// the stars, and put back into the frame where that has turned to by then. The frame's origin
/// is carried round the axis at the rim's speed, which the motion is taken apart from, so that
/// on a big ring nothing small is the difference of two large.
@compute @workgroup_size(64)
fn fly(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= HOOPS || hoops[id.x].alive == 0u) {
        return;
    }
    let dt = sheet.advance;
    let w = sheet.spin;
    let sine = sines(w * dt);
    let s = sine.x;
    let sagged = sine.y;
    let c = 1.0 - sagged;
    let origin = sheet.radius * vec3(w * dt * s - sagged, 0.0, sine_lag(w * dt));
    hoops[id.x].flown += length(hoops[id.x].run[0].xyz) * dt;
    for (var k = 0u; k < RIM; k++) {
        let p = hoops[id.x].rim[k].xyz;
        let among_stars = hoops[id.x].run[k].xyz + vec3(w * p.z, 0.0, -w * p.x);
        let landed = turned(p + among_stars * dt, -s, c) + origin;
        let carried = turned(among_stars, -s, c) + sheet.radius * vec3(w * s, 0.0, w * sagged);
        hoops[id.x].rim[k] = vec4(landed, 0.0);
        hoops[id.x].run[k] = vec4(carried - vec3(w * landed.z, 0.0, -w * landed.x), 0.0);
    }
}

/// Where a point of the frame the surface is drawn in lies on the sheet: in cells from its
/// first corner, round the ring and along it, and how high over the glass.
fn placed(p: vec3<f32>) -> vec3<f32> {
    let out = vec2(sheet.radius + p.x, p.z);
    let arc = sheet.radius * atan2(out.y, out.x);
    // the glass's radius less how far from the axis, worked out without taking one from the other
    let height = -(2.0 * sheet.radius * p.x + p.x * p.x + p.z * p.z) / (length(out) + sheet.radius);
    return vec3((arc - sheet.origin.x) / sheet.cell.x, (p.y - sheet.origin.y) / sheet.cell.y, height);
}

fn cell_at(place: vec2<f32>) -> vec2<u32> {
    var i = i32(floor(place.x));
    let count = i32(sheet.used.x);
    if (sheet.closed == 1u) {
        i = ((i % count) + count) % count;
    }
    return vec2(
        u32(clamp(i, 0, count - 1)),
        u32(clamp(i32(floor(place.y)), 0, i32(sheet.used.y) - 1)),
    );
}

/// How much air a jet plunging into a pool takes under with it, by its water: as Bin measured
/// of plunging jets, more the faster and the longer the jet, and none below the speed at which
/// the pool's surface closes smoothly round it.
fn taken_in(speed: f32, width: f32, flown: f32) -> f32 {
    if (speed < SMOOTH_ENTRY) {
        return 0.0;
    }
    let gravity = sheet.spin * sheet.spin * sheet.radius;
    let froude = speed * speed / max(gravity * width, 1e-9);
    return 0.04 * pow(froude, 0.28) * pow(flown / width, 0.4);
}

/// The hoops that have come down give the sheet their water, one after another, so that no
/// two of them are ever at one cell at once.
@compute @workgroup_size(1)
fn land() {
    for (var h = 0u; h < HOOPS; h++) {
        if (hoops[h].alive == 0u) {
            continue;
        }
        var middle = vec3(0.0);
        var run = vec3(0.0);
        for (var k = 0u; k < RIM; k++) {
            middle += hoops[h].rim[k].xyz / f32(RIM);
            run += hoops[h].run[k].xyz / f32(RIM);
        }
        let place = placed(middle);
        let under = cell_at(place.xy);
        let floor = middle_floor(under);
        // it flies on until the water has closed over all of it, or its middle is at the bed
        var top = place.z;
        for (var k = 0u; k < RIM; k++) {
            top = max(top, placed(hoops[h].rim[k].xyz).z);
        }
        if (place.z > floor && top > floor + depth_of(after[slot(under)].x, floor)) {
            continue;
        }
        // how wide it comes down: as far as its rim lies from its middle over the floor
        var reach = 0.0;
        for (var k = 0u; k < RIM; k++) {
            let from_middle = (placed(hoops[h].rim[k].xyz).xy - place.xy) * sheet.cell;
            reach = max(reach, length(from_middle));
        }
        reach = max(reach, 0.5 * max(sheet.cell.x, sheet.cell.y));
        let cells = vec2<i32>(ceil(vec2(reach) / sheet.cell));
        var shares = 0.0;
        for (var j = -cells.y; j <= cells.y; j++) {
            for (var i = -cells.x; i <= cells.x; i++) {
                let from_middle = vec2<f32>(vec2(i, j)) * sheet.cell;
                shares += select(0.0, 1.0, length(from_middle) <= reach);
            }
        }
        // its run along the floor there: spinward round the ring, and along the axis
        let turn = atan2(middle.z, sheet.radius + middle.x);
        let spinward = vec3(-sin(turn), 0.0, cos(turn));
        let outward = vec3(cos(turn), 0.0, sin(turn));
        let along_floor = vec2(dot(run, spinward), run.y);
        let down = max(dot(run, outward), 0.0);
        let each = hoops[h].water / (shares * sheet.cell.x * sheet.cell.y);
        let air = each * taken_in(length(run), 2.0 * reach, hoops[h].flown);
        for (var j = -cells.y; j <= cells.y; j++) {
            for (var i = -cells.x; i <= cells.x; i++) {
                let from_middle = vec2<f32>(vec2(i, j)) * sheet.cell;
                if (length(from_middle) > reach) {
                    continue;
                }
                let c = slot(cell_at(place.xy + vec2<f32>(vec2(i, j))));
                let landing = along_floor + down * from_middle / reach;
                after[c] += vec4(each, each * landing, air);
            }
        }
        hoops[h].alive = 0u;
        hoops[h].water = 0.0;
    }
}

/// The water there is in the air, kept after the water in each row of the sheet.
@compute @workgroup_size(1)
fn weigh() {
    var water = 0.0;
    for (var h = 0u; h < HOOPS; h++) {
        if (hoops[h].alive == 1u) {
            water += hoops[h].water;
        }
    }
    for (var d = 0u; d < DRAINS; d++) {
        let drain = drains[d];
        for (var j = 0u; j < drain.cells.y; j++) {
            for (var i = 0u; i < drain.cells.x; i++) {
                if (drain.reach > 0.0 && in_drain(drain, vec2(i, j))) {
                    water += drained[slot(drain_cell(drain, vec2(i, j)))].x;
                }
            }
        }
    }
    held[sheet.stored.y] = water;
}

/// The tube a jet's hoops are the ribs of: a vertex at every point of every hoop, facing out
/// from the hoop's middle across its run and told how thick the water is there, which is how
/// far across the hoop is; and the triangles between each hoop and the one born after it.
@compute @workgroup_size(64)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let h = id.x;
    if (h >= HOOPS || hoops[h].alive == 0u) {
        return;
    }
    var middle = vec3(0.0);
    for (var k = 0u; k < RIM; k++) {
        middle += hoops[h].rim[k].xyz / f32(RIM);
    }
    var across = 0.0;
    for (var k = 0u; k < RIM; k++) {
        across += 2.0 * length(hoops[h].rim[k].xyz - middle) / f32(RIM);
    }
    for (var k = 0u; k < RIM; k++) {
        let p = hoops[h].rim[k].xyz;
        let run = hoops[h].run[k].xyz;
        let out = p - middle;
        let along = run * dot(out, run) / max(dot(run, run), 1e-12);
        var vertex: SurfaceVertex;
        vertex.position = vec4(p, 0.0);
        vertex.normal = vec4(normalize(out - along), across);
        vertex.velocity = vec4(run, 0.0);
        jet_vertices[h * RIM + k] = vertex;
    }
    let next = (h + 1u) % HOOPS;
    if (hoops[next].alive == 0u || hoops[next].born != hoops[h].born + 1u
        || hoops[next].drain != hoops[h].drain) {
        return;
    }
    let first = atomicAdd(&jet_counters[1], 6u * RIM);
    for (var k = 0u; k < RIM; k++) {
        let on = (k + 1u) % RIM;
        var corners = array<u32, 6>(
            h * RIM + k, h * RIM + on, next * RIM + k,
            next * RIM + k, h * RIM + on, next * RIM + on,
        );
        for (var c = 0u; c < 6u; c++) {
            jet_indices[first + 6u * k + c] = corners[c];
        }
    }
}
