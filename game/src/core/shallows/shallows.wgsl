// Water that lies on the ground, as the height its face stands at over each cell of a chart of
// the ground and the flow through each cell's two low faces, averaged over its depth, in metres
// and seconds. Staggered and momentum-conserving (Stelling & Duinmeijer 2003): a lake at rest is
// an exact solution, a bore loses what a bore loses, and a front wets and dries ground without a
// special case. The pressure beyond the water's weight is carried as its value at the bed, with
// a quadratic profile over the depth, which gives a wave the speed its length has in water this
// deep (the dispersion of Serre, Green and Naghdi); it is found by sweeps, each step's from the
// last step's. Where a face rises faster than a wave can carry it the front breaks: there the
// pressure is the weight alone and the front a bore (Smit, Zijlema & Stelling 2013).
//
// What the ground is comes from the module `ground`, which whoever owns the ground provides,
// with a chart whose first axis is `x` and second `y`, and heights measured square to it:
//   ground_bed(at)               how high the ground stands at a place on the chart
//   ground_weight(height)        the pull toward the ground at a height, per unit mass
//   ground_turning(height, flow) what the frame's turning and the chart's curving do to a flow
//                                (along x, along y, up), as an acceleration in the same axes
//   ground_stretch(height)       how long a cell's two sides are at a height, over what they
//                                are on the chart
//   ground_held(height)          the water a unit of the chart holds under a height
//   ground_raised(held)          the height under which a unit of the chart holds that much
#import ground::{ground_bed, ground_weight, ground_turning, ground_stretch, ground_held, ground_raised}
#import shallows_chart::{shallows, Cell, Pressing, WALL, cell_count, cell_of, slot, stood}

@group(0) @binding(1) var<storage, read_write> bed: array<f32>;
@group(0) @binding(2) var<storage, read_write> cells: array<Cell>;
// the flows once weight, carrying, turning and drag have had a step at them, and whether each
// of the two faces is wet
@group(0) @binding(3) var<storage, read_write> pushed: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> pressing: array<Pressing>;
// the share of what its faces would take out of a cell that the cell has to give, and the depth
// it had
@group(0) @binding(5) var<storage, read_write> giving: array<vec2<f32>>;
// where on the chart, and how many cubic metres
@group(0) @binding(6) var<storage, read> poured: array<vec4<f32>>;
// the cubic metres each row of the chart holds
@group(0) @binding(7) var<storage, read_write> rows: array<f32>;

const DRY: f32 = 1e-5;
// water shallower than this has no pressure beyond its weight
const FILM: f32 = 0.05;
const BREAKS_AT: f32 = 0.6;
const BREAKS_ON_AT: f32 = 0.3;
const OVER: f32 = 1.0;
// water a cell is owed beyond its face that a face could tell, which is given it though
// nothing passes its faces
const OWED_SHOWS: f32 = 1e-4;

fn middle(c: vec2<i32>) -> vec2<f32> {
    return shallows.low + (vec2<f32>(c) + vec2(0.5)) * shallows.cell;
}

fn depth(s: i32) -> f32 {
    return max(cells[s].face - bed[s], 0.0);
}

fn step_along(axis: u32) -> vec2<i32> {
    return vec2<i32>(i32(axis == 0u), i32(axis == 1u));
}

/// The depth of the water passing the face between two cells: that of the cell it comes from,
/// or, where nothing moves, what stands over the higher of the two beds.
fn depth_passing(low: i32, high: i32, flow: f32) -> f32 {
    if (low == WALL || high == WALL) {
        return 0.0;
    }
    if (flow > 0.0) {
        return depth(low);
    }
    if (flow < 0.0) {
        return depth(high);
    }
    return max(max(cells[low].face, cells[high].face) - max(bed[low], bed[high]), 0.0);
}

/// The flow through a cell's low face along an axis, none off the chart.
fn flow_at(c: vec2<i32>, axis: u32) -> f32 {
    let s = slot(c);
    if (s == WALL) {
        return 0.0;
    }
    return cells[s].flow[axis];
}

/// The water passing a cell's low face along an axis, per unit of the chart across it.
fn passing(c: vec2<i32>, axis: u32) -> f32 {
    let high = slot(c);
    let low = slot(c - step_along(axis));
    if (high == WALL || low == WALL) {
        return 0.0;
    }
    let flow = cells[high].flow[axis];
    let deep = depth_passing(low, high, flow);
    let at = 0.5 * (bed[low] + bed[high]) + 0.5 * deep;
    return flow * deep * ground_stretch(at)[1u - axis];
}

/// How the pressure at the bed of the cell below a face and of the cell above it slows the flow
/// through the face, per second: its weight on the face by the quadratic profile, and its push
/// along a bed that slopes.
fn leans(low: i32, high: i32, axis: u32) -> vec2<f32> {
    if (low == WALL || high == WALL) {
        return vec2(0.0);
    }
    let between = 0.5 * (depth(low) + depth(high));
    if (between <= DRY) {
        return vec2(0.0);
    }
    let at = 0.5 * (bed[low] + bed[high]) + 0.5 * between;
    let long = shallows.cell[axis] * ground_stretch(at)[axis] * between;
    let slope = 0.5 * (bed[high] - bed[low]);
    return vec2(-2.0 / 3.0 * depth(low) + slope, 2.0 / 3.0 * depth(high) + slope) / long;
}

fn is_deep(s: i32) -> bool {
    return s != WALL && depth(s) > FILM && pressing[s].breaking == 0.0;
}

fn pressure_at(s: i32) -> f32 {
    if (s == WALL) {
        return 0.0;
    }
    return pressing[s].at_bed;
}

/// The ground under every cell. Water over ground that is raised or dug keeps its volume, and
/// water over ground left alone is left alone, to the last bit.
@compute @workgroup_size(64)
fn lay(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let ground = ground_bed(middle(cell_of(id.x)));
    if (shallows.restoring == 1u) {
        bed[id.x] = ground;
        cells[id.x].face = max(cells[id.x].face, ground);
        return;
    }
    if (ground == bed[id.x]) {
        return;
    }
    let held = max(ground_held(cells[id.x].face) - ground_held(bed[id.x]), 0.0) + pressing[id.x].owed;
    let now = stood(held, ground);
    bed[id.x] = ground;
    cells[id.x].face = now.face;
    pressing[id.x].owed = now.owed;
}

/// Stand the water to a level that may tilt along the chart, at rest.
@compute @workgroup_size(64)
fn stand(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let level = shallows.standing.x + dot(shallows.standing.yz, middle(cell_of(id.x)));
    cells[id.x] = Cell(max(bed[id.x], level), 0.0, vec2(0.0));
    pressing[id.x] = Pressing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
}

/// Pour what is to be poured, each into the cell it lands in.
@compute @workgroup_size(1)
fn pour() {
    for (var k = 0u; k < shallows.pouring; k++) {
        let c = vec2<i32>(floor((poured[k].xy - shallows.low) / shallows.cell));
        let s = slot(c);
        if (s == WALL) {
            continue;
        }
        let held = ground_held(cells[s].face) - ground_held(bed[s]) + pressing[s].owed;
        let more = poured[k].z / (shallows.cell.x * shallows.cell.y);
        let now = stood(max(held + more, 0.0), bed[s]);
        cells[s].face = now.face;
        pressing[s].owed = now.owed;
    }
}

/// The water a face stands for over a step, in depth: what of it stays, what comes to it, and
/// the flow that comes with that.
struct Carried {
    kept: f32,
    came: f32,
    brought: f32,
}

/// One side of the water a face stands for lets this much into it a second, which comes with
/// the flow of the face on that side, or lets it out.
fn carry(water: ptr<function, Carried>, into: f32, theirs: f32) {
    if (into > 0.0) {
        (*water).came += shallows.dt * into;
        (*water).brought += shallows.dt * into * theirs;
    } else {
        (*water).kept += shallows.dt * into;
    }
}

/// What a step of the water's weight, of what the flow carries of itself, of the frame's
/// turning and of the bed's drag makes of the flow through one low face of a cell.
fn push_face(c: vec2<i32>, axis: u32) -> vec2<f32> {
    let other = 1u - axis;
    let along = step_along(axis);
    let across = step_along(other);
    let high = slot(c);
    let low = slot(c - along);
    if (low == WALL) {
        return vec2(0.0);
    }
    let flow = cells[high].flow[axis];
    let deep = depth_passing(low, high, flow);
    let between = 0.5 * (depth(low) + depth(high));
    if (deep <= DRY || between <= DRY) {
        return vec2(0.0);
    }
    let surface = 0.5 * (cells[low].face + cells[high].face);
    let at = 0.5 * (bed[low] + bed[high]) + 0.5 * between;
    let stretch = ground_stretch(at);
    let long = shallows.cell[axis] * stretch[axis];
    let wide = shallows.cell[other] * stretch[other];

    // what the flow carries of itself: the water the face stands for, half of each of the two
    // cells, keeps what does not leave it through the middles of the cells and the two corners
    // the face ends in, and takes up what comes in with the flow it comes with
    var water = Carried(between, 0.0, 0.0);
    let through_low = 0.5 * (passing(c - along, axis) + passing(c, axis)) / long;
    let through_high = 0.5 * (passing(c, axis) + passing(c + along, axis)) / long;
    let by_low = 0.5 * (passing(c - along, other) + passing(c, other)) / wide;
    let by_high = 0.5 * (passing(c - along + across, other) + passing(c + across, other)) / wide;
    carry(&water, through_low, flow_at(c - along, axis));
    carry(&water, -through_high, flow_at(c + along, axis));
    carry(&water, by_low, flow_at(c - across, axis));
    carry(&water, -by_high, flow_at(c + across, axis));
    let kept = max(water.kept, 0.0);
    var carried = flow;
    if (kept + water.came > DRY) {
        carried = (kept * flow + water.brought) / (kept + water.came);
    }

    // the flow across the face, and the climbing at it, as the cells round it have them
    let sideways = 0.25 * (flow_at(c, other) + flow_at(c + across, other)
        + flow_at(c - along, other) + flow_at(c - along + across, other));
    let climbing = 0.25 * (cells[low].climbing + cells[high].climbing);
    var moving = vec3(0.0, 0.0, climbing);
    moving[axis] = flow;
    moving[other] = sideways;
    let turning = ground_turning(at, moving);
    let weight = ground_weight(surface) - turning.z;
    var next = carried - shallows.dt * weight * (cells[high].face - cells[low].face) / long
        + shallows.dt * turning[axis];
    next /= 1.0 + shallows.dt * shallows.bed_friction * length(vec2(flow, sideways)) / max(deep, 1e-3);
    return vec2(next, 1.0);
}

@compute @workgroup_size(64)
fn push(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let c = cell_of(id.x);
    let x = push_face(c, 0u);
    let y = push_face(c, 1u);
    pushed[id.x] = vec4(x.x, y.x, x.y, y.y);
}

fn pushed_at(c: vec2<i32>, axis: u32) -> f32 {
    let s = slot(c);
    if (s == WALL) {
        return 0.0;
    }
    return pushed[s][axis];
}

/// The speed the bed's slope gives the water at the bed, by the flows through a cell's faces.
fn climbing_at_bed(c: vec2<i32>, once_pushed: bool) -> f32 {
    var out = 0.0;
    for (var axis = 0u; axis < 2u; axis++) {
        let along = step_along(axis);
        let low = slot(c - along);
        let high = slot(c + along);
        if (low == WALL || high == WALL) {
            continue;
        }
        let slope = 0.5 * (bed[high] - bed[low]) / shallows.cell[axis];
        var flow = 0.5 * (flow_at(c, axis) + flow_at(c + along, axis));
        if (once_pushed) {
            flow = 0.5 * (pushed_at(c, axis) + pushed_at(c + along, axis));
        }
        out += flow * slope;
    }
    return out;
}

/// The system of the pressure at the bed: what the pushed flows let into a cell beyond what
/// its face's climbing takes up.
@compute @workgroup_size(64)
fn want(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let c = cell_of(id.x);
    let deep = depth(s);
    var mine = pressing[s];
    let at_bed = climbing_at_bed(c, true);
    mine.climbed = cells[s].climbing - (at_bed - climbing_at_bed(c, false));
    if (deep <= FILM || mine.breaking != 0.0) {
        mine.at_bed = 0.0;
        mine.wanted = 0.0;
        mine.own = 1.0;
        pressing[s] = mine;
        return;
    }
    let stretch = ground_stretch(bed[s] + 0.5 * deep);
    var let_in = (mine.climbed - at_bed) / deep;
    var own = 2.0 * shallows.dt / (deep * deep);
    for (var axis = 0u; axis < 2u; axis++) {
        let along = step_along(axis);
        let long = shallows.cell[axis] * stretch[axis];
        let_in += (pushed_at(c + along, axis) - pushed_at(c, axis)) / long;
        let below = leans(slot(c - along), s, axis);
        let above = leans(s, slot(c + along), axis);
        own += shallows.dt * (below.y - above.x) / long;
    }
    mine.wanted = -let_in;
    mine.own = own;
    pressing[s] = mine;
}

fn sweep(thread: u32, colour: i32) {
    if (thread >= cell_count()) {
        return;
    }
    let s = i32(thread);
    let c = cell_of(thread);
    if (((c.x + c.y) & 1) != colour || !is_deep(s)) {
        return;
    }
    let deep = depth(s);
    let stretch = ground_stretch(bed[s] + 0.5 * deep);
    var sum = pressing[s].wanted;
    for (var axis = 0u; axis < 2u; axis++) {
        let along = step_along(axis);
        let long = shallows.cell[axis] * stretch[axis];
        let low = slot(c - along);
        let high = slot(c + along);
        let below = leans(low, s, axis);
        let above = leans(s, high, axis);
        if (is_deep(high)) {
            sum += shallows.dt * above.y / long * pressure_at(high);
        }
        if (is_deep(low)) {
            sum -= shallows.dt * below.x / long * pressure_at(low);
        }
    }
    pressing[s].at_bed = mix(pressing[s].at_bed, sum / pressing[s].own, OVER);
}

@compute @workgroup_size(64)
fn sweep_red(@builtin(global_invocation_id) id: vec3<u32>) {
    sweep(id.x, 0);
}

@compute @workgroup_size(64)
fn sweep_black(@builtin(global_invocation_id) id: vec3<u32>) {
    sweep(id.x, 1);
}

/// The flows and the face's climbing give way to the pressure.
@compute @workgroup_size(64)
fn yield_to(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let c = cell_of(id.x);
    var flow = pushed[s].xy;
    for (var axis = 0u; axis < 2u; axis++) {
        let low = slot(c - step_along(axis));
        if (pushed[s][2u + axis] == 0.0) {
            flow[axis] = 0.0;
            continue;
        }
        let lean = leans(low, s, axis);
        var slowed = 0.0;
        if (is_deep(low)) {
            slowed += lean.x * pressure_at(low);
        }
        if (is_deep(s)) {
            slowed += lean.y * pressure_at(s);
        }
        flow[axis] -= shallows.dt * slowed;
    }
    var climbing = 0.0;
    if (is_deep(s)) {
        climbing = pressing[s].climbed + 2.0 * shallows.dt * pressing[s].at_bed / depth(s);
    }
    cells[s].flow = flow;
    cells[s].climbing = climbing;
}

/// The water each row of the chart holds, for whoever reads the rows to sum them where a sum
/// of so many keeps its figures.
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= shallows.size.y) {
        return;
    }
    var held = 0.0;
    for (var x = 0u; x < shallows.size.x; x++) {
        let s = id.x * shallows.size.x + x;
        held += max(ground_held(cells[s].face) - ground_held(bed[s]), 0.0);
    }
    rows[id.x] = held * shallows.cell.x * shallows.cell.y;
}

/// No cell gives more than it holds: what its faces would take out of it is scaled to that.
@compute @workgroup_size(64)
fn share(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let c = cell_of(id.x);
    var leaving = 0.0;
    for (var axis = 0u; axis < 2u; axis++) {
        let along = step_along(axis);
        leaving += (max(passing(c + along, axis), 0.0) + max(-passing(c, axis), 0.0)) / shallows.cell[axis];
    }
    let held = max(ground_held(cells[s].face) - ground_held(bed[s]), 0.0);
    giving[s] = vec2(min(1.0, held / max(shallows.dt * leaving, 1e-30)), depth(s));
}

/// The water a cell's low face passes once the cell it leaves has said what it can give. The
/// depths are those `share` kept, the faces being on their way to where the water takes them.
fn passed(c: vec2<i32>, axis: u32) -> f32 {
    let high = slot(c);
    let low = slot(c - step_along(axis));
    if (high == WALL || low == WALL) {
        return 0.0;
    }
    let flow = cells[high].flow[axis];
    if (flow == 0.0) {
        return 0.0;
    }
    let giver = select(high, low, flow > 0.0);
    let deep = giving[giver].y;
    let at = 0.5 * (bed[low] + bed[high]) + 0.5 * deep;
    return flow * deep * ground_stretch(at)[1u - axis] * giving[giver].x;
}

/// The faces rise and fall by what the flows pass, and a front that rises faster than a wave
/// can carry it breaks, until it rises slowly again.
@compute @workgroup_size(64)
fn rise(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let s = i32(id.x);
    let c = cell_of(id.x);
    var let_in = 0.0;
    for (var axis = 0u; axis < 2u; axis++) {
        let_in += (passed(c, axis) - passed(c + step_along(axis), axis)) / shallows.cell[axis];
    }
    let was = cells[s].face;
    var face = was;
    if (let_in != 0.0 || abs(pressing[s].owed) > OWED_SHOWS) {
        let held = ground_held(was) - ground_held(bed[s]) + pressing[s].owed + shallows.dt * let_in;
        let now = stood(max(held, 0.0), bed[s]);
        face = now.face;
        pressing[s].owed = now.owed;
    }
    cells[s].face = face;
    let rising = (face - was) / shallows.dt;
    let wave = sqrt(ground_weight(face) * max(face - bed[s], DRY));
    let breaks = rising > BREAKS_AT * wave || (pressing[s].breaking != 0.0 && rising > BREAKS_ON_AT * wave);
    pressing[s].breaking = f32(breaks);
}
