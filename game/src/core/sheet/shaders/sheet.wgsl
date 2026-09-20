// Water lying on the floor of a spun ring as a sheet: how much of it stands over each cell of
// the floor, and how it runs round the ring and along its axis. It is solved as shallow water
// is, in the ring's own turning frame, by the central-upwind scheme of Kurganov and Petrova
// (2007): the level is rebuilt across each cell from its neighbours', what crosses each face is
// found from the two states meeting there and the fastest waves either can send, and no cell
// is let give more water than it holds. The floor's heights stand at the cells' corners, so a
// face's floor is the same from both sides of it and a level lake lies still exactly.
//
// The ring's weight is no constant: what a column presses with is its share of the ring's
// potential, turned as fast as the ring and the water's own run round it together turn, and
// the floor's push on the water is what a still column's would differ by from one side of the
// cell to the other, which is what leaves the lake at rest whatever the floor does under it.
//
// A cell is a cell of the glass: further in, the ring is narrower round, and a cell holds the
// less water for it. What is kept for a cell is the water over it by the glass's area, so that
// what is added and what crosses the faces adds up to the water there is, exactly.

struct Sheet {
    // cells round the ring and along it: as the buffers are laid out, and as are in use
    stored: vec2<u32>,
    used: vec2<u32>,
    // a cell's arc on the glass and its width along the axis, in metres
    cell: vec2<f32>,
    // where the first corner lies from the site the surface is drawn about: metres of arc
    // round the glass, and metres along the axis
    origin: vec2<f32>,
    radius: f32,
    spin: f32,
    // the seconds this frame adds
    advance: f32,
    // whether the cells in use close on themselves round the ring
    closed: u32,
    // water poured this frame: the first cell it falls in, how many round and along, and the
    // water each takes, by the glass's area
    pour_first: vec2<u32>,
    pour_cells: vec2<u32>,
    // how fast it runs over the floor as it lands, round the ring and along it, where the
    // middle of it lands, in cells from the first, how fast what it came down with spreads it
    // from there, and how far from the middle it falls at all, in metres
    pour_run: vec2<f32>,
    pour_middle: vec2<f32>,
    pour_spread: f32,
    pour_reach: f32,
    pour: f32,
    // the first of the cells whose water is reported back
    watch_first: vec2<u32>,
}

// the seconds still to be stepped, the step being taken, how quickly the quickest wave of all
// crosses a cell, per second, and how quickly that of each row does
struct Clock {
    remaining: f32,
    dt: f32,
    quickest: f32,
    rows: array<f32>,
}

struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    velocity: vec4<f32>,
}

@group(0) @binding(0) var<uniform> sheet: Sheet;
// the floor's height over the glass at every corner, a row of them along the axis after another
@group(0) @binding(1) var<storage, read> bed: array<f32>;
// x: the water over a cell by the glass's area; yz: that times its run round and along
@group(0) @binding(2) var<storage, read> before: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> after: array<vec4<f32>>;
// what crosses each cell's far face round the ring, and along it: water, the run across the
// face, and the run along it; and with the first, how long the cell's water lasts what leaves
@group(0) @binding(4) var<storage, read_write> crossing_round: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> crossing_along: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> clock: Clock;
@group(0) @binding(7) var<storage, read_write> vertices: array<SurfaceVertex>;
@group(0) @binding(8) var<storage, read_write> indices: array<u32>;
// vertex count, index count, and two the water's shader has no use for here
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;
// the water in each row, in cubic metres
@group(0) @binding(10) var<storage, read_write> held: array<f32>;
// how many workgroups visit the cells for the step being taken: none when none is taken. The
// kernels that are sent out by it cannot have it bound, so it is kept apart from the clock.
@group(0) @binding(11) var<storage, read_write> threads: vec3<u32>;
// the water round the corners of a block of cells, for whoever wades there: first which cell
// the block begins at and how quickly the sheet's quickest wave crosses a cell, which says how
// many steps the next frames will take, then for each corner how deep the water is, the level it stands at over
// the glass, and its run round the ring and along it
@group(0) @binding(12) var<storage, read_write> watched: array<vec4<f32>>;
const WATCHED: u32 = 16u;

const STEEPEST: f32 = 1.3;
// water thinner than the fourth root of this runs slower than its momentum says
const THIN: f32 = 1e-9;
const COURANT: f32 = 0.2;
const LONGEST_STEP: f32 = 1.0 / 120.0;
const MOST_BEHIND: f32 = 0.05;
// Manning's roughness of earth and short grass, which is measured under the Earth's gravity
// whatever the water weighs here: what a rough floor takes of the water's run is no matter of
// the water's weight. Then water's kinematic viscosity.
const ROUGHNESS: f32 = 0.024;
const MANNINGS_GRAVITY: f32 = 9.80665;
const VISCOSITY: f32 = 1.0e-6;
const NEVER: f32 = 1e30;
// water shallower than this is not drawn
const DRAWN: f32 = 1e-4;
const ROUND: u32 = 0u;
const ALONG: u32 = 1u;

fn slot(at: vec2<u32>) -> u32 {
    return at.y * sheet.stored.x + at.x;
}

fn corner(at: vec2<u32>) -> f32 {
    return bed[at.y * (sheet.stored.x + 1u) + at.x];
}

/// How much narrower round the ring is this far in from the glass.
fn narrowing(height: f32) -> f32 {
    return 1.0 - height / sheet.radius;
}

/// The water over a cell by the glass's area, from how deep it stands on a floor, and back.
fn water_of(depth: f32, floor: f32) -> f32 {
    return depth * narrowing(floor + 0.5 * depth);
}

fn depth_of(water: f32, floor: f32) -> f32 {
    let narrow = narrowing(floor);
    return 2.0 * water / (narrow + sqrt(max(narrow * narrow - 2.0 * water / sheet.radius, 0.0)));
}

/// How fast water runs, from its momentum: as that says, but for water too thin to trust.
fn run_of(water: f32, momentum: f32) -> f32 {
    let w4 = water * water * water * water;
    return sqrt(2.0) * water * momentum / sqrt(w4 + max(w4, THIN));
}

/// What a column this deep presses with across a face, a face across the ring's turning and
/// one across its axis, by the glass's length: the ring's potential summed over the face.
fn push(depth: f32, floor: f32, turn: f32, axis: u32) -> f32 {
    let narrow = select(
        narrowing(floor + depth * 2.0 / 3.0),
        narrowing(floor + 0.5 * depth) * narrowing(floor + 0.5 * depth),
        axis == ALONG,
    );
    return 0.5 * turn * turn * sheet.radius * depth * depth * narrow;
}

/// The cell `by` cells on from one, round the ring or along it, unless the sheet ends first.
struct Beside {
    at: vec2<u32>,
    there: bool,
}

fn beside(at: vec2<u32>, axis: u32, by: i32) -> Beside {
    let count = i32(sheet.used[axis]);
    var to = i32(at[axis]) + by;
    var there = to >= 0 && to < count;
    if (axis == ROUND && sheet.closed == 1u) {
        to = (to + count) % count;
        there = true;
    }
    var out = at;
    out[axis] = u32(max(to, 0));
    return Beside(out, there);
}

/// The floor under a cell: at the middle of its near and far faces across an axis, and at its
/// own middle.
struct Floor {
    near: f32,
    far: f32,
    middle: f32,
}

fn floor_under(at: vec2<u32>, axis: u32) -> Floor {
    let b00 = corner(at);
    let b10 = corner(at + vec2(1u, 0u));
    let b01 = corner(at + vec2(0u, 1u));
    let b11 = corner(at + vec2(1u, 1u));
    let middle = 0.25 * (b00 + b10 + b01 + b11);
    if (axis == ROUND) {
        return Floor(0.5 * (b00 + b01), 0.5 * (b10 + b11), middle);
    }
    return Floor(0.5 * (b00 + b10), 0.5 * (b01 + b11), middle);
}

/// A cell's water as the scheme sees it across an axis: its level, its momentum across the
/// axis's faces and along them, how deep it is, how fast it and the ring turn together, and
/// its weight.
struct Standing {
    level: f32,
    across: f32,
    along: f32,
    depth: f32,
    turn: f32,
    weight: f32,
}

fn standing(at: vec2<u32>, axis: u32) -> Standing {
    let state = before[slot(at)];
    let floor = floor_under(at, axis).middle;
    let depth = depth_of(state.x, floor);
    let middle = sheet.radius - floor - 0.5 * depth;
    let turn = sheet.spin + run_of(state.x, state.y) / middle;
    return Standing(
        floor + depth,
        select(state.y, state.z, axis == ALONG),
        select(state.z, state.y, axis == ALONG),
        depth,
        turn,
        turn * turn * middle,
    );
}

fn gentlest(a: f32, b: f32, c: f32) -> f32 {
    if (a > 0.0 && b > 0.0 && c > 0.0) {
        return min(a, min(b, c));
    }
    if (a < 0.0 && b < 0.0 && c < 0.0) {
        return max(a, max(b, c));
    }
    return 0.0;
}

fn sloped(behind: f32, here: f32, ahead: f32) -> f32 {
    return gentlest(STEEPEST * (here - behind), 0.5 * (ahead - behind), STEEPEST * (ahead - here));
}

/// A cell's water rebuilt at its near and far faces across an axis, and the level it would
/// stand at were it still.
struct Rebuilt {
    level: vec2<f32>,
    across: vec2<f32>,
    along: vec2<f32>,
    floor: vec2<f32>,
    still: f32,
    turn: f32,
    weight: f32,
}

fn rebuilt(at: vec2<u32>, axis: u32) -> Rebuilt {
    let here = standing(at, axis);
    var behind = here;
    var ahead = here;
    let back = beside(at, axis, -1);
    let on = beside(at, axis, 1);
    if (back.there) {
        behind = standing(back.at, axis);
    }
    if (on.there) {
        ahead = standing(on.at, axis);
    }
    let floor = floor_under(at, axis);
    let level = sloped(behind.level, here.level, ahead.level);
    let across = sloped(behind.across, here.across, ahead.across);
    let along = sloped(behind.along, here.along, ahead.along);
    var out: Rebuilt;
    out.level = here.level + vec2(-0.5, 0.5) * level;
    out.across = here.across + vec2(-0.5, 0.5) * across;
    out.along = here.along + vec2(-0.5, 0.5) * along;
    out.floor = vec2(floor.near, floor.far);
    out.still = here.level;
    out.turn = here.turn;
    out.weight = here.weight;
    // no face's water lies under its floor
    if (out.level.y < floor.far) {
        out.level = vec2(2.0 * here.level - floor.far, floor.far);
    }
    if (out.level.x < floor.near) {
        out.level = vec2(floor.near, 2.0 * here.level - floor.near);
    }
    // water that does not cover the cell lies in the low end of it, level
    let fall = abs(floor.far - floor.near);
    if (here.depth < 0.5 * fall) {
        let lying = min(floor.near, floor.far) + sqrt(2.0 * here.depth * fall);
        out.level = select(vec2(floor.near, lying), vec2(lying, floor.far), floor.near <= floor.far);
        out.across = vec2(here.across);
        out.along = vec2(here.along);
        out.still = lying;
    }
    return out;
}

/// What a wall gives back: what would cross the face were the water beyond it the water at it
/// coming the other way. No water crosses, none of its run along the wall is lost, and the wall
/// pushes as hard as the water at it presses and runs at it.
fn walled(cell: Rebuilt, far: bool, axis: u32) -> vec3<f32> {
    let floor = select(cell.floor.x, cell.floor.y, far);
    let depth = max(select(cell.level.x, cell.level.y, far) - floor, 0.0);
    let water = water_of(depth, floor);
    let run = run_of(water, select(cell.across.x, cell.across.y, far));
    let face = select(water, depth, axis == ROUND);
    let quickest = abs(run) + sqrt(cell.weight * depth);
    let toward = select(-1.0, 1.0, far);
    return vec3(
        0.0,
        face * run * run + push(depth, floor, cell.turn, axis) + toward * quickest * face * run,
        0.0,
    );
}

/// What crosses the face between a cell's far side and the next cell's near side.
fn crossing(low: Rebuilt, high: Rebuilt, axis: u32) -> vec3<f32> {
    let floor = low.floor.y;
    let depths = max(vec2(low.level.y, high.level.x) - floor, vec2(0.0));
    let waters = vec2(water_of(depths.x, floor), water_of(depths.y, floor));
    let runs = vec2(run_of(waters.x, low.across.y), run_of(waters.y, high.across.x));
    let sideways = vec2(run_of(waters.x, low.along.y), run_of(waters.y, high.along.x));
    let waves = sqrt(0.5 * (low.weight + high.weight) * depths);
    // a face across the ring's turning is as tall as the water is deep, and the cells either
    // side of it as long as the ring is round where the water lies
    var faces = waters;
    var lengths = vec2(1.0);
    if (axis == ROUND) {
        faces = depths;
        lengths = 1.0 / vec2(narrowing(floor + 0.5 * depths.x), narrowing(floor + 0.5 * depths.y));
    }
    let fastest = max(max((runs.x + waves.x) * lengths.x, (runs.y + waves.y) * lengths.y), 0.0);
    let slowest = min(min((runs.x - waves.x) * lengths.x, (runs.y - waves.y) * lengths.y), 0.0);
    if (fastest - slowest < 1e-12) {
        return vec3(0.0);
    }
    let turn = 0.5 * (low.turn + high.turn);
    let from_low = vec3(
        faces.x * runs.x,
        faces.x * runs.x * runs.x + push(depths.x, floor, turn, axis),
        faces.x * runs.x * sideways.x,
    );
    let from_high = vec3(
        faces.y * runs.y,
        faces.y * runs.y * runs.y + push(depths.y, floor, turn, axis),
        faces.y * runs.y * sideways.y,
    );
    let kept_low = vec3(waters.x, waters.x * runs.x, waters.x * sideways.x);
    let kept_high = vec3(waters.y, waters.y * runs.y, waters.y * sideways.y);
    return (fastest * from_low - slowest * from_high + fastest * slowest * (kept_high - kept_low))
        / (fastest - slowest);
}

fn in_use(at: vec3<u32>) -> bool {
    return at.x < sheet.used.x && at.y < sheet.used.y;
}

/// How much of poured water falls this far from the middle of where it falls, as a share of
/// how far it falls at all: most in the middle, and none at the rim. `Sheet::pour` sums the
/// same over the cells to pour exactly what it was asked to.
fn falling(share: f32) -> f32 {
    let within = max(1.0 - share * share, 0.0);
    return within * within;
}

/// The frame's seconds are owed, and the frame's water falls where it was poured. Water that
/// lands on a floor keeps its speed: what it had over the floor it runs on with, and what it
/// came down with turns outward from where it landed, the more the further out it lands.
@compute @workgroup_size(8, 8)
fn pour(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u && id.y == 0u) {
        clock.remaining = min(clock.remaining + sheet.advance, MOST_BEHIND);
    }
    if (id.x >= sheet.pour_cells.x || id.y >= sheet.pour_cells.y) {
        return;
    }
    var at = sheet.pour_first + id.xy;
    if (sheet.closed == 1u) {
        at.x = at.x % sheet.used.x;
    }
    if (in_use(vec3(at, 0u))) {
        let from_middle = (vec2<f32>(id.xy) + 0.5 - sheet.pour_middle) * sheet.cell;
        let run = sheet.pour_run + sheet.pour_spread * from_middle / sheet.pour_reach;
        let water = sheet.pour * falling(length(from_middle) / sheet.pour_reach);
        after[slot(at)] += vec4(water, water * run, 0.0);
    }
}

/// How quickly the quickest wave of each row crosses a cell.
@compute @workgroup_size(64)
fn quickest(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= sheet.used.y) {
        return;
    }
    var rate = 0.0;
    for (var k = 0u; k < sheet.used.x; k++) {
        let at = vec2(k, id.x);
        let state = before[slot(at)];
        if (state.x <= 0.0) {
            continue;
        }
        let here = standing(at, ROUND);
        let wave = sqrt(here.weight * here.depth);
        let round = (abs(run_of(state.x, state.y)) + wave)
            / (narrowing(here.level - 0.5 * here.depth) * sheet.cell.x);
        let along = (abs(run_of(state.x, state.z)) + wave) / sheet.cell.y;
        rate = max(rate, max(round, along));
    }
    clock.rows[id.x] = rate;
}

/// The step to take: no longer than is owed, and none a wave could cross a cell in.
@compute @workgroup_size(1)
fn pace() {
    var rate = 0.0;
    for (var l = 0u; l < sheet.used.y; l++) {
        rate = max(rate, clock.rows[l]);
    }
    clock.quickest = rate;
    var dt = min(clock.remaining, LONGEST_STEP);
    if (rate > 0.0) {
        dt = min(dt, COURANT / rate);
    } else {
        // nothing to step: the time is not owed to water there is none of
        clock.remaining = 0.0;
        dt = 0.0;
    }
    if (dt < 1e-6) {
        dt = 0.0;
    }
    clock.remaining = max(clock.remaining - dt, 0.0);
    clock.dt = dt;
    threads = select(
        vec3(0u),
        vec3((sheet.used.x + 7u) / 8u, (sheet.used.y + 7u) / 8u, 1u),
        dt > 0.0,
    );
}

@compute @workgroup_size(8, 8)
fn cross(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!in_use(id)) {
        return;
    }
    let at = id.xy;
    for (var axis = ROUND; axis <= ALONG; axis++) {
        let here = rebuilt(at, axis);
        let on = beside(at, axis, 1);
        var crosses = walled(here, true, axis);
        if (on.there) {
            crosses = crossing(here, rebuilt(on.at, axis), axis);
        }
        if (axis == ROUND) {
            crossing_round[slot(at)] = vec4(crosses, NEVER);
        } else {
            crossing_along[slot(at)] = vec4(crosses, 0.0);
        }
    }
}

/// What crosses a cell's near face across an axis: what crossed the far face of the cell
/// before it, or what the wall there gives back.
fn crossed_before(at: vec2<u32>, axis: u32) -> vec4<f32> {
    let back = beside(at, axis, -1);
    if (!back.there) {
        return vec4(walled(rebuilt(at, axis), false, axis), NEVER);
    }
    if (axis == ROUND) {
        return crossing_round[slot(back.at)];
    }
    return vec4(crossing_along[slot(back.at)].xyz, crossing_round[slot(back.at)].w);
}

/// How long each cell's water lasts all that leaves it.
@compute @workgroup_size(8, 8)
fn drain(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!in_use(id)) {
        return;
    }
    let at = id.xy;
    let c = slot(at);
    let leaving = (max(crossing_round[c].x, 0.0) + max(-crossed_before(at, ROUND).x, 0.0)) / sheet.cell.x
        + (max(crossing_along[c].x, 0.0) + max(-crossed_before(at, ALONG).x, 0.0)) / sheet.cell.y;
    crossing_round[c].w = select(NEVER, before[c].x / max(leaving, 1e-30), leaving > 1e-30);
}

/// The water that crosses a face, held to what the cell it leaves can give over the step.
fn given(water: f32, lasts_low: f32, lasts_high: f32) -> f32 {
    return water * min(1.0, select(lasts_high, lasts_low, water >= 0.0) / clock.dt);
}

/// A cell's state a step on from `before`: what crossed its faces, what the floor pushed it
/// with, and what the floor's roughness and the water's own stickiness took of its run.
fn stepped(at: vec2<u32>) -> vec4<f32> {
    let c = slot(at);
    let dt = clock.dt;
    let lasts = crossing_round[c].w;
    let far_round = crossing_round[c];
    let far_along = crossing_along[c];
    let near_round = crossed_before(at, ROUND);
    let near_along = crossed_before(at, ALONG);
    let on_round = beside(at, ROUND, 1);
    let on_along = beside(at, ALONG, 1);
    let lasts_on_round = select(NEVER, crossing_round[slot(on_round.at)].w, on_round.there);
    let lasts_on_along = select(NEVER, crossing_round[slot(on_along.at)].w, on_along.there);
    let water = vec4(
        given(near_round.x, near_round.w, lasts),
        given(far_round.x, lasts, lasts_on_round),
        given(near_along.x, near_along.w, lasts),
        given(far_along.x, lasts, lasts_on_along),
    );
    let round = rebuilt(at, ROUND);
    let along = rebuilt(at, ALONG);
    let floor_round = push(max(round.still - round.floor.y, 0.0), round.floor.y, round.turn, ROUND)
        - push(max(round.still - round.floor.x, 0.0), round.floor.x, round.turn, ROUND);
    let floor_along = push(max(along.still - along.floor.y, 0.0), along.floor.y, along.turn, ALONG)
        - push(max(along.still - along.floor.x, 0.0), along.floor.x, along.turn, ALONG);
    var state = before[c];
    state.x += dt * ((water.x - water.y) / sheet.cell.x + (water.z - water.w) / sheet.cell.y);
    state.y += dt * ((near_round.y - far_round.y + floor_round) / sheet.cell.x
        + (near_along.z - far_along.z) / sheet.cell.y);
    state.z += dt * ((near_round.z - far_round.z) / sheet.cell.x
        + (near_along.y - far_along.y + floor_along) / sheet.cell.y);
    state.x = max(state.x, 0.0);
    if (state.x <= 0.0) {
        return vec4(0.0);
    }
    let floor = floor_under(at, ROUND).middle;
    let depth = depth_of(state.x, floor);
    let run = vec2(run_of(state.x, state.y), run_of(state.x, state.z));
    let drag = MANNINGS_GRAVITY * ROUGHNESS * ROUGHNESS * length(run) / pow(depth, 4.0 / 3.0)
        + 3.0 * VISCOSITY / (depth * depth);
    return vec4(state.x, state.yz / (1.0 + dt * drag), 0.0);
}

/// The first of a step's two halves: a whole step on.
@compute @workgroup_size(8, 8)
fn step_once(@builtin(global_invocation_id) id: vec3<u32>) {
    if (in_use(id)) {
        after[slot(id.xy)] = stepped(id.xy);
    }
}

/// The second: another whole step on from the first, and the mean of that and where the step
/// began, which `after` still holds.
@compute @workgroup_size(8, 8)
fn step_twice(@builtin(global_invocation_id) id: vec3<u32>) {
    if (in_use(id)) {
        let c = slot(id.xy);
        after[c] = 0.5 * (after[c] + stepped(id.xy));
    }
}

/// The level of the water round a corner of the floor: the mean of the water's in the cells
/// that meet there, and how it runs; nothing where none of them holds any.
struct Met {
    level: f32,
    run: vec2<f32>,
    wet: bool,
}

fn met_at(at: vec2<u32>) -> Met {
    var level = 0.0;
    var run = vec2(0.0);
    var wet = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let round = beside(at, ROUND, i - 1);
            let cell = beside(round.at, ALONG, j - 1);
            if (!round.there || !cell.there) {
                continue;
            }
            let state = before[slot(cell.at)];
            let floor = floor_under(cell.at, ROUND).middle;
            let depth = depth_of(state.x, floor);
            if (depth > DRAWN) {
                level += floor + depth;
                run += vec2(run_of(state.x, state.y), run_of(state.x, state.z));
                wet += 1.0;
            }
        }
    }
    if (wet == 0.0) {
        return Met(corner(at), vec2(0.0), false);
    }
    return Met(max(level / wet, corner(at)), run / wet, true);
}

fn corner_number(at: vec2<u32>) -> u32 {
    return at.y * (sheet.stored.x + 1u) + at.x;
}

fn corners_in_use(at: vec3<u32>) -> bool {
    return at.x <= sheet.used.x && at.y <= sheet.used.y;
}

/// The surface's corners: where each stands, about the site the surface is drawn about, and
/// how the water under it runs. Its level rides in the run's fourth place for `face`.
@compute @workgroup_size(8, 8)
fn raise(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!corners_in_use(id)) {
        return;
    }
    // the corner past the last cell of a ring closed on itself is the first corner again
    var at = id.xy;
    if (sheet.closed == 1u) {
        at.x = at.x % sheet.used.x;
    }
    let met = met_at(at);
    let turn = (sheet.origin.x + f32(id.x) * sheet.cell.x) / sheet.radius;
    let along = sheet.origin.y + f32(id.y) * sheet.cell.y;
    let half = sin(0.5 * turn);
    let inward = vec3(-cos(turn), 0.0, -sin(turn));
    let spinward = vec3(-sin(turn), 0.0, cos(turn));
    let wall = vec3(-2.0 * sheet.radius * half * half, along, sheet.radius * sin(turn));
    var vertex: SurfaceVertex;
    vertex.position = vec4(wall + inward * met.level, 0.0);
    vertex.normal = vec4(inward, met.level - corner(at));
    vertex.velocity = vec4(spinward * met.run.x + vec3(0.0, met.run.y, 0.0), met.level);
    vertices[corner_number(id.xy)] = vertex;
}

/// Which way the surface faces at each corner, and the two triangles over each cell that has
/// water at a corner.
@compute @workgroup_size(8, 8)
fn face(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!corners_in_use(id)) {
        return;
    }
    let at = id.xy;
    let n = corner_number(at);
    // the corners either side, which round a ring closed on itself are never wanting
    let last = sheet.used;
    var back = vec2(select(at.x - 1u, at.x, at.x == 0u), select(at.y - 1u, at.y, at.y == 0u));
    var on = min(at + vec2(1u), last);
    var apart = vec2<f32>(on - back);
    if (sheet.closed == 1u) {
        back.x = select(at.x - 1u, last.x - 1u, at.x == 0u);
        on.x = select(at.x + 1u, 1u, at.x == last.x);
        apart.x = 2.0;
    }
    let level = vertices[n].velocity.w;
    let round = (vertices[corner_number(vec2(on.x, at.y))].velocity.w
        - vertices[corner_number(vec2(back.x, at.y))].velocity.w)
        / (apart.x * sheet.cell.x * narrowing(level));
    let along = (vertices[corner_number(vec2(at.x, on.y))].velocity.w
        - vertices[corner_number(vec2(at.x, back.y))].velocity.w)
        / (apart.y * sheet.cell.y);
    let inward = vertices[n].normal.xyz;
    let spinward = vec3(inward.z, 0.0, -inward.x);
    vertices[n].normal = vec4(
        normalize(inward - spinward * round - vec3(0.0, along, 0.0)),
        vertices[n].normal.w,
    );
    if (at.x == sheet.used.x || at.y == sheet.used.y) {
        return;
    }
    var corners = array<u32, 4>(n, n + 1u, n + sheet.stored.x + 1u, n + sheet.stored.x + 2u);
    var deepest = 0.0;
    for (var k = 0u; k < 4u; k++) {
        deepest = max(deepest, vertices[corners[k]].normal.w);
    }
    if (deepest <= 0.0) {
        return;
    }
    let first = atomicAdd(&counters[1], 6u);
    var order = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    for (var k = 0u; k < 6u; k++) {
        indices[first + k] = corners[order[k]];
    }
}

/// The water round the corners of the watched block.
@compute @workgroup_size(8, 8)
fn report(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x > WATCHED || id.y > WATCHED) {
        return;
    }
    var at = sheet.watch_first + id.xy;
    if (sheet.closed == 1u) {
        at.x = at.x % sheet.used.x;
    }
    var found = vec4(0.0);
    if (corners_in_use(vec3(at, 0u))) {
        let met = met_at(at);
        found = vec4(met.level - corner(at), met.level, met.run);
    }
    watched[0] = vec4(vec2<f32>(sheet.watch_first), clock.quickest, 0.0);
    watched[1u + id.y * (WATCHED + 1u) + id.x] = found;
}

/// The water in each row.
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= sheet.stored.y) {
        return;
    }
    var water = 0.0;
    if (id.x < sheet.used.y) {
        for (var k = 0u; k < sheet.used.x; k++) {
            water += before[slot(vec2(k, id.x))].x;
        }
    }
    held[id.x] = water * sheet.cell.x * sheet.cell.y;
}
