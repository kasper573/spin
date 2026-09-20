// What keeps the water from being squeezed: a grid of cells laid over it, two spacings to a
// side, that the particles hand their motion to and take it back from once the grid has made
// it one that squeezes nothing. The handing over keeps each particle's own turning and
// shearing (Jiang et al. 2015) and, where all the faces round it are the water's, all that
// they told it (Fu et al. 2017), so that water loses next to nothing of its motion to being
// handed back and forth. The walls leave each face of a cell open by the share of it that no
// solid covers (Batty et al. 2007). At the water's face the pressure is the air's (Enright et
// al. 2003), and where that face is the height functions of the volume-of-fluid methods say:
// as far up a column of cells as the water in the column fills it, which says exactly where a
// flat face lies however it is tilted to the grid, and says it by a sum, in which a little
// more or less water is never more than a little more or less height. Pressure is one unknown
// to a cell of water, relaxed from what the water carried from the step before, so water at
// rest asks for next to nothing. A second unknown to the cell shifts the water to where it is
// as dense as water is (Kugelstadt et al. 2019).
//
// Only cells near water exist: each is claimed in a hashed table by the particles round it,
// listed once, and visited through the list. A particle claims the eight cells whose middles
// it is within a cell of, which are the ones its water can fill, and the cells across their
// faces: so a cell of water has all six of the cells round it, and, a cell owning the three
// of its faces toward the origin, every face within a particle's reach has its cell.
//
// Pressures are kept as the speed they would give across one cell in one step, which is what
// they are wanted for.
#import vessel::{vessel_gravity, Confined, VESSEL_OPENINGS, vessel_confine, vessel_star_velocity, vessel_through}
#import fluid_common::{params, Bodies, GpuBody, cell_key, cell_slot, neighbour_cell}

@group(0) @binding(1) var<storage, read_write> position: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
// xyz: where free flight alone landed the particle
@group(0) @binding(4) var<storage, read> velocity_next: array<vec4<f32>>;
// xyz: where the walls left it; w: how fast the water round it is coming together, once known
@group(0) @binding(5) var<storage, read_write> pred_in: array<vec4<f32>>;
// xyz: which way the water round it lies; w: how full of water the place is
@group(0) @binding(6) var<storage, read_write> pred_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> contact: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
@group(0) @binding(12) var<storage, read> key: array<u32>;
// six to a particle, two to each part of its velocity: how that part changes across the
// particle along each axis, then how those changes change in turn, across each pair of axes
// (xy, yz, zx) and across all three. With the part itself that is the whole of what the
// eight faces round the particle told it, so a particle hands back to them what it took.
// Beside the first is the pressure the particle was under
@group(0) @binding(15) var<storage, read_write> affine: array<vec4<f32>>;

@group(2) @binding(0) var<uniform> bodies: Bodies;

@group(3) @binding(1) var<storage, read_write> grid_keys: array<atomic<u32>>;
// the cells there are, and from halfway on those of them with the water's face by a solid
@group(3) @binding(2) var<storage, read_write> grid_list: array<u32>;
// how many cells are listed, and how many of them by a shore
@group(3) @binding(3) var<storage, read_write> grid_counters: array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> cells: array<Cell>;
@group(3) @binding(5) var<storage, read_write> links: array<Link>;
// the workgroups that visit the listed cells, then those that visit the ones by a shore
@group(3) @binding(6) var<storage, read_write> grid_dispatch: array<u32>;

struct Cell {
    // the water's velocity through the three faces the cell owns; w: how full of water it is
    flow: vec4<f32>,
    // how far the water is to be shifted through each face to be as dense as water is, in
    // cells; w: how fast the water is closing
    shift: vec4<f32>,
    // the share of each face no solid covers; w: the push that shifts the water, as a pressure
    open: vec4<f32>,
    // the velocity of whatever solid covers each face; w: which faces the pressure has set
    solid: vec4<f32>,
    // the pressure, what it has to answer for, what it is divided by, and 1 for water
    pressure: vec4<f32>,
    // x: how much of the cell's reach is taken up, by water and by solids alike
    // w: the share of the cell's reach that solids take up, until what of the cell's water is
    // to be shifted out of it is known, which it then is
    filled: vec4<f32>,
    // how far under the water's face the cell's middle is, in cells, along each axis; w: the
    // axis the face lies most squarely across, which way along it the air lies, and along
    // which axes a column of cells told the depth
    level: vec4<f32>,
    // how the depth under the face changes from cell to cell along the two axes after the one
    // the face lies most squarely across; w: what the push that shifts the water is divided by
    slope: vec4<f32>,
    // the two walls of the vessel nearest the cell's middle, each taken as flat; xyz: which
    // way it faces, w: how far off it is, in cells
    walls: array<vec4<f32>, 2>,
}

/// The cells across a cell's six faces, toward the origin and away along each axis in turn,
/// and what each one's pressure counts for in this one's.
struct Link {
    to: array<u32, 6>,
    counts: array<f32, 6>,
}

// A cell's key is its place among the `KEY_SPAN` cells a side that the water may reach, as
// far either way from the frame's origin as the coarsening of far-flung water lets it get;
// counted from one, as no key at all is nothing. The cube of the span is the most that fits.
const KEY_SPAN: u32 = 1536u;
const KEY_ORIGIN: i32 = 768;
const PROBES: u32 = 64u;
const NONE: u32 = 0xffffffffu;
const WORKGROUP: u32 = 64u;
// how far from a wall a cell's centre is before none of the cell is asked about it, in cells
const CLEAR_OF_WALLS: f32 = 1.8;
const WALL_REACH: f32 = 2.0;
// the nearest the water's face is taken to be to the centre of a cell it has filled, in cells
const NEAREST_FACE: f32 = 0.01;
// the least of a cell's reach a wall beside it is taken to leave it
const LEAST_BESIDE: f32 = 0.05;
// how full the cell at the wet end of a column of five must be for the column to stand on
// water, and for the cells past that end to have nothing to add but that they are full: a
// face's fill rises to full within a cell of where it has risen this far, however the face
// is tilted
const FULL_BELOW: f32 = 0.8;
// Gauss's three points on the half of a cell's reach from its middle out, and their weights
// times how the reach thins out toward its end
const GAUSS_POINT: array<f32, 3> = array<f32, 3>(0.11270167, 0.5, 0.88729833);
const GAUSS_WEIGHT: array<f32, 3> = array<f32, 3>(0.24647176, 0.22222222, 0.03130602);
// Gauss's two points on the half of a cell's reach from its middle out, and their weights
// times how the reach thins out toward its end
const LINE_AT: array<f32, 2> = array<f32, 2>(0.21132487, 0.78867513);
const LINE_WEIGHT: array<f32, 2> = array<f32, 2>(0.39433757, 0.10566243);
// the lines through a cell's reach that those points make, four by four
const LINES: u32 = 16u;
// how many of Newton's steps put the water's face where a column's water fills it to
const NEWTON_STEPS: u32 = 8u;
// the share of what a cell holds over or under its fill that is shifted out of it or into
// it in a step: all of it at once would shift the water by every bit of chance in how its
// particles lie
const SETTLED_IN_A_STEP: f32 = 0.2;
// what a cell's level says beside the axis the water's face lies most squarely across: which
// way along it the air lies, along which axes a column told the depth, and whether the cell
// is the water's
const AIR_IS_UP: u32 = 4u;
const TOLD_ALONG: u32 = 8u;
const IS_WATER: u32 = 64u;
// how much of what the solids leave of a cell's reach must be water for the cell to be the
// water's: as much as is under a face that passes through the middle of a cell clear of them
const HALF_FULL: f32 = 0.5;
// the shallowest sheet of water a particle against a wall stands for, in spacings
const SHALLOWEST: f32 = 0.1;
// the share of what a cell reads over or under its fill, within what chance alone could make
// it read, that is shifted in a step
const EVENED_OUT_IN_A_STEP: f32 = 0.02;
// how full the cells either side of a cell must be, along the axis the water's face lies most
// squarely across, for the cell to be so deep in the water that all its reach is under the face
const DEEP: f32 = 0.9;
// how little a solid's face may turn toward a line through a cell for the line to run along
// it: a quarter, which is a wall within fifteen degrees of the line
const RUNS_ALONG: f32 = 0.26;
// how squarely a solid faces along an axis for it to face along it rather than across it
const FACES_ALONG: f32 = 0.7;
// how far from a cell's middle the water's face may be for the column of five cells about the
// cell to have the face in it, in cells: all but the last hair of the column's reach
const FACE_IN_THE_COLUMN: f32 = 2.45;
// how far under the water's face a cell is taken to be where nothing tells of a face over it,
// in cells: further than its reach
const FAR_UNDER: f32 = 2.0;
// how far from its fill a cell may read by the chance of how its particles lie
const BY_CHANCE: f32 = 0.05;
// how near a solid a corner of a cell may be and count as in it, in cells: a wall that lies
// along a face of the grid, as the vessel's often do, stands to one side of it or the other
// by nothing but rounding, and has to cover the face for certain
const HAIR: f32 = 0.001;
// the share of a face a solid must leave open for any water to be let through it: what flows
// through so little is nothing the pressure has a hold on
const BARELY_OPEN: f32 = 0.01;
// the share of a face a solid must leave open for the face to count as wholly open
const WHOLLY_OPEN: f32 = 0.99;
// how near a cell's middle a solid is before it takes any of the cell's reach, in cells: the
// reach is a cell either way, and its corners are further off than that
const SOLID_REACH: f32 = 1.5;
// all eight faces round a particle set by the pressure: only then does the particle carry how
// its velocity's changes change in turn, the finest things the faces tell, which faces that
// stand in for others, or for a solid, do not tell at all
const ALL_EIGHT: u32 = 0xffu;
// over-relaxation of the pressure's sweeps
const OVER: f32 = 1.5;

fn grid_key(c: vec3<i32>) -> u32 {
    let k = c + vec3(KEY_ORIGIN);
    if (any(k < vec3(0)) || any(k >= vec3(i32(KEY_SPAN)))) {
        return 0u;
    }
    return 1u + u32(k.x) + KEY_SPAN * (u32(k.y) + KEY_SPAN * u32(k.z));
}

fn grid_coords(k: u32) -> vec3<i32> {
    let at = k - 1u;
    return vec3<i32>(vec3(at % KEY_SPAN, (at / KEY_SPAN) % KEY_SPAN, at / (KEY_SPAN * KEY_SPAN))) - vec3(KEY_ORIGIN);
}

fn grid_slot(k: u32) -> u32 {
    var h = k;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h & params.grid_mask;
}

/// Make a cell exist, listing it if this is the first it was asked for.
fn claim(c: vec3<i32>) {
    let k = grid_key(c);
    if (k == 0u) {
        return;
    }
    var slot = grid_slot(k);
    for (var probe = 0u; probe < PROBES; probe++) {
        let r = atomicCompareExchangeWeak(&grid_keys[slot], 0u, k);
        if (r.exchanged) {
            let at = atomicAdd(&grid_counters[0], 1u);
            grid_list[at] = slot;
            return;
        }
        if (r.old_value == k) {
            return;
        }
        if (r.old_value != 0u) {
            slot = (slot + 1u) & params.grid_mask;
        }
    }
}

/// Where a cell is kept, if it exists.
fn find(c: vec3<i32>) -> u32 {
    let k = grid_key(c);
    if (k == 0u) {
        return NONE;
    }
    var slot = grid_slot(k);
    for (var probe = 0u; probe < PROBES; probe++) {
        let found = atomicLoad(&grid_keys[slot]);
        if (found == k) {
            return slot;
        }
        if (found == 0u) {
            return NONE;
        }
        slot = (slot + 1u) & params.grid_mask;
    }
    return NONE;
}

@compute @workgroup_size(64)
fn occupy(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let first = vec3<i32>(floor(pred_in[i].xyz * params.inv_cell - vec3(0.5)));
    for (var n = 0u; n < 8u; n++) {
        claim(first + vec3<i32>(vec3(n & 1u, (n >> 1u) & 1u, n >> 2u)));
    }
    for (var axis = 0u; axis < 3u; axis++) {
        for (var n = 0u; n < 4u; n++) {
            var o = vec3(0);
            o[(axis + 1u) % 3u] = i32(n & 1u);
            o[(axis + 2u) % 3u] = i32(n >> 1u);
            o[axis] = -1;
            claim(first + o);
            o[axis] = 2;
            claim(first + o);
        }
    }
}

@compute @workgroup_size(1)
fn muster() {
    let count = atomicLoad(&grid_counters[0]);
    grid_dispatch[0] = (count + WORKGROUP - 1u) / WORKGROUP;
    grid_dispatch[1] = 1u;
    grid_dispatch[2] = 1u;
}

@compute @workgroup_size(1)
fn muster_shores() {
    let count = atomicLoad(&grid_counters[1]);
    grid_dispatch[4] = (count + WORKGROUP - 1u) / WORKGROUP;
    grid_dispatch[5] = 1u;
    grid_dispatch[6] = 1u;
}

/// The listed cell a thread visits: where it is kept and which it is, or none.
struct Visit {
    slot: u32,
    c: vec3<i32>,
    there: bool,
}

fn visit(thread: u32) -> Visit {
    return visit_listed(thread, 0u);
}

/// The cell by a shore that a thread visits.
fn visit_shore(thread: u32) -> Visit {
    return visit_listed(thread, 1u);
}

fn visit_listed(thread: u32, list: u32) -> Visit {
    if (thread >= atomicLoad(&grid_counters[list])) {
        return Visit(0u, vec3(0), false);
    }
    let slot = grid_list[list * (arrayLength(&grid_list) / 2u) + thread];
    return Visit(slot, grid_coords(atomicLoad(&grid_keys[slot])), true);
}

fn tent(d: f32) -> f32 {
    return max(1.0 - abs(d), 0.0);
}

fn tent3(d: vec3<f32>) -> f32 {
    return tent(d.x) * tent(d.y) * tent(d.z);
}

/// How much one part of a particle's velocity differs a way off from it, by what it carries
/// of how that part changes across it.
/// The share of a particle's parcel of water that is within a cell's reach, `d` cells from
/// the cell's middle. The parcel is a spacing across, not a point: handed to the cells as
/// one, water lying evenly on its lattice reads exactly as full as it is, right up to the face
/// or the wall it ends at half a spacing past its last particle, where points would read a few
/// parts in a hundred off by where among the cells they happen to lie.
fn parcel_within(d: vec3<f32>) -> f32 {
    let across = params.spacing * params.inv_cell;
    let share = vec3(under(d.x + 0.5 * across) - under(d.x - 0.5 * across), under(d.y + 0.5 * across) - under(d.y - 0.5 * across), under(d.z + 0.5 * across) - under(d.z - 0.5 * across)) / across;
    return share.x * share.y * share.z;
}

fn carried_to(j: u32, axis: u32, off: vec3<f32>) -> f32 {
    let slopes = affine[6u * j + 2u * axis].xyz;
    let twists = affine[6u * j + 2u * axis + 1u];
    return dot(slopes, off) + dot(twists.xyz, off.xyz * off.yzx) + twists.w * off.x * off.y * off.z;
}

/// Hand the particles' motion to a cell's faces, each particle weighing on a face by how near
/// it is, and its water to the cell's middle.
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let base = vec3<f32>(at.c) * params.h;
    let parcel = params.spacing * params.spacing * params.spacing * params.inv_cell * params.inv_cell * params.inv_cell;
    var moved = vec3(0.0);
    var said = vec3(0.0);
    var fill = 0.0;
    var pressed = 0.0;
    var pressing = 0.0;
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(at.c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            if (key[j] != k) {
                continue;
            }
            let x = pred_in[j].xyz;
            let v = velocity[j].xyz;
            let rel = (x - base) * params.inv_cell;
            let middle = parcel_within(rel - vec3(0.5));
            fill += middle * parcel;
            pressed += middle * affine[6u * j].w;
            pressing += middle;
            let w = vec3(tent3(rel - vec3(0.0, 0.5, 0.5)), tent3(rel - vec3(0.5, 0.0, 0.5)), tent3(rel - vec3(0.5, 0.5, 0.0)));
            if (w.x + w.y + w.z <= 0.0) {
                continue;
            }
            let to_x = base + vec3(0.0, 0.5, 0.5) * params.h - x;
            let to_y = base + vec3(0.5, 0.0, 0.5) * params.h - x;
            let to_z = base + vec3(0.5, 0.5, 0.0) * params.h - x;
            let there = vec3(v.x + carried_to(j, 0u, to_x), v.y + carried_to(j, 1u, to_y), v.z + carried_to(j, 2u, to_z));
            moved += w * there;
            said += w;
        }
    }
    cells[at.slot].flow = vec4(moved / max(said, vec3(1e-9)), fill);
    cells[at.slot].shift = vec4(0.0);
    var carried = 0.0;
    if (pressing > 0.0) {
        carried = pressed / pressing;
    }
    cells[at.slot].pressure = vec4(carried, 0.0, 0.0, 0.0);
}

/// How far a point is from the nearest wall of the vessel, within `WALL_REACH` cells.
fn clear_of_walls(p: vec3<f32>) -> f32 {
    let reach = WALL_REACH * params.h;
    let c = vessel_confine(p, reach);
    let moved = c.p - p;
    var clear = reach;
    if (c.first.w > 0.0) {
        clear = min(clear, reach - dot(moved, c.first.xyz));
    }
    if (c.second.w > 0.0) {
        clear = min(clear, reach - dot(moved, c.second.xyz));
    }
    return clear;
}

/// A solid near a point: how far off it is, in cells, and which way it faces.
struct Solid {
    clear: f32,
    normal: vec3<f32>,
}

/// The walls of the vessel and the body nearest a point, within `WALL_REACH` cells.
fn solids_round(p: vec3<f32>) -> array<Solid, 3> {
    let reach = WALL_REACH * params.h;
    let c = vessel_confine(p, reach);
    let moved = c.p - p;
    var out = array<Solid, 3>(Solid(WALL_REACH, vec3(0.0)), Solid(WALL_REACH, vec3(0.0)), body_nearest(p));
    if (c.first.w > 0.0) {
        out[0] = Solid((reach - dot(moved, c.first.xyz)) * params.inv_cell, c.first.xyz);
    }
    if (c.second.w > 0.0) {
        out[1] = Solid((reach - dot(moved, c.second.xyz)) * params.inv_cell, c.second.xyz);
    }
    return out;
}

/// The same round a cell's middle, of the walls that were found for the cell.
fn solids_of(slot: u32, c: vec3<i32>) -> array<Solid, 3> {
    let first = cells[slot].walls[0];
    let second = cells[slot].walls[1];
    return array<Solid, 3>(Solid(first.w, first.xyz), Solid(second.w, second.xyz), body_nearest((vec3<f32>(c) + vec3(0.5)) * params.h));
}

fn body_nearest(p: vec3<f32>) -> Solid {
    var out = Solid(WALL_REACH, vec3(0.0));
    for (var b = 0u; b < params.body_count; b++) {
        if (bodies.items[b].position.w <= 0.0) {
            continue;
        }
        let bulk = bulk_of(b);
        let off = p - bulk.xyz;
        let apart = length(off);
        let clear = (apart - bulk.w) * params.inv_cell;
        if (clear < out.clear && apart > 1e-9) {
            out = Solid(clear, off / apart);
        }
    }
    return out;
}

fn bulk_of(b: u32) -> vec4<f32> {
    let body = bodies.items[b];
    let centre = body.position.xyz + vec3(dot(body.row_x.xyz, body.shape.xyz), dot(body.row_y.xyz, body.shape.xyz), dot(body.row_z.xyz, body.shape.xyz));
    return vec4(centre, body.shape.w);
}

/// How far a point is from the nearest solid, a wall or a body; and which body, or none.
struct Clear {
    by: f32,
    body: u32,
}

fn clear_of_solids(p: vec3<f32>) -> Clear {
    var out = Clear(clear_of_walls(p), NONE);
    for (var b = 0u; b < params.body_count; b++) {
        if (bodies.items[b].position.w <= 0.0) {
            continue;
        }
        let bulk = bulk_of(b);
        let off = length(p - bulk.xyz) - bulk.w;
        if (off < out.by) {
            out = Clear(off, b);
        }
    }
    return out;
}

/// The share of an edge between two corners that is clear, given how clear each is.
fn clear_along(a: f32, b: f32) -> f32 {
    return a / (a - b);
}

/// The share of a square that is clear of solids, given how clear its corners are, taken
/// round it in turn.
fn open_share(round: array<f32, 4>) -> f32 {
    var corners = round;
    var clear = 0u;
    for (var k = 0u; k < 4u; k++) {
        clear += u32(corners[k] > 0.0);
    }
    if (clear == 0u) {
        return 0.0;
    }
    if (clear == 4u) {
        return 1.0;
    }
    // one corner clear, or one covered, is a triangle cut off at that corner
    if (clear == 1u || clear == 3u) {
        let lone = clear == 1u;
        for (var k = 0u; k < 4u; k++) {
            if ((corners[k] > 0.0) == lone) {
                let next = corners[(k + 1u) % 4u];
                let last = corners[(k + 3u) % 4u];
                let cut = 0.5 * clear_along(corners[k], next) * clear_along(corners[k], last);
                return select(1.0 - cut, cut, lone);
            }
        }
    }
    for (var k = 0u; k < 4u; k++) {
        let next = (k + 1u) % 4u;
        if (corners[k] > 0.0 && corners[next] > 0.0) {
            // two corners on one side: a strip between the edges that leave them
            return 0.5 * (clear_along(corners[k], corners[(k + 3u) % 4u]) + clear_along(corners[next], corners[(k + 2u) % 4u]));
        }
    }
    // two opposite corners: a triangle at each
    var share = 0.0;
    for (var k = 0u; k < 4u; k++) {
        if (corners[k] > 0.0) {
            share += 0.5 * clear_along(corners[k], corners[(k + 1u) % 4u]) * clear_along(corners[k], corners[(k + 3u) % 4u]);
        }
    }
    return share;
}

/// The share of a cell's reach that lies under a flat face square to an axis, `d` cells
/// above the cell's middle: the cell's reach is a cell either way and thins out from its
/// middle, as the particles' water is handed to it.
fn under(d: f32) -> f32 {
    let t = clamp(d, -1.0, 1.0);
    if (t >= 0.0) {
        return 1.0 - 0.5 * (1.0 - t) * (1.0 - t);
    }
    return 0.5 * (1.0 + t) * (1.0 + t);
}

/// The same under a flat face that is square to `normal`: a cell's reach is the product of
/// its reach along each axis, so its share under the face is the share along the axis the
/// face lies most squarely across, summed over where along the other two the face is met.
/// The sum is Gauss's, three points to each half of the reach, where it thins out evenly.
fn under_tilted(d: f32, normal: vec3<f32>) -> f32 {
    let m = abs(normal);
    var squarest = m.x;
    var others = m.yz;
    if (m.y >= m.x && m.y >= m.z) {
        squarest = m.y;
        others = m.xz;
    } else if (m.z >= m.x && m.z >= m.y) {
        squarest = m.z;
        others = m.xy;
    }
    return under_sloping(d / squarest, others / squarest);
}

/// The same by how far over the cell's middle the face is along the axis it lies most squarely
/// across, and how it slopes along the other two, in cells to the cell.
fn under_sloping(over: f32, slopes: vec2<f32>) -> f32 {
    var total = 0.0;
    for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j < 6u; j++) {
            total += GAUSS_WEIGHT[i % 3u] * GAUSS_WEIGHT[j % 3u] * under(over - slopes.x * gauss_point(i) - slopes.y * gauss_point(j));
        }
    }
    return total;
}

fn gauss_point(i: u32) -> f32 {
    return select(GAUSS_POINT[i % 3u], -GAUSS_POINT[i % 3u], i >= 3u);
}

/// How far under a flat face a cell's middle is that the face fills this much of, in cells.
fn depth_filling(filled: f32) -> f32 {
    if (filled >= 0.5) {
        return 1.0 - sqrt(2.0 * max(1.0 - filled, 0.0));
    }
    return sqrt(2.0 * max(filled, 0.0)) - 1.0;
}

fn solid_velocity(b: u32, p: vec3<f32>) -> vec3<f32> {
    if (b == NONE) {
        return vec3(0.0);
    }
    let body = bodies.items[b];
    return body.velocity.xyz + cross(body.angular.xyz, p - body.position.xyz);
}

/// What of a cell's faces the walls and the bodies leave open, and how they move.
@compute @workgroup_size(64)
fn walls(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let base = vec3<f32>(at.c) * params.h;
    let middle = clear_of_solids(base + vec3(0.5) * params.h);
    let fill = cells[at.slot].flow.w;
    if (middle.by > CLEAR_OF_WALLS * params.h) {
        cells[at.slot].open = vec4(1.0);
        cells[at.slot].solid = vec4(0.0);
        cells[at.slot].filled = vec4(vec3(fill), 0.0);
        cells[at.slot].walls = array<vec4<f32>, 2>(vec4(0.0, 0.0, 0.0, WALL_REACH), vec4(0.0, 0.0, 0.0, WALL_REACH));
        return;
    }
    let solids = solids_round(base + vec3(0.5) * params.h);
    cells[at.slot].walls = array<vec4<f32>, 2>(vec4(solids[0].normal, solids[0].clear), vec4(solids[1].normal, solids[1].clear));
    var nearest = 0u;
    for (var k = 1u; k < 3u; k++) {
        if (solids[k].clear < solids[nearest].clear) {
            nearest = k;
        }
    }
    let facing = abs(solids[nearest].normal);
    var along = 0u;
    if (facing.y >= facing.x && facing.y >= facing.z) {
        along = 1u;
    } else if (facing.z >= facing.x && facing.z >= facing.y) {
        along = 2u;
    }
    let left_to_water = waters_share(FaceAcross(along, 1.0, FAR_UNDER, vec2(0.0)), solids);
    cells[at.slot].filled = vec4(vec3(fill + 1.0 - left_to_water), 1.0 - left_to_water);
    let h = params.h;
    let hair = HAIR * h;
    let c000 = clear_of_solids(base).by - hair;
    let c100 = clear_of_solids(base + vec3(h, 0.0, 0.0)).by - hair;
    let c010 = clear_of_solids(base + vec3(0.0, h, 0.0)).by - hair;
    let c001 = clear_of_solids(base + vec3(0.0, 0.0, h)).by - hair;
    let c110 = clear_of_solids(base + vec3(h, h, 0.0)).by - hair;
    let c101 = clear_of_solids(base + vec3(h, 0.0, h)).by - hair;
    let c011 = clear_of_solids(base + vec3(0.0, h, h)).by - hair;
    let open = vec3(open_share(array<f32, 4>(c000, c010, c011, c001)), open_share(array<f32, 4>(c000, c100, c101, c001)), open_share(array<f32, 4>(c000, c100, c110, c010)));
    cells[at.slot].open = vec4(open * step(vec3(BARELY_OPEN), open), 0.0);
    let by_x = clear_of_solids(base + vec3(0.0, 0.5, 0.5) * h);
    let by_y = clear_of_solids(base + vec3(0.5, 0.0, 0.5) * h);
    let by_z = clear_of_solids(base + vec3(0.5, 0.5, 0.0) * h);
    cells[at.slot].solid = vec4(solid_velocity(by_x.body, base + vec3(0.0, 0.5, 0.5) * h).x, solid_velocity(by_y.body, base + vec3(0.5, 0.0, 0.5) * h).y, solid_velocity(by_z.body, base + vec3(0.5, 0.5, 0.0) * h).z, 0.0);
}

/// Whether a cell is one of the water's, for the pressure to keep its water from being
/// squeezed.
fn is_water(slot: u32) -> bool {
    if (slot == NONE) {
        return false;
    }
    return (u32(cells[slot].level.w) & IS_WATER) != 0u;
}

/// Whether a cell's middle is inside a wall of the vessel. There is no air there for water to
/// have a face against: what of such a cell's faces is open lets water through as it comes,
/// and neither holds it back nor draws it on.
fn in_a_wall(slot: u32) -> bool {
    return slot != NONE && min(cells[slot].walls[0].w, cells[slot].walls[1].w) < 0.0;
}

/// The axis the water's face lies most squarely across at a cell.
fn squarest_of(level: vec4<f32>) -> u32 {
    return u32(level.w) & 3u;
}

/// Whether a column of cells along an axis told how far under the water's face a cell is.
fn told_along(level: vec4<f32>, axis: u32) -> bool {
    return (u32(level.w) & (TOLD_ALONG << axis)) != 0u;
}

/// The water's face over a point: how far under it the point is, in cells, measured square
/// to the face, which way is deeper, and whether a column of cells told so or only the
/// cell's own fill. The point is as far under the face as its cell's middle is, and as much
/// further or nearer as the face's depth changes from there to it. Where there is no water
/// about, it is a cell over the face.
struct FaceOver {
    under: f32,
    deeper: vec3<f32>,
    told: bool,
}

fn face_over(q: vec3<f32>) -> FaceOver {
    let at = q * params.inv_cell;
    let c = vec3<i32>(floor(at));
    let slot = find(c);
    if (slot == NONE) {
        return FaceOver(-1.0, vec3(0.0), false);
    }
    let level = cells[slot].level;
    let axis = squarest_of(level);
    let off = at - vec3<f32>(c) - vec3(0.5);
    var deepening = vec3(0.0);
    deepening[axis] = select(1.0, -1.0, (u32(level.w) & AIR_IS_UP) != 0u);
    deepening[(axis + 1u) % 3u] = cells[slot].slope.x;
    deepening[(axis + 2u) % 3u] = cells[slot].slope.y;
    let steepness = length(deepening);
    return FaceOver((level[axis] + dot(deepening, off)) / steepness, deepening / steepness, told_along(level, axis));
}

/// How much a particle's parcel of water is the water's, to be moved as the pressure moves
/// the water, given how far under the water's face its middle is. Water above the water's
/// face has nothing to hold it up: a parcel whose middle is under the face is held wholly, as
/// the pressure holds up all the water under the face alike, and one whose middle is over it
/// is in flight by the share of the half of it that is over the face altogether. The parcel
/// is a spacing across.
fn held_by_water(under_face: f32) -> f32 {
    return clamp(1.0 + 2.0 * under_face * params.h / params.spacing, 0.0, 1.0);
}

/// A wall drags on what runs along it as a rough bed does on a river: by a share of the
/// water's dynamic pressure, which slows the water over it by that share of its speed squared
/// over its depth, taken here in the closed form that can never turn it round. A particle
/// against the wall stands for the water between the wall and the water's face, half a
/// spacing under it and as much as it is under the face over it, and for no more than the
/// spacing of water that is its own: what is deeper is other particles' to be slowed for.
fn dragged_by_wall(v: vec3<f32>, under_face: f32) -> vec3<f32> {
    let depth = clamp(0.5 + under_face * params.h / params.spacing, SHALLOWEST, 1.0);
    return v / (1.0 + params.wall_friction * length(v) / depth);
}

/// How far past the middle of a cell of water its face lies, toward a cell that is not
/// water, in cells: both cells' middles are so far under or over the face, measured along
/// the axis the face lies most squarely across, and the face crosses between them where that
/// comes to nothing.
fn face_past(water: u32, beyond: u32) -> f32 {
    let axis = squarest_of(cells[water].level);
    let under_it = cells[water].level[axis];
    var over_it = 1.0;
    if (beyond != NONE) {
        over_it = -cells[beyond].level[axis];
    }
    return clamp(under_it / max(under_it + over_it, 1e-6), NEAREST_FACE, 1.0);
}

/// How much of a cell's reach is taken up, by water or by solids; a cell no particle is near
/// is all taken up if its middle is inside a solid, and else empty.
fn taken_up(slot: u32, c: vec3<i32>) -> f32 {
    if (slot == NONE) {
        return select(0.0, 1.0, clear_of_solids((vec3<f32>(c) + vec3(0.5)) * params.h).by < 0.0);
    }
    return cells[slot].filled.x;
}

/// How much of a cell's reach is water.
fn water_in(slot: u32) -> f32 {
    if (slot == NONE) {
        return 0.0;
    }
    return cells[slot].flow.w;
}

/// The column of five cells about a cell along an axis, from the origin's end on: where each
/// is kept, how much of each is taken up, and which way along the axis the air lies.
struct Column {
    cells: array<u32, 5>,
    taken: array<f32, 5>,
    air_is_up: bool,
}

fn column_about(at: Visit, axis: u32) -> Column {
    var e = vec3(0);
    e[axis] = 1;
    var column: Column;
    for (var k = 0u; k < 5u; k++) {
        let c = at.c + (i32(k) - 2) * e;
        column.cells[k] = select(find(c), at.slot, k == 2u);
        column.taken[k] = taken_up(column.cells[k], c);
    }
    let rise = column.taken[3] - column.taken[1];
    // deep in the water nothing rises either way but by chance, and the face is taken to lie
    // as it does over water at rest, across the way things fall
    let weight = vessel_gravity((vec3<f32>(at.c) + vec3(0.5)) * params.h);
    column.air_is_up = select(rise <= 0.0, weight[axis] < 0.0, abs(rise) < BY_CHANCE);
    return column;
}

/// Whether a cell is the water's: its middle is under the water's face, and what the solids
/// leave of it is half full.
fn under_the_face(slot: u32, depth: f32) -> bool {
    return depth >= 0.0 && cells[slot].flow.w >= HALF_FULL * (1.0 - cells[slot].filled.w);
}

/// Find the cells round a cell, and how far under the water's face its middle is: along each
/// axis, the column of five cells about it says how high the water stands in that column, so
/// long as the column stands on water or on a solid. Where it does not, as in a drop or a
/// sheet of water in the air, the cell's own fill has to say.
///
/// What is taken up of the cells of a column, by water and by solids alike, sums to how high
/// the water stands only if every solid in it is under the water's face. Ground that rises
/// out of the water and a wall beside it are not: the cells with such a column are listed as
/// by a shore, for `level_shores` to put right.
@compute @workgroup_size(64)
fn level(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    var link: Link;
    for (var face = 0u; face < 6u; face++) {
        link.to[face] = find(across(at.c, face));
        link.counts[face] = 0.0;
    }
    links[at.slot] = link;
    let me = cells[at.slot];
    var depths = vec3(0.0);
    var steepest = -1.0;
    var squarest = 0u;
    var air_is_up = true;
    var told = 0u;
    var by_a_shore = false;
    let weight = vessel_gravity((vec3<f32>(at.c) + vec3(0.5)) * params.h);
    let falling = weight / max(length(weight), 1e-9);
    for (var axis = 0u; axis < 3u; axis++) {
        let column = column_about(at, axis);
        let taken = column.taken;
        if (select(taken[4], taken[0], column.air_is_up) >= FULL_BELOW) {
            depths[axis] = taken[0] + taken[1] + taken[2] + taken[3] + taken[4] - 2.5;
            told |= TOLD_ALONG << axis;
            if (!by_a_shore && face_by_a_solid(me, depths[axis])) {
                let face = FaceAcross(axis, select(1.0, -1.0, column.air_is_up), depths[axis], vec2(0.0));
                by_a_shore = solids_over(at.c, column.cells, face);
            }
        } else {
            depths[axis] = depth_filling(min(me.flow.w / max(1.0 - me.filled.w, LEAST_BESIDE), 1.0));
        }
        let steepness = abs(taken[3] - taken[1]) + BY_CHANCE * abs(falling[axis]);
        if (steepness > steepest) {
            steepest = steepness;
            squarest = axis;
            air_is_up = column.air_is_up;
        }
    }
    var marks = squarest | told;
    if (air_is_up) {
        marks |= AIR_IS_UP;
    }
    if (under_the_face(at.slot, depths[squarest])) {
        marks |= IS_WATER;
    }
    cells[at.slot].level = vec4(depths, f32(marks));
    cells[at.slot].open.w = 0.0;
    if (by_a_shore) {
        grid_list[arrayLength(&grid_list) / 2u + atomicAdd(&grid_counters[1], 1u)] = at.slot;
    }
}

/// Whether a column that told a cell's depth has the water's face in it, and not only water
/// or only air from end to end, where the cell has solids within its reach.
fn face_by_a_solid(cell: Cell, depth: f32) -> bool {
    return cell.filled.w > 0.0 && abs(depth) < FACE_IN_THE_COLUMN;
}

/// Whether any solid in a column of cells stands over the water's face, anywhere the lines
/// that `share_along` sums run: what the column's cells have taken up then sums to something
/// other than how high the water stands.
fn solids_over(c: vec3<i32>, column: array<u32, 5>, across: FaceAcross) -> bool {
    for (var k = 0u; k < 5u; k++) {
        let slot = column[k];
        if (slot == NONE || cells[slot].filled.w == 0.0) {
            continue;
        }
        var e = vec3(0);
        e[across.axis] = i32(k) - 2;
        let solids = solids_of(slot, c + e);
        let under_this = across.under + across.deeper * (f32(k) - 2.0);
        for (var s = 0u; s < 3u; s++) {
            if (solids[s].clear >= SOLID_REACH) {
                continue;
            }
            let n = solids[s].normal;
            let clearing = -across.deeper * n[across.axis];
            if (clearing < RUNS_ALONG) {
                return true;
            }
            let furthest = LINE_AT[1] * (abs(n[(across.axis + 1u) % 3u]) + abs(n[(across.axis + 2u) % 3u]));
            if (min((furthest - solids[s].clear) / clearing, 1.0) > under_this) {
                return true;
            }
        }
    }
    return false;
}

/// Put the water's face, for the cells by a shore, where the water of each column that told
/// of it fills what the solids leave under it.
@compute @workgroup_size(64)
fn level_shores(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit_shore(id.x);
    if (!at.there) {
        return;
    }
    let me = cells[at.slot];
    var level = me.level;
    for (var axis = 0u; axis < 3u; axis++) {
        if (!told_along(level, axis) || !face_by_a_solid(me, level[axis])) {
            continue;
        }
        let column = column_about(at, axis);
        let face = FaceAcross(axis, select(1.0, -1.0, column.air_is_up), level[axis], vec2(0.0));
        level[axis] = face_holding(at.c, column.cells, face);
    }
    var marks = u32(level.w) & ~IS_WATER;
    if (under_the_face(at.slot, level[squarest_of(level)])) {
        marks |= IS_WATER;
    }
    cells[at.slot].level = vec4(level.xyz, f32(marks));
}

/// Which of a cell's faces the water flows through, those with water to either side of them
/// that no solid covers altogether; and how the water's face slopes at the cell, which the
/// columns beside the one that told its depth tell.
@compute @workgroup_size(64)
fn wetted(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let wet = is_water(at.slot);
    var faces = 0u;
    for (var axis = 0u; axis < 3u; axis++) {
        if (cells[at.slot].open[axis] >= WHOLLY_OPEN && (wet || is_water(links[at.slot].to[2u * axis]))) {
            faces |= 1u << axis;
        }
    }
    cells[at.slot].solid.w = f32(faces);
    let level = cells[at.slot].level;
    let squarest = squarest_of(level);
    var slopes = vec2(0.0);
    for (var n = 0u; n < 2u; n++) {
        let beside = (squarest + 1u + n) % 3u;
        let lo = links[at.slot].to[2u * beside];
        let hi = links[at.slot].to[2u * beside + 1u];
        var span = 0.0;
        var one_side = level[squarest];
        var other_side = level[squarest];
        if (lo != NONE && told_along(cells[lo].level, squarest)) {
            one_side = cells[lo].level[squarest];
            span += 1.0;
        }
        if (hi != NONE && told_along(cells[hi].level, squarest)) {
            other_side = cells[hi].level[squarest];
            span += 1.0;
        }
        slopes[n] = (other_side - one_side) / max(span, 1.0);
    }
    cells[at.slot].slope = vec4(slopes, 0.0, 0.0);
}

/// What one of the three fields the smoothing goes through holds at a cell.
fn smoothing(slot: u32, stage: u32) -> vec3<f32> {
    if (stage == 0u) {
        return cells[slot].flow.xyz;
    }
    if (stage == 1u) {
        return cells[slot].shift.xyz;
    }
    return cells[slot].pressure.yzw;
}

/// How unevenly a field lies across a cell's faces: along each axis, how far the two like
/// faces either side are from lying in a line with the face between them, where the water
/// flows through all three. A field that changes evenly, as the flow of water turning as one
/// or sheared evenly does, is not uneven at all; one that swaps its sign from each face to
/// the next, the finest thing the faces can hold, is as uneven as it is large.
fn unevenness(slot: u32, stage: u32) -> vec3<f32> {
    let link = links[slot];
    let mine = smoothing(slot, stage);
    let faces = u32(cells[slot].solid.w);
    var out = vec3(0.0);
    for (var along = 0u; along < 3u; along++) {
        let lo = link.to[2u * along];
        let hi = link.to[2u * along + 1u];
        if (lo == NONE || hi == NONE) {
            continue;
        }
        let all = faces & u32(cells[lo].solid.w) & u32(cells[hi].solid.w);
        let bent = (smoothing(lo, stage) + smoothing(hi, stage) - 2.0 * mine) / 12.0;
        for (var axis = 0u; axis < 3u; axis++) {
            if (((all >> axis) & 1u) == 1u) {
                out[axis] += bent[axis];
            }
        }
    }
    return out;
}

/// Take out of the flow what the grid cannot tell from chance. How particles happen to lie
/// among the cells jolts the water a little at every step, mostly a cell or two across, and
/// water whose motion the particles carry without loss never loses that either, so it has to
/// be taken out; in water itself such motion breaks up into ever smaller eddies and ends as
/// heat. Taking the unevenness of the unevenness of the unevenness leaves a wave its height
/// but for the sixth power of how few cells it is long: all of what swaps its sign from face
/// to face goes in a step, a ten-thousandth of a wave eight cells long, and of one
/// twenty-five long a ten-millionth.
@compute @workgroup_size(64)
fn smooth_once(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    cells[at.slot].shift = vec4(unevenness(at.slot, 0u), 0.0);
}

@compute @workgroup_size(64)
fn smooth_twice(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    cells[at.slot].pressure = vec4(cells[at.slot].pressure.x, unevenness(at.slot, 1u));
}

@compute @workgroup_size(64)
fn smooth_thrice(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    cells[at.slot].flow = vec4(cells[at.slot].flow.xyz + unevenness(at.slot, 2u), cells[at.slot].flow.w);
}

fn across(c: vec3<i32>, face: u32) -> vec3<i32> {
    var offset = vec3(0);
    offset[face / 2u] = select(-1, 1, (face & 1u) == 1u);
    return c + offset;
}

/// Write down what a cell's pressure has to answer for: what its faces let in, and what the
/// solids covering them push in.
@compute @workgroup_size(64)
fn system(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let me = cells[at.slot];
    let wet = is_water(at.slot);
    var link = links[at.slot];
    var divided_by = 0.0;
    var shift_divided_by = 0.0;
    var let_in = 0.0;
    var opened = 0.0;
    for (var face = 0u; face < 6u; face++) {
        let axis = face / 2u;
        let other = link.to[face];
        var open = me.open[axis];
        var through = me.flow[axis];
        var pushed = me.solid[axis];
        var inward = 1.0;
        if ((face & 1u) == 1u) {
            inward = -1.0;
            open = 1.0;
            through = 0.0;
            pushed = 0.0;
            if (other != NONE) {
                open = cells[other].open[axis];
                through = cells[other].flow[axis];
                pushed = cells[other].solid[axis];
            }
        }
        let_in += inward * (open * through + (1.0 - open) * pushed);
        opened += open / 6.0;
        if (is_water(other)) {
            link.counts[face] = open;
            divided_by += open;
            shift_divided_by += open;
        } else if (wet && !in_a_wall(other)) {
            divided_by += open / face_past(at.slot, other);
        }
    }
    links[at.slot] = link;
    var water = 0.0;
    if (wet && divided_by > 1e-6) {
        water = 1.0;
    }
    // a cell the solids leave little of has as little water to be shifted
    cells[at.slot].filled.w = over_its_fill(at.c, at.slot, link) * opened;
    cells[at.slot].slope.w = shift_divided_by;
    cells[at.slot].shift.w = water * max(let_in, 0.0) * params.inv_cell;
    cells[at.slot].pressure = vec4(me.pressure.x * water, let_in, divided_by, water);
}

/// How much of what a cell holds over or under its fill is to be shifted out of it or into it
/// in a step. A cell is full when its water and the solids within its reach take up all of
/// that reach that is under the water's face. How full it reads is a sum over the few
/// particles within its reach, good to a few parts in a hundred, so what it reads within that
/// of full may be chance. What is beyond chance is too much water, and is shifted out. What
/// is within it is shifted so slowly that only what stays when the chances have evened out
/// gets shifted, and only where it is known how much of the cell's reach is under the face:
/// where a column told how far under the face the cell is, or the cell is so deep in the
/// water that all of it is. Anywhere else, as in a drop or a sheet of water in the air, a cell
/// can only be told to be too full.
///
/// What is shifted is shifted between the water's cells and never out through the water's
/// face: the face is where it is for how much water there is under it, so putting the water
/// right is a matter of moving it about under the face, and water let out through one side
/// of a body of it and not another would move the body.
fn over_its_fill(c: vec3<i32>, slot: u32, link: Link) -> f32 {
    let level = cells[slot].level;
    let axis = squarest_of(level);
    var known = told_along(level, axis);
    if (!known) {
        var e = vec3(0);
        e[axis] = 1;
        let before = taken_up(link.to[2u * axis], c - e);
        let after = taken_up(link.to[2u * axis + 1u], c + e);
        known = min(before, after) >= DEEP;
    }
    var solids = array<Solid, 3>(Solid(WALL_REACH, vec3(0.0)), Solid(WALL_REACH, vec3(0.0)), Solid(WALL_REACH, vec3(0.0)));
    if (cells[slot].filled.w > 0.0) {
        solids = solids_of(slot, c);
    }
    var face = FaceAcross(axis, select(1.0, -1.0, (u32(level.w) & AIR_IS_UP) != 0u), FAR_UNDER, vec2(0.0));
    if (told_along(level, axis)) {
        face.under = level[axis];
        face.slopes = cells[slot].slope.xy;
    }
    let over = cells[slot].flow.w - waters_share(face, solids);
    var shifted = SETTLED_IN_A_STEP * max(over - BY_CHANCE, 0.0);
    if (known) {
        shifted += EVENED_OUT_IN_A_STEP * clamp(over, -BY_CHANCE, BY_CHANCE);
    }
    return shifted;
}

/// The water's face at a cell, as a column of cells along the axis it lies most squarely
/// across told of it: which way along the axis is deeper, how far under the face the cell's
/// middle is, and how that changes along the other two axes, all in cells.
struct FaceAcross {
    axis: u32,
    deeper: f32,
    under: f32,
    slopes: vec2<f32>,
}

/// The stretch of a line through a cell that is the water's: along the axis the water's face
/// lies across, from the solids under the water up to the face, or to the solids over the
/// water where those come first, in cells from the cell's middle toward the air. The line
/// runs `beside` the cell's middle, and the solids are taken as flat.
struct Stretch {
    bottom: f32,
    top: f32,
}

fn waters_stretch(face: FaceAcross, beside: vec2<f32>, solids: array<Solid, 3>) -> Stretch {
    var out = Stretch(-FAR_UNDER - 1.0, face.under + dot(face.slopes, beside));
    for (var k = 0u; k < 3u; k++) {
        if (solids[k].clear >= SOLID_REACH) {
            continue;
        }
        let n = solids[k].normal;
        let clear = solids[k].clear + n[(face.axis + 1u) % 3u] * beside.x + n[(face.axis + 2u) % 3u] * beside.y;
        // how fast the line comes clear of the solid toward the air
        let clearing = -face.deeper * n[face.axis];
        if (clearing >= RUNS_ALONG) {
            out.bottom = max(out.bottom, -clear / clearing);
        } else if (clearing <= -RUNS_ALONG) {
            out.top = min(out.top, -clear / clearing);
        }
    }
    return out;
}

/// The share of a cell's reach that is the water's to fill: what is under the water's face
/// and clear of the solids. The lines through the reach are summed as `under_sloping` sums
/// them. Ground that rises through the water's face takes none of the water's share where it
/// stands over the face, which is what a share of the whole reach taken off for it would get
/// wrong all along a shore.
fn waters_share(face: FaceAcross, solids: array<Solid, 3>) -> f32 {
    return share_along(face, solids) * share_beside(face.axis, solids);
}

/// What the solids that lines along an axis run along leave of a cell's reach. Such a solid
/// cuts a line off whole or not at all, which a sum over a few lines tells badly: it takes
/// its share off the reach at every height instead.
fn share_beside(axis: u32, solids: array<Solid, 3>) -> f32 {
    var beside = 1.0;
    for (var k = 0u; k < 3u; k++) {
        if (solids[k].clear < SOLID_REACH && abs(solids[k].normal[axis]) < RUNS_ALONG) {
            beside *= under_tilted(solids[k].clear, solids[k].normal);
        }
    }
    return beside;
}

/// The share of a cell's reach that is under the water's face and clear of the solids its
/// lines run into. How much of a line is the water's changes across the reach no faster than
/// the reach thins out, between the few places where a line comes to an end of the reach or
/// of the water, so Gauss's two points to each half of the reach sum it as well as more do.
fn share_along(face: FaceAcross, solids: array<Solid, 3>) -> f32 {
    var total = 0.0;
    for (var l = 0u; l < LINES; l++) {
        let line = waters_stretch(face, line_beside(l), solids);
        total += line_weight(l) * max(under(line.top) - under(line.bottom), 0.0);
    }
    return total;
}

/// Where beside a cell's middle the lines that sum its reach run, and what each counts for.
fn line_beside(l: u32) -> vec2<f32> {
    let i = l / 4u;
    let j = l % 4u;
    return vec2(select(LINE_AT[i % 2u], -LINE_AT[i % 2u], i >= 2u), select(LINE_AT[j % 2u], -LINE_AT[j % 2u], j >= 2u));
}

fn line_weight(l: u32) -> f32 {
    return LINE_WEIGHT[(l / 4u) % 2u] * LINE_WEIGHT[l % 2u];
}

/// How far under the water's face a cell's middle is for the column of five cells about it to
/// hold the water it does, the face taken as square to the column: each cell of the column
/// holds what the solids within its own reach leave of what of it is under the face. More
/// water puts the face higher and never lower, and by as much more as the reach is wide where
/// the face cuts it, so Newton's steps close in on the height, kept between the heights that
/// were found to hold too little and too much.
fn face_holding(c: vec3<i32>, column: array<u32, 5>, across: FaceAcross) -> f32 {
    var water = 0.0;
    var left = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    var beside = array<f32, 5>(1.0, 1.0, 1.0, 1.0, 1.0);
    // each line's water with the face far over it: the share under where it comes clear of
    // the solids under the water, and how high the solids over the water let it reach
    var lines: array<array<Stretch, LINES>, 5>;
    let brimming = FaceAcross(across.axis, across.deeper, FAR_UNDER, vec2(0.0));
    for (var k = 0u; k < 5u; k++) {
        let slot = column[k];
        if (slot == NONE) {
            continue;
        }
        water += cells[slot].flow.w;
        left[k] = 1.0 - cells[slot].filled.w;
        if (left[k] < 1.0) {
            var e = vec3(0);
            e[across.axis] = i32(k) - 2;
            let solids = solids_of(slot, c + e);
            beside[k] = share_beside(across.axis, solids);
            for (var l = 0u; l < LINES; l++) {
                let line = waters_stretch(brimming, line_beside(l), solids);
                lines[k][l] = Stretch(under(line.bottom), line.top);
            }
        }
    }
    var lowest = -FAR_UNDER - 1.0;
    var highest = FAR_UNDER + 1.0;
    var under_face = clamp(across.under, lowest, highest);
    for (var closer = 0u; closer < NEWTON_STEPS; closer++) {
        var held = 0.0;
        var widening = 0.0;
        for (var k = 0u; k < 5u; k++) {
            if (column[k] == NONE) {
                continue;
            }
            // how far toward the air the cell is from the middle one
            let under_this = under_face + across.deeper * (f32(k) - 2.0);
            if (under_this >= 1.0) {
                held += left[k];
            } else if (under_this > -1.0 && left[k] == 1.0) {
                held += under(under_this);
                widening += tent(under_this);
            } else if (under_this > -1.0) {
                let up_to_face = under(under_this);
                for (var l = 0u; l < LINES; l++) {
                    let line = lines[k][l];
                    let faced = under_this < line.top;
                    let wet = select(under(line.top), up_to_face, faced) - line.bottom;
                    if (wet > 0.0) {
                        held += beside[k] * line_weight(l) * wet;
                        widening += select(0.0, beside[k] * line_weight(l) * tent(under_this), faced);
                    }
                }
            }
        }
        if (held > water) {
            highest = under_face;
        } else {
            lowest = under_face;
        }
        var next = under_face + (water - held) / max(widening, LEAST_BESIDE);
        if (next <= lowest || next >= highest) {
            next = 0.5 * (lowest + highest);
        }
        under_face = next;
    }
    return under_face;
}

/// One sweep of the pressure over the cells of one colour of a checkerboard, each from the
/// cells of the other colour round it. Water holds together under no more pull than the air
/// presses it with.
fn relax(thread: u32, colour: i32) {
    let at = visit(thread);
    if (!at.there || ((at.c.x + at.c.y + at.c.z) & 1) != colour) {
        return;
    }
    let me = cells[at.slot].pressure;
    if (me.w == 0.0) {
        return;
    }
    var sum = me.y;
    for (var face = 0u; face < 6u; face++) {
        let counts = links[at.slot].counts[face];
        if (counts > 0.0) {
            let other = cells[links[at.slot].to[face]].pressure;
            sum += counts * other.x * other.w;
        }
    }
    cells[at.slot].pressure.x = max(mix(me.x, sum / me.z, OVER), -params.hold);
}

@compute @workgroup_size(64)
fn relax_red(@builtin(global_invocation_id) id: vec3<u32>) {
    relax(id.x, 0);
}

@compute @workgroup_size(64)
fn relax_black(@builtin(global_invocation_id) id: vec3<u32>) {
    relax(id.x, 1);
}

/// The same sweep for the push that shifts the water to where it is as dense as water is:
/// particles carried by a flow that the grid keeps from being squeezed are still, in the
/// end, a little squeezed and a little spread by how they are carried, and water that has
/// spread never comes together again by itself. What a cell holds over or under its fill is
/// what this push has to answer for, as what the faces let in is what the pressure has to;
/// it moves the particles and leaves their velocities alone (Kugelstadt et al. 2019).
fn settle(thread: u32, colour: i32) {
    let at = visit(thread);
    if (!at.there || ((at.c.x + at.c.y + at.c.z) & 1) != colour) {
        return;
    }
    let me = cells[at.slot].pressure;
    if (me.w == 0.0) {
        return;
    }
    var sum = cells[at.slot].filled.w;
    for (var face = 0u; face < 6u; face++) {
        let counts = links[at.slot].counts[face];
        if (counts > 0.0) {
            let other = links[at.slot].to[face];
            sum += counts * cells[other].open.w * cells[other].pressure.w;
        }
    }
    let divided_by = cells[at.slot].slope.w;
    if (divided_by > 1e-6) {
        cells[at.slot].open.w = mix(cells[at.slot].open.w, sum / divided_by, OVER);
    }
}

@compute @workgroup_size(64)
fn settle_red(@builtin(global_invocation_id) id: vec3<u32>) {
    settle(id.x, 0);
}

@compute @workgroup_size(64)
fn settle_black(@builtin(global_invocation_id) id: vec3<u32>) {
    settle(id.x, 1);
}

/// Let the pressure have its way with the faces a cell owns. A face a solid covers altogether
/// is none of the water's, which slips along a wall and is kept out of it particle by
/// particle: what the water does beside such a face is for the faces round it to say.
@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let me = cells[at.slot];
    var flow = me.flow.xyz;
    var shift = vec3(0.0);
    var settled = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let other = links[at.slot].to[2u * axis];
        var theirs = vec4(0.0);
        var their_push = 0.0;
        if (other != NONE) {
            theirs = cells[other].pressure;
            their_push = cells[other].open.w;
        }
        if ((me.pressure.w == 0.0 && theirs.w == 0.0) || me.open[axis] <= 0.0) {
            continue;
        }
        if ((me.pressure.w == 0.0 && in_a_wall(at.slot)) || (theirs.w == 0.0 && in_a_wall(other))) {
            continue;
        }
        settled += f32(1u << axis);
        if (me.pressure.w > 0.0 && theirs.w > 0.0) {
            flow[axis] -= me.pressure.x - theirs.x;
            shift[axis] = their_push - me.open.w;
        } else if (me.pressure.w > 0.0) {
            flow[axis] -= me.pressure.x / face_past(at.slot, other);
        } else {
            flow[axis] += theirs.x / face_past(other, at.slot);
        }
    }
    cells[at.slot].shift = vec4(shift, me.shift.w);
    cells[at.slot].flow = vec4(flow, me.flow.w);
    cells[at.slot].solid.w = settled;
}

/// Tell the water's flow on to the faces round the water's that the pressure has not set, each
/// taking the mean of the like faces next to it that have a flow: so that what a particle
/// reads at the edge of the water is one field, the same for every particle that reads it,
/// and handing motion back and forth between particles and faces adds nothing to it. A face
/// the first telling reaches is marked with its bit eight times over, one the second reaches
/// sixty-four times.
fn tell_on(thread: u32, layer: u32) {
    let at = visit(thread);
    if (!at.there) {
        return;
    }
    let link = links[at.slot];
    var marks = u32(cells[at.slot].solid.w);
    let has = marks | (marks >> 3u) | (marks >> 6u);
    // what a face may be told by: a set face, or one an earlier telling reached
    let tellers = select(0x3fu, 0x7u, layer == 0u);
    for (var axis = 0u; axis < 3u; axis++) {
        if (((has >> axis) & 1u) == 1u) {
            continue;
        }
        var total = 0.0;
        var count = 0.0;
        for (var face = 0u; face < 6u; face++) {
            let other = link.to[face];
            if (other == NONE) {
                continue;
            }
            let theirs = u32(cells[other].solid.w) & tellers;
            if ((((theirs | (theirs >> 3u)) >> axis) & 1u) == 1u) {
                total += cells[other].flow[axis];
                count += 1.0;
            }
        }
        if (count > 0.0) {
            cells[at.slot].flow[axis] = total / count;
            marks |= (select(64u, 8u, layer == 0u)) << axis;
        }
    }
    cells[at.slot].solid.w = f32(marks);
}

@compute @workgroup_size(64)
fn tell_on_once(@builtin(global_invocation_id) id: vec3<u32>) {
    tell_on(id.x, 0u);
}

@compute @workgroup_size(64)
fn tell_on_twice(@builtin(global_invocation_id) id: vec3<u32>) {
    tell_on(id.x, 1u);
}

/// What the water's flow is told on to a face that a solid covers altogether is how the water
/// slips along the solid there: the flow beside it with what of it goes into the solid, or
/// comes out of it, taken out, which is what no water does.
@compute @workgroup_size(64)
fn slip(@builtin(global_invocation_id) id: vec3<u32>) {
    let at = visit(id.x);
    if (!at.there) {
        return;
    }
    let me = cells[at.slot];
    let marks = u32(me.solid.w);
    let told = (marks >> 3u) | (marks >> 6u);
    var flow = me.flow.xyz;
    for (var axis = 0u; axis < 3u; axis++) {
        if (me.open[axis] > 0.0 || ((told >> axis) & 1u) == 0u) {
            continue;
        }
        var half = vec3(0.5);
        half[axis] = 0.0;
        let middle = (vec3<f32>(at.c) + half) * params.h;
        let solids = solids_round(middle);
        var nearest = 0u;
        for (var k = 1u; k < 3u; k++) {
            if (solids[k].clear < solids[nearest].clear) {
                nearest = k;
            }
        }
        let n = solids[nearest].normal;
        var solid = vec3(0.0);
        if (nearest == 2u) {
            solid = solid_velocity(clear_of_solids(middle).body, middle);
        }
        flow[axis] -= dot(me.flow.xyz - solid, n) * n[axis];
    }
    cells[at.slot].flow = vec4(flow, me.flow.w);
}

/// Keep the particle out of solid bodies (a tunnelling guard).
fn exclude_from_bodies(q_in: vec3<f32>) -> vec3<f32> {
    var q = q_in;
    for (var b = 0u; b < params.body_count; b++) {
        if (bodies.items[b].position.w <= 0.0) {
            continue;
        }
        let bulk = bulk_of(b);
        let keep_out = bulk.w + 0.3 * params.spacing;
        q = kept_out_of(q, bulk.xyz, keep_out);
        for (var opening = 0u; opening < VESSEL_OPENINGS; opening++) {
            // what of the body has gone in at an opening of the vessel is beyond it
            let beyond = vessel_through(bulk.xyz, opening, bulk.w);
            if (beyond.there) {
                q = kept_out_of(q, beyond.p, keep_out);
            }
        }
    }
    return q;
}

/// A point put `keep_out` clear of a centre it is nearer than that to.
fn kept_out_of(q: vec3<f32>, centre: vec3<f32>, keep_out: f32) -> vec3<f32> {
    let rc = q - centre;
    let r2 = dot(rc, rc);
    if (r2 >= keep_out * keep_out) {
        return q;
    }
    let len = sqrt(r2);
    var n = vec3(0.0, 1.0, 0.0);
    if (len > 1e-9) {
        n = rc / len;
    }
    return q + n * (keep_out - len);
}

/// The safety clamp on speed, on the speed among the stars: nothing the frame does can bring
/// it down on water that is only at rest.
fn clamp_speed(p: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let rest = vessel_star_velocity(p);
    let among_stars = v - rest;
    let s2 = dot(among_stars, among_stars);
    if (s2 > params.max_speed * params.max_speed) {
        return rest + among_stars * (params.max_speed / sqrt(s2));
    }
    return v;
}

/// The eight faces round a particle that carry one part of its velocity: what flows through
/// those that have a flow, which those are, and which of them the pressure set, the rest
/// having been told theirs by the set ones next to them. What flows through a face is known
/// where the pressure set it, and where a solid that faces along the face's axis covers it
/// and lets through only its own motion; what the rest were told is a guess at it.
struct Faces {
    flows: array<f32, 8>,
    shifts: array<f32, 8>,
    settled: u32,
    flowing: u32,
    known: u32,
    first: vec3<i32>,
    within: vec3<f32>,
}

fn faces_round(x: vec3<f32>, axis: u32) -> Faces {
    var half = vec3(0.5);
    half[axis] = 0.0;
    let rel = x * params.inv_cell - half;
    var out: Faces;
    out.first = vec3<i32>(floor(rel));
    out.within = rel - vec3<f32>(out.first);
    for (var n = 0u; n < 8u; n++) {
        let slot = find(out.first + vec3<i32>(vec3(n & 1u, (n >> 1u) & 1u, n >> 2u)));
        if (slot == NONE) {
            continue;
        }
        let cell = cells[slot];
        let marks = u32(cell.solid.w);
        if ((((marks | (marks >> 3u) | (marks >> 6u)) >> axis) & 1u) == 0u) {
            continue;
        }
        out.flows[n] = cell.flow[axis];
        out.flowing |= 1u << n;
        let nearest = select(cell.walls[1], cell.walls[0], cell.walls[0].w <= cell.walls[1].w);
        if (cell.open[axis] <= 0.0 && abs(nearest[axis]) >= FACES_ALONG) {
            out.known |= 1u << n;
        }
        if (((marks >> axis) & 1u) == 1u) {
            out.shifts[n] = cell.shift[axis];
            out.settled |= 1u << n;
            out.known |= 1u << n;
        }
    }
    return out;
}

/// What stands in for a face that is not `among` those to be read: the mean of the faces
/// nearest it among the eight that are, so that what is read changes across the particle no
/// faster than they differ, however few they are, and as they do along the ways in which
/// they are told.
fn stand_in(faces: Faces, n: u32, among: u32) -> f32 {
    var nearest = 4u;
    var total = 0.0;
    var count = 0.0;
    for (var m = 0u; m < 8u; m++) {
        if ((among & (1u << m)) == 0u) {
            continue;
        }
        let apart = countOneBits(n ^ m);
        if (apart < nearest) {
            nearest = apart;
            total = 0.0;
            count = 0.0;
        }
        if (apart == nearest) {
            total += faces.flows[m];
            count += 1.0;
        }
    }
    return total / max(count, 1.0);
}

/// What a particle takes back from the faces round it of one part of its velocity: the part
/// itself, how it changes across the particle, how those changes change in turn, and how far
/// the particle is shifted along this axis, in cells.
///
/// How the part changes across the particle is taken from the faces whose flow is known
/// alone. Where the pressure holds no sway, as in a sheet of water too thin for the cells,
/// the flow still has in it the fall of a step that the walls have yet to stop, while the
/// faces the walls cover have not: a change from the one to the other is no motion of the
/// water's, and a particle that carried it would hand it back as one, more of it each step.
struct Taken {
    part: f32,
    across: vec3<f32>,
    twists: vec4<f32>,
    shifted: f32,
}

fn take(faces: Faces) -> Taken {
    let f = faces.within;
    var out = Taken(0.0, vec3(0.0), vec4(0.0), 0.0);
    for (var n = 0u; n < 8u; n++) {
        let o = vec3(n & 1u, (n >> 1u) & 1u, n >> 2u);
        let wx = select(1.0 - f.x, f.x, o.x == 1u);
        let wy = select(1.0 - f.y, f.y, o.y == 1u);
        let wz = select(1.0 - f.z, f.z, o.z == 1u);
        let way = vec3<f32>(o) * 2.0 - 1.0;
        var flow = faces.flows[n];
        if ((faces.flowing & (1u << n)) == 0u) {
            flow = stand_in(faces, n, faces.flowing);
        }
        out.shifted += wx * wy * wz * faces.shifts[n];
        out.part += wx * wy * wz * flow;
        if ((faces.known & (1u << n)) == 0u) {
            flow = stand_in(faces, n, faces.known);
        }
        out.across += vec3(way.x * wy * wz, wx * way.y * wz, wx * wy * way.z) * flow;
        out.twists += vec4(way.x * way.y * wz, wx * way.y * way.z, way.x * wy * way.z, way.x * way.y * way.z) * flow;
    }
    out.across *= params.inv_cell;
    out.twists *= vec4(vec3(params.inv_cell * params.inv_cell), params.inv_cell * params.inv_cell * params.inv_cell);
    return out;
}

/// What a particle reads at the cells' middles: the pressure it is under, how full of water
/// the place is and which way the water lies, and how fast the water is closing on it.
struct Read {
    pressure: f32,
    full: f32,
    toward_water: vec3<f32>,
    closing: f32,
}

fn read_at(x: vec3<f32>) -> Read {
    let rel = x * params.inv_cell - vec3(0.5);
    let first = vec3<i32>(floor(rel));
    let f = rel - vec3<f32>(first);
    var out = Read(0.0, 0.0, vec3(0.0), 0.0);
    var under = 0.0;
    for (var n = 0u; n < 8u; n++) {
        let o = vec3<i32>(vec3(n & 1u, (n >> 1u) & 1u, n >> 2u));
        let slot = find(first + o);
        if (slot == NONE) {
            continue;
        }
        let wx = select(1.0 - f.x, f.x, o.x == 1);
        let wy = select(1.0 - f.y, f.y, o.y == 1);
        let wz = select(1.0 - f.z, f.z, o.z == 1);
        let w = wx * wy * wz;
        let cell = cells[slot];
        let way = vec3<f32>(o) * 2.0 - 1.0;
        out.full += w * cell.flow.w;
        out.toward_water += vec3(way.x * wy * wz, wx * way.y * wz, wx * wy * way.z) * (cell.flow.w * params.inv_cell);
        out.closing += w * cell.shift.w;
        out.pressure += w * cell.pressure.x * cell.pressure.w;
        under += w * cell.pressure.w;
    }
    if (under > 0.0) {
        out.pressure /= under;
    }
    return out;
}

/// A particle's velocity with what of its water's motion goes into a wall it has met taken
/// out. The particle's water reaches as far as the wall, half a spacing from its middle, and
/// water that closes on a wall slows toward it as fast as it spreads along it, there being
/// nowhere else for it to go: so it is still where it meets the wall if the particle carries
/// as much spreading as its middle closes on the wall by, which is no motion into the wall
/// and takes nothing out. Whatever more the middle closes by is motion into the wall.
fn stopped_at_the_wall(v: vec3<f32>, across: array<vec3<f32>, 3>, wall: vec4<f32>) -> vec3<f32> {
    if (wall.w <= 0.0) {
        return v;
    }
    let n = wall.xyz;
    let toward = vec3(dot(across[0], n), dot(across[1], n), dot(across[2], n));
    let spreading = across[0].x + across[1].y + across[2].z - dot(toward, n);
    let into_the_wall = min(dot(v, n) + params.margin * max(spreading, 0.0), 0.0);
    return v - into_the_wall * n;
}

/// What a step adds to the motion of water lying on a wall in a sheet too thin for the cells'
/// pressure to hold: its weight presses it against the wall, and where its face is not level
/// with the wall that weight runs it out along the wall toward where it lies thinner, as the
/// pressure does with water deep enough for it.
fn run_out(q: vec3<f32>, wall: vec4<f32>, face: FaceOver) -> vec3<f32> {
    if (wall.w <= 0.0 || !face.told) {
        return vec3(0.0);
    }
    let pressing = max(-dot(vessel_gravity(q), wall.xyz), 0.0);
    let tilted = -face.deeper - wall.xyz;
    return pressing * (tilted - dot(tilted, wall.xyz) * wall.xyz) * params.dt;
}

/// A velocity with a share of what of it goes into the walls a particle has met taken out.
fn stopped(v_in: vec3<f32>, first: vec4<f32>, second: vec4<f32>, share: f32) -> vec3<f32> {
    var v = v_in;
    if (first.w > 0.0) {
        v -= share * min(dot(v, first.xyz), 0.0) * first.xyz;
    }
    if (second.w > 0.0) {
        v -= share * min(dot(v, second.xyz), 0.0) * second.xyz;
    }
    return v;
}

/// A step ends with each particle taking back from the grid the motion the pressure has left
/// it: all of it where half the faces round the particle are water's or more, as they are for
/// the water at a body of water's face, and less the fewer are, the rest being the particle's
/// own flight, which the pressure had no say in. A drop on its own flies on exactly as it flew.
///
/// What the pressure added it added all through the step, so the particle ends half the
/// step's worth of it further on than where its flight landed it; and the velocity that built
/// up was turned aside by the frame's turning as the fall it undid was: the velocity by the
/// frame's turning of twice that push, the place by a third of the step's worth of that. What
/// stands still among the stars moves through the frame as the frame turns, so how that
/// differs across the push is the turning of it.
///
/// A wall, which stands still in the vessel's frame, lets nothing through it: it stops what
/// of a particle's own flight goes into it, and what of the motion the particle takes from
/// the grid goes into it where the particle's water meets it. The pressure turns the water
/// along the walls only as well as the faces round a wall tell of it, and a face that a wall
/// leaves little of open tells badly.
@compute @workgroup_size(64)
fn transfer(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let q = pred_in[i].xyz;
    let flown = velocity[i].xyz;
    var v = flown;
    let round = array<Faces, 3>(faces_round(q, 0u), faces_round(q, 1u), faces_round(q, 2u));
    let face = face_over(q);
    let under_face = face.under;
    let own = stopped(flown, contact[2u * i], contact[2u * i + 1u], 1.0) + run_out(q, contact[2u * i], face);
    // only a particle with faces of the water's round it has any water to be held by
    var held = 0.0;
    if ((round[0].settled | round[1].settled | round[2].settled) != 0u) {
        held = held_by_water(under_face);
    }
    var shifted = vec3(0.0);
    var across = array<vec3<f32>, 3>(vec3(0.0), vec3(0.0), vec3(0.0));
    for (var axis = 0u; axis < 3u; axis++) {
        v[axis] = own[axis];
        if (round[axis].flowing == 0u) {
            affine[6u * i + 2u * axis] = vec4(0.0);
            affine[6u * i + 2u * axis + 1u] = vec4(0.0);
            continue;
        }
        let taken = take(round[axis]);
        let share = held;
        v[axis] = mix(own[axis], taken.part, share);
        across[axis] = taken.across * share;
        affine[6u * i + 2u * axis] = vec4(across[axis], 0.0);
        affine[6u * i + 2u * axis + 1u] = taken.twists * select(0.0, 1.0, round[axis].settled == ALL_EIGHT);
        shifted[axis] = taken.shifted * share * params.h;
    }
    let at = read_at(q);
    affine[6u * i].w = at.pressure;
    let landed = velocity_next[i].xyz;
    // what the walls stopped of the particle's flight they have moved it for already
    let pushed = 0.5 * (v - own) * params.dt;
    let turned = vessel_star_velocity(landed + 0.5 * (v - flown) * params.dt) - vessel_star_velocity(landed);
    // All of a parcel is water, so all of it belongs under the water's face, as all of it
    // belongs within the walls: what of a parcel the water holds stands over a face that a
    // column of cells told of settles under it, in its own time, as water that is too close
    // is shifted apart.
    if (face.told) {
        let standing_over = max(0.5 * params.spacing * params.inv_cell - under_face, 0.0);
        shifted += face.deeper * (EVENED_OUT_IN_A_STEP * held * standing_over * params.h);
    }
    let confined = vessel_confine(exclude_from_bodies(q + pushed + turned * (2.0 / 3.0 * params.dt) + shifted), params.margin);
    var first = contact[2u * i];
    var second = contact[2u * i + 1u];
    if (confined.first.w > 0.0) {
        second = first;
        first = confined.first;
        if (confined.second.w > 0.0) {
            second = confined.second;
        }
    }
    v = stopped_at_the_wall(v, across, first);
    v = stopped_at_the_wall(v, across, second);
    if (first.w > 0.0) {
        v = dragged_by_wall(v, under_face);
    }
    // The frame turns under the particle for as far as it is moved from where its flight
    // landed it. What pushes it there, the water or a wall, pushes as the frame turns, which
    // turns the push as well: twice the frame's turn over the way. What only shifts it, to
    // keep the water as dense as water is, leaves its motion among the stars alone: once.
    // Left out, water lying on the floor of a spinning vessel is left behind by it a little
    // more with every step.
    v += 2.0 * (vessel_star_velocity(confined.p) - vessel_star_velocity(landed));
    v -= vessel_star_velocity(landed + shifted) - vessel_star_velocity(landed);
    contact[2u * i] = first;
    contact[2u * i + 1u] = second;
    position[i] = vec4(confined.p, position[i].w);
    velocity[i] = vec4(clamp_speed(confined.p, v), velocity[i].w);
    pred_in[i].w = at.closing;
    pred_out[i] = vec4(at.toward_water, at.full);
}
