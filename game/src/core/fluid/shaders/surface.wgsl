// Isosurface of the particle cloud with naive surface nets on a grid fixed to the vessel. The
// grid is cut into blocks of BLOCK³ cells; every particle marks the blocks its splat reaches in
// a hash table, and one workgroup per marked block splats the scalar field at the block's
// corners into workgroup memory and places one vertex in every cell the surface crosses. The
// crossed cells go into a second hash table, and one quad is laid per crossing edge, joining the
// vertices of the four cells round it through that table. Nothing is kept per block, so what
// the tables hold grows only with the particles, however scattered they are. Every kernel binds
// at most eight storage buffers, the least a WebGPU device promises. Vertices come out
// in the vessel's frame, so the mesh turns with the vessel between extractions.
#import vessel::{vessel_to_world, vessel_from_world}
#import fluid_common::{params, coords_of, cell_key, cell_slot, neighbour_cell}

struct SurfaceParams {
    cell: f32,
    inv_r2: f32,
    iso: f32,
    max_vertices: u32,
    max_indices: u32,
    max_blocks: u32,
    table_mask: u32,
    cell_mask: u32,
}

struct Vertex {
    // xyz: position in the vessel's frame, w: foam
    position: vec4<f32>,
    // xyz: normal, w: the key of the cell the vertex sits in, as bits
    normal: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> position: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
@group(0) @binding(12) var<storage, read> key: array<u32>;

@group(3) @binding(0) var<uniform> surface: SurfaceParams;
@group(3) @binding(1) var<storage, read_write> vertices: array<Vertex>;
@group(3) @binding(2) var<storage, read_write> indices: array<u32>;
@group(3) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> table: array<atomic<u32>>;
@group(3) @binding(5) var<storage, read_write> table_index: array<u32>;
@group(3) @binding(6) var<storage, read_write> blocks: array<u32>;
@group(3) @binding(7) var<storage, read_write> dispatch: array<u32>;
@group(3) @binding(8) var<storage, read_write> cell_table: array<atomic<u32>>;
@group(3) @binding(9) var<storage, read_write> cell_value: array<u32>;

const BLOCK: i32 = 4;
const CORNERS_PER_BLOCK: u32 = 125u;
const CELLS_PER_BLOCK: u32 = 64u;
const COUNTER_VERTICES: u32 = 0u;
const COUNTER_INDICES: u32 = 1u;
const COUNTER_BLOCKS: u32 = 2u;
// the second indirect dispatch, for the kernel that runs per crossed cell
const DISPATCH_CELLS: u32 = 4u;
const NO_VERTEX: u32 = 0xffffffffu;
const NO_BLOCK: u32 = 0xffffffffu;
const NO_CELL: u32 = 0xffffffffu;
// block coordinates are packed ten bits each about this origin, the top bit marking a used slot
const KEY_ORIGIN: i32 = 512;
const KEY_USED: u32 = 0x80000000u;
const MAX_PROBES: u32 = 64u;
// a dispatch axis holds at most 65535 workgroups, so the blocks are spread over two
const DISPATCH_ROW: u32 = 32768u;

var<workgroup> field: array<vec2<f32>, 125>;

fn block_key(b: vec3<i32>) -> u32 {
    let c = b + vec3(KEY_ORIGIN);
    if (any(c < vec3(0)) || any(c >= vec3(1024))) {
        return 0u;
    }
    return KEY_USED | (u32(c.x) << 20u) | (u32(c.y) << 10u) | u32(c.z);
}

fn block_coords(key: u32) -> vec3<i32> {
    return vec3<i32>(i32((key >> 20u) & 1023u), i32((key >> 10u) & 1023u), i32(key & 1023u)) - vec3(KEY_ORIGIN);
}

fn key_slot(key: u32) -> u32 {
    return ((key & 0x3fffffffu) * 2654435761u >> 12u) & surface.table_mask;
}

/// Claim a slot for the block, unless it has one.
fn mark_block(b: vec3<i32>) {
    let key = block_key(b);
    if (key == 0u) {
        return;
    }
    var slot = key_slot(key);
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
        let r = atomicCompareExchangeWeak(&table[slot], 0u, key);
        if (r.exchanged || r.old_value == key) {
            return;
        }
        if (r.old_value != 0u) {
            slot = (slot + 1u) & surface.table_mask;
        }
    }
}

/// The index of the block, or NO_BLOCK if the water never touched it.
fn find_block(b: vec3<i32>) -> u32 {
    let key = block_key(b);
    if (key == 0u) {
        return NO_BLOCK;
    }
    var slot = key_slot(key);
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
        let found = atomicLoad(&table[slot]);
        if (found == key) {
            return table_index[slot];
        }
        if (found == 0u) {
            return NO_BLOCK;
        }
        slot = (slot + 1u) & surface.table_mask;
    }
    return NO_BLOCK;
}

/// A cell is named by its block's index and its place in the block.
fn cell_key_of(block: u32, cell: u32) -> u32 {
    return KEY_USED | (block << 6u) | cell;
}

fn cell_slot_of(key: u32) -> u32 {
    return ((key & 0x7fffffffu) * 2654435761u >> 12u) & surface.cell_mask;
}

/// Record a crossed cell: its vertex and which of its corners are inside.
fn insert_cell(key: u32, value: u32) {
    var slot = cell_slot_of(key);
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
        let r = atomicCompareExchangeWeak(&cell_table[slot], 0u, key);
        if (r.exchanged) {
            cell_value[slot] = value;
            return;
        }
        if (r.old_value != 0u) {
            slot = (slot + 1u) & surface.cell_mask;
        }
    }
}

fn find_cell(key: u32) -> u32 {
    var slot = cell_slot_of(key);
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
        let found = atomicLoad(&cell_table[slot]);
        if (found == key) {
            return cell_value[slot];
        }
        if (found == 0u) {
            return NO_CELL;
        }
        slot = (slot + 1u) & surface.cell_mask;
    }
    return NO_CELL;
}

/// Every particle marks the blocks whose corners its splat reaches. The splat is zero at its
/// radius, so only corners strictly inside it count, which keeps it to two blocks per axis.
@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let p = vessel_from_world(position[i].xyz) / surface.cell;
    let reach = 2.0;
    let lo = vec3<i32>(floor(p - reach)) + vec3(1);
    let hi = vec3<i32>(ceil(p + reach)) - vec3(1);
    let b_lo = (lo - vec3(1)) >> vec3(2u);
    let b_hi = hi >> vec3(2u);
    for (var x = b_lo.x; x <= b_hi.x; x++) {
        for (var y = b_lo.y; y <= b_hi.y; y++) {
            for (var z = b_lo.z; z <= b_hi.z; z++) {
                mark_block(vec3(x, y, z));
            }
        }
    }
}

/// Number the marked blocks.
@compute @workgroup_size(64)
fn list(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if (slot > surface.table_mask) {
        return;
    }
    let key = atomicLoad(&table[slot]);
    if (key == 0u) {
        return;
    }
    let index = atomicAdd(&counters[COUNTER_BLOCKS], 1u);
    if (index >= surface.max_blocks) {
        table_index[slot] = NO_BLOCK;
        return;
    }
    table_index[slot] = index;
    blocks[index] = key;
}

/// One workgroup per marked block for the extraction.
@compute @workgroup_size(1)
fn prepare_dispatch() {
    let count = min(atomicLoad(&counters[COUNTER_BLOCKS]), surface.max_blocks);
    dispatch[0] = min(count, DISPATCH_ROW);
    dispatch[1] = max((count + DISPATCH_ROW - 1u) / DISPATCH_ROW, 1u);
    dispatch[2] = 1u;
}

fn corner_position(block: vec3<i32>, corner: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(block * BLOCK + corner) * surface.cell;
}

/// Splatted density and foam at a corner of a block: the smooth kernel of every particle within
/// reach.
fn splat(b: vec3<i32>, corner: vec3<i32>) -> vec2<f32> {
    let p = vessel_to_world(corner_position(b, corner));
    var d = 0.0;
    var f = 0.0;
    let c = coords_of(p);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let s = position[j];
            if (kj != k) {
                continue;
            }
            let r = p - s.xyz;
            let q = dot(r, r) * surface.inv_r2;
            if (q < 1.0) {
                let w = (1.0 - q) * (1.0 - q);
                d += w;
                f += w * s.w;
            }
        }
    }
    return vec2(d, f);
}

fn corner_at(c: vec3<i32>) -> vec2<f32> {
    return field[u32((c.x * 5 + c.y) * 5 + c.z)];
}

fn cell_coords(cell: u32) -> vec3<i32> {
    return vec3<i32>(i32(cell / 16u), i32((cell / 4u) % 4u), i32(cell % 4u));
}

const OFFSETS = array<vec3<i32>, 8>(
    vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0),
    vec3(0, 0, 1), vec3(1, 0, 1), vec3(0, 1, 1), vec3(1, 1, 1),
);
const EDGES = array<vec2<u32>, 12>(
    vec2(0u, 1u), vec2(2u, 3u), vec2(4u, 5u), vec2(6u, 7u),
    vec2(0u, 2u), vec2(1u, 3u), vec2(4u, 6u), vec2(5u, 7u),
    vec2(0u, 4u), vec2(1u, 5u), vec2(2u, 6u), vec2(3u, 7u),
);

/// The field at every corner of a block, then one vertex per cell the surface passes through,
/// at the mean of its edge crossings.
@compute @workgroup_size(125)
fn extract(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let block = group.x + group.y * DISPATCH_ROW;
    let listed = block < min(atomicLoad(&counters[COUNTER_BLOCKS]), surface.max_blocks);
    var b = vec3(0);
    if (listed) {
        b = block_coords(blocks[block]);
        field[t] = splat(b, vec3<i32>(i32(t / 25u), i32((t / 5u) % 5u), i32(t % 5u)));
    }
    workgroupBarrier();
    if (!listed || t >= CELLS_PER_BLOCK) {
        return;
    }
    let cell = cell_coords(t);
    var d: array<f32, 8>;
    var mask = 0u;
    var foam = 0.0;
    var weight = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let corner = corner_at(cell + OFFSETS[k]);
        d[k] = corner.x;
        foam += corner.y;
        weight += corner.x;
        if (corner.x > surface.iso) {
            mask |= 1u << k;
        }
    }
    if (mask == 0u || mask == 255u) {
        return;
    }
    var sum = vec3(0.0);
    var crossings = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = EDGES[e].x;
        let c = EDGES[e].y;
        if (((mask >> a) & 1u) == ((mask >> c) & 1u)) {
            continue;
        }
        let s = (surface.iso - d[a]) / (d[c] - d[a]);
        sum += mix(vec3<f32>(OFFSETS[a]), vec3<f32>(OFFSETS[c]), s);
        crossings += 1.0;
    }
    let slot = atomicAdd(&counters[COUNTER_VERTICES], 1u);
    if (slot >= surface.max_vertices) {
        return;
    }
    let p = (vec3<f32>(b * BLOCK + cell) + sum / crossings) * surface.cell;
    let gradient = vec3(
        (d[1] - d[0]) + (d[3] - d[2]) + (d[5] - d[4]) + (d[7] - d[6]),
        (d[2] - d[0]) + (d[3] - d[1]) + (d[6] - d[4]) + (d[7] - d[5]),
        (d[4] - d[0]) + (d[5] - d[1]) + (d[6] - d[2]) + (d[7] - d[3]),
    );
    let normal = -gradient / max(length(gradient), 1e-9);
    var f = 0.0;
    if (weight > 0.0) {
        f = foam / weight;
    }
    let key = cell_key_of(block, t);
    vertices[slot] = Vertex(vec4(p, f), vec4(normal, bitcast<f32>(key)));
    insert_cell(key, (slot << 8u) | mask);
}

/// One workgroup per sixty-four crossed cells for the quads.
@compute @workgroup_size(1)
fn prepare_quads() {
    let count = min(atomicLoad(&counters[COUNTER_VERTICES]), surface.max_vertices);
    dispatch[DISPATCH_CELLS] = (count + 63u) / 64u;
    dispatch[DISPATCH_CELLS + 1u] = 1u;
    dispatch[DISPATCH_CELLS + 2u] = 1u;
}

/// The vertex of a cell given relative to a block, which may lie in a neighbouring block.
fn vertex_of(block: u32, b: vec3<i32>, cell: vec3<i32>) -> u32 {
    let shift = vec3<i32>(floor(vec3<f32>(cell) / f32(BLOCK)));
    var other = block;
    if (any(shift != vec3(0))) {
        other = find_block(b + shift);
        if (other == NO_BLOCK) {
            return NO_VERTEX;
        }
    }
    let local = cell - shift * BLOCK;
    let value = find_cell(cell_key_of(other, u32((local.x * 4 + local.y) * 4 + local.z)));
    if (value == NO_CELL) {
        return NO_VERTEX;
    }
    return value >> 8u;
}

fn emit_quad(v: vec4<u32>, inside: bool) {
    if (any(v == vec4(NO_VERTEX))) {
        return;
    }
    var q = v;
    if (!inside) {
        q = vec4(v.x, v.w, v.z, v.y);
    }
    let slot = atomicAdd(&counters[COUNTER_INDICES], 6u);
    if (slot + 6u > surface.max_indices) {
        return;
    }
    indices[slot] = q.x;
    indices[slot + 1u] = q.y;
    indices[slot + 2u] = q.z;
    indices[slot + 3u] = q.x;
    indices[slot + 4u] = q.z;
    indices[slot + 5u] = q.w;
}

/// One quad per edge leaving a crossed cell's first corner that crosses the surface, joining
/// the vertices of the four cells around the edge.
@compute @workgroup_size(64)
fn quads(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= min(atomicLoad(&counters[COUNTER_VERTICES]), surface.max_vertices)) {
        return;
    }
    let key = bitcast<u32>(vertices[i].normal.w);
    let block = (key & 0x7fffffffu) >> 6u;
    let b = block_coords(blocks[block]);
    let c = cell_coords(key & 63u);
    let mask = find_cell(key) & 0xffu;
    let inside = (mask & 1u) != 0u;
    let own = i;
    if ((((mask >> 1u) & 1u) != 0u) != inside) {
        emit_quad(vec4(
            own,
            vertex_of(block, b, c + vec3(0, -1, 0)),
            vertex_of(block, b, c + vec3(0, -1, -1)),
            vertex_of(block, b, c + vec3(0, 0, -1)),
        ), inside);
    }
    if ((((mask >> 2u) & 1u) != 0u) != inside) {
        emit_quad(vec4(
            own,
            vertex_of(block, b, c + vec3(0, 0, -1)),
            vertex_of(block, b, c + vec3(-1, 0, -1)),
            vertex_of(block, b, c + vec3(-1, 0, 0)),
        ), inside);
    }
    if ((((mask >> 4u) & 1u) != 0u) != inside) {
        emit_quad(vec4(
            own,
            vertex_of(block, b, c + vec3(-1, 0, 0)),
            vertex_of(block, b, c + vec3(-1, -1, 0)),
            vertex_of(block, b, c + vec3(0, -1, 0)),
        ), inside);
    }
}
