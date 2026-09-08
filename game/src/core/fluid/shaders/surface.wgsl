// Isosurface of the particle cloud with naive surface nets: particles are gathered into a scalar
// field on the corners of a fine grid, every cell the surface crosses gets one vertex, and every
// crossing edge one quad. Vertices and indices land in buffers the water material draws directly.
#import fluid_common::{params, cell_at, coords_of}

struct SurfaceParams {
    // xyz: grid origin, w: cell size
    origin: vec4<f32>,
    // xyz: cells per axis, w: corners in total
    dims: vec4<i32>,
    iso: f32,
    inv_r2: f32,
    max_vertices: u32,
    max_indices: u32,
}

struct Vertex {
    // xyz: position, w: foam
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> position: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;

@group(3) @binding(0) var<uniform> surface: SurfaceParams;
@group(3) @binding(1) var<storage, read_write> corners: array<vec2<f32>>;
@group(3) @binding(2) var<storage, read_write> cell_vertex: array<u32>;
@group(3) @binding(3) var<storage, read_write> vertices: array<Vertex>;
@group(3) @binding(4) var<storage, read_write> indices: array<u32>;
@group(3) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;

const NO_VERTEX: u32 = 0xffffffffu;

fn corner_index(i: i32, j: i32, k: i32) -> u32 {
    let d = surface.dims.xyz + vec3(1);
    return u32((i * d.y + j) * d.z + k);
}

fn cell_index(i: i32, j: i32, k: i32) -> u32 {
    let d = surface.dims.xyz;
    return u32((i * d.y + j) * d.z + k);
}

fn corner_coords(index: u32) -> vec3<i32> {
    let d = surface.dims.xyz + vec3(1);
    let n = i32(index);
    let k = n % d.z;
    let j = (n / d.z) % d.y;
    let i = n / (d.z * d.y);
    return vec3(i, j, k);
}

fn cell_coords(index: u32) -> vec3<i32> {
    let d = surface.dims.xyz;
    let n = i32(index);
    let k = n % d.z;
    let j = (n / d.z) % d.y;
    let i = n / (d.z * d.y);
    return vec3(i, j, k);
}

/// Splatted density and foam at one corner: the smooth kernel of every particle within reach.
@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= u32(surface.dims.w)) {
        return;
    }
    let p = surface.origin.xyz + vec3<f32>(corner_coords(n)) * surface.origin.w;
    var d = 0.0;
    var f = 0.0;
    let c = coords_of(p);
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let ci = cell_at(c + vec3(dx, dy, dz));
                if (ci < 0) {
                    continue;
                }
                let end = cell_start[ci + 1];
                for (var j = cell_start[ci]; j < end; j++) {
                    let s = position[j];
                    let r = p - s.xyz;
                    let q = dot(r, r) * surface.inv_r2;
                    if (q < 1.0) {
                        let w = (1.0 - q) * (1.0 - q);
                        d += w;
                        f += w * s.w;
                    }
                }
            }
        }
    }
    corners[n] = vec2(d, f);
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

/// One vertex per cell the surface passes through, at the mean of its edge crossings.
@compute @workgroup_size(64)
fn place_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= u32(surface.dims.x * surface.dims.y * surface.dims.z)) {
        return;
    }
    let base = cell_coords(n);
    var d: array<f32, 8>;
    var mask = 0u;
    var foam = 0.0;
    var weight = 0.0;
    for (var b = 0u; b < 8u; b++) {
        let o = base + OFFSETS[b];
        let corner = corners[corner_index(o.x, o.y, o.z)];
        d[b] = corner.x;
        foam += corner.y;
        weight += corner.x;
        if (corner.x > surface.iso) {
            mask |= 1u << b;
        }
    }
    if (mask == 0u || mask == 255u) {
        cell_vertex[n] = NO_VERTEX;
        return;
    }
    var sum = vec3(0.0);
    var crossings = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = EDGES[e].x;
        let b = EDGES[e].y;
        if (((mask >> a) & 1u) == ((mask >> b) & 1u)) {
            continue;
        }
        let t = (surface.iso - d[a]) / (d[b] - d[a]);
        sum += mix(vec3<f32>(OFFSETS[a]), vec3<f32>(OFFSETS[b]), t);
        crossings += 1.0;
    }
    let slot = atomicAdd(&counters[0], 1u);
    if (slot >= surface.max_vertices) {
        cell_vertex[n] = NO_VERTEX;
        return;
    }
    let p = surface.origin.xyz + (vec3<f32>(base) + sum / crossings) * surface.origin.w;
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
    vertices[slot] = Vertex(vec4(p, f), vec4(normal, 0.0));
    cell_vertex[n] = slot;
}

fn emit_quad(cells: vec4<u32>, inside: bool) {
    let v = vec4(cell_vertex[cells.x], cell_vertex[cells.y], cell_vertex[cells.z], cell_vertex[cells.w]);
    if (any(v == vec4(NO_VERTEX))) {
        return;
    }
    var q = v;
    if (!inside) {
        q = vec4(v.x, v.w, v.z, v.y);
    }
    let slot = atomicAdd(&counters[1], 6u);
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

/// One quad per edge leaving a cell's corner that crosses the surface, joining the vertices of
/// the four cells around the edge.
@compute @workgroup_size(64)
fn place_quads(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let d = surface.dims.xyz;
    if (n >= u32(d.x * d.y * d.z)) {
        return;
    }
    let c = cell_coords(n);
    let i = c.x;
    let j = c.y;
    let k = c.z;
    let inside = corners[corner_index(i, j, k)].x > surface.iso;
    if (i + 1 < d.x && j > 0 && k > 0 && (corners[corner_index(i + 1, j, k)].x > surface.iso) != inside) {
        emit_quad(vec4(cell_index(i, j, k), cell_index(i, j - 1, k), cell_index(i, j - 1, k - 1), cell_index(i, j, k - 1)), inside);
    }
    if (j + 1 < d.y && i > 0 && k > 0 && (corners[corner_index(i, j + 1, k)].x > surface.iso) != inside) {
        emit_quad(vec4(cell_index(i, j, k), cell_index(i, j, k - 1), cell_index(i - 1, j, k - 1), cell_index(i - 1, j, k)), inside);
    }
    if (k + 1 < d.z && i > 0 && j > 0 && (corners[corner_index(i, j, k + 1)].x > surface.iso) != inside) {
        emit_quad(vec4(cell_index(i, j, k), cell_index(i - 1, j, k), cell_index(i - 1, j - 1, k), cell_index(i, j - 1, k)), inside);
    }
}
