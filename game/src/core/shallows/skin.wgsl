// The water on the ground as a surface to draw: a sheet over the water's face and a sheet under
// it on the bed, meeting at the shore, so that the two close round the water as the surface of
// the water in flight closes round that. A vertex stands at every corner of the chart's cells,
// as high as the faces of the wet cells that meet there stand; a dry cell's triangles are of no
// size. The vertices are as the water's surface has them, for what draws the one to draw the
// other: in the frame, the units and the clock the `ground` module carries them into.
#import ground::{ground_point, ground_carried}
#import shallows_chart::{shallows, Cell, WALL, cell_count, slot}

struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    velocity: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> bed: array<f32>;
@group(0) @binding(2) var<storage, read> cells: array<Cell>;
@group(0) @binding(3) var<storage, read_write> vertices: array<SurfaceVertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
// vertex count, index count, then two the water in flight counts its own by
@group(0) @binding(5) var<storage, read_write> counters: array<u32>;

fn is_wet(s: i32) -> bool {
    return s != WALL && cells[s].face > bed[s];
}

fn corners() -> vec2<u32> {
    return shallows.size + vec2(1u);
}

/// The two vertices of a corner of the chart: on the water's face, then on the bed.
@compute @workgroup_size(64)
fn skin_corners(@builtin(global_invocation_id) id: vec3<u32>) {
    let across = corners();
    if (id.x >= across.x * across.y) {
        return;
    }
    if (id.x == 0u) {
        counters[0] = 2u * across.x * across.y;
        counters[1] = 12u * cell_count();
        counters[2] = 0u;
        counters[3] = 0u;
    }
    let corner = vec2<i32>(i32(id.x % across.x), i32(id.x / across.x));
    var face = 0.0;
    var ground = 0.0;
    var flow = vec2(0.0);
    var climbing = 0.0;
    var wet = 0.0;
    var met = 0.0;
    // how the face lies across the corner, by the wet cells either side of it along each axis
    var rise = vec2(0.0);
    for (var n = 0u; n < 4u; n++) {
        let s = slot(corner - vec2<i32>(i32(n & 1u), i32(n >> 1u)));
        if (s == WALL) {
            continue;
        }
        ground += bed[s];
        met += 1.0;
        if (is_wet(s)) {
            face += cells[s].face;
            flow += cells[s].flow;
            climbing += cells[s].climbing;
            wet += 1.0;
            rise += cells[s].face * (vec2(1.0) - 2.0 * vec2(f32(n & 1u), f32(n >> 1u)));
        }
    }
    ground /= max(met, 1.0);
    var top = ground;
    var lies = vec2(0.0);
    if (wet > 0.0) {
        top = max(face / wet, ground);
        flow /= wet;
        climbing /= wet;
    }
    if (wet == 4.0) {
        lies = 0.5 * rise / shallows.cell;
    }
    let at = shallows.low + vec2<f32>(corner) * shallows.cell;
    let up = normalize(ground_carried(at, vec3(-lies, 1.0)));
    let moving = ground_carried(at, vec3(flow, climbing));
    vertices[2u * id.x] = SurfaceVertex(vec4(ground_point(at, top), 0.0), vec4(up, 0.0), vec4(moving, 0.0));
    vertices[2u * id.x + 1u] = SurfaceVertex(
        vec4(ground_point(at, ground), 0.0),
        vec4(-normalize(ground_carried(at, vec3(0.0, 0.0, 1.0))), 0.0),
        vec4(0.0),
    );
}

/// A cell's four triangles: two of the sheet over it, two of the sheet under it, wound so that
/// each faces out of the water.
@compute @workgroup_size(64)
fn skin_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cell_count()) {
        return;
    }
    let c = vec2<u32>(id.x % shallows.size.x, id.x / shallows.size.x);
    let across = corners().x;
    let first = 12u * id.x;
    if (!is_wet(i32(id.x))) {
        for (var k = 0u; k < 12u; k++) {
            indices[first + k] = 0u;
        }
        return;
    }
    let a = 2u * (c.y * across + c.x);
    let b = 2u * (c.y * across + c.x + 1u);
    let d = 2u * ((c.y + 1u) * across + c.x);
    let e = 2u * ((c.y + 1u) * across + c.x + 1u);
    let over = array<u32, 6>(a, b, e, a, e, d);
    for (var k = 0u; k < 6u; k++) {
        indices[first + k] = over[k];
        indices[first + 6u + k] = over[5u - k] + 1u;
    }
}
