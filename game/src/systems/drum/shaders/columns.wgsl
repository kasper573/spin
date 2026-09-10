// The water over every column of the ground: for each cell of the landscape's grid, how high
// above the glass the water reaches there and which way it flows, gathered from the particles,
// so that the ground can be lit by what sunlight comes down to it through the water.
#import vessel::drum

struct Survey {
    count: u32,
}

@group(1) @binding(2) var<uniform> survey: Survey;
@group(1) @binding(3) var<storage, read> position: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> velocity: array<vec4<f32>>;
// four words per column: the height reached above the glass, the particles counted, and the
// flow round the ring and along its axis summed over them, the flow in fixed point
@group(1) @binding(5) var<storage, read_write> columns: array<atomic<u32>>;

const FIXED: f32 = 256.0;

@compute @workgroup_size(64)
fn survey_columns(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= survey.count) {
        return;
    }
    let p = position[i].xyz;
    let height = drum.radius - length(p.xz);
    if (height <= 0.0) {
        return;
    }
    let segments = f32(drum.segments);
    let segment = u32((atan2(p.z, p.x) / drum.dphi + segments) % segments);
    let row = u32(clamp((p.y + drum.half_width) / drum.dy + 0.5, 0.0, f32(drum.rows) - 1.0));
    let c = (segment * drum.rows + row) * 4u;
    atomicMax(&columns[c], u32(height * FIXED));
    atomicAdd(&columns[c + 1u], 1u);
    let v = velocity[i].xyz;
    let spinward = normalize(vec3(-p.z, 0.0, p.x));
    atomicAdd(&columns[c + 2u], bitcast<u32>(i32(dot(v, spinward) * FIXED)));
    atomicAdd(&columns[c + 3u], bitcast<u32>(i32(v.y * FIXED)));
}
