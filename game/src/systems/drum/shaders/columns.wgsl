// The water over the ground: for every column of the water's own grain standing over the
// ground, how high above the glass the water reaches there, how many particles stand in it and
// which way they flow, gathered from the particles into a table keyed by where the column is
// from the water's site, so that the ground can be lit by what sunlight comes down to it
// through the water; and how far from its site the water reaches, which sets how fine it may be.
#import vessel::{drum, wall}

struct Survey {
    count: u32,
    // how many columns there are round the ring, or 0 when there are too many for the water to
    // reach round it, and how many of them a unit of arc spans
    columns_round: u32,
    per_unit: f32,
    mask: u32,
}

@group(1) @binding(3) var<uniform> survey: Survey;
@group(1) @binding(4) var<storage, read> position: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> velocity: array<vec4<f32>>;
// the farthest the water reaches from its site in metres, as the bits of the number, then five
// words per column: its key, the height reached above the glass, the particles counted, and the
// flow round the ring and along its axis summed over them, the height and flow in fixed point
@group(1) @binding(6) var<storage, read_write> columns: array<atomic<u32>>;

const FIXED: f32 = 256.0;
const HEADER: u32 = 4u;
const WORDS: u32 = 5u;
const USED: u32 = 0x80000000u;
const MOST_PROBES: u32 = 64u;
// a key tells apart this many columns either way round the ring and along the axis from the
// water's site, far more than the water ever reaches
const KEYED_ROUND: i32 = 16384;
const KEYED_ALONG: i32 = 32768;

// the farthest any particle of the workgroup is from the water's site, gathered here first so
// that the whole water does not queue on the one word it ends up in
var<workgroup> farthest: atomic<u32>;

@compute @workgroup_size(64)
fn survey_columns(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let i = id.x;
    if (local == 0u) {
        atomicStore(&farthest, 0u);
    }
    workgroupBarrier();
    // a positive float's bits run in the same order as the float
    if (i < survey.count) {
        atomicMax(&farthest, bitcast<u32>(length(position[i].xyz) / drum.per_metre));
    }
    workgroupBarrier();
    if (local == 0u) {
        atomicMax(&columns[0], atomicLoad(&farthest));
    }
    if (i >= survey.count) {
        return;
    }
    let p = position[i].xyz;
    let height = wall(p).height;
    if (height <= 0.0) {
        return;
    }
    var round = i32(floor(drum.radius * atan2(p.z, drum.radius + p.x) * survey.per_unit));
    if (survey.columns_round > 0u) {
        let n = i32(survey.columns_round);
        round = ((round + n / 2) % n + n) % n - n / 2;
    }
    let along = i32(floor(p.y));
    if (round < -KEYED_ROUND || round >= KEYED_ROUND || along < -KEYED_ALONG || along >= KEYED_ALONG) {
        return;
    }
    let key = USED | ((u32(round) & 0x7fffu) << 16u) | (u32(along) & 0xffffu);
    var slot = ((key * 2654435761u) >> 12u) & survey.mask;
    for (var probe = 0u; probe < MOST_PROBES; probe++) {
        let c = HEADER + slot * WORDS;
        let r = atomicCompareExchangeWeak(&columns[c], 0u, key);
        if (r.exchanged || r.old_value == key) {
            atomicMax(&columns[c + 1u], u32(height * FIXED));
            atomicAdd(&columns[c + 2u], 1u);
            let v = velocity[i].xyz;
            let spinward = normalize(vec3(-p.z, 0.0, drum.radius + p.x));
            atomicAdd(&columns[c + 3u], bitcast<u32>(i32(dot(v, spinward) * FIXED)));
            atomicAdd(&columns[c + 4u], bitcast<u32>(i32(v.y * FIXED)));
            return;
        }
        if (r.old_value != 0u) {
            slot = (slot + 1u) & survey.mask;
        }
    }
}
