// Spray: water too fine for the particles to carry, which is what the air tears off a parcel
// of water flying through it and what is thrown up where water runs into water. The solver
// knows how much of each parcel the air has torn off it and where water is meeting water; what
// that water then does is below its grain, and is followed here as motes, each a cluster of
// drops of one width let go with the water's velocity. A mote flies through the vessel's frame
// exactly as the water does, is slowed by the air as drops of its width are, and is done when
// it comes down on a wall or falls back into water. Motes are what the spray is drawn from and
// push on nothing: the water they stand for is still carried by the particle it came off.
#import fluid_common::{params, spray_of, coords_of, cell_key, cell_slot, neighbour_cell}
#import vessel::{vessel_confine, vessel_flight, vessel_has_air, vessel_gone_through, vessel_turned}

@group(0) @binding(1) var<storage, read> position: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> velocity_next: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
@group(0) @binding(12) var<storage, read> key: array<u32>;
// two entries per mote: xyz where it is, w how wide its drops are; xyz its velocity, w how
// long it has flown, nought for a mote that is done
@group(3) @binding(13) var<storage, read_write> motes: array<vec4<f32>>;

// as many motes as particles: see `MAX_MOTES`
const MOTES: u32 = 65536u;
// how many motes a parcel is torn into by the time the air has torn all of it off
const PER_PARCEL: f32 = 16.0;
// the share of its own water that water which is all froth throws up as spray each second of
// the water's own clock, once it moves faster than the second of the speeds below, and none
// under the first. A mote holds as much water however it was come by, so this is what keeps
// the spray in the air from holding more water than was thrown into it.
const THROWN_SHARE: f32 = 0.02;
const THROWING: vec2<f32> = vec2(3.0, 9.0);
// a mote is let go within a parcel's width of its particle, with the particle's velocity and
// this share of its speed any way
const PARCEL: f32 = 1.2407;
const SCATTERED: f32 = 0.35;
// the widest drops a mote is made of, in the narrowest the air tears water to
const WIDEST: f32 = 20.0;
const BALL_DRAG: f32 = 0.5;
// a mote with this many particles within a spacing of it has fallen back into the water
const IN_WATER: u32 = 4u;

fn lot(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 747796405u + 2891336453u;
    let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}

fn lot3(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3(lot(seed), lot(seed), lot(seed)) - 0.5;
}

/// Let go the motes a particle sheds over a step: what the air tore off it, and what it throws
/// up as froth on the move. A mote takes the place of one that is done, or of any once none is.
@compute @workgroup_size(64)
fn shed(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    if (position[i].w < 0.0) {
        return;
    }
    let v = velocity_next[i].xyz;
    let speed = length(v);
    let torn = max(spray_of(velocity_next[i].w) - spray_of(velocity[i].w), 0.0);
    let thrown = THROWN_SHARE * PER_PARCEL * position[i].w * smoothstep(THROWING.x, THROWING.y, speed) * params.dt;
    let expected = PER_PARCEL * torn + thrown;
    if (expected <= 0.0) {
        return;
    }
    var seed = bitcast<u32>(position[i].x) ^ (bitcast<u32>(position[i].y) * 31u) ^ (bitcast<u32>(v.z) * 131u) ^ i;
    let count = min(u32(expected + lot(&seed)), 4u);
    var width = WIDEST * params.finest_drop;
    if (params.shatter > 0.0) {
        width = clamp(params.shatter / max(speed * speed, 1e-12), params.finest_drop, width);
    }
    for (var k = 0u; k < count; k++) {
        var slot = u32(lot(&seed) * f32(MOTES)) % MOTES;
        if (motes[2u * slot + 1u].w > 0.0) {
            slot = u32(lot(&seed) * f32(MOTES)) % MOTES;
        }
        motes[2u * slot] = vec4(position[i].xyz + lot3(&seed) * PARCEL, width);
        motes[2u * slot + 1u] = vec4(v + lot3(&seed) * (2.0 * SCATTERED * speed), params.dt);
    }
}

/// Whether a point is in the water: among its particles rather than by one of them.
fn in_water(p: vec3<f32>) -> bool {
    let c = coords_of(p);
    var near = 0u;
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let r = p - position[j].xyz;
            if (key[j] == k && dot(r, r) < params.spacing * params.spacing) {
                near++;
            }
        }
    }
    return near >= IN_WATER;
}

/// Carry a mote through a step.
@compute @workgroup_size(64)
fn fly(@builtin(global_invocation_id) id: vec3<u32>) {
    let m = id.x;
    if (m >= MOTES) {
        return;
    }
    let age = motes[2u * m + 1u].w;
    if (age <= 0.0) {
        return;
    }
    let flight = vessel_flight(motes[2u * m].xyz, motes[2u * m + 1u].xyz, params.dt);
    var p = flight.p;
    var v = flight.v;
    let gone = vessel_gone_through(p);
    if (gone.there) {
        p = gone.p;
        v = vessel_turned(v, gone.opening);
    }
    if (params.air > 0.0 && vessel_has_air(p)) {
        // a drop is slowed as a ball of its width is: by three quarters of a ball's drag
        // coefficient, times the air's density against the water's, times its speed squared,
        // over its width
        let slowed = 0.75 * BALL_DRAG * params.air / motes[2u * m].w * length(v) * params.dt;
        v /= 1.0 + slowed;
    }
    let landed = vessel_confine(p, 0.0).first.w > 0.0;
    if (landed || params.count == 0u || in_water(p)) {
        motes[2u * m + 1u].w = 0.0;
        return;
    }
    motes[2u * m] = vec4(p, motes[2u * m].w);
    motes[2u * m + 1u] = vec4(v, age + params.dt);
}
