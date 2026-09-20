// The water's particles on the GPU, one thread each: flown through the vessel's frame, binned
// by hashed grid cell so that the particles of a cell are found together, and weathered by the
// air, the walls and the bodies once the grid (see `grid.wgsl`) has kept them from being
// squeezed.
//
// The water lives in its vessel's frame, whose walls stand still, and flies through it exactly
// as the `vessel` module says. That module must define:
//   vessel_confine(p, margin) -> Confined     a point put back inside, and the walls it met
//   vessel_flight(p, v, dt) -> Flight         where free flight over dt lands, and at what velocity
//   vessel_star_velocity(p) -> vec3<f32>      the velocity here of something at rest among the stars
//   vessel_has_air(p) -> bool                 whether the vessel's air, at rest in it, is here
// and, for the bodies' buoyancy, vessel_gravity(p), the frame's acceleration of a point at rest.
// Where the vessel's openings lead is asked of it in `views.wgsl`.
#import vessel::{Confined, Flight, VESSEL_OPENINGS, vessel_confine, vessel_flight, vessel_star_velocity, vessel_has_air, vessel_gone_through, vessel_through, vessel_turned}
#import fluid_common::{TOLD_APART, spray_of, carrying, params, Bodies, GpuBody, Boundary, SampleState, coords_of, cell_key, cell_slot, neighbour_cell}

@group(0) @binding(1) var<storage, read_write> position: array<vec4<f32>>;
// xyz: the velocity, w: what `fluid_common` says a particle carries beside it
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> position_sorted: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> velocity_next: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> pred_in: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> pred_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> contact: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
// per particle: its slot, its place within the slot, and its cell key
@group(0) @binding(10) var<storage, read_write> slot: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read> pending: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> key: array<u32>;
@group(0) @binding(14) var<storage, read> sites: array<vec4<f32>>;
// six to a particle: see `grid.wgsl`
@group(0) @binding(15) var<storage, read_write> affine: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> affine_sorted: array<vec4<f32>>;

@group(2) @binding(0) var<uniform> bodies: Bodies;
@group(2) @binding(2) var<storage, read> boundary: array<Boundary>;
@group(2) @binding(3) var<storage, read> sample_state: array<SampleState>;

// the agitation a particle starts foaming at and the range it foams fully over
const FOAM_ONSET: f32 = 1.875;
const FOAM_RANGE: f32 = 5.0;
// how long water takes to froth where it meets, and froth to clear, in seconds of the water's
// own clock
const FROTHS_IN: f32 = 0.06;
const BURSTS_IN: f32 = 1.1;
// how much of the space round it is water for a particle that is on its own, and for one that
// is among others: a pair a spacing apart fill a twelfth of it, the face of a body of water
// two fifths
const ALONE: f32 = 0.1;
const AMONG: f32 = 0.3;
// how wide a particle's water is as a ball on its own, in spacings, and a ball's drag
// coefficient
const PARCEL: f32 = 1.2407;
const BALL_DRAG: f32 = 0.5;
// how steeply how full of water a place is falls off across the flat face of a body of
// water, per spacing: from full to empty over the two cells the particles' water is handed to
const FLAT_FACE: f32 = 0.25;

struct Predicted {
    // where free flight alone lands the particle, at what velocity, and whether through an
    // opening
    landed: vec3<f32>,
    through: bool,
    v: vec3<f32>,
    // where the walls leave it
    q: vec3<f32>,
    first: vec4<f32>,
    second: vec4<f32>,
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

fn predict_particle(i: u32) -> Predicted {
    let p = position[i].xyz;
    let flight = vessel_flight(p, velocity[i].xyz, params.dt);
    let v = clamp_speed(flight.p, flight.v);
    var target_p = p + (flight.p - p) + (v - flight.v) * params.dt;
    let gone = vessel_gone_through(target_p);
    var carried = v;
    if (gone.there) {
        // what goes in at an opening comes out of the other with its motion turned as the two lie
        target_p = gone.p;
        carried = vessel_turned(v, gone.opening);
    }
    let c = vessel_confine(target_p, params.margin);
    var out: Predicted;
    out.landed = target_p;
    out.through = gone.there;
    out.q = c.p;
    out.v = carried;
    out.first = c.first;
    out.second = c.second;
    return out;
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let cell = coords_of(predict_particle(i).q);
    let ci = cell_slot(cell);
    slot[i] = vec4(ci, atomicAdd(&cell_count[ci], 1u), cell_key(cell), 0u);
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let s = slot[i];
    let dest = cell_start[s.x] + s.y;
    position_sorted[dest] = position[i];
    velocity_next[dest] = velocity[i];
    key[dest] = s.z;
}

@compute @workgroup_size(64)
fn scatter_affine(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let s = slot[i];
    let dest = cell_start[s.x] + s.y;
    for (var row = 0u; row < 6u; row++) {
        affine_sorted[6u * dest + row] = affine[6u * i + row];
    }
}

/// Keep every other particle of the sorted water, in place of copying the sort back: with the
/// particles twice as heavy and the kernel twice as wide, the water is as dense as before. The
/// kept particles are rescaled into the coarser water's units.
@compute @workgroup_size(64)
fn thin(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.pending) {
        return;
    }
    let p = position_sorted[2u * i];
    position[i] = vec4(p.xyz * params.thin_scale, p.w);
    velocity[i] = vec4(velocity_next[2u * i].xyz * params.thin_scale_v, velocity_next[2u * i].w);
    // a change of velocity across a length, across two and across three
    let across = params.thin_scale_v / params.thin_scale;
    let twice = across / params.thin_scale;
    for (var part = 0u; part < 3u; part++) {
        let slopes = affine_sorted[12u * i + 2u * part];
        let twists = affine_sorted[12u * i + 2u * part + 1u];
        affine[6u * i + 2u * part] = vec4(slopes.xyz * across, slopes.w * params.thin_scale_v);
        affine[6u * i + 2u * part + 1u] = vec4(twists.xyz * twice, twists.w * twice / params.thin_scale);
    }
}

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let pr = predict_particle(i);
    if (pr.through) {
        // what has gone through an opening has been turned, and how its velocity changed
        // across it is not carried over
        let pressed = affine[6u * i].w;
        for (var row = 0u; row < 6u; row++) {
            affine[6u * i + row] = vec4(0.0);
        }
        affine[6u * i].w = pressed;
    }
    velocity[i] = vec4(pr.v, velocity[i].w);
    velocity_next[i] = vec4(pr.landed, 0.0);
    pred_in[i] = vec4(pr.q, 0.0);
    contact[2u * i] = pr.first;
    contact[2u * i + 1u] = pr.second;
}

@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.pending) {
        return;
    }
    let dest = params.count + i;
    position[dest] = pending[2u * i];
    velocity[dest] = pending[2u * i + 1u];
    at_rest_within(dest);
}

/// Water that joins has yet to be told how its velocity changes across it, or what presses it.
fn at_rest_within(i: u32) {
    for (var row = 0u; row < 6u; row++) {
        affine[6u * i + row] = vec4(0.0);
    }
}

// how many lattice sites one placement may offer, and how near an existing particle makes a
// site taken
const SITES: u32 = 4096u;
const TAKEN: f32 = 0.85;
var<workgroup> site_free: array<u32, SITES>;

/// Whether no particle of the sorted water is on a site.
fn site_is_free(q: vec3<f32>) -> bool {
    let c = coords_of(q);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            if (key[j] != k) {
                continue;
            }
            let r = q - position_sorted[j].xyz;
            if (dot(r, r) < TAKEN * TAKEN) {
                return false;
            }
        }
    }
    return true;
}

/// What a particle joining the water carries beside its velocity: a number of its own to be told
/// apart by.
fn told_apart(site: vec3<f32>) -> f32 {
    return floor(fract(sin(dot(site, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * TOLD_APART);
}

/// Settle the water joining at rest onto the free sites nearest where it was placed: the water
/// is put down as gently as it can be, at its rest spacing, pushing nothing aside that it need
/// not. Water there is no free site for is water forced in faster than the water there gets
/// away, as from a source under pressure: it goes in between the sites, spread evenly over all
/// of those offered rather than heaped on the nearest, so that the water already there is
/// crowded a little everywhere and pushes out on all sides, and nowhere so much at once that
/// it is thrown.
@compute @workgroup_size(256)
fn join(@builtin(local_invocation_id) local: vec3<u32>) {
    let offered = min(params.candidates, SITES);
    for (var i = local.x; i < offered; i += 256u) {
        site_free[i] = u32(site_is_free(sites[i].xyz));
    }
    workgroupBarrier();
    if (local.x != 0u) {
        return;
    }
    var placed = 0u;
    for (var i = 0u; i < offered && placed < params.pending; i++) {
        if (site_free[i] == 1u) {
            position[params.count + placed] = vec4(sites[i].xyz, 0.0);
            velocity[params.count + placed] = vec4(0.0, 0.0, 0.0, told_apart(sites[i].xyz));
            at_rest_within(params.count + placed);
            placed++;
        }
    }
    let forced = params.pending - placed;
    if (forced == 0u || offered == 0u) {
        return;
    }
    // every so many sites takes one, starting somewhere else each time
    let every = f32(offered) / f32(forced);
    let first = fract(f32(params.count) * 0.61803399) * every;
    for (var k = 0u; k < forced; k++) {
        let site = sites[min(u32(first + f32(k) * every), offered - 1u)].xyz;
        let lot = vec3<u32>(vec3(told_apart(site), told_apart(site.yzx), told_apart(site.zxy))) & vec3(1u);
        let between = site + (vec3<f32>(lot) - 0.5) * params.spacing;
        position[params.count + placed] = vec4(between, 0.0);
        velocity[params.count + placed] = vec4(0.0, 0.0, 0.0, told_apart(between));
        at_rest_within(params.count + placed);
        placed++;
    }
}

fn poly(r2: f32) -> f32 {
    let t = params.h_sq - r2;
    return params.poly * t * t * t;
}

/// Whether body `b` is solid and close enough to `q` for its samples to matter.
fn body_near(b: u32, q: vec3<f32>) -> bool {
    let body = bodies.items[b];
    if (body.position.w <= 0.0) {
        return false;
    }
    let reach = body.extra.w + params.h;
    let d = q - body.position.xyz;
    if (dot(d, d) < reach * reach) {
        return true;
    }
    // what of the body has gone in at an opening of the vessel is beyond it
    for (var opening = 0u; opening < VESSEL_OPENINGS; opening++) {
        let beyond = vessel_through(body.position.xyz, opening, body.extra.w);
        let e = q - beyond.p;
        if (beyond.there && dot(e, e) < reach * reach) {
            return true;
        }
    }
    return false;
}

/// What the air, the walls and the bodies do to a particle once the grid has kept the water
/// from being squeezed: the air's drag; no-slip drag toward bodies (their share is accumulated
/// per sample in the bodies pass); the reaction to buoyancy; how much of the particle is froth;
/// and a rough bed's drag. Writes the new velocity beside the drag scale this particle applied,
/// which the bodies pass needs too.
@compute @workgroup_size(64)
fn weather(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let xi = position[i].xyz;
    let ui = velocity[i].xyz;
    // what the grid read out to the particle: how fast the water is closing on it, which way
    // the water round it lies, and how full of water the place is
    let closing = pred_in[i].w;
    let toward_water = pred_out[i].xyz;
    let filled = pred_out[i].w;
    var accel = vec3(0.0);
    // agitation: how fast the water round the particle is coming together onto it, which is
    // water running into water and folding air in, as water that only shears, turns or is
    // drawn out does not; and slip against the walls. In spacings per second of the water's
    // own clock, so that big water foams at the pace it moves
    var agitation = 2.0 * closing * params.h;
    if (contact[2u * i].w > 0.0) {
        agitation += 0.5 * length(ui);
    }
    let wanted = clamp((agitation - FOAM_ONSET) / FOAM_RANGE, 0.0, 1.0);
    var foam = position[i].w;
    // next to none of the space round a particle on its own is water, and most of it is
    // inside a body of water
    let alone = 1.0 - smoothstep(ALONE, AMONG, filled);
    // air is beaten into water as fast as the water meets, and the bubbles of the froth rise
    // out of it and burst in their own time
    let over = select(BURSTS_IN, FROTHS_IN, wanted > foam);
    foam += (wanted - foam) * (1.0 - exp(-params.dt / over));
    // spray that falls back into a body of water is water again
    var spray = spray_of(velocity[i].w) * alone;
    // and spray that comes down on the ground or the glass runs together on it
    if (contact[2u * i].w > 0.0) {
        spray = 0.0;
    }
    if (params.air > 0.0 && vessel_has_air(xi)) {
        let speed = length(ui);
        // the widest drop the air leaves whole at this speed
        let whole = clamp(params.shatter / max(speed * speed, 1e-12), params.finest_drop, PARCEL);
        if (whole < PARCEL) {
            spray = min(spray + alone * params.breakup * speed, 1.0);
        }
        // a parcel on its own is slowed as a ball of its water is, and what the air has torn off
        // it as drops of the width the air leaves whole: by three quarters of the drag
        // coefficient of a ball, times the air's density against the water's, times the speed
        // squared, over the width
        let widths = mix(1.0 / PARCEL, 1.0 / whole, spray);
        let on_its_own = 0.75 * BALL_DRAG * params.air * widths * speed * params.dt;
        // the face a body of water turns to the air takes the air's dynamic pressure, square
        // on, which presses it in: Newton's law of resistance, by which a ball is dragged as
        // balls are
        let facing = length(toward_water);
        var pressed = vec3(0.0);
        if (facing > 1e-6 && speed > 1e-6) {
            let outward = -toward_water / facing;
            let square_on = max(dot(outward, ui) / speed, 0.0);
            let exposed = min(facing / FLAT_FACE, 1.0);
            pressed = -outward * (BALL_DRAG * params.air * speed * speed * square_on * square_on * exposed / params.spacing * params.dt);
        }
        accel += mix(pressed, -ui * (on_its_own / (1.0 + on_its_own)), alone);
    }
    position[i].w = foam;

    // no-slip drag toward the bodies, limited so this particle relaxes at most fully in one substep
    var wsum = 0.0;
    for (var b = 0u; b < params.body_count; b++) {
        if (!body_near(b, xi)) {
            continue;
        }
        let slots = bodies.items[b].slots;
        let end = slots.z + slots.y;
        for (var k = slots.z; k < end; k++) {
            let s = boundary[k].pos;
            let r = xi - s.xyz;
            let r2 = dot(r, r);
            if (r2 < params.h_sq) {
                wsum += params.body_drag * poly(r2) * (s.w / params.rest_density);
            }
        }
    }
    var scale = 1.0;
    if (wsum > 1.0) {
        scale = 1.0 / wsum;
    }
    for (var b = 0u; b < params.body_count; b++) {
        if (!body_near(b, xi)) {
            continue;
        }
        let slots = bodies.items[b].slots;
        let end = slots.z + slots.y;
        for (var k = slots.z; k < end; k++) {
            let s = boundary[k];
            let r = xi - s.pos.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let kernel = poly(r2);
            let w = params.body_drag * kernel * (s.pos.w / params.rest_density) * scale;
            accel += (s.vel.xyz - ui) * w;
            // reaction to buoyancy: share of -F_b, weighted by this particle's kernel contribution
            let state = sample_state[k];
            if (state.field.w > 1.0) {
                accel -= state.force.xyz * (kernel * params.dt / state.field.w);
            }
        }
    }
    let after = ui + accel;
    pred_out[i].w = scale;
    velocity_next[i] = vec4(after, carrying(velocity[i].w, spray));
}
