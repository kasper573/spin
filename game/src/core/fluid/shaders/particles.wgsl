// Position-based fluid (Macklin & Müller 2013) on the GPU, one thread per particle. Particles are
// kept sorted by hashed grid cell so a particle's neighbours are in the 27 cells around it.
// Boundary samples of solid bodies (Akinci et al. 2012) contribute to density and its gradient
// like heavy particles.
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
#import fluid_views::{VIEWS, View, view_of, apart, brought_back}
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
// the most of a neighbour's velocity a particle takes on in a step
const MOST_MIXED: f32 = 0.5;
// how much of the space round it is water for a particle in a whole layer of them on a wall,
// and the shallowest sheet of water a particle stands for, in spacings
const WHOLE_LAYER: f32 = 0.42;
const SHALLOWEST: f32 = 0.1;
const LEAST_MIXED: f32 = 0.03;
// how far to one side of a particle its neighbours lie, by the sum of the kernel's slopes
// toward them, for one well inside a body of water and for one near its face, where that
// sum is some four tenths
const INSIDE: vec2<f32> = vec2(0.2, 0.38);
// how far short of settled water's density water stands on the lattice it is put down on,
// which is not thinned water, and is not pulled together for it
const ON_ITS_LATTICE: f32 = 0.03;
// the rings of a square lattice about one of its points: how far off they are, squared, in
// spacings, and how many points are on each
const LATTICE_RINGS = array<vec2<f32>, 5>(vec2(0.0, 1.0), vec2(1.0, 4.0), vec2(2.0, 4.0), vec2(4.0, 4.0), vec2(5.0, 8.0));
// how much of the space round it is water for a particle that is on its own, and for one that
// is among others: a pair a spacing apart fill a twelfth of it, the face of a body of water
// two fifths
const ALONE: f32 = 0.1;
const AMONG: f32 = 0.3;
// how wide a particle's water is as a ball on its own, in spacings, and a ball's drag
// coefficient
const PARCEL: f32 = 1.2407;
const BALL_DRAG: f32 = 0.5;
// the gradient of how full of water a place is, at the flat face of a body of water
const FLAT_FACE: f32 = 0.6;

struct Predicted {
    // where free flight alone lands the particle, and at what velocity
    landed: vec3<f32>,
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
}

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let pr = predict_particle(i);
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
        placed++;
    }
}

fn poly(r2: f32) -> f32 {
    let t = params.h_sq - r2;
    return params.poly * t * t * t;
}

/// The kernel the solver holds the water's density with, whose slope `spiky_gradient` is.
fn crowding(len: f32) -> f32 {
    let hr = params.h - len;
    return params.crowding * hr * hr * hr;
}

fn spiky_gradient(r: vec3<f32>, len: f32) -> vec3<f32> {
    let hr = params.h - len;
    return r * (params.spiky * hr * hr / len);
}

/// The walls within a kernel's reach of a point: their inward normals, each with how far the
/// wall reaches into the kernel.
fn walls_beside(q: vec3<f32>) -> Confined {
    var beside = vessel_confine(q, params.h);
    let moved = beside.p - q;
    beside.first.w = min(dot(moved, beside.first.xyz), params.h) * beside.first.w;
    beside.second.w = min(dot(moved, beside.second.xyz), params.h) * beside.second.w;
    return beside;
}

/// What a wall adds to the density of water beside it, as a share of the rest density, and
/// how steeply that falls with the distance from it. Water only fills the kernel of a
/// particle on the near side of a wall, and would have to crowd against the wall to come to its
/// rest density there, were the wall not to count as so much water at rest: water standing on
/// its lattice beyond the wall, as water put down at rest stands on its lattice before it, a
/// layer at a time, each layer the kernel summed over the rings of a square lattice.
fn wall_fill(reach: f32) -> vec2<f32> {
    let s2 = params.spacing * params.spacing;
    var fill = vec2(0.0);
    for (var layer = 0u; layer < 2u; layer++) {
        let z = params.h - reach + (f32(layer) + 0.5) * params.spacing;
        for (var ring = 0u; ring < 5u; ring++) {
            let off = sqrt(z * z + LATTICE_RINGS[ring].x * s2);
            let t = params.h - off;
            if (t > 0.0) {
                fill += LATTICE_RINGS[ring].y * vec2(t * t * t, 3.0 * t * t * z / off);
            }
        }
    }
    return fill * params.crowding;
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

@compute @workgroup_size(64)
fn lambda(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let qi = pred_in[i].xyz;
    let mr = params.mass / params.rest_density;
    var dens = params.mass * crowding(0.0);
    var grad = vec3(0.0);
    var sum = 0.0;
    for (var seen = 0u; seen < VIEWS; seen++) {
        let view = view_of(qi, seen, params.h);
        if (!view.there) {
            continue;
        }
        let c = coords_of(view.q);
        for (var n = 0u; n < 27u; n++) {
            let cell = neighbour_cell(c, n);
            let k = cell_key(cell);
            let ci = cell_slot(cell);
            let end = cell_start[ci + 1u];
            for (var j = cell_start[ci]; j < end; j++) {
                let kj = key[j];
                let pj = pred_in[j];
                if (kj != k || (j == i && seen == 0u)) {
                    continue;
                }
                let found = apart(view, pj.xyz);
                let r = found.xyz;
                let r2 = dot(r, r);
                if (found.w == 0.0 || r2 >= params.h_sq) {
                    continue;
                }
                let len = sqrt(r2);
                dens += params.mass * crowding(len);
                if (len < 1e-6) {
                    continue;
                }
                let g = spiky_gradient(r, len) * mr;
                grad += g;
                sum += dot(g, g);
            }
        }
    }
    for (var b = 0u; b < params.body_count; b++) {
        if (!body_near(b, qi)) {
            continue;
        }
        let slots = bodies.items[b].slots;
        let end = slots.z + slots.y;
        for (var k = slots.z; k < end; k++) {
            let s = boundary[k].pos;
            let r = qi - s.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            dens += s.w * poly(r2);
            let len = sqrt(r2);
            if (len < 1e-6) {
                continue;
            }
            grad += spiky_gradient(r, len) * (s.w / params.rest_density);
        }
    }
    let beside = walls_beside(qi);
    if (beside.first.w > 0.0) {
        let fill = wall_fill(beside.first.w);
        dens += params.rest_density * fill.x;
        grad -= beside.first.xyz * fill.y;
    }
    if (beside.second.w > 0.0) {
        let fill = wall_fill(beside.second.w);
        dens += params.rest_density * fill.x;
        grad -= beside.second.xyz * fill.y;
    }
    sum += dot(grad, grad);
    // Water no more thins than it squashes: what would pull a body of it apart, a stream
    // speeding up as it falls, draws it in from the sides instead, for as long as the air
    // presses on it harder than it is pulled. Past that it comes apart, as it does at once
    // with no air round it. Only water inside a body of it can be told to be thinned: at the
    // face of one it is short of neighbours whether it is or not, all of them lying to one
    // side of it, and pulling on it for that would be a skin on the water a great deal
    // stronger than the one it has.
    var constraint = dens / params.rest_density - 1.0;
    if (constraint < 0.0) {
        let inside = 1.0 - smoothstep(INSIDE.x, INSIDE.y, length(grad));
        constraint = max(min(constraint + ON_ITS_LATTICE, 0.0), -params.hold) * inside;
    }
    pred_in[i].w = -constraint / (sum + params.eps_lambda);
}

fn push_contact(i: u32, n: vec4<f32>) {
    if (n.w <= 0.0) {
        return;
    }
    if (contact[2u * i].w <= 0.0) {
        contact[2u * i] = n;
    } else if (contact[2u * i + 1u].w <= 0.0) {
        contact[2u * i + 1u] = n;
    }
}

/// Keep the particle out of solid bodies (a tunnelling guard).
fn exclude_from_bodies(q_in: vec3<f32>) -> vec3<f32> {
    var q = q_in;
    let d = params.spacing;
    for (var b = 0u; b < params.body_count; b++) {
        let body = bodies.items[b];
        if (body.position.w <= 0.0) {
            continue;
        }
        let centre = body.position.xyz + vec3(dot(body.row_x.xyz, body.shape.xyz), dot(body.row_y.xyz, body.shape.xyz), dot(body.row_z.xyz, body.shape.xyz));
        let keep_out = body.shape.w + 0.3 * d;
        q = kept_out_of(q, centre, keep_out);
        for (var opening = 0u; opening < VESSEL_OPENINGS; opening++) {
            let beyond = vessel_through(centre, opening, body.shape.w);
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

@compute @workgroup_size(64)
fn delta(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let pi = pred_in[i];
    let qi = pi.xyz;
    let li = pi.w;
    let mr = params.mass / params.rest_density;
    var dq = vec3(0.0);
    for (var seen = 0u; seen < VIEWS; seen++) {
        let view = view_of(qi, seen, params.h);
        if (!view.there) {
            continue;
        }
        let c = coords_of(view.q);
        for (var n = 0u; n < 27u; n++) {
            let cell = neighbour_cell(c, n);
            let k = cell_key(cell);
            let ci = cell_slot(cell);
            let end = cell_start[ci + 1u];
            for (var j = cell_start[ci]; j < end; j++) {
                let kj = key[j];
                let pj = pred_in[j];
                if (kj != k || (j == i && seen == 0u)) {
                    continue;
                }
                let found = apart(view, pj.xyz);
                let r = found.xyz;
                let r2 = dot(r, r);
                if (found.w == 0.0 || r2 >= params.h_sq) {
                    continue;
                }
                let len = sqrt(r2);
                if (len < 1e-6) {
                    continue;
                }
                dq += spiky_gradient(r, len) * (mr * (li + pj.w));
            }
        }
    }
    for (var b = 0u; b < params.body_count; b++) {
        if (!body_near(b, qi)) {
            continue;
        }
        let slots = bodies.items[b].slots;
        let end = slots.z + slots.y;
        for (var k = slots.z; k < end; k++) {
            let s = boundary[k].pos;
            let r = qi - s.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let len = sqrt(r2);
            if (len < 1e-6) {
                continue;
            }
            dq += spiky_gradient(r, len) * (s.w / params.rest_density * li);
        }
    }
    let beside = walls_beside(qi);
    if (beside.first.w > 0.0) {
        dq -= beside.first.xyz * (wall_fill(beside.first.w).y * li);
    }
    if (beside.second.w > 0.0) {
        dq -= beside.second.xyz * (wall_fill(beside.second.w).y * li);
    }
    let dl = dot(dq, dq);
    if (dl > params.max_delta * params.max_delta) {
        dq *= params.max_delta / sqrt(dl);
    }
    let confined = vessel_confine(qi + dq, params.margin);
    push_contact(i, confined.first);
    push_contact(i, confined.second);
    pred_out[i] = vec4(exclude_from_bodies(confined.p), 0.0);
}

/// A step ends where the solver put the particle, at the velocity its free flight ended at and
/// what the solver's push adds to it: the push over the step. Free flight is exact in where it
/// lands and in how fast, so water nothing pushes on keeps both; the distance covered over the
/// step would be the velocity half way through it, and would leave out half of what the frame
/// did to the water in every step.
///
/// A push counts as a knock at the end of the step, which every push between two particles is
/// alike for both, so that no water gains or loses momentum to its own pushes. Water held
/// still is so left with half a step's fall to carry into the next, which the next undoes.
///
/// Whatever pushed, the walls and the water round it, pushed all through the step, and the
/// velocity that built up was turned aside by the frame's turning as the fall it undid was: the
/// velocity by the frame's turning of twice the push, the place by a third of the step's worth
/// of that. The solver knows nothing of it and pushes straight, which would leave water held
/// still in a turning vessel creeping round it. What stands still among the stars moves
/// through the frame as the frame turns, so how that differs across the push is the turning
/// of it.
///
/// A wall, which stands still in the vessel's frame, lets nothing through it.
@compute @workgroup_size(64)
fn update_velocities(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let landed = velocity_next[i].xyz;
    let pushed = pred_in[i].xyz - landed;
    let turned = vessel_star_velocity(landed + pushed) - vessel_star_velocity(landed);
    let q = pred_in[i].xyz + turned * (2.0 / 3.0 * params.dt);
    var v = velocity[i].xyz + pushed / params.dt + 2.0 * turned;
    position[i] = vec4(q, position[i].w);
    let first = contact[2u * i];
    let second = contact[2u * i + 1u];
    if (first.w > 0.0) {
        let vn0 = dot(v, first.xyz);
        if (vn0 < 0.0) {
            v -= vn0 * first.xyz;
        }
        if (second.w > 0.0) {
            let vn1 = dot(v, second.xyz);
            if (vn1 < 0.0) {
                v -= vn1 * second.xyz;
            }
        }
    }
    velocity[i] = vec4(clamp_speed(q, v), velocity[i].w);
}

/// How fast the water at a particle is being squeezed, over how readily it gives: what the
/// solver's pushes have left of motion that crowds water already at its rest density, which
/// water, not being squeezable, cannot have. The mixing pass takes it out as a pressure
/// would, between neighbours, so that none of the water's momentum, its shear or its turning
/// is touched by it. Water below its rest density is coming together freely, and is let.
@compute @workgroup_size(64)
fn squeeze(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let xi = position[i].xyz;
    let ui = velocity[i].xyz;
    let mr = params.mass / params.rest_density;
    var dens = params.mass * crowding(0.0);
    var grad = vec3(0.0);
    var sum = 0.0;
    var closing = 0.0;
    for (var seen = 0u; seen < VIEWS; seen++) {
        let view = view_of(xi, seen, params.h);
        if (!view.there) {
            continue;
        }
        let c = coords_of(view.q);
        for (var n = 0u; n < 27u; n++) {
            let cell = neighbour_cell(c, n);
            let k = cell_key(cell);
            let ci = cell_slot(cell);
            let end = cell_start[ci + 1u];
            for (var j = cell_start[ci]; j < end; j++) {
                let kj = key[j];
                let pj = position[j];
                let vj = velocity[j];
                if (kj != k || (j == i && seen == 0u)) {
                    continue;
                }
                let found = apart(view, pj.xyz);
                let r2 = dot(found.xyz, found.xyz);
                if (found.w == 0.0 || r2 >= params.h_sq) {
                    continue;
                }
                let len = sqrt(r2);
                dens += params.mass * crowding(len);
                if (len < 1e-6) {
                    continue;
                }
                let g = spiky_gradient(found.xyz, len) * mr;
                grad += g;
                sum += dot(g, g);
                closing += dot(ui - brought_back(view, vj.xyz), g);
            }
        }
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
            dens += s.pos.w * poly(r2);
            let len = sqrt(r2);
            if (len < 1e-6) {
                continue;
            }
            let g = spiky_gradient(r, len) * (s.pos.w / params.rest_density);
            grad += g;
            closing += dot(ui - s.vel.xyz, g);
        }
    }
    let beside = walls_beside(xi);
    if (beside.first.w > 0.0) {
        let fill = wall_fill(beside.first.w);
        dens += params.rest_density * fill.x;
        grad -= beside.first.xyz * fill.y;
        closing -= dot(ui, beside.first.xyz) * fill.y;
    }
    if (beside.second.w > 0.0) {
        let fill = wall_fill(beside.second.w);
        dens += params.rest_density * fill.x;
        grad -= beside.second.xyz * fill.y;
        closing -= dot(ui, beside.second.xyz) * fill.y;
    }
    var held = 0.0;
    if ((dens >= 0.8 * params.rest_density && closing > 0.0) || dens >= params.rest_density) {
        held = closing / (sum + dot(grad, grad) + params.eps_lambda);
    }
    pred_in[i].w = held;
}

/// The mixing of momentum by eddies too small for the particles to show, as XSPH smoothing in
/// proportion to how fast neighbours go by each other; the air's drag; no-slip drag toward
/// bodies (their share is accumulated per sample in the bodies pass); the reaction to
/// buoyancy; and how much of the particle is froth. Writes the new velocity
/// beside the drag scale this particle applied, which the bodies pass needs too.
@compute @workgroup_size(64)
fn viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let xi = position[i].xyz;
    let ui = velocity[i].xyz;
    let held = pred_in[i].w;
    let mr = params.mass / params.rest_density;
    var accel = vec3(0.0);
    var meeting = 0.0;
    var sw = 0.0;
    // which way the water round the particle lies, weighted as the kernel's slope is: what
    // the gradient of how full of water the place is comes to
    var toward_water = vec3(0.0);
    for (var seen = 0u; seen < VIEWS; seen++) {
        let view = view_of(xi, seen, params.h);
        if (!view.there) {
            continue;
        }
        let c = coords_of(view.q);
        for (var n = 0u; n < 27u; n++) {
            let cell = neighbour_cell(c, n);
            let k = cell_key(cell);
            let ci = cell_slot(cell);
            let end = cell_start[ci + 1u];
            for (var j = cell_start[ci]; j < end; j++) {
                let kj = key[j];
                let pj = position[j];
                let vj = velocity[j];
                if (kj != k || (j == i && seen == 0u)) {
                    continue;
                }
                let found = apart(view, pj.xyz);
                let r2 = dot(found.xyz, found.xyz);
                if (found.w == 0.0 || r2 >= params.h_sq) {
                    continue;
                }
                let w_zero = poly(r2);
                let dv = brought_back(view, vj.xyz) - ui;
                let between = found.xyz * inverseSqrt(max(r2, 1e-12));
                accel += between * (dot(dv, between) * clamp(5.0 * params.eddy * length(dv), LEAST_MIXED, MOST_MIXED) * mr * w_zero);
                let len = sqrt(r2);
                if (len > 1e-6) {
                    accel -= spiky_gradient(found.xyz, len) * (mr * (held + pred_in[j].w));
                }
                meeting += dot(dv, between) * w_zero;
                sw += w_zero;
                let t = params.h_sq - r2;
                toward_water -= found.xyz * (t * t);
            }
        }
    }
    let beside = walls_beside(xi);
    if (beside.first.w > 0.0) {
        accel += beside.first.xyz * (wall_fill(beside.first.w).y * held);
    }
    if (beside.second.w > 0.0) {
        accel += beside.second.xyz * (wall_fill(beside.second.w).y * held);
    }
    // agitation: how fast the water round the particle is coming together onto it, which is
    // water running into water and folding air in, as water that only shears, turns or is
    // drawn out does not; and slip against the walls. In spacings per second of the water's
    // own clock, so that big water foams at the pace it moves
    var agitation = 0.0;
    if (sw > 0.0) {
        agitation = 2.0 * max(meeting, 0.0) / sw;
    }
    if (contact[2u * i].w > 0.0) {
        agitation += 0.5 * length(ui);
    }
    let wanted = clamp((agitation - FOAM_ONSET) / FOAM_RANGE, 0.0, 1.0);
    var foam = position[i].w;
    // how much of the space round the particle is water: next to none for one on its own, and
    // most of it inside a body of water
    let filled = sw * mr;
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
        let facing = 6.0 * params.poly * mr * length(toward_water);
        var pressed = vec3(0.0);
        if (facing > 1e-6 && speed > 1e-6) {
            let outward = -toward_water / length(toward_water);
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
            let len = sqrt(r2);
            if (len > 1e-6) {
                accel -= spiky_gradient(r, len) * (s.pos.w / params.rest_density * held);
            }
            // reaction to buoyancy: share of -F_b, weighted by this particle's kernel contribution
            let state = sample_state[k];
            if (state.field.w > 1.0) {
                accel -= state.force.xyz * (kernel * params.dt / state.field.w);
            }
        }
    }
    // A wall drags on what runs along it as a rough bed does on a river: by a share of the
    // water's dynamic pressure, which slows the water over it by that share of its speed
    // squared over its depth, taken here in the closed form that can never turn it round. A
    // particle against the wall stands for a layer of water a spacing deep where the layer is
    // whole, and for as much shallower a sheet as it has fewer neighbours in it.
    var after = ui + accel;
    if (contact[2u * i].w > 0.0) {
        let depth = clamp(filled / WHOLE_LAYER, SHALLOWEST, 1.0);
        after /= 1.0 + params.wall_friction * length(after) / depth;
    }
    pred_out[i].w = scale;
    velocity_next[i] = vec4(after, carrying(velocity[i].w, spray));
}
