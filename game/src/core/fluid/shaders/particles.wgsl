// Position-based fluid (Macklin & Müller 2013) on the GPU, one thread per particle. Particles are
// kept sorted by hashed grid cell so a particle's neighbours are in the 27 cells around it.
// Boundary samples of solid bodies (Akinci et al. 2012) contribute to density and its gradient
// like heavy particles.
//
// The water lives in its vessel's frame, whose walls stand still, and feels the frame's motion
// as the accelerations the `vessel` module reports. That module must define:
//   vessel_confine(p, margin) -> Confined     a point put back inside, and the walls it met
//   vessel_gravity(p) -> vec3<f32>            the frame's acceleration of a point at rest
//   vessel_coriolis(v, dt) -> vec3<f32>       a velocity after dt of the frame's turning
//   vessel_has_air(p) -> bool                 whether the vessel's air, at rest in it, is here
#import vessel::{Confined, vessel_confine, vessel_gravity, vessel_coriolis, vessel_has_air}
#import fluid_common::{params, Bodies, GpuBody, Boundary, SampleState, coords_of, cell_key, cell_slot, neighbour_cell}

@group(0) @binding(1) var<storage, read_write> position: array<vec4<f32>>;
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

struct Predicted {
    q: vec3<f32>,
    v: vec3<f32>,
    first: vec4<f32>,
    second: vec4<f32>,
}

fn clamp_speed(v: vec3<f32>) -> vec3<f32> {
    let s2 = dot(v, v);
    if (s2 > params.max_speed * params.max_speed) {
        return v * (params.max_speed / sqrt(s2));
    }
    return v;
}

fn predict_particle(i: u32) -> Predicted {
    let p = position[i].xyz;
    var v = velocity[i].xyz;
    v += vessel_gravity(p) * params.dt;
    v = vessel_coriolis(v, params.dt);
    if (params.air_k > 0.0 && vessel_has_air(p)) {
        v -= v * params.air_k;
    }
    v = clamp_speed(v);
    let c = vessel_confine(p + v * params.dt, params.margin);
    var out: Predicted;
    out.q = c.p;
    out.v = v;
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
    velocity[i] = vec4(velocity_next[2u * i].xyz * params.thin_scale_v, 0.0);
}

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let pr = predict_particle(i);
    velocity[i] = vec4(pr.v, 0.0);
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

/// Settle the water joining at rest onto the free sites nearest where it was placed, and only
/// once those run out onto taken ones: the water is put down as gently as it can be, at its
/// rest spacing, pushing nothing aside that it need not.
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
    for (var round = 0u; round < 2u; round++) {
        let wanted = 1u - round;
        for (var i = 0u; i < offered && placed < params.pending; i++) {
            if (site_free[i] == wanted) {
                position[params.count + placed] = vec4(sites[i].xyz, 0.0);
                velocity[params.count + placed] = vec4(0.0);
                placed++;
            }
        }
    }
}

fn poly(r2: f32) -> f32 {
    let t = params.h_sq - r2;
    return params.poly * t * t * t;
}

fn spiky_gradient(r: vec3<f32>, len: f32) -> vec3<f32> {
    let hr = params.h - len;
    return r * (params.spiky * hr * hr / len);
}

/// Whether body `b` is solid and close enough to `q` for its samples to matter.
fn body_near(b: u32, q: vec3<f32>) -> bool {
    let body = bodies.items[b];
    if (body.position.w <= 0.0) {
        return false;
    }
    let reach = body.extra.w + params.h;
    let d = q - body.position.xyz;
    return dot(d, d) < reach * reach;
}

@compute @workgroup_size(64)
fn lambda(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let qi = pred_in[i].xyz;
    let mr = params.mass / params.rest_density;
    var dens = params.mass * params.w_zero;
    var grad = vec3(0.0);
    var sum = 0.0;
    let c = coords_of(qi);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let pj = pred_in[j];
            if (kj != k || j == i) {
                continue;
            }
            let r = qi - pj.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            dens += params.mass * poly(r2);
            let len = sqrt(r2);
            if (len < 1e-6) {
                continue;
            }
            let g = spiky_gradient(r, len) * mr;
            grad += g;
            sum += dot(g, g);
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
    sum += dot(grad, grad);
    // no negative pressure: sparse water is free-flying spray
    let constraint = max(dens / params.rest_density - 1.0, 0.0);
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
        let radius = body.shape.w;
        let rc = q - centre;
        let r2 = dot(rc, rc);
        let keep_out = radius + 0.3 * d;
        if (r2 >= keep_out * keep_out) {
            continue;
        }
        let len = sqrt(r2);
        var n = vec3(0.0, 1.0, 0.0);
        if (len > 1e-9) {
            n = rc / len;
        }
        q += n * (keep_out - len);
    }
    return q;
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
    let c = coords_of(qi);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let pj = pred_in[j];
            if (kj != k || j == i) {
                continue;
            }
            let r = qi - pj.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let len = sqrt(r2);
            if (len < 1e-6) {
                continue;
            }
            let wr = poly(r2) / params.scorr_wq;
            let w2 = wr * wr;
            let scorr = -params.scorr_k * w2 * w2;
            dq += spiky_gradient(r, len) * (mr * (li + pj.w + scorr));
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
    let dl = dot(dq, dq);
    if (dl > params.max_delta * params.max_delta) {
        dq *= params.max_delta / sqrt(dl);
    }
    let confined = vessel_confine(qi + dq, params.margin);
    push_contact(i, confined.first);
    push_contact(i, confined.second);
    pred_out[i] = vec4(exclude_from_bodies(confined.p), 0.0);
}

/// Velocities from positions; wall contact = no penetration + viscous drag toward the wall, which
/// stands still in the vessel's frame.
@compute @workgroup_size(64)
fn update_velocities(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let q = pred_in[i].xyz;
    let old = position[i];
    var v = (q - old.xyz) / params.dt;
    position[i] = vec4(q, old.w);
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
        v *= params.wall_keep;
    }
    velocity[i] = vec4(clamp_speed(v), 0.0);
}

/// XSPH viscosity, no-slip drag toward bodies (their share is accumulated per sample in the
/// bodies pass), the reaction to buoyancy, and the visual foam estimate. Writes the new velocity
/// beside the drag scale this particle applied, which the bodies pass needs too.
@compute @workgroup_size(64)
fn viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let xi = position[i].xyz;
    let ui = velocity[i].xyz;
    var accel = vec3(0.0);
    var slip = vec3(0.0);
    var sw = 0.0;
    let c = coords_of(xi);
    for (var n = 0u; n < 27u; n++) {
        let cell = neighbour_cell(c, n);
        let k = cell_key(cell);
        let ci = cell_slot(cell);
        let end = cell_start[ci + 1u];
        for (var j = cell_start[ci]; j < end; j++) {
            let kj = key[j];
            let pj = position[j];
            let vj = velocity[j];
            if (kj != k || j == i) {
                continue;
            }
            let r = xi - pj.xyz;
            let r2 = dot(r, r);
            if (r2 >= params.h_sq) {
                continue;
            }
            let w_zero = poly(r2);
            let dv = vj.xyz - ui;
            accel += dv * (params.viscosity * w_zero);
            slip += dv * w_zero;
            sw += w_zero;
        }
    }
    // agitation: relative motion against neighbours and slip against the walls, in spacings
    // per second of the water's own clock, so that big water foams at the pace it moves
    var agitation = 0.0;
    if (sw > 0.0) {
        agitation = length(slip) / sw;
    }
    if (contact[2u * i].w > 0.0) {
        agitation += 0.5 * length(ui);
    }
    let wanted = clamp((agitation - FOAM_ONSET) / FOAM_RANGE, 0.0, 1.0);
    let foam = position[i].w;
    if (wanted > foam) {
        position[i].w = foam + (wanted - foam) * 0.25;
    } else {
        position[i].w = foam + (wanted - foam) * 0.015;
    }

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
    velocity_next[i] = vec4(ui + accel, scale);
}
