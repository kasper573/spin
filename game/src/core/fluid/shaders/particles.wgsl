// Position-based fluid (Macklin & Müller 2013) on the GPU, one thread per particle. Particles are
// kept sorted by grid cell so a particle's neighbours are the 27 cells around it. Boundary samples
// of solid bodies (Akinci et al. 2012) contribute to density and its gradient like heavy particles.
#import vessel::{Confined, vessel_confine, vessel_wall_velocity, vessel_air_velocity}
#import fluid_common::{params, Bodies, GpuBody, Boundary, SampleState, cell_at, coords_of, cell_of}

@group(0) @binding(1) var<storage, read_write> position: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> position_sorted: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> velocity_next: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> pred_in: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> pred_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> contact: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> cell_start: array<u32>;
@group(0) @binding(10) var<storage, read_write> slot: array<u32>;
@group(0) @binding(11) var<storage, read> pending: array<vec4<f32>>;

@group(2) @binding(0) var<uniform> bodies: Bodies;
@group(2) @binding(2) var<storage, read> boundary: array<Boundary>;
@group(2) @binding(3) var<storage, read> sample_state: array<SampleState>;

const SLOT_BITS: u32 = 18u;
const SLOT_MASK: u32 = 0x3ffffu;

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
    if (params.air_k > 0.0) {
        let air = vessel_air_velocity(p);
        if (air.w > 0.0) {
            v += (air.xyz - v) * params.air_k;
        }
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
    let ci = u32(cell_of(predict_particle(i).q));
    let s = atomicAdd(&cell_count[ci], 1u);
    slot[i] = (ci << SLOT_BITS) | (s & SLOT_MASK);
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let packed = slot[i];
    let dest = cell_start[packed >> SLOT_BITS] + (packed & SLOT_MASK);
    position_sorted[dest] = position[i];
    velocity_next[dest] = velocity[i];
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
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let ci = cell_at(c + vec3(dx, dy, dz));
                if (ci < 0) {
                    continue;
                }
                let end = cell_start[ci + 1];
                for (var j = cell_start[ci]; j < end; j++) {
                    if (j == i) {
                        continue;
                    }
                    let r = qi - pred_in[j].xyz;
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
        let r = q - body.position.xyz;
        if (body.slots.w == 0u) {
            let half = body.shape.xyz;
            let e = half + vec3(0.3, 0.4, 0.3) * d;
            if (dot(r, r) > dot(e, e)) {
                continue;
            }
            let l = vec3(dot(body.row_x.xyz, r), dot(body.row_y.xyz, r), dot(body.row_z.xyz, r));
            let a = abs(l);
            if (any(a >= e)) {
                continue;
            }
            let pen = e - a;
            var axis = vec3(0.0);
            var amount = 0.0;
            if (pen.y <= pen.x && pen.y <= pen.z) {
                axis = vec3(body.row_x.y, body.row_y.y, body.row_z.y);
                amount = pen.y * sign(l.y);
            } else if (pen.x <= pen.z) {
                axis = vec3(body.row_x.x, body.row_y.x, body.row_z.x);
                amount = pen.x * sign(l.x);
            } else {
                axis = vec3(body.row_x.z, body.row_y.z, body.row_z.z);
                amount = pen.z * sign(l.z);
            }
            q += axis * amount;
        } else {
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
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let ci = cell_at(c + vec3(dx, dy, dz));
                if (ci < 0) {
                    continue;
                }
                let end = cell_start[ci + 1];
                for (var j = cell_start[ci]; j < end; j++) {
                    if (j == i) {
                        continue;
                    }
                    let pj = pred_in[j];
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

/// Velocities from positions; wall contact = no penetration + viscous drag toward the wall's speed.
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
        let wall = vessel_wall_velocity(q);
        var rv = v - wall;
        let vn0 = dot(rv, first.xyz);
        if (vn0 < 0.0) {
            rv -= vn0 * first.xyz;
        }
        if (second.w > 0.0) {
            let vn1 = dot(rv, second.xyz);
            if (vn1 < 0.0) {
                rv -= vn1 * second.xyz;
            }
        }
        v = wall + rv * params.wall_keep;
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
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let ci = cell_at(c + vec3(dx, dy, dz));
                if (ci < 0) {
                    continue;
                }
                let end = cell_start[ci + 1];
                for (var j = cell_start[ci]; j < end; j++) {
                    if (j == i) {
                        continue;
                    }
                    let r = xi - position[j].xyz;
                    let r2 = dot(r, r);
                    if (r2 >= params.h_sq) {
                        continue;
                    }
                    let w_zero = poly(r2);
                    let dv = velocity[j].xyz - ui;
                    accel += dv * (params.viscosity * w_zero);
                    slip += dv * w_zero;
                    sw += w_zero;
                }
            }
        }
    }
    // agitation: relative motion against neighbours and slip against the walls
    var agitation = 0.0;
    if (sw > 0.0) {
        agitation = length(slip) / sw;
    }
    if (contact[2u * i].w > 0.0) {
        agitation += 0.5 * length(ui - vessel_wall_velocity(xi));
    }
    let wanted = clamp((agitation - 0.6) / 1.6, 0.0, 1.0);
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
