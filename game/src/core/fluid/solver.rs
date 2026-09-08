use super::{
    Boundary, EPS_LAMBDA, Fluid, FluidParams, H, H2, ITERATIONS, MAX_BOUNDARY_NEIGHBOURS,
    MAX_BOUNDARY_SAMPLES, MAX_DELTA, MAX_NEIGHBOURS, MAX_SPEED, PARTICLE_MASS, PARTICLE_SPACING,
    POLY6, REST_DENSITY, SCORR_K, SCORR_WQ, SPIKY, W0, WET_REF,
};
use crate::core::rigid::{Body, BodyShape, Collider, collide_pair, collide_vessel};
use crate::core::vessel::Vessel;

const MARGIN: f32 = PARTICLE_SPACING * 0.5;

pub fn step(
    f: &mut Fluid,
    dt: f32,
    vessel: &impl Vessel,
    shapes: &[BodyShape],
    bodies: &mut [Body],
    p: &FluidParams,
) {
    step_particles(f, dt, vessel, shapes, bodies, p);
    step_bodies(dt as f64, vessel, shapes, bodies, p);
}

/// Predict, solve constraints, then derive velocities with wall contact and viscosity.
fn step_particles(
    f: &mut Fluid,
    dt: f32,
    vessel: &impl Vessel,
    shapes: &[BodyShape],
    bodies: &mut [Body],
    p: &FluidParams,
) {
    gather_boundary(&mut f.boundary, bodies, shapes);
    let n = f.n;
    if n == 0 {
        return;
    }
    let air_k = if p.air { dt / p.air_tau.0 } else { 0.0 };
    for i in 0..n {
        let (mut ux, mut uy, mut uz) = (f.vx[i], f.vy[i], f.vz[i]);
        if air_k > 0.0
            && let Some(a) = vessel.air_velocity([f.x[i] as f64, f.y[i] as f64, f.z[i] as f64])
        {
            ux += (a[0] as f32 - ux) * air_k;
            uy += (a[1] as f32 - uy) * air_k;
            uz += (a[2] as f32 - uz) * air_k;
        }
        (ux, uy, uz) = clamp_speed(ux, uy, uz);
        f.vx[i] = ux;
        f.vy[i] = uy;
        f.vz[i] = uz;
        let mut q = [f.x[i] + ux * dt, f.y[i] + uy * dt, f.z[i] + uz * dt];
        f.contact[i] = vessel.confine(&mut q, MARGIN);
        [f.px[i], f.py[i], f.pz[i]] = q;
    }
    f.grid.build(&f.px, &f.py, &f.pz, n);
    f.sort_by_cell();
    if f.boundary.n > 0 {
        let b = &mut f.boundary;
        b.grid.build(&b.x, &b.y, &b.z, b.n);
    }
    find_neighbours(f);
    for _ in 0..ITERATIONS {
        compute_lambda(f);
        apply_delta(f, vessel, bodies, shapes);
    }
    update_velocities(f, dt, vessel, p);
    if f.boundary.n > 0 {
        apply_buoyancy(f, bodies, shapes, dt);
    }
    apply_viscosity_and_drag(f, bodies, vessel, dt, p);
}

/// Apply the water's impulses, air and wet damping, integrate, and resolve contacts.
fn step_bodies(
    dt: f64,
    vessel: &impl Vessel,
    shapes: &[BodyShape],
    bodies: &mut [Body],
    p: &FluidParams,
) {
    let air_k = if p.air { dt / p.air_tau.0 as f64 } else { 0.0 };
    let spin = vessel.angular_velocity();
    let spin_mag = (spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]).sqrt();
    for b in bodies.iter_mut() {
        // water may change a body's velocity by a few times the vessel's artificial gravity per substep
        let reach = shapes[b.shape].reach();
        let max_dv = (20.0 + 4.0 * spin_mag * spin_mag * reach) * dt;
        b.apply_accumulated(max_dv);
        let wet_k = (b.wet * p.wet_spin_rate.0 as f64 * dt).min(1.0);
        if wet_k > 0.0 {
            relax(&mut b.w, &spin, wet_k);
        }
        if air_k > 0.0
            && let Some(a) = vessel.air_velocity(b.p)
        {
            relax(&mut b.v, &a, air_k);
            relax(&mut b.w, &spin, air_k);
        }
        b.integrate(dt);
    }
    for b in bodies.iter_mut().filter(|b| b.solid) {
        collide_vessel(b, &shapes[b.shape], vessel, p.restitution, p.body_friction);
    }
    let n = bodies.len();
    for a in 0..n {
        for b in (a + 1)..n {
            if !(bodies[a].solid && bodies[b].solid) {
                continue;
            }
            collide_pair(bodies, a, b, shapes, p.restitution, p.body_friction);
            collide_pair(bodies, b, a, shapes, p.restitution, p.body_friction);
        }
    }
}

/// Move `v` a fraction `k` of the way toward `target`.
#[inline]
fn relax(v: &mut [f64; 3], target: &[f64; 3], k: f64) {
    for (x, t) in v.iter_mut().zip(target) {
        *x += (t - *x) * k;
    }
}

#[inline]
fn clamp_speed(ux: f32, uy: f32, uz: f32) -> (f32, f32, f32) {
    let s2 = ux * ux + uy * uy + uz * uz;
    if s2 > MAX_SPEED * MAX_SPEED {
        let s = MAX_SPEED / s2.sqrt();
        (ux * s, uy * s, uz * s)
    } else {
        (ux, uy, uz)
    }
}

fn gather_boundary(b: &mut Boundary, bodies: &[Body], shapes: &[BodyShape]) {
    b.n = 0;
    for (bi, body) in bodies.iter().enumerate().filter(|(_, b)| b.solid) {
        for t in &shapes[body.shape].samples {
            if b.n >= MAX_BOUNDARY_SAMPLES {
                return;
            }
            let wp = body.to_world(&[t[0] as f64, t[1] as f64, t[2] as f64]);
            let wv = body.point_velocity(&wp);
            let k = b.n;
            b.n += 1;
            b.x[k] = wp[0] as f32;
            b.y[k] = wp[1] as f32;
            b.z[k] = wp[2] as f32;
            b.vx[k] = wv[0] as f32;
            b.vy[k] = wv[1] as f32;
            b.vz[k] = wv[2] as f32;
            b.psi[k] = t[3];
            b.body[k] = bi as u32;
        }
    }
}

fn find_neighbours(f: &mut Fluid) {
    let n = f.n;
    let b = &f.boundary;
    let has_b = b.n > 0;
    for i in 0..n {
        let (xi, yi, zi) = (f.px[i], f.py[i], f.pz[i]);
        let [cx, cy, cz] = f.grid.coords(xi, yi, zi);
        let mut c = 0usize;
        let mut cb = 0usize;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(h) = f.grid.cell_index([cx + dx, cy + dy, cz + dz]) else {
                        continue;
                    };
                    let (s, e) = (f.grid.start[h] as usize, f.grid.start[h + 1] as usize);
                    for k in s..e {
                        let j = f.grid.entries[k] as usize;
                        if j == i {
                            continue;
                        }
                        let (rx, ry, rz) = (xi - f.px[j], yi - f.py[j], zi - f.pz[j]);
                        if rx * rx + ry * ry + rz * rz < H2 && c < MAX_NEIGHBOURS {
                            f.nb[i * MAX_NEIGHBOURS + c] = j as u32;
                            c += 1;
                        }
                    }
                    if !has_b {
                        continue;
                    }
                    let (s, e) = (b.grid.start[h] as usize, b.grid.start[h + 1] as usize);
                    for k in s..e {
                        let bj = b.grid.entries[k] as usize;
                        let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
                        if rx * rx + ry * ry + rz * rz < H2 && cb < MAX_BOUNDARY_NEIGHBOURS {
                            f.bnb[i * MAX_BOUNDARY_NEIGHBOURS + cb] = bj as u32;
                            cb += 1;
                        }
                    }
                }
            }
        }
        f.nbc[i] = c as u16;
        f.bnbc[i] = cb as u16;
    }
}

fn compute_lambda(f: &mut Fluid) {
    let mr = PARTICLE_MASS / REST_DENSITY;
    let b = &f.boundary;
    for i in 0..f.n {
        let (xi, yi, zi) = (f.px[i], f.py[i], f.pz[i]);
        let mut dens = PARTICLE_MASS * W0;
        let (mut gx, mut gy, mut gz, mut sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let base = i * MAX_NEIGHBOURS;
        for k in 0..f.nbc[i] as usize {
            let j = f.nb[base + k] as usize;
            let (rx, ry, rz) = (xi - f.px[j], yi - f.py[j], zi - f.pz[j]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let t = H2 - r2;
            dens += PARTICLE_MASS * POLY6 * t * t * t;
            let r = r2.sqrt();
            if r < 1e-6 {
                continue;
            }
            let hr = H - r;
            let g = (SPIKY * hr * hr / r) * mr;
            let (jx, jy, jz) = (g * rx, g * ry, g * rz);
            gx += jx;
            gy += jy;
            gz += jz;
            sum += jx * jx + jy * jy + jz * jz;
        }
        let bbase = i * MAX_BOUNDARY_NEIGHBOURS;
        for k in 0..f.bnbc[i] as usize {
            let bj = f.bnb[bbase + k] as usize;
            let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let t = H2 - r2;
            dens += b.psi[bj] * POLY6 * t * t * t;
            let r = r2.sqrt();
            if r < 1e-6 {
                continue;
            }
            let hr = H - r;
            let g = (SPIKY * hr * hr / r) * (b.psi[bj] / REST_DENSITY);
            gx += g * rx;
            gy += g * ry;
            gz += g * rz;
        }
        sum += gx * gx + gy * gy + gz * gz;
        // no negative pressure: sparse water is free-flying spray
        let c = (dens / REST_DENSITY - 1.0).max(0.0);
        f.lam[i] = -c / (sum + EPS_LAMBDA);
    }
}

fn apply_delta(f: &mut Fluid, vessel: &impl Vessel, bodies: &mut [Body], shapes: &[BodyShape]) {
    let mr = PARTICLE_MASS / REST_DENSITY;
    for i in 0..f.n {
        let b = &f.boundary;
        let (xi, yi, zi, li) = (f.px[i], f.py[i], f.pz[i], f.lam[i]);
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        let base = i * MAX_NEIGHBOURS;
        for k in 0..f.nbc[i] as usize {
            let j = f.nb[base + k] as usize;
            let (rx, ry, rz) = (xi - f.px[j], yi - f.py[j], zi - f.pz[j]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let r = r2.sqrt();
            if r < 1e-6 {
                continue;
            }
            let t = H2 - r2;
            let wr = POLY6 * t * t * t / SCORR_WQ;
            let w2 = wr * wr;
            let sc = -SCORR_K * w2 * w2;
            let hr = H - r;
            let g = (SPIKY * hr * hr / r) * mr * (li + f.lam[j] + sc);
            ax += g * rx;
            ay += g * ry;
            az += g * rz;
        }
        let bbase = i * MAX_BOUNDARY_NEIGHBOURS;
        for k in 0..f.bnbc[i] as usize {
            let bj = f.bnb[bbase + k] as usize;
            let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let r = r2.sqrt();
            if r < 1e-6 {
                continue;
            }
            let hr = H - r;
            let g = (SPIKY * hr * hr / r) * (b.psi[bj] / REST_DENSITY) * li;
            ax += g * rx;
            ay += g * ry;
            az += g * rz;
        }
        let dl = ax * ax + ay * ay + az * az;
        if dl > MAX_DELTA * MAX_DELTA {
            let s = MAX_DELTA / dl.sqrt();
            ax *= s;
            ay *= s;
            az *= s;
        }
        f.dx[i] = ax;
        f.dy[i] = ay;
        f.dz[i] = az;
    }
    for i in 0..f.n {
        let mut q = [f.px[i] + f.dx[i], f.py[i] + f.dy[i], f.pz[i] + f.dz[i]];
        let c = vessel.confine(&mut q, MARGIN);
        [f.px[i], f.py[i], f.pz[i]] = q;
        for n in c.iter() {
            f.contact[i].push(*n);
        }
    }
    exclude_from_bodies(f, bodies, shapes);
}

/// Hard exclusion of particles from body volumes (tunnelling guard); also counts submerged samples.
fn exclude_from_bodies(f: &mut Fluid, bodies: &mut [Body], shapes: &[BodyShape]) {
    for body in bodies.iter_mut() {
        if !body.solid {
            body.submerged = 0;
            continue;
        }
        match shapes[body.shape].collider {
            Collider::Box { half } => exclude_from_box(f, body, half),
            Collider::Sphere { radius, centre } => exclude_from_sphere(f, body, radius, centre),
        }
    }
}

fn exclude_from_box(f: &mut Fluid, body: &mut Body, half: [f64; 3]) {
    let d = PARTICLE_SPACING as f64;
    let m = body.m;
    let c = body.p;
    let [hx, hy, hz] = half;
    let (ex, ey, ez) = (hx + 0.3 * d, hy + 0.4 * d, hz + 0.3 * d);
    let reach2 = ex * ex + ey * ey + ez * ez;
    let mut sub = 0u32;
    for i in 0..f.n {
        let rx = f.px[i] as f64 - c[0];
        let ry = f.py[i] as f64 - c[1];
        let rz = f.pz[i] as f64 - c[2];
        if rx * rx + ry * ry + rz * rz > reach2 {
            continue;
        }
        let lx = m[0] * rx + m[3] * ry + m[6] * rz;
        let ly = m[1] * rx + m[4] * ry + m[7] * rz;
        let lz = m[2] * rx + m[5] * ry + m[8] * rz;
        let (ax, ay, az) = (lx.abs(), ly.abs(), lz.abs());
        if ax < hx + d && az < hz + d && ay < hy + 1.25 * d {
            sub += 1;
        }
        if ax >= ex || ay >= ey || az >= ez {
            continue;
        }
        let (pxn, pyn, pzn) = (ex - ax, ey - ay, ez - az);
        let (k, amt) = if pyn <= pxn && pyn <= pzn {
            (1, pyn * ly.signum())
        } else if pxn <= pzn {
            (0, pxn * lx.signum())
        } else {
            (2, pzn * lz.signum())
        };
        f.px[i] += (m[k] * amt) as f32;
        f.py[i] += (m[3 + k] * amt) as f32;
        f.pz[i] += (m[6 + k] * amt) as f32;
    }
    body.submerged = sub;
}

fn exclude_from_sphere(f: &mut Fluid, body: &mut Body, radius: f64, centre: [f64; 3]) {
    let d = PARTICLE_SPACING as f64;
    let c = body.to_world(&centre);
    let keep_out = radius + 0.3 * d;
    let count_in = radius + d;
    let mut sub = 0u32;
    for i in 0..f.n {
        let rx = f.px[i] as f64 - c[0];
        let ry = f.py[i] as f64 - c[1];
        let rz = f.pz[i] as f64 - c[2];
        let r2 = rx * rx + ry * ry + rz * rz;
        if r2 > count_in * count_in {
            continue;
        }
        sub += 1;
        if r2 >= keep_out * keep_out {
            continue;
        }
        let r = r2.sqrt();
        let (nx, ny, nz) = if r > 1e-9 {
            (rx / r, ry / r, rz / r)
        } else {
            (0.0, 1.0, 0.0)
        };
        let amt = keep_out - r;
        f.px[i] += (nx * amt) as f32;
        f.py[i] += (ny * amt) as f32;
        f.pz[i] += (nz * amt) as f32;
    }
    body.submerged = sub;
}

/// Velocities from positions; wall contact = no penetration + viscous drag toward the wall's speed.
fn update_velocities(f: &mut Fluid, dt: f32, vessel: &impl Vessel, p: &FluidParams) {
    let invdt = 1.0 / dt;
    let keep = 1.0 - p.wall_friction;
    for i in 0..f.n {
        let mut ux = (f.px[i] - f.x[i]) * invdt;
        let mut uy = (f.py[i] - f.y[i]) * invdt;
        let mut uz = (f.pz[i] - f.z[i]) * invdt;
        f.x[i] = f.px[i];
        f.y[i] = f.py[i];
        f.z[i] = f.pz[i];
        let c = f.contact[i];
        if !c.is_empty() {
            let w = vessel.wall_velocity([f.x[i] as f64, f.y[i] as f64, f.z[i] as f64]);
            let (wx, wy, wz) = (w[0] as f32, w[1] as f32, w[2] as f32);
            let (mut rvx, mut rvy, mut rvz) = (ux - wx, uy - wy, uz - wz);
            for n in c.iter() {
                let vn = rvx * n[0] + rvy * n[1] + rvz * n[2];
                if vn < 0.0 {
                    rvx -= vn * n[0];
                    rvy -= vn * n[1];
                    rvz -= vn * n[2];
                }
            }
            ux = wx + rvx * keep;
            uy = wy + rvy * keep;
            uz = wz + rvz * keep;
        }
        (ux, uy, uz) = clamp_speed(ux, uy, uz);
        f.vx[i] = ux;
        f.vy[i] = uy;
        f.vz[i] = uz;
    }
}

/// Hydrostatic buoyancy on bodies. PBF pressure is a per-step correction, not a depth-integrated
/// pressure, so Archimedes is added explicitly: local water density at each sample point gives
/// wetness, local water swirl gives the pressure gradient (rho * v_t^2 / r, pointing inward).
fn apply_buoyancy(f: &mut Fluid, bodies: &mut [Body], shapes: &[BodyShape], dt: f32) {
    let b = &mut f.boundary;
    for k in 0..b.n {
        b.rho[k] = 0.0;
        b.fvx[k] = 0.0;
        b.fvy[k] = 0.0;
        b.fvz[k] = 0.0;
    }
    for i in 0..f.n {
        let bcnt = f.bnbc[i] as usize;
        if bcnt == 0 {
            continue;
        }
        let (xi, yi, zi) = (f.x[i], f.y[i], f.z[i]);
        let bbase = i * MAX_BOUNDARY_NEIGHBOURS;
        for k in 0..bcnt {
            let bj = f.bnb[bbase + k] as usize;
            let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let t = H2 - r2;
            let w = PARTICLE_MASS * POLY6 * t * t * t;
            b.rho[bj] += w;
            b.fvx[bj] += w * f.vx[i];
            b.fvy[bj] += w * f.vy[i];
            b.fvz[bj] += w * f.vz[i];
        }
    }
    for body in bodies.iter_mut() {
        body.wet = 0.0;
    }
    for k in 0..b.n {
        let rho = b.rho[k];
        if rho < 1.0 {
            b.fx[k] = 0.0;
            b.fz[k] = 0.0;
            continue;
        }
        let wet = (rho / WET_REF).min(1.0);
        let body = &mut bodies[b.body[k] as usize];
        let shape = &shapes[body.shape];
        body.wet += wet as f64 / shape.samples.len() as f64;
        let (bx, bz) = (b.x[k], b.z[k]);
        let rr = (bx * bx + bz * bz).sqrt().max(1e-6);
        let (tx, tz) = (bz / rr, -bx / rr);
        let vt = (b.fvx[k] * tx + b.fvz[k] * tz) / rho;
        let ac = vt * vt / rr;
        let fmag = REST_DENSITY * shape.volume_per_sample as f32 * wet * ac;
        let (fxx, fzz) = (-fmag * bx / rr, -fmag * bz / rr);
        b.fx[k] = fxx;
        b.fz[k] = fzz;
        body.accumulate(
            [(fxx * dt) as f64, 0.0, (fzz * dt) as f64],
            [bx as f64, b.y[k] as f64, bz as f64],
        );
    }
}

/// XSPH viscosity, no-slip drag against bodies (momentum-conserving) and the visual foam estimate.
fn apply_viscosity_and_drag(
    f: &mut Fluid,
    bodies: &mut [Body],
    vessel: &impl Vessel,
    dt: f32,
    p: &FluidParams,
) {
    let cv = p.viscosity * PARTICLE_MASS / REST_DENSITY;
    let cd = p.body_drag;
    for i in 0..f.n {
        let b = &f.boundary;
        let (xi, yi, zi) = (f.x[i], f.y[i], f.z[i]);
        let (ui, vi, wi) = (f.vx[i], f.vy[i], f.vz[i]);
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        let (mut sx, mut sy, mut sz, mut sw) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let base = i * MAX_NEIGHBOURS;
        let cnt = f.nbc[i] as usize;
        for k in 0..cnt {
            let j = f.nb[base + k] as usize;
            let (rx, ry, rz) = (xi - f.x[j], yi - f.y[j], zi - f.z[j]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let t = H2 - r2;
            let w0 = POLY6 * t * t * t;
            let w = cv * w0;
            let (dvx, dvy, dvz) = (f.vx[j] - ui, f.vy[j] - vi, f.vz[j] - wi);
            ax += dvx * w;
            ay += dvy * w;
            az += dvz * w;
            sx += dvx * w0;
            sy += dvy * w0;
            sz += dvz * w0;
            sw += w0;
        }
        // agitation: relative motion against neighbours and slip against the walls
        let mut ag = if sw > 0.0 {
            (sx * sx + sy * sy + sz * sz).sqrt() / sw
        } else {
            0.0
        };
        if !f.contact[i].is_empty() {
            let wv = vessel.wall_velocity([xi as f64, yi as f64, zi as f64]);
            let (sx, sy, sz) = (ui - wv[0] as f32, vi - wv[1] as f32, wi - wv[2] as f32);
            ag += 0.5 * (sx * sx + sy * sy + sz * sz).sqrt();
        }
        let target = ((ag - 0.6) / 1.6).clamp(0.0, 1.0);
        let fo = f.foam[i];
        f.foam[i] = if target > fo {
            fo + (target - fo) * 0.25
        } else {
            fo + (target - fo) * 0.015
        };

        // no-slip drag toward the bodies, limited so this particle relaxes at most fully in one substep
        let bbase = i * MAX_BOUNDARY_NEIGHBOURS;
        let bcnt = f.bnbc[i] as usize;
        let mut wsum = 0.0f32;
        for k in 0..bcnt {
            let bj = f.bnb[bbase + k] as usize;
            let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 < H2 {
                let t = H2 - r2;
                wsum += cd * POLY6 * t * t * t * (b.psi[bj] / REST_DENSITY);
            }
        }
        let scale = if wsum > 1.0 { 1.0 / wsum } else { 1.0 };
        for k in 0..bcnt {
            let bj = f.bnb[bbase + k] as usize;
            let (rx, ry, rz) = (xi - b.x[bj], yi - b.y[bj], zi - b.z[bj]);
            let r2 = rx * rx + ry * ry + rz * rz;
            if r2 >= H2 {
                continue;
            }
            let t = H2 - r2;
            let w = cd * POLY6 * t * t * t * (b.psi[bj] / REST_DENSITY) * scale;
            let (dvx, dvy, dvz) = (
                (b.vx[bj] - ui) * w,
                (b.vy[bj] - vi) * w,
                (b.vz[bj] - wi) * w,
            );
            ax += dvx;
            ay += dvy;
            az += dvz;
            bodies[b.body[bj] as usize].accumulate_drag(
                [
                    (-PARTICLE_MASS * dvx) as f64,
                    (-PARTICLE_MASS * dvy) as f64,
                    (-PARTICLE_MASS * dvz) as f64,
                ],
                [b.x[bj] as f64, b.y[bj] as f64, b.z[bj] as f64],
                (PARTICLE_MASS * w) as f64,
            );
            // reaction to buoyancy: share of -F_b, weighted by this particle's kernel contribution
            let rb = b.rho[bj];
            if rb > 1.0 {
                let sh = POLY6 * t * t * t * dt / rb;
                ax -= b.fx[bj] * sh;
                az -= b.fz[bj] * sh;
            }
        }
        f.dx[i] = ax;
        f.dy[i] = ay;
        f.dz[i] = az;
    }
    for i in 0..f.n {
        f.vx[i] += f.dx[i];
        f.vy[i] += f.dy[i];
        f.vz[i] += f.dz[i];
    }
}
