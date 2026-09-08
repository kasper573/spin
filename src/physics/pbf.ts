/* Position-based fluids (Macklin & Müller 2013) in the inertial frame, with Akinci-style
   boundary coupling to the rafts and a projection onto the inside of the glass drum. */
import {
  D,
  EPS_LAMBDA,
  H,
  H2,
  ITERS,
  MASS,
  MAX_DP,
  MAX_SPEED,
  MAXB,
  MAXBP,
  MAXN,
  POLY6,
  RHO0,
  RMAX,
  SCORR_K,
  SCORR_WQ,
  SPIKY,
  W0,
  WET_REF,
  YMAX,
} from './constants';
import type { Boundary, Fluid } from './fluid';
import { cellHash } from './grid';
import type { Landscape } from './landscape';
import type { SimParams } from './params';
import { RAFT_TEMPLATE, type Raft } from './raft';
import { vec3 } from './math';

const CONTACT_FLOOR = 1;
const CONTACT_WALL_POS = 4;
const CONTACT_WALL_NEG = 8;

/** Terrain and wheel angle for the substep being solved. */
interface Walls {
  land: Landscape;
  theta: number;
}

const sN = new Float64Array(3);

/** Clamp a predicted position into the drum and out of the terrain; returns contact flags. */
function projectWheel(
  px: Float32Array,
  py: Float32Array,
  pz: Float32Array,
  i: number,
  W: Walls,
): number {
  let c = 0;
  if (W.land.empty) {
    const x = px[i],
      z = pz[i],
      r = Math.sqrt(x * x + z * z);
    if (r > RMAX) {
      const s = RMAX / r;
      px[i] = x * s;
      pz[i] = z * s;
      c |= CONTACT_FLOOR;
    }
  } else {
    for (let k = 0; k < 2; k++) {
      const pen = W.land.penetration(px[i], py[i], pz[i], W.theta, D * 0.5, sN);
      if (pen <= 0) break;
      px[i] += sN[0] * pen;
      py[i] += sN[1] * pen;
      pz[i] += sN[2] * pen;
      c |= CONTACT_FLOOR;
    }
  }
  if (py[i] > YMAX) {
    py[i] = YMAX;
    c |= CONTACT_WALL_POS;
  } else if (py[i] < -YMAX) {
    py[i] = -YMAX;
    c |= CONTACT_WALL_NEG;
  }
  return c;
}

const sWp = vec3(),
  sWv = vec3();

function gatherBoundary(B: Boundary, rafts: Raft[]): void {
  B.n = 0;
  for (let ri = 0; ri < rafts.length; ri++) {
    const r = rafts[ri];
    for (const t of RAFT_TEMPLATE) {
      if (B.n >= MAXBP) return;
      r.toWorld(t, sWp);
      r.pointVel(sWp, sWv);
      const b = B.n++;
      B.x[b] = sWp[0];
      B.y[b] = sWp[1];
      B.z[b] = sWp[2];
      B.vx[b] = sWv[0];
      B.vy[b] = sWv[1];
      B.vz[b] = sWv[2];
      B.psi[b] = t[3];
      B.raft[b] = ri;
    }
  }
}

function findNeighbors(F: Fluid, B: Boundary): void {
  const { px, py, pz, nb, nbc, bnb, bnbc, n } = F;
  const g = F.grid,
    bg = B.grid,
    inv = 1 / H,
    hasB = B.n > 0;
  for (let i = 0; i < n; i++) {
    const xi = px[i],
      yi = py[i],
      zi = pz[i];
    const cx = Math.floor(xi * inv),
      cy = Math.floor(yi * inv),
      cz = Math.floor(zi * inv);
    let c = 0,
      cb = 0;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const h = cellHash(cx + dx, cy + dy, cz + dz);
          for (let k = g.start[h], ke = g.start[h + 1]; k < ke; k++) {
            const j = g.entries[k];
            if (j === i) continue;
            const rx = xi - px[j],
              ry = yi - py[j],
              rz = zi - pz[j];
            if (rx * rx + ry * ry + rz * rz < H2 && c < MAXN) nb[i * MAXN + c++] = j;
          }
          if (!hasB) continue;
          for (let k = bg.start[h], ke = bg.start[h + 1]; k < ke; k++) {
            const b = bg.entries[k];
            const rx = xi - B.x[b],
              ry = yi - B.y[b],
              rz = zi - B.z[b];
            if (rx * rx + ry * ry + rz * rz < H2 && cb < MAXB) bnb[i * MAXB + cb++] = b;
          }
        }
      }
    }
    nbc[i] = c;
    bnbc[i] = cb;
  }
}

function computeLambda(F: Fluid, B: Boundary): void {
  const { px, py, pz, lam, rho, nb, nbc, bnb, bnbc, n } = F;
  const mr = MASS / RHO0;
  for (let i = 0; i < n; i++) {
    const xi = px[i],
      yi = py[i],
      zi = pz[i];
    let dens = MASS * W0,
      gx = 0,
      gy = 0,
      gz = 0,
      sum = 0;
    const base = i * MAXN,
      cnt = nbc[i];
    for (let k = 0; k < cnt; k++) {
      const j = nb[base + k];
      const rx = xi - px[j],
        ry = yi - py[j],
        rz = zi - pz[j],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const t = H2 - r2;
      dens += MASS * POLY6 * t * t * t;
      const r = Math.sqrt(r2);
      if (r < 1e-6) continue;
      const hr = H - r,
        g = ((SPIKY * hr * hr) / r) * mr;
      const jx = g * rx,
        jy = g * ry,
        jz = g * rz;
      gx += jx;
      gy += jy;
      gz += jz;
      sum += jx * jx + jy * jy + jz * jz;
    }
    const bbase = i * MAXB,
      bcnt = bnbc[i];
    for (let k = 0; k < bcnt; k++) {
      const b = bnb[bbase + k];
      const rx = xi - B.x[b],
        ry = yi - B.y[b],
        rz = zi - B.z[b],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const t = H2 - r2;
      dens += B.psi[b] * POLY6 * t * t * t;
      const r = Math.sqrt(r2);
      if (r < 1e-6) continue;
      const hr = H - r,
        g = ((SPIKY * hr * hr) / r) * (B.psi[b] / RHO0);
      gx += g * rx;
      gy += g * ry;
      gz += g * rz;
    }
    sum += gx * gx + gy * gy + gz * gz;
    rho[i] = dens;
    const C = Math.max(dens / RHO0 - 1, 0); // no negative pressure: sparse water is free-flying spray
    lam[i] = -C / (sum + EPS_LAMBDA);
  }
}

function applyDelta(F: Fluid, B: Boundary, rafts: Raft[], W: Walls): void {
  const { px, py, pz, lam, dx, dy, dz, nb, nbc, bnb, bnbc, contact, n } = F;
  const mr = MASS / RHO0;
  for (let i = 0; i < n; i++) {
    const xi = px[i],
      yi = py[i],
      zi = pz[i],
      li = lam[i];
    let ax = 0,
      ay = 0,
      az = 0;
    const base = i * MAXN,
      cnt = nbc[i];
    for (let k = 0; k < cnt; k++) {
      const j = nb[base + k];
      const rx = xi - px[j],
        ry = yi - py[j],
        rz = zi - pz[j],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const r = Math.sqrt(r2);
      if (r < 1e-6) continue;
      const t = H2 - r2,
        wr = (POLY6 * t * t * t) / SCORR_WQ,
        w2 = wr * wr;
      const sc = -SCORR_K * w2 * w2;
      const hr = H - r,
        g = ((SPIKY * hr * hr) / r) * mr * (li + lam[j] + sc);
      ax += g * rx;
      ay += g * ry;
      az += g * rz;
    }
    const bbase = i * MAXB,
      bcnt = bnbc[i];
    for (let k = 0; k < bcnt; k++) {
      const b = bnb[bbase + k];
      const rx = xi - B.x[b],
        ry = yi - B.y[b],
        rz = zi - B.z[b],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const r = Math.sqrt(r2);
      if (r < 1e-6) continue;
      const hr = H - r,
        g = ((SPIKY * hr * hr) / r) * (B.psi[b] / RHO0) * li;
      ax += g * rx;
      ay += g * ry;
      az += g * rz;
    }
    const dl = ax * ax + ay * ay + az * az;
    if (dl > MAX_DP * MAX_DP) {
      const s = MAX_DP / Math.sqrt(dl);
      ax *= s;
      ay *= s;
      az *= s;
    }
    dx[i] = ax;
    dy[i] = ay;
    dz[i] = az;
  }
  for (let i = 0; i < n; i++) {
    px[i] += dx[i];
    py[i] += dy[i];
    pz[i] += dz[i];
    contact[i] |= projectWheel(px, py, pz, i, W);
  }
  excludeFromRafts(F, rafts);
}

/** Hard exclusion of particles from raft volumes (tunnelling guard); also counts submerged samples. */
function excludeFromRafts(F: Fluid, rafts: Raft[]): void {
  const { px, py, pz, n } = F;
  for (let ri = 0; ri < rafts.length; ri++) {
    const r = rafts[ri],
      m = r.m,
      c = r.p;
    const ex = r.hx + 0.06,
      ey = r.hy + 0.08,
      ez = r.hz + 0.06,
      reach = Math.hypot(ex, ey, ez);
    let sub = 0;
    for (let i = 0; i < n; i++) {
      const rx = px[i] - c[0],
        ry = py[i] - c[1],
        rz = pz[i] - c[2];
      if (rx * rx + ry * ry + rz * rz > reach * reach) continue;
      const lx = m[0] * rx + m[3] * ry + m[6] * rz;
      const ly = m[1] * rx + m[4] * ry + m[7] * rz;
      const lz = m[2] * rx + m[5] * ry + m[8] * rz;
      const ax = Math.abs(lx),
        ay = Math.abs(ly),
        az = Math.abs(lz);
      if (ax < r.hx + 0.2 && az < r.hz + 0.2 && ay < r.hy + 0.25) sub++;
      if (ax >= ex || ay >= ey || az >= ez) continue;
      const pxn = ex - ax,
        pyn = ey - ay,
        pzn = ez - az;
      let k: number, amt: number;
      if (pyn <= pxn && pyn <= pzn) {
        k = 1;
        amt = pyn * (ly >= 0 ? 1 : -1);
      } else if (pxn <= pzn) {
        k = 0;
        amt = pxn * (lx >= 0 ? 1 : -1);
      } else {
        k = 2;
        amt = pzn * (lz >= 0 ? 1 : -1);
      }
      px[i] += m[k] * amt;
      py[i] += m[3 + k] * amt;
      pz[i] += m[6 + k] * amt;
    }
    r.submerged = sub;
  }
}

/** Predict, solve constraints, then derive velocities with glass contact and viscosity. */
export function stepFluid(
  F: Fluid,
  B: Boundary,
  dt: number,
  omega: number,
  theta: number,
  land: Landscape,
  rafts: Raft[],
  P: SimParams,
): void {
  const n = F.n;
  const W: Walls = { land, theta };
  gatherBoundary(B, rafts);
  if (n === 0) return;
  const { x, y, z, vx, vy, vz, px, py, pz, contact } = F;
  const airK = P.air ? dt / P.airTau : 0;
  for (let i = 0; i < n; i++) {
    let ux = vx[i],
      uy = vy[i],
      uz = vz[i];
    if (airK > 0) {
      const ax = omega * z[i],
        az = -omega * x[i];
      ux += (ax - ux) * airK;
      uy -= uy * airK;
      uz += (az - uz) * airK;
    }
    const s2 = ux * ux + uy * uy + uz * uz;
    if (s2 > MAX_SPEED * MAX_SPEED) {
      const s = MAX_SPEED / Math.sqrt(s2);
      ux *= s;
      uy *= s;
      uz *= s;
    }
    vx[i] = ux;
    vy[i] = uy;
    vz[i] = uz;
    px[i] = x[i] + ux * dt;
    py[i] = y[i] + uy * dt;
    pz[i] = z[i] + uz * dt;
    contact[i] = projectWheel(px, py, pz, i, W);
  }
  F.grid.build(px, py, pz, n);
  if (B.n > 0) B.grid.build(B.x, B.y, B.z, B.n);
  findNeighbors(F, B);
  for (let it = 0; it < ITERS; it++) {
    computeLambda(F, B);
    applyDelta(F, B, rafts, W);
  }
  updateVelocities(F, dt, omega, P, W);
  if (B.n > 0) applyBuoyancy(F, B, rafts, dt);
  applyViscosityAndDrag(F, B, rafts, omega, dt, P);
}

/** Velocities from positions; wall contact = no penetration + viscous drag toward the wall's speed. */
function updateVelocities(F: Fluid, dt: number, omega: number, P: SimParams, W: Walls): void {
  const { x, y, z, vx, vy, vz, px, py, pz, contact, n } = F;
  const invdt = 1 / dt,
    keep = 1 - P.wallFriction;
  for (let i = 0; i < n; i++) {
    let ux = (px[i] - x[i]) * invdt,
      uy = (py[i] - y[i]) * invdt,
      uz = (pz[i] - z[i]) * invdt;
    x[i] = px[i];
    y[i] = py[i];
    z[i] = pz[i];
    const c = contact[i];
    if (c) {
      const xi = x[i],
        zi = z[i],
        wx = omega * zi,
        wz = -omega * xi;
      let rvx = ux - wx,
        rvy = uy,
        rvz = uz - wz;
      if (c & CONTACT_FLOOR) {
        let nx: number, ny: number, nz: number;
        if (W.land.empty) {
          const r = Math.sqrt(xi * xi + zi * zi) || 1;
          nx = -xi / r;
          ny = 0;
          nz = -zi / r;
        } else {
          W.land.penetration(xi, y[i], zi, W.theta, D * 0.5, sN);
          nx = sN[0];
          ny = sN[1];
          nz = sN[2];
        }
        const vn = rvx * nx + rvy * ny + rvz * nz;
        if (vn < 0) {
          rvx -= vn * nx;
          rvy -= vn * ny;
          rvz -= vn * nz;
        }
      }
      if (c & CONTACT_WALL_POS && rvy > 0) rvy = 0;
      if (c & CONTACT_WALL_NEG && rvy < 0) rvy = 0;
      ux = wx + rvx * keep;
      uy = rvy * keep;
      uz = wz + rvz * keep;
    }
    const s2 = ux * ux + uy * uy + uz * uz;
    if (s2 > MAX_SPEED * MAX_SPEED) {
      const s = MAX_SPEED / Math.sqrt(s2);
      ux *= s;
      uy *= s;
      uz *= s;
    }
    vx[i] = ux;
    vy[i] = uy;
    vz[i] = uz;
  }
}

/* Hydrostatic buoyancy on rafts. PBF pressure is a per-step correction, not a depth-integrated
   pressure, so Archimedes is added explicitly: local water density at each raft sample point gives
   wetness, local water swirl gives the pressure gradient (rho * v_t^2 / r, pointing inward). */
function applyBuoyancy(F: Fluid, B: Boundary, rafts: Raft[], dt: number): void {
  const { x, y, z, vx, vy, vz, bnb, bnbc, n } = F;
  for (let b = 0; b < B.n; b++) {
    B.rho[b] = 0;
    B.fvx[b] = B.fvy[b] = B.fvz[b] = 0;
  }
  for (let i = 0; i < n; i++) {
    const bbase = i * MAXB,
      bcnt = bnbc[i];
    if (!bcnt) continue;
    const xi = x[i],
      yi = y[i],
      zi = z[i];
    for (let k = 0; k < bcnt; k++) {
      const b = bnb[bbase + k];
      const rx = xi - B.x[b],
        ry = yi - B.y[b],
        rz = zi - B.z[b],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const t = H2 - r2,
        w = MASS * POLY6 * t * t * t;
      B.rho[b] += w;
      B.fvx[b] += w * vx[i];
      B.fvy[b] += w * vy[i];
      B.fvz[b] += w * vz[i];
    }
  }
  for (const r of rafts) r.wet = 0;
  for (let b = 0; b < B.n; b++) {
    const rho = B.rho[b];
    if (rho < 1) {
      B.fx[b] = B.fy[b] = B.fz[b] = 0;
      continue;
    }
    const wet = Math.min(1, rho / WET_REF);
    const raft = rafts[B.raft[b]];
    raft.wet += wet / RAFT_TEMPLATE.length;
    const bx = B.x[b],
      bz = B.z[b],
      r = Math.hypot(bx, bz) || 1e-6;
    const tx = bz / r,
      tz = -bx / r;
    const vt = (B.fvx[b] * tx + B.fvz[b] * tz) / rho;
    const ac = (vt * vt) / r;
    const fmag = RHO0 * raft.volPerSample * wet * ac;
    const fxx = (-fmag * bx) / r,
      fzz = (-fmag * bz) / r;
    B.fx[b] = fxx;
    B.fy[b] = 0;
    B.fz[b] = fzz;
    raft.accumulate(fxx * dt, 0, fzz * dt, bx, B.y[b], bz);
  }
}

/** XSPH viscosity, no-slip drag against rafts (momentum-conserving) and the visual foam estimate. */
function applyViscosityAndDrag(
  F: Fluid,
  B: Boundary,
  rafts: Raft[],
  omega: number,
  dt: number,
  P: SimParams,
): void {
  const { x, y, z, vx, vy, vz, dx, dy, dz, nb, nbc, bnb, bnbc, contact, foam, n } = F;
  const cv = (P.viscosity * MASS) / RHO0,
    cd = P.raftDrag;
  for (let i = 0; i < n; i++) {
    const xi = x[i],
      yi = y[i],
      zi = z[i],
      ui = vx[i],
      vi = vy[i],
      wi = vz[i];
    let ax = 0,
      ay = 0,
      az = 0,
      sx = 0,
      sy = 0,
      sz = 0,
      sw = 0;
    const base = i * MAXN,
      cnt = nbc[i];
    for (let k = 0; k < cnt; k++) {
      const j = nb[base + k];
      const rx = xi - x[j],
        ry = yi - y[j],
        rz = zi - z[j],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const t = H2 - r2,
        w0 = POLY6 * t * t * t,
        w = cv * w0;
      ax += (vx[j] - ui) * w;
      ay += (vy[j] - vi) * w;
      az += (vz[j] - wi) * w;
      sx += (vx[j] - ui) * w0;
      sy += (vy[j] - vi) * w0;
      sz += (vz[j] - wi) * w0;
      sw += w0;
    }
    // agitation: relative motion against neighbours and slip against the glass
    let ag = sw > 0 ? Math.hypot(sx, sy, sz) / sw : 0;
    if (contact[i]) ag += 0.5 * Math.hypot(ui - omega * zi, vi, wi + omega * xi);
    const target = Math.max(0, Math.min(1, (ag - 0.6) / 1.6));
    const fo = foam[i];
    foam[i] = target > fo ? fo + (target - fo) * 0.25 : fo + (target - fo) * 0.015;

    const bbase = i * MAXB,
      bcnt = bnbc[i];
    // no-slip drag toward the rafts, limited so this particle relaxes at most fully in one substep
    let wsum = 0;
    for (let k = 0; k < bcnt; k++) {
      const b = bnb[bbase + k];
      const rx = xi - B.x[b],
        ry = yi - B.y[b],
        rz = zi - B.z[b],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 < H2) {
        const t = H2 - r2;
        wsum += cd * POLY6 * t * t * t * (B.psi[b] / RHO0);
      }
    }
    const scale = wsum > 1 ? 1 / wsum : 1;
    for (let k = 0; k < bcnt; k++) {
      const b = bnb[bbase + k];
      const rx = xi - B.x[b],
        ry = yi - B.y[b],
        rz = zi - B.z[b],
        r2 = rx * rx + ry * ry + rz * rz;
      if (r2 >= H2) continue;
      const t = H2 - r2,
        w = cd * POLY6 * t * t * t * (B.psi[b] / RHO0) * scale;
      const dvx = (B.vx[b] - ui) * w,
        dvy = (B.vy[b] - vi) * w,
        dvz = (B.vz[b] - wi) * w;
      ax += dvx;
      ay += dvy;
      az += dvz;
      rafts[B.raft[b]].accumulateDrag(
        -MASS * dvx,
        -MASS * dvy,
        -MASS * dvz,
        B.x[b],
        B.y[b],
        B.z[b],
        MASS * w,
      );
      // reaction to buoyancy: share of -F_b, weighted by this particle's kernel contribution
      const rb = B.rho[b];
      if (rb > 1) {
        const sh = (POLY6 * t * t * t * dt) / rb;
        ax -= B.fx[b] * sh;
        az -= B.fz[b] * sh;
      }
    }
    dx[i] = ax;
    dy[i] = ay;
    dz[i] = az;
  }
  for (let i = 0; i < n; i++) {
    vx[i] += dx[i];
    vy[i] += dy[i];
    vz[i] += dz[i];
  }
}
