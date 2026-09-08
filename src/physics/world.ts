import { MAX_RAFTS, RMAX, YMAX } from './constants';
import { collideRaftPair, collideWheel } from './contacts';
import { Boundary, Fluid } from './fluid';
import { quatFromBasis, vec3, type Vec3 } from './math';
import { defaultParams, type SimParams } from './params';
import { stepFluid } from './pbf';
import { Raft } from './raft';

export interface SimState {
  fluid: Fluid;
  boundary: Boundary;
  rafts: Raft[];
  /** Drum angular speed and its target (rad/s), accumulated angle and simulated time. */
  omega: number;
  omegaTarget: number;
  theta: number;
  time: number;
  params: SimParams;
}

export function makeState(): SimState {
  return {
    fluid: new Fluid(),
    boundary: new Boundary(),
    rafts: [],
    omega: 0,
    omegaTarget: 0,
    theta: 0,
    time: 0,
    params: defaultParams(),
  };
}

export function step(S: SimState, dt: number): void {
  const P = S.params;
  const dO = S.omegaTarget - S.omega,
    md = P.spinAccel * dt;
  S.omega += Math.max(-md, Math.min(md, dO));
  S.theta += S.omega * dt;
  S.time += dt;
  const rafts = S.rafts,
    omega = S.omega;
  stepFluid(S.fluid, S.boundary, dt, omega, rafts, P);
  const airK = P.air ? dt / P.airTau : 0;
  for (const r of rafts) {
    r.applyAcc();
    if (airK > 0) {
      r.v[0] += (omega * r.p[2] - r.v[0]) * airK;
      r.v[1] -= r.v[1] * airK;
      r.v[2] += (-omega * r.p[0] - r.v[2]) * airK;
      r.w[0] -= r.w[0] * airK;
      r.w[1] += (omega - r.w[1]) * airK;
      r.w[2] -= r.w[2] * airK;
    }
    r.integrate(dt);
  }
  for (const r of rafts) collideWheel(r, omega, P);
  for (let a = 0; a < rafts.length; a++) {
    for (let b = a + 1; b < rafts.length; b++) {
      collideRaftPair(rafts[a], rafts[b], P);
      collideRaftPair(rafts[b], rafts[a], P);
    }
  }
}

/** Velocity of the glass at a point (x, z). */
export function wheelVelocity(S: SimState, x: number, z: number): Vec3 {
  return vec3(S.omega * z, 0, -S.omega * x);
}

/** Basis for a raft lying flat against the floor at angle a. */
export function floorBasis(a: number) {
  return quatFromBasis([-Math.sin(a), 0, Math.cos(a)], [Math.cos(a), 0, Math.sin(a)], [0, 1, 0]);
}

export function spawnRaft(
  S: SimState,
  x: number,
  y: number,
  z: number,
  matchWheel: boolean,
): Raft | null {
  if (S.rafts.length >= MAX_RAFTS) return null;
  const q = floorBasis(Math.atan2(z, x));
  const v = matchWheel ? wheelVelocity(S, x, z) : vec3();
  const w = matchWheel ? vec3(0, S.omega, 0) : vec3();
  const r = new Raft(vec3(x, y, z), q, v, w);
  S.rafts.push(r);
  return r;
}

/** Inject up to `count` particles in a small cloud around a point; returns how many were added. */
export function injectAt(
  S: SimState,
  x: number,
  y: number,
  z: number,
  count: number,
  matchWheel: boolean,
  rng: () => number = Math.random,
): number {
  let added = 0;
  for (let k = 0; k < count; k++) {
    let px = x + (rng() - 0.5) * 0.5,
      py = y + (rng() - 0.5) * 0.5,
      pz = z + (rng() - 0.5) * 0.5;
    const r = Math.hypot(px, pz);
    if (r > RMAX - 0.05) {
      const rc = RMAX - 0.05;
      px *= rc / r;
      pz *= rc / r;
    }
    py = Math.max(-YMAX + 0.05, Math.min(YMAX - 0.05, py));
    const v = matchWheel ? wheelVelocity(S, px, pz) : vec3();
    const ok = S.fluid.add(
      px,
      py,
      pz,
      v[0] + (rng() - 0.5) * 0.3,
      v[1] + (rng() - 0.5) * 0.3,
      v[2] + (rng() - 0.5) * 0.3,
    );
    if (!ok) break;
    added++;
  }
  return added;
}
