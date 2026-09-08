import { MAXP } from './constants';
import { Raft } from './raft';
import { makeState, type SimState } from './world';

/** Plain-data snapshot of the simulation, with typed arrays for the bulk fields. */
export interface SimSnapshot {
  omega: number;
  omegaTarget: number;
  theta: number;
  time: number;
  fluid: {
    n: number;
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
    vx: Float32Array;
    vy: Float32Array;
    vz: Float32Array;
    foam: Float32Array;
  };
  rafts: { p: number[]; q: number[]; v: number[]; w: number[] }[];
  landscape: Float32Array;
}

export function serializeState(S: SimState): SimSnapshot {
  const F = S.fluid,
    n = F.n;
  const cut = (a: Float32Array) => a.slice(0, n);
  return {
    omega: S.omega,
    omegaTarget: S.omegaTarget,
    theta: S.theta,
    time: S.time,
    fluid: {
      n,
      x: cut(F.x),
      y: cut(F.y),
      z: cut(F.z),
      vx: cut(F.vx),
      vy: cut(F.vy),
      vz: cut(F.vz),
      foam: cut(F.foam),
    },
    rafts: S.rafts.map((r) => ({
      p: Array.from(r.p),
      q: Array.from(r.q),
      v: Array.from(r.v),
      w: Array.from(r.w),
    })),
    landscape: S.landscape.h.slice(),
  };
}

/** Rebuild a simulation from a snapshot; malformed parts fall back to defaults. */
export function deserializeState(snap: SimSnapshot): SimState {
  const S = makeState();
  S.omega = snap.omega || 0;
  S.omegaTarget = snap.omegaTarget || 0;
  S.theta = snap.theta || 0;
  S.time = snap.time || 0;
  const f = snap.fluid,
    n = Math.min(
      MAXP,
      f.n | 0,
      f.x.length,
      f.y.length,
      f.z.length,
      f.vx.length,
      f.vy.length,
      f.vz.length,
    );
  for (let i = 0; i < n; i++) S.fluid.add(f.x[i], f.y[i], f.z[i], f.vx[i], f.vy[i], f.vz[i]);
  if (f.foam.length >= n) S.fluid.foam.set(f.foam.subarray(0, n));
  for (const r of snap.rafts) {
    if (r.p.length !== 3 || r.q.length !== 4 || r.v.length !== 3 || r.w.length !== 3) continue;
    S.rafts.push(
      new Raft(
        Float64Array.from(r.p),
        Float64Array.from(r.q),
        Float64Array.from(r.v),
        Float64Array.from(r.w),
      ),
    );
  }
  if (snap.landscape.length === S.landscape.h.length) S.landscape.load(snap.landscape);
  return S;
}
