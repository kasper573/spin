/* Raft contacts: sequential impulses with Coulomb friction against the moving glass and other rafts. */
import { HALF_W, R_OUT } from './constants';
import { cross, dot, mat3mul, vec3, type Vec3 } from './math';
import type { Landscape } from './landscape';
import type { SimParams } from './params';
import { RAFT_PTS, type Raft } from './raft';

const sWp = vec3(),
  sVp = vec3(),
  sR = vec3(),
  sT = vec3(),
  sT2 = vec3(),
  sN = vec3(),
  sJ = vec3(),
  sRv = vec3();

/** Inverse effective mass of the raft at lever arm r along direction n. */
function effMass(raft: Raft, r: Vec3, n: Vec3): number {
  cross(r, n, sT);
  mat3mul(raft.iw, sT, sT2);
  cross(sT2, r, sT);
  return raft.invM + dot(n, sT);
}

/** Relative velocity of a raft point against the glass moving at omega. */
function relativeToWheel(raft: Raft, wp: Vec3, omega: number, o: Vec3): Vec3 {
  raft.pointVel(wp, sVp);
  o[0] = sVp[0] - omega * wp[2];
  o[1] = sVp[1];
  o[2] = sVp[2] + omega * wp[0];
  return o;
}

function resolveWallContact(
  raft: Raft,
  wp: Vec3,
  n: Vec3,
  pen: number,
  omega: number,
  e: number,
  mu: number,
): void {
  sR[0] = wp[0] - raft.p[0];
  sR[1] = wp[1] - raft.p[1];
  sR[2] = wp[2] - raft.p[2];
  const rv = relativeToWheel(raft, wp, omega, sRv);
  const vn = dot(rv, n);
  if (vn < 0) {
    const kn = effMass(raft, sR, n);
    const rest = vn < -0.5 ? e : 0;
    const j = (-(1 + rest) * vn) / kn;
    sJ[0] = j * n[0];
    sJ[1] = j * n[1];
    sJ[2] = j * n[2];
    raft.applyImpulse(sJ, wp);
    relativeToWheel(raft, wp, omega, rv);
    const vn2 = dot(rv, n);
    const tx = rv[0] - vn2 * n[0],
      ty = rv[1] - vn2 * n[1],
      tz = rv[2] - vn2 * n[2];
    const vt = Math.hypot(tx, ty, tz);
    if (vt > 1e-6) {
      sT2[0] = tx / vt;
      sT2[1] = ty / vt;
      sT2[2] = tz / vt;
      const kt = effMass(raft, sR, sT2);
      const jt = Math.min(mu * j, vt / kt);
      sJ[0] = -jt * sT2[0];
      sJ[1] = -jt * sT2[1];
      sJ[2] = -jt * sT2[2];
      raft.applyImpulse(sJ, wp);
    }
  }
  const corr = Math.min(pen, 0.05) * 0.4;
  raft.p[0] += n[0] * corr;
  raft.p[1] += n[1] * corr;
  raft.p[2] += n[2] * corr;
}

export function collideWheel(
  raft: Raft,
  omega: number,
  theta: number,
  land: Landscape,
  P: SimParams,
): void {
  for (let pass = 0; pass < 2; pass++) {
    for (const lp of RAFT_PTS) {
      raft.toWorld(lp, sWp);
      let pen: number;
      if (land.empty) {
        const x = sWp[0],
          z = sWp[2],
          r = Math.sqrt(x * x + z * z) || 1e-9;
        pen = r - R_OUT;
        sN[0] = -x / r;
        sN[1] = 0;
        sN[2] = -z / r;
      } else {
        pen = land.penetration(sWp[0], sWp[1], sWp[2], theta, 0, sN);
      }
      if (pen > 0) {
        resolveWallContact(raft, sWp, sN, pen, omega, P.restitution, P.raftFriction);
        raft.toWorld(lp, sWp);
      }
      if (sWp[1] > HALF_W) {
        sN[0] = 0;
        sN[1] = -1;
        sN[2] = 0;
        resolveWallContact(raft, sWp, sN, sWp[1] - HALF_W, omega, P.restitution, P.raftFriction);
      } else if (sWp[1] < -HALF_W) {
        sN[0] = 0;
        sN[1] = 1;
        sN[2] = 0;
        resolveWallContact(raft, sWp, sN, -HALF_W - sWp[1], omega, P.restitution, P.raftFriction);
      }
    }
  }
}

const sL = vec3(),
  sRA = vec3(),
  sRB = vec3(),
  sVa = vec3(),
  sVb = vec3(),
  sJt = vec3();

/** Sample points of A against the box of B. Call twice with the roles swapped for symmetry. */
export function collideRaftPair(A: Raft, Bft: Raft, P: SimParams): void {
  for (const lp of RAFT_PTS) {
    A.toWorld(lp, sWp);
    Bft.toLocal(sWp, sL);
    const ax = Math.abs(sL[0]),
      ay = Math.abs(sL[1]),
      az = Math.abs(sL[2]);
    if (ax >= Bft.hx || ay >= Bft.hy || az >= Bft.hz) continue;
    const px = Bft.hx - ax,
      py = Bft.hy - ay,
      pz = Bft.hz - az;
    let k: number, pen: number, sgn: number;
    if (py <= px && py <= pz) {
      k = 1;
      pen = py;
      sgn = sL[1] >= 0 ? 1 : -1;
    } else if (px <= pz) {
      k = 0;
      pen = px;
      sgn = sL[0] >= 0 ? 1 : -1;
    } else {
      k = 2;
      pen = pz;
      sgn = sL[2] >= 0 ? 1 : -1;
    }
    const m = Bft.m;
    sN[0] = m[k] * sgn;
    sN[1] = m[3 + k] * sgn;
    sN[2] = m[6 + k] * sgn;
    sRA[0] = sWp[0] - A.p[0];
    sRA[1] = sWp[1] - A.p[1];
    sRA[2] = sWp[2] - A.p[2];
    sRB[0] = sWp[0] - Bft.p[0];
    sRB[1] = sWp[1] - Bft.p[1];
    sRB[2] = sWp[2] - Bft.p[2];
    A.pointVel(sWp, sVa);
    Bft.pointVel(sWp, sVb);
    sRv[0] = sVa[0] - sVb[0];
    sRv[1] = sVa[1] - sVb[1];
    sRv[2] = sVa[2] - sVb[2];
    const vn = dot(sRv, sN);
    if (vn < 0) {
      const kk = effMass(A, sRA, sN) + effMass(Bft, sRB, sN);
      const j = (-(1 + (vn < -0.5 ? P.restitution : 0)) * vn) / kk;
      sJ[0] = j * sN[0];
      sJ[1] = j * sN[1];
      sJ[2] = j * sN[2];
      A.applyImpulse(sJ, sWp);
      sJ[0] = -sJ[0];
      sJ[1] = -sJ[1];
      sJ[2] = -sJ[2];
      Bft.applyImpulse(sJ, sWp);
      A.pointVel(sWp, sVa);
      Bft.pointVel(sWp, sVb);
      sRv[0] = sVa[0] - sVb[0];
      sRv[1] = sVa[1] - sVb[1];
      sRv[2] = sVa[2] - sVb[2];
      const vn2 = dot(sRv, sN);
      const tx = sRv[0] - vn2 * sN[0],
        ty = sRv[1] - vn2 * sN[1],
        tz = sRv[2] - vn2 * sN[2];
      const vt = Math.hypot(tx, ty, tz);
      if (vt > 1e-6) {
        sT2[0] = tx / vt;
        sT2[1] = ty / vt;
        sT2[2] = tz / vt;
        const kt = effMass(A, sRA, sT2) + effMass(Bft, sRB, sT2);
        const jt = Math.min(P.raftFriction * j, vt / kt);
        sJt[0] = -jt * sT2[0];
        sJt[1] = -jt * sT2[1];
        sJt[2] = -jt * sT2[2];
        A.applyImpulse(sJt, sWp);
        sJt[0] = -sJt[0];
        sJt[1] = -sJt[1];
        sJt[2] = -sJt[2];
        Bft.applyImpulse(sJt, sWp);
      }
    }
    const c = Math.min(pen, 0.05) * 0.25;
    A.p[0] += sN[0] * c;
    A.p[1] += sN[1] * c;
    A.p[2] += sN[2] * c;
    Bft.p[0] -= sN[0] * c;
    Bft.p[1] -= sN[1] * c;
    Bft.p[2] -= sN[2] * c;
  }
}
