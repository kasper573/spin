import { H2, MAX_SPEED, POLY6, RAFT_L, RAFT_RHO, RAFT_SPACING, RAFT_T, RHO0 } from './constants';
import { cross, mat3mul, quat, vec3, type Quat, type Vec3 } from './math';

/** Boundary sample points on the board's mid-plane: [x, y, z, Ψ]. */
export const RAFT_TEMPLATE: ReadonlyArray<Readonly<[number, number, number, number]>> = (() => {
  const pts: [number, number, number, number][] = [];
  const k = Math.round(RAFT_L / RAFT_SPACING);
  for (let i = 0; i <= k; i++) {
    for (let j = 0; j <= k; j++) {
      pts.push([-RAFT_L / 2 + i * RAFT_SPACING, 0, -RAFT_L / 2 + j * RAFT_SPACING, 0]);
    }
  }
  for (const p of pts) {
    let s = 0;
    for (const q of pts) {
      const dx = p[0] - q[0],
        dz = p[2] - q[2],
        r2 = dx * dx + dz * dz;
      if (r2 < H2) {
        const t = H2 - r2;
        s += POLY6 * t * t * t;
      }
    }
    p[3] = RHO0 / s;
  }
  return pts;
})();

/** Contact sample points on the board: corners, edge midpoints and face centres. */
export const RAFT_PTS: ReadonlyArray<Vec3> = (() => {
  const hx = RAFT_L / 2,
    hy = RAFT_T / 2,
    hz = RAFT_L / 2;
  const pts: Vec3[] = [];
  for (const sx of [-1, 1])
    for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push(vec3(sx * hx, sy * hy, sz * hz));
  for (const sy of [-1, 1]) {
    pts.push(
      vec3(0, sy * hy, hz),
      vec3(0, sy * hy, -hz),
      vec3(hx, sy * hy, 0),
      vec3(-hx, sy * hy, 0),
      vec3(0, sy * hy, 0),
    );
  }
  return pts;
})();

/** Rigid square wooden board. */
export class Raft {
  readonly hx = RAFT_L / 2;
  readonly hy = RAFT_T / 2;
  readonly hz = RAFT_L / 2;
  readonly M = RAFT_RHO * RAFT_L * RAFT_L * RAFT_T;
  readonly invM = 1 / this.M;
  /** Inverse body-space inertia (diagonal) and its largest component. */
  readonly invI: Vec3;
  readonly invIMax: number;
  readonly p: Vec3;
  readonly q: Quat;
  readonly v: Vec3;
  readonly w: Vec3;
  /** Rotation matrix (row-major) and world-space inverse inertia. */
  readonly m = new Float64Array(9);
  readonly iw = new Float64Array(9);
  readonly accJ = vec3();
  readonly accL = vec3();
  /** Drag impulses are kept apart so their relaxation can be limited to one full step. */
  readonly dragJ = vec3();
  readonly dragL = vec3();
  private dragK = 0;
  private dragKa = 0;
  submerged = 0;
  wet = 0;
  readonly volPerSample = (RAFT_L * RAFT_L * RAFT_T) / RAFT_TEMPLATE.length;

  constructor(p: Vec3, q?: Quat, v?: Vec3, w?: Vec3) {
    const a = RAFT_L,
      b = RAFT_T,
      c = RAFT_L;
    this.invI = vec3(
      12 / (this.M * (b * b + c * c)),
      12 / (this.M * (a * a + c * c)),
      12 / (this.M * (a * a + b * b)),
    );
    this.invIMax = Math.max(this.invI[0], this.invI[1], this.invI[2]);
    this.p = Float64Array.from(p);
    this.q = q ? Float64Array.from(q) : quat();
    this.v = v ? Float64Array.from(v) : vec3();
    this.w = w ? Float64Array.from(w) : vec3();
    this.updateRot();
  }

  updateRot(): void {
    const [x, y, z, w] = this.q,
      m = this.m;
    m[0] = 1 - 2 * (y * y + z * z);
    m[1] = 2 * (x * y - z * w);
    m[2] = 2 * (x * z + y * w);
    m[3] = 2 * (x * y + z * w);
    m[4] = 1 - 2 * (x * x + z * z);
    m[5] = 2 * (y * z - x * w);
    m[6] = 2 * (x * z - y * w);
    m[7] = 2 * (y * z + x * w);
    m[8] = 1 - 2 * (x * x + y * y);
    const I = this.invI,
      iw = this.iw;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += m[a * 3 + k] * I[k] * m[b * 3 + k];
        iw[a * 3 + b] = s;
      }
    }
  }

  toWorld(l: ArrayLike<number>, o: Vec3): Vec3 {
    const m = this.m,
      p = this.p;
    o[0] = m[0] * l[0] + m[1] * l[1] + m[2] * l[2] + p[0];
    o[1] = m[3] * l[0] + m[4] * l[1] + m[5] * l[2] + p[1];
    o[2] = m[6] * l[0] + m[7] * l[1] + m[8] * l[2] + p[2];
    return o;
  }

  toLocal(wp: ArrayLike<number>, o: Vec3): Vec3 {
    const m = this.m,
      r0 = wp[0] - this.p[0],
      r1 = wp[1] - this.p[1],
      r2 = wp[2] - this.p[2];
    o[0] = m[0] * r0 + m[3] * r1 + m[6] * r2;
    o[1] = m[1] * r0 + m[4] * r1 + m[7] * r2;
    o[2] = m[2] * r0 + m[5] * r1 + m[8] * r2;
    return o;
  }

  pointVel(wp: ArrayLike<number>, o: Vec3): Vec3 {
    const rx = wp[0] - this.p[0],
      ry = wp[1] - this.p[1],
      rz = wp[2] - this.p[2],
      w = this.w;
    o[0] = w[1] * rz - w[2] * ry + this.v[0];
    o[1] = w[2] * rx - w[0] * rz + this.v[1];
    o[2] = w[0] * ry - w[1] * rx + this.v[2];
    return o;
  }

  private readonly sR = vec3();
  private readonly sLv = vec3();
  private readonly sDw = vec3();

  applyImpulse(J: ArrayLike<number>, at: ArrayLike<number>): void {
    this.v[0] += J[0] * this.invM;
    this.v[1] += J[1] * this.invM;
    this.v[2] += J[2] * this.invM;
    const r = this.sR;
    r[0] = at[0] - this.p[0];
    r[1] = at[1] - this.p[1];
    r[2] = at[2] - this.p[2];
    const dw = mat3mul(this.iw, cross(r, J, this.sLv), this.sDw);
    this.w[0] += dw[0];
    this.w[1] += dw[1];
    this.w[2] += dw[2];
  }

  /** Accumulate an impulse applied at a world point, to be applied once per substep. */
  accumulate(jx: number, jy: number, jz: number, ax: number, ay: number, az: number): void {
    this.accJ[0] += jx;
    this.accJ[1] += jy;
    this.accJ[2] += jz;
    const rx = ax - this.p[0],
      ry = ay - this.p[1],
      rz = az - this.p[2];
    this.accL[0] += ry * jz - rz * jy;
    this.accL[1] += rz * jx - rx * jz;
    this.accL[2] += rx * jy - ry * jx;
  }

  /**
   * Accumulate a drag impulse from one particle–sample pair; `mw` is the pair's coupling mass, used
   * to measure how far the summed drag would relax this body in a single substep.
   */
  accumulateDrag(
    jx: number,
    jy: number,
    jz: number,
    ax: number,
    ay: number,
    az: number,
    mw: number,
  ): void {
    this.dragJ[0] += jx;
    this.dragJ[1] += jy;
    this.dragJ[2] += jz;
    const rx = ax - this.p[0],
      ry = ay - this.p[1],
      rz = az - this.p[2];
    this.dragL[0] += ry * jz - rz * jy;
    this.dragL[1] += rz * jx - rx * jz;
    this.dragL[2] += rx * jy - ry * jx;
    this.dragK += mw * this.invM;
    this.dragKa += mw * (rx * rx + ry * ry + rz * rz) * this.invIMax;
  }

  /** Apply the accumulated water impulse, limited to `maxDv` of velocity change. */
  applyAcc(maxDv: number): void {
    const dJ = this.dragJ,
      dL = this.dragL;
    const sLin = this.dragK > 1 ? 1 / this.dragK : 1,
      sAng = this.dragKa > 1 ? 1 / this.dragKa : 1;
    this.v[0] += dJ[0] * this.invM * sLin;
    this.v[1] += dJ[1] * this.invM * sLin;
    this.v[2] += dJ[2] * this.invM * sLin;
    const ddw = mat3mul(this.iw, dL, this.sDw);
    this.w[0] += ddw[0] * sAng;
    this.w[1] += ddw[1] * sAng;
    this.w[2] += ddw[2] * sAng;
    dJ.fill(0);
    dL.fill(0);
    this.dragK = this.dragKa = 0;
    const J = this.accJ,
      L = this.accL,
      cap = this.M * maxDv;
    const jm = Math.hypot(J[0], J[1], J[2]);
    if (jm > cap) {
      const s = cap / jm;
      J[0] *= s;
      J[1] *= s;
      J[2] *= s;
      L[0] *= s;
      L[1] *= s;
      L[2] *= s;
    }
    this.v[0] += J[0] * this.invM;
    this.v[1] += J[1] * this.invM;
    this.v[2] += J[2] * this.invM;
    const dw = mat3mul(this.iw, L, this.sDw);
    this.w[0] += dw[0];
    this.w[1] += dw[1];
    this.w[2] += dw[2];
    J.fill(0);
    L.fill(0);
  }

  integrate(dt: number): void {
    const v = this.v,
      w = this.w,
      q = this.q,
      p = this.p;
    const sp = Math.hypot(v[0], v[1], v[2]);
    if (sp > MAX_SPEED) {
      const s = MAX_SPEED / sp;
      v[0] *= s;
      v[1] *= s;
      v[2] *= s;
    }
    const ws = Math.hypot(w[0], w[1], w[2]);
    if (ws > 25) {
      const s = 25 / ws;
      w[0] *= s;
      w[1] *= s;
      w[2] *= s;
    }
    p[0] += v[0] * dt;
    p[1] += v[1] * dt;
    p[2] += v[2] * dt;
    const qx = q[0],
      qy = q[1],
      qz = q[2],
      qw = q[3];
    q[0] += 0.5 * (w[0] * qw + w[1] * qz - w[2] * qy) * dt;
    q[1] += 0.5 * (w[1] * qw + w[2] * qx - w[0] * qz) * dt;
    q[2] += 0.5 * (w[2] * qw + w[0] * qy - w[1] * qx) * dt;
    q[3] += 0.5 * (-w[0] * qx - w[1] * qy - w[2] * qz) * dt;
    const l = Math.hypot(q[0], q[1], q[2], q[3]);
    q[0] /= l;
    q[1] /= l;
    q[2] /= l;
    q[3] /= l;
    this.updateRot();
  }
}
