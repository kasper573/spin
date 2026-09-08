import { MAXB, MAXBP, MAXN, MAXP, RHO0 } from './constants';
import { Grid } from './grid';

/** Structure-of-arrays fluid particle store. */
export class Fluid {
  n = 0;
  readonly x = new Float32Array(MAXP);
  readonly y = new Float32Array(MAXP);
  readonly z = new Float32Array(MAXP);
  readonly vx = new Float32Array(MAXP);
  readonly vy = new Float32Array(MAXP);
  readonly vz = new Float32Array(MAXP);
  readonly px = new Float32Array(MAXP);
  readonly py = new Float32Array(MAXP);
  readonly pz = new Float32Array(MAXP);
  readonly lam = new Float32Array(MAXP);
  readonly dx = new Float32Array(MAXP);
  readonly dy = new Float32Array(MAXP);
  readonly dz = new Float32Array(MAXP);
  readonly rho = new Float32Array(MAXP);
  /** Visual agitation in [0, 1]; drives the foam rendering. */
  readonly foam = new Float32Array(MAXP);
  readonly contact = new Uint8Array(MAXP);
  readonly nb = new Int32Array(MAXP * MAXN);
  readonly nbc = new Int32Array(MAXP);
  readonly bnb = new Int32Array(MAXP * MAXB);
  readonly bnbc = new Int32Array(MAXP);
  readonly grid = new Grid(MAXP);

  add(x: number, y: number, z: number, vx: number, vy: number, vz: number): boolean {
    if (this.n >= MAXP) return false;
    const i = this.n++;
    this.x[i] = x;
    this.y[i] = y;
    this.z[i] = z;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.vz[i] = vz;
    this.contact[i] = 0;
    this.rho[i] = RHO0;
    this.foam[i] = 0.25;
    return true;
  }

  remove(i: number): void {
    const n = --this.n;
    if (i === n) return;
    this.x[i] = this.x[n];
    this.y[i] = this.y[n];
    this.z[i] = this.z[n];
    this.vx[i] = this.vx[n];
    this.vy[i] = this.vy[n];
    this.vz[i] = this.vz[n];
    this.rho[i] = this.rho[n];
    this.contact[i] = this.contact[n];
    this.foam[i] = this.foam[n];
  }

  clear(): void {
    this.n = 0;
  }
}

/** Boundary particles sampled from every raft each substep (Akinci-style coupling). */
export class Boundary {
  n = 0;
  readonly x = new Float32Array(MAXBP);
  readonly y = new Float32Array(MAXBP);
  readonly z = new Float32Array(MAXBP);
  readonly vx = new Float32Array(MAXBP);
  readonly vy = new Float32Array(MAXBP);
  readonly vz = new Float32Array(MAXBP);
  /** Boundary pseudo-mass Ψ. */
  readonly psi = new Float32Array(MAXBP);
  readonly raft = new Int32Array(MAXBP);
  /** Fluid density and momentum seen at the sample, and the buoyancy force applied there. */
  readonly rho = new Float32Array(MAXBP);
  readonly fvx = new Float32Array(MAXBP);
  readonly fvy = new Float32Array(MAXBP);
  readonly fvz = new Float32Array(MAXBP);
  readonly fx = new Float32Array(MAXBP);
  readonly fy = new Float32Array(MAXBP);
  readonly fz = new Float32Array(MAXBP);
  readonly grid = new Grid(MAXBP);
}
