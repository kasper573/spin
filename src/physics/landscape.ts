import { HALF_W, R_OUT } from './constants';

/** Grid resolution around the drum and across its width. */
export const LAND_NT = 256;
export const LAND_NY = 32;
export const LAND_HMAX = R_OUT - 1.0;

const TWO_PI = Math.PI * 2;
const DPHI = TWO_PI / LAND_NT;
const DY = (2 * HALF_W) / (LAND_NY - 1);

export interface LandSample {
  /** Height above the glass floor (m). */
  h: number;
  /** Partial derivatives with respect to the wheel angle (rad) and axial position (m). */
  dPhi: number;
  dY: number;
}

/** Angle of a world point in the wheel's own frame. */
export function wheelAngle(x: number, z: number, theta: number): number {
  return Math.atan2(z, x) + theta;
}

/**
 * Sculptable terrain on the inside of the drum floor: a periodic heightfield in the wheel's frame,
 * measured inward from the glass. Height zero means bare glass.
 */
export class Landscape {
  readonly h = new Float32Array(LAND_NT * LAND_NY);
  /** Incremented on every change so renderers can rebuild lazily. */
  version = 0;
  /** True while every height is zero, letting hot loops skip the terrain entirely. */
  empty = true;

  reset(): void {
    this.h.fill(0);
    this.empty = true;
    this.version++;
  }

  /** Replace all heights (clamped to the valid range). */
  load(h: ArrayLike<number>): void {
    let any = false;
    for (let i = 0; i < this.h.length; i++) {
      const v = Math.max(0, Math.min(LAND_HMAX, h[i] || 0));
      this.h[i] = v;
      if (v > 0) any = true;
    }
    this.empty = !any;
    this.version++;
  }

  /** Bilinear height and gradient at wheel angle phi and axial position y. */
  sample(phi: number, y: number, out: LandSample): LandSample {
    const u = (((phi / DPHI) % LAND_NT) + LAND_NT) % LAND_NT;
    const i0 = Math.floor(u),
      fu = u - i0,
      i1 = (i0 + 1) % LAND_NT;
    const v = Math.max(0, Math.min(LAND_NY - 1 - 1e-6, (y + HALF_W) / DY));
    const j0 = Math.floor(v),
      fv = v - j0,
      j1 = j0 + 1;
    const h = this.h;
    const h00 = h[i0 * LAND_NY + j0],
      h10 = h[i1 * LAND_NY + j0];
    const h01 = h[i0 * LAND_NY + j1],
      h11 = h[i1 * LAND_NY + j1];
    out.h = (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv;
    out.dPhi = ((h10 - h00) * (1 - fv) + (h11 - h01) * fv) / DPHI;
    out.dY = ((h01 - h00) * (1 - fu) + (h11 - h10) * fu) / DY;
    return out;
  }

  /**
   * Signed penetration of a world point into the terrain (positive = inside), with the inward
   * unit normal of the terrain surface written to `n`. `margin` shifts the surface inward.
   */
  penetration(
    x: number,
    y: number,
    z: number,
    theta: number,
    margin: number,
    n: Float64Array | Float32Array,
  ): number {
    const r = Math.hypot(x, z);
    if (r < 1e-6) return -R_OUT;
    const s = this.sample(wheelAngle(x, z, theta), y, TMP);
    const f = r - (R_OUT - s.h - margin);
    // f = r - R(phi, y): gradient in world space, then normalised
    const gr = 1,
      gPhi = s.dPhi / r,
      gY = s.dY;
    const gx = (gr * x - gPhi * z) / r,
      gz = (gr * z + gPhi * x) / r,
      gy = gY;
    // ∇f = r̂ + (dh/dφ / r) φ̂ + (dh/dy) ŷ with φ̂ = (-z, 0, x) / r
    const len = Math.hypot(gx, gy, gz) || 1;
    n[0] = -gx / len;
    n[1] = -gy / len;
    n[2] = -gz / len;
    return f / len;
  }

  /** Raise (amount > 0) or lower the terrain around a wheel-frame point with a smooth brush. */
  sculpt(phi: number, y: number, radius: number, amount: number): void {
    const ci = phi / DPHI,
      cj = (y + HALF_W) / DY;
    const ri = Math.ceil(radius / (R_OUT * DPHI)),
      rj = Math.ceil(radius / DY);
    const r2 = radius * radius;
    for (let di = -ri; di <= ri; di++) {
      const i = (((Math.round(ci) + di) % LAND_NT) + LAND_NT) % LAND_NT;
      let dPhi = (i - ci) * DPHI;
      dPhi = ((((dPhi + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;
      const ds = R_OUT * dPhi;
      for (let dj = -rj; dj <= rj; dj++) {
        const j = Math.round(cj) + dj;
        if (j < 0 || j >= LAND_NY) continue;
        const dy = (j - cj) * DY,
          d2 = ds * ds + dy * dy;
        if (d2 >= r2) continue;
        const t = 1 - d2 / r2,
          w = t * t;
        const k = i * LAND_NY + j;
        this.h[k] = Math.max(0, Math.min(LAND_HMAX, this.h[k] + amount * w));
      }
    }
    this.empty = false;
    this.version++;
  }
}

const TMP: LandSample = { h: 0, dPhi: 0, dY: 0 };
