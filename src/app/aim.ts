import { Vector3 } from 'three';
import { HALF_W, R_OUT } from '../physics/constants';
import type { Landscape } from '../physics/landscape';

export interface Aim {
  /** Point on the drum's inner surface or terrain hit by the ray. */
  point: Vector3;
  /** Inward-facing surface normal at that point. */
  normal: Vector3;
}

const EPS = 1e-6;
const MARCH_STEP = 0.04;
const tmpN = new Float64Array(3);

/**
 * Casts a ray through the drum. Returns the first terrain surface it meets, or otherwise the
 * point where it leaves the glass interior: the far wall or cap the crosshair rests on, whether
 * the camera is inside or outside the drum. Null if the ray misses.
 */
export function aimAtDrum(origin: Vector3, dir: Vector3, land?: Landscape, theta = 0): Aim | null {
  let tIn = -Infinity,
    tOut = Infinity;

  // infinite cylinder |xz| <= R
  const a = dir.x * dir.x + dir.z * dir.z;
  const b = 2 * (origin.x * dir.x + origin.z * dir.z);
  const c = origin.x * origin.x + origin.z * origin.z - R_OUT * R_OUT;
  if (a > EPS) {
    const disc = b * b - 4 * a * c;
    if (disc < 0) return null;
    const s = Math.sqrt(disc);
    tIn = (-b - s) / (2 * a);
    tOut = (-b + s) / (2 * a);
  } else if (c > 0) {
    return null;
  }

  // slab |y| <= HALF_W
  if (Math.abs(dir.y) > EPS) {
    const t0 = (-HALF_W - origin.y) / dir.y,
      t1 = (HALF_W - origin.y) / dir.y;
    tIn = Math.max(tIn, Math.min(t0, t1));
    tOut = Math.min(tOut, Math.max(t0, t1));
  } else if (Math.abs(origin.y) > HALF_W) {
    return null;
  }

  const tStart = Math.max(tIn, 0);
  if (tOut <= tStart) return null;

  if (land && !land.empty) {
    const hit = marchTerrain(origin, dir, tStart, tOut, land, theta);
    if (hit) return hit;
  }

  const point = origin.clone().addScaledVector(dir, tOut);
  const normal = new Vector3();
  if (Math.abs(Math.abs(point.y) - HALF_W) < 1e-4 * HALF_W) {
    normal.set(0, point.y > 0 ? -1 : 1, 0);
  } else {
    const r = Math.hypot(point.x, point.z) || 1;
    normal.set(-point.x / r, 0, -point.z / r);
  }
  return { point, normal };
}

function penetrationAt(
  origin: Vector3,
  dir: Vector3,
  t: number,
  land: Landscape,
  theta: number,
): number {
  return land.penetration(
    origin.x + dir.x * t,
    origin.y + dir.y * t,
    origin.z + dir.z * t,
    theta,
    0,
    tmpN,
  );
}

/** Fixed-step march along the ray, refined by bisection at the first terrain crossing. */
function marchTerrain(
  origin: Vector3,
  dir: Vector3,
  t0: number,
  t1: number,
  land: Landscape,
  theta: number,
): Aim | null {
  let tPrev = t0;
  if (penetrationAt(origin, dir, t0, land, theta) > 0) return null;
  for (let t = t0 + MARCH_STEP; t < t1; t += MARCH_STEP) {
    const inside = penetrationAt(origin, dir, t, land, theta) > 0;
    if (inside) {
      let lo = tPrev,
        hi = t;
      for (let k = 0; k < 8; k++) {
        const mid = (lo + hi) / 2;
        if (penetrationAt(origin, dir, mid, land, theta) > 0) hi = mid;
        else lo = mid;
      }
      penetrationAt(origin, dir, lo, land, theta);
      const point = origin.clone().addScaledVector(dir, lo);
      return { point, normal: new Vector3(tmpN[0], tmpN[1], tmpN[2]) };
    }
    tPrev = t;
  }
  return null;
}
