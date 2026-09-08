import { Vector3 } from 'three';
import { HALF_W, R_OUT } from '../physics/constants';

export interface Aim {
  /** Point on the drum's inner surface hit by the ray. */
  point: Vector3;
  /** Inward-facing surface normal at that point. */
  normal: Vector3;
}

const EPS = 1e-6;

/**
 * Casts a ray through the drum and returns where it leaves the interior: the far wall or cap the
 * crosshair rests on, whether the camera is inside or outside the glass. Null if the ray misses.
 */
export function aimAtDrum(origin: Vector3, dir: Vector3): Aim | null {
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

  if (tOut <= Math.max(tIn, 0)) return null;
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
