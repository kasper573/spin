export type Vec3 = Float64Array;
export type Quat = Float64Array;

export const vec3 = (x = 0, y = 0, z = 0): Vec3 => Float64Array.of(x, y, z);
export const quat = (x = 0, y = 0, z = 0, w = 1): Quat => Float64Array.of(x, y, z, w);

export function cross(a: ArrayLike<number>, b: ArrayLike<number>, o: Vec3): Vec3 {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
  return o;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Row-major 3x3 matrix times vector. */
export function mat3mul(m: ArrayLike<number>, v: ArrayLike<number>, o: Vec3): Vec3 {
  o[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  o[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  o[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
  return o;
}

/** Quaternion whose rotation maps local axes onto the given orthonormal basis (columns). */
export function quatFromBasis(
  ex: ArrayLike<number>,
  ey: ArrayLike<number>,
  ez: ArrayLike<number>,
): Quat {
  const m0 = ex[0],
    m1 = ey[0],
    m2 = ez[0];
  const m3 = ex[1],
    m4 = ey[1],
    m5 = ez[1];
  const m6 = ex[2],
    m7 = ey[2],
    m8 = ez[2];
  const tr = m0 + m4 + m8;
  let x: number, y: number, z: number, w: number, s: number;
  if (tr > 0) {
    s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (m7 - m5) / s;
    y = (m2 - m6) / s;
    z = (m3 - m1) / s;
  } else if (m0 > m4 && m0 > m8) {
    s = Math.sqrt(1 + m0 - m4 - m8) * 2;
    w = (m7 - m5) / s;
    x = 0.25 * s;
    y = (m1 + m3) / s;
    z = (m2 + m6) / s;
  } else if (m4 > m8) {
    s = Math.sqrt(1 + m4 - m0 - m8) * 2;
    w = (m2 - m6) / s;
    x = (m1 + m3) / s;
    y = 0.25 * s;
    z = (m5 + m7) / s;
  } else {
    s = Math.sqrt(1 + m8 - m0 - m4) * 2;
    w = (m3 - m1) / s;
    x = (m2 + m6) / s;
    y = (m5 + m7) / s;
    z = 0.25 * s;
  }
  return quat(x, y, z, w);
}
