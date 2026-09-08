import { BufferAttribute, BufferGeometry, DoubleSide, Mesh, MeshStandardMaterial } from 'three';
import { HALF_W, R_OUT } from '../physics/constants';
import { LAND_HMAX, LAND_NT, LAND_NY, type Landscape } from '../physics/landscape';

const VT = LAND_NT + 1; // duplicate the seam column
const EPS_H = 0.003; // cells below this stay bare glass and are not drawn
const SURFACE_VERTS = VT * LAND_NY;
// two skirt strips per cap (terrain edge row and floor row), so the terrain reads as a solid body
const SKIRT_BASE = SURFACE_VERTS;
const TOTAL_VERTS = SURFACE_VERTS + 4 * VT;
const MAX_INDICES = (LAND_NT * (LAND_NY - 1) + 2 * LAND_NT) * 6;

// height-graded palette: wet sand → grass → rock; cut faces are darker earth
const SAND = [0.74, 0.66, 0.5];
const GRASS = [0.38, 0.6, 0.3];
const ROCK = [0.5, 0.48, 0.47];
const EARTH = [0.42, 0.34, 0.26];

function shade(t: number, out: Float32Array, k: number): void {
  const a = t < 0.35 ? SAND : GRASS,
    b = t < 0.35 ? GRASS : ROCK,
    f = t < 0.35 ? t / 0.35 : (t - 0.35) / 0.65;
  out[k] = a[0] + (b[0] - a[0]) * f;
  out[k + 1] = a[1] + (b[1] - a[1]) * f;
  out[k + 2] = a[2] + (b[2] - a[2]) * f;
}

/** Mesh of the sculpted terrain in the wheel's frame; rebuilt whenever the heightfield changes. */
export class LandscapeMesh {
  readonly mesh: Mesh;
  private readonly geometry = new BufferGeometry();
  private readonly positions = new Float32Array(TOTAL_VERTS * 3);
  private readonly colors = new Float32Array(TOTAL_VERTS * 3);
  private readonly indices = new Uint32Array(MAX_INDICES);
  private version = -1;

  constructor() {
    this.geometry.setAttribute('position', new BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('color', new BufferAttribute(this.colors, 3));
    this.geometry.setIndex(new BufferAttribute(this.indices, 1));
    this.geometry.setDrawRange(0, 0);
    this.mesh = new Mesh(
      this.geometry,
      new MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.95,
        metalness: 0,
        side: DoubleSide,
      }),
    );
    this.mesh.frustumCulled = false;
  }

  sync(land: Landscape): void {
    if (land.version === this.version) return;
    this.version = land.version;
    const pos = this.positions,
      col = this.colors,
      h = land.h;
    for (let i = 0; i < VT; i++) {
      const phi = (i / LAND_NT) * Math.PI * 2,
        c = Math.cos(phi),
        s = Math.sin(phi);
      const src = (i % LAND_NT) * LAND_NY;
      for (let j = 0; j < LAND_NY; j++) {
        const hv = h[src + j],
          r = R_OUT - hv,
          k = (i * LAND_NY + j) * 3;
        pos[k] = c * r;
        pos[k + 1] = -HALF_W + (j / (LAND_NY - 1)) * 2 * HALF_W;
        pos[k + 2] = s * r;
        shade(hv / LAND_HMAX, col, k);
      }
      // skirts: for each cap, a vertex on the terrain edge and one on the glass floor below it
      for (let cap = 0; cap < 2; cap++) {
        const j = cap === 0 ? 0 : LAND_NY - 1,
          y = cap === 0 ? -HALF_W : HALF_W,
          hv = h[src + j];
        const top = (SKIRT_BASE + cap * 2 * VT + i) * 3,
          bottom = (SKIRT_BASE + (cap * 2 + 1) * VT + i) * 3;
        pos[top] = c * (R_OUT - hv);
        pos[top + 1] = y;
        pos[top + 2] = s * (R_OUT - hv);
        pos[bottom] = c * R_OUT;
        pos[bottom + 1] = y;
        pos[bottom + 2] = s * R_OUT;
        for (let q = 0; q < 3; q++) {
          col[top + q] = EARTH[q];
          col[bottom + q] = EARTH[q] * 0.8;
        }
      }
    }
    let n = 0;
    const idx = this.indices;
    for (let i = 0; i < LAND_NT; i++) {
      const i1 = (i + 1) % LAND_NT;
      for (let j = 0; j < LAND_NY - 1; j++) {
        const hmax = Math.max(
          h[i * LAND_NY + j],
          h[i * LAND_NY + j + 1],
          h[i1 * LAND_NY + j],
          h[i1 * LAND_NY + j + 1],
        );
        if (hmax < EPS_H) continue;
        const a = i * LAND_NY + j,
          b = (i + 1) * LAND_NY + j,
          c = a + 1,
          d = b + 1;
        // wound so the face normal points toward the drum axis
        idx[n++] = a;
        idx[n++] = c;
        idx[n++] = b;
        idx[n++] = b;
        idx[n++] = c;
        idx[n++] = d;
      }
      for (let cap = 0; cap < 2; cap++) {
        const j = cap === 0 ? 0 : LAND_NY - 1;
        if (Math.max(h[i * LAND_NY + j], h[i1 * LAND_NY + j]) < EPS_H) continue;
        const t0 = SKIRT_BASE + cap * 2 * VT + i,
          t1 = t0 + 1,
          b0 = t0 + VT,
          b1 = b0 + 1;
        idx[n++] = t0;
        idx[n++] = b0;
        idx[n++] = t1;
        idx[n++] = t1;
        idx[n++] = b0;
        idx[n++] = b1;
      }
    }
    this.geometry.setDrawRange(0, n);
    const g = this.geometry;
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    if (g.index) g.index.needsUpdate = true;
    g.computeVertexNormals();
  }
}
