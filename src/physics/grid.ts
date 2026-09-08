import { H } from './constants';

const GRID_BITS = 15;
const GRID_SIZE = 1 << GRID_BITS;
const GRID_MASK = GRID_SIZE - 1;

export function cellHash(ix: number, iy: number, iz: number): number {
  return (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) & GRID_MASK;
}

/** Counting-sort spatial hash over kernel-sized cells. */
export class Grid {
  readonly start = new Int32Array(GRID_SIZE + 1);
  readonly count = new Int32Array(GRID_SIZE);
  readonly entries: Int32Array;
  readonly hash: Int32Array;
  n = 0;

  constructor(cap: number) {
    this.entries = new Int32Array(cap);
    this.hash = new Int32Array(cap);
  }

  build(x: Float32Array, y: Float32Array, z: Float32Array, n: number): void {
    this.n = n;
    const cnt = this.count,
      hs = this.hash,
      st = this.start,
      en = this.entries,
      inv = 1 / H;
    cnt.fill(0);
    for (let i = 0; i < n; i++) {
      const h = cellHash(Math.floor(x[i] * inv), Math.floor(y[i] * inv), Math.floor(z[i] * inv));
      hs[i] = h;
      cnt[h]++;
    }
    st[0] = 0;
    for (let c = 0; c < GRID_SIZE; c++) st[c + 1] = st[c] + cnt[c];
    cnt.fill(0);
    for (let i = 0; i < n; i++) {
      const h = hs[i];
      en[st[h] + cnt[h]++] = i;
    }
  }
}
