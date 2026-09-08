import type { CameraSnapshot } from '../render/camera';
import type { SimSnapshot } from '../physics/serialize';
import type { Settings } from './settings';

const KEY = 'spin-gravity-wheel/v1';

export interface Snapshot {
  settings: Partial<Settings>;
  sim: SimSnapshot;
  camera: CameraSnapshot;
}

/** Storage form: typed arrays are base64 strings. */
type Stored = Omit<Snapshot, 'sim'> & {
  sim: Omit<SimSnapshot, 'fluid' | 'landscape'> & {
    fluid: Record<'x' | 'y' | 'z' | 'vx' | 'vy' | 'vz' | 'foam', string> & { n: number };
    landscape: string;
  };
};

function encode(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + 0x8000)));
  }
  return btoa(s);
}

function decode(s: unknown): Float32Array {
  if (typeof s !== 'string') return new Float32Array(0);
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, Math.floor(bytes.length / 4));
}

export function saveSnapshot(snap: Snapshot): void {
  const f = snap.sim.fluid;
  const stored: Stored = {
    settings: snap.settings,
    camera: snap.camera,
    sim: {
      ...snap.sim,
      fluid: {
        n: f.n,
        x: encode(f.x),
        y: encode(f.y),
        z: encode(f.z),
        vx: encode(f.vx),
        vy: encode(f.vy),
        vz: encode(f.vz),
        foam: encode(f.foam),
      },
      landscape: encode(snap.sim.landscape),
    },
  };
  try {
    localStorage.setItem(KEY, JSON.stringify(stored));
  } catch {
    // storage may be full or unavailable; the simulation keeps running without persistence
  }
}

export function loadSnapshot(): Snapshot | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Stored;
    if (!s || typeof s !== 'object' || !s.sim || !s.sim.fluid) return null;
    const f = s.sim.fluid;
    return {
      settings: s.settings && typeof s.settings === 'object' ? s.settings : {},
      camera: s.camera,
      sim: {
        omega: +s.sim.omega,
        omegaTarget: +s.sim.omegaTarget,
        theta: +s.sim.theta,
        time: +s.sim.time,
        fluid: {
          n: f.n | 0,
          x: decode(f.x),
          y: decode(f.y),
          z: decode(f.z),
          vx: decode(f.vx),
          vy: decode(f.vy),
          vz: decode(f.vz),
          foam: decode(f.foam),
        },
        rafts: Array.isArray(s.sim.rafts) ? s.sim.rafts : [],
        landscape: decode(s.sim.landscape),
      },
    };
  } catch {
    return null;
  }
}

export function clearSnapshot(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}

/** Settings restored from storage, if any; read once at startup. */
export const savedSnapshot: Snapshot | null = loadSnapshot();
