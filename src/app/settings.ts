import { createStore } from 'solid-js/store';
import { savedSnapshot } from './persistence';

export interface Settings {
  /** Target drum spin (rad/s). */
  spin: number;
  /** Injection rate in litres per second. */
  flow: number;
  /** New water and rafts start moving with the glass. */
  matchWheel: boolean;
  viscosity: number;
  wallFriction: number;
  raftFriction: number;
  air: boolean;
  /** Landscape brush radius (m) and rate of height change (m/s). */
  brushSize: number;
  brushRate: number;
  paused: boolean;
}

export interface Readouts {
  omega: number;
  particles: number;
  rafts: number;
  fps: number;
  /** Fraction of real time the simulation keeps up with (1 = full speed). */
  simRate: number;
  /** Pointer lock held: mouse and keys steer the camera. */
  controlsActive: boolean;
}

export const SPIN_MAX = 3;
export const SPIN_STEP = 0.25;

export const defaultSettings = (): Settings => ({
  spin: 0,
  flow: 500,
  matchWheel: true,
  viscosity: 0.15,
  wallFriction: 0.5,
  raftFriction: 0.45,
  air: true,
  brushSize: 0.6,
  brushRate: 0.8,
  paused: false,
});

/** Only known keys with the right type survive from storage. */
function sanitize(saved: Partial<Settings> | undefined): Settings {
  const defaults = defaultSettings();
  const out: Record<string, unknown> = { ...defaults };
  if (saved) {
    for (const k of Object.keys(defaults) as (keyof Settings)[]) {
      const v = saved[k];
      if (typeof v === typeof defaults[k] && (typeof v !== 'number' || Number.isFinite(v)))
        out[k] = v;
    }
  }
  return out as unknown as Settings;
}

export const [settings, setSettings] = createStore<Settings>(sanitize(savedSnapshot?.settings));
export const [readouts, setReadouts] = createStore<Readouts>({
  omega: 0,
  particles: 0,
  rafts: 0,
  fps: 0,
  simRate: 1,
  controlsActive: false,
});
