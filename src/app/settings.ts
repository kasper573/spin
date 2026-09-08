import { createStore } from 'solid-js/store';

/** What the left mouse button does. */
export type Tool = 'inject' | 'drain';

export interface Settings {
  /** Target drum spin (rad/s). */
  spin: number;
  /** Injection rate (particles per second). */
  flow: number;
  /** New water and rafts start moving with the glass. */
  matchWheel: boolean;
  viscosity: number;
  wallFriction: number;
  raftFriction: number;
  air: boolean;
  tool: Tool;
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
  flow: 60,
  matchWheel: true,
  viscosity: 0.15,
  wallFriction: 0.5,
  raftFriction: 0.45,
  air: true,
  tool: 'inject',
  paused: false,
});

export const [settings, setSettings] = createStore<Settings>(defaultSettings());
export const [readouts, setReadouts] = createStore<Readouts>({
  omega: 0,
  particles: 0,
  rafts: 0,
  fps: 0,
  simRate: 1,
  controlsActive: false,
});
