import { Vector3 } from 'three';
import { MASS, RAFT_T } from '../physics/constants';
import { wheelAngle } from '../physics/landscape';
import { deserializeState, serializeState } from '../physics/serialize';
import { injectAt, makeState, spawnRaft, step, type SimState } from '../physics/world';
import type { MarkerKind } from '../render/markers';
import { View } from '../render/view';
import { aimAtDrum, type Aim } from './aim';
import { Controls } from './controls';
import { savedSnapshot, saveSnapshot } from './persistence';
import { setReadouts, setSettings, settings } from './settings';

const SUBSTEP = 1 / 120;
const MAX_SUBSTEPS = 3;
const READOUT_INTERVAL = 100; // ms
const SAVE_INTERVAL = 2000; // ms
const INJECT_DEPTH = 0.4; // how far inside the aimed surface water appears
const RAFT_OFFSET = RAFT_T / 2 + 0.04; // raft centre above the aimed surface

/** Owns the simulation state, the renderer, the controls and the frame loop. */
export class Simulation {
  private state: SimState = makeState();
  private readonly view: View;
  private readonly controls: Controls;
  private aim: Aim | null = null;
  private readonly rayDir = new Vector3();
  private raf = 0;
  private last = performance.now();
  private acc = 0;
  private toolAcc = 0;
  private fps = { t: 0, n: 0, value: 0 };
  private simRate = 1;
  private lastReadout = 0;
  private lastSave = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.view = new View(canvas);
    this.controls = new Controls(canvas, this.view.fly.keys, {
      look: (dx, dy) => this.view.fly.look(dx, dy),
      wheel: (dy) => this.view.fly.adjustSpeed(dy),
      secondary: () => this.placeRaft(),
      lockChange: (locked) => setReadouts('controlsActive', locked),
    });
    window.addEventListener('resize', this.onResize);
    window.addEventListener('pagehide', this.save);
    document.addEventListener('visibilitychange', this.onVisibility);
    if (savedSnapshot) {
      this.state = deserializeState(savedSnapshot.sim);
      if (savedSnapshot.camera) this.view.fly.restore(savedSnapshot.camera);
    }
  }

  start(): void {
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('pagehide', this.save);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.save();
    this.controls.dispose();
    this.view.renderer.dispose();
  }

  clearWater(): void {
    this.state.fluid.clear();
  }

  clearRafts(): void {
    this.state.rafts.length = 0;
  }

  resetLandscape(): void {
    this.state.landscape.reset();
  }

  /** Inject `count` particles around a point inside the drum. */
  inject(x: number, y: number, z: number, count: number): void {
    injectAt(this.state, x, y, z, count, settings.matchWheel);
  }

  /** Run the physics forward synchronously without rendering (tests and scripting). */
  advance(seconds: number): void {
    this.syncParams();
    const n = Math.round(seconds / SUBSTEP);
    for (let i = 0; i < n; i++) step(this.state, SUBSTEP);
  }

  reset(): void {
    this.state = makeState();
    setSettings('spin', 0);
  }

  get litres(): number {
    return Math.round(this.state.fluid.n * MASS);
  }

  private readonly onResize = (): void => this.view.resize();

  private readonly onVisibility = (): void => {
    if (document.visibilityState === 'hidden') this.save();
  };

  /** Persist settings, simulation and camera to localStorage. */
  readonly save = (): void => {
    this.lastSave = performance.now();
    saveSnapshot({
      settings: { ...settings },
      sim: serializeState(this.state),
      camera: this.view.fly.snapshot(),
    });
  };

  private updateAim(): void {
    const cam = this.view.fly.camera;
    cam.getWorldDirection(this.rayDir);
    this.aim = aimAtDrum(cam.position, this.rayDir, this.state.landscape, this.state.theta);
  }

  private placeRaft(): void {
    if (settings.paused) return;
    this.updateAim();
    if (!this.aim) return;
    const p = this.aim.point.clone().addScaledVector(this.aim.normal, RAFT_OFFSET);
    const n = this.aim.normal;
    spawnRaft(this.state, [p.x, p.y, p.z], [n.x, n.y, n.z], settings.matchWheel);
  }

  private applyTools(dt: number): void {
    if (!this.aim) return;
    if (this.controls.primary) {
      this.toolAcc += settings.flow * dt;
      const k = Math.floor(this.toolAcc);
      this.toolAcc -= k;
      if (k > 0) {
        const p = this.aim.point.clone().addScaledVector(this.aim.normal, INJECT_DEPTH);
        injectAt(this.state, p.x, p.y, p.z, k, settings.matchWheel);
      }
    }
    if (this.controls.tertiary) {
      const p = this.aim.point;
      const amount = settings.brushRate * dt * (this.controls.modifier ? -1 : 1);
      this.state.landscape.sculpt(
        wheelAngle(p.x, p.z, this.state.theta),
        p.y,
        settings.brushSize,
        amount,
      );
    }
  }

  private syncParams(): void {
    const S = this.state,
      P = S.params;
    S.omegaTarget = settings.spin;
    P.viscosity = settings.viscosity;
    P.wallFriction = settings.wallFriction;
    P.raftFriction = settings.raftFriction;
    P.air = settings.air;
  }

  private readonly frame = (now: number): void => {
    this.raf = requestAnimationFrame(this.frame);
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const fps = this.fps;
    fps.t += dt;
    fps.n++;
    if (fps.t > 0.5) {
      fps.value = fps.n / fps.t;
      fps.t = 0;
      fps.n = 0;
    }

    this.syncParams();
    this.view.fly.update(dt);
    this.updateAim();

    if (!settings.paused) {
      this.applyTools(dt);
      this.acc += dt;
      let steps = 0;
      while (this.acc >= SUBSTEP && steps < MAX_SUBSTEPS) {
        step(this.state, SUBSTEP);
        this.acc -= SUBSTEP;
        steps++;
      }
      if (steps === MAX_SUBSTEPS) {
        this.simRate = 0.9 * this.simRate + 0.1 * ((steps * SUBSTEP) / dt);
        this.acc = 0;
      } else {
        this.simRate = 0.9 * this.simRate + 0.1;
      }
    }

    const marker: MarkerKind = this.aim && this.controls.locked ? 'aim' : 'none';
    const aim = this.aim ?? NO_AIM;
    this.view.render(this.state, marker, aim.point, aim.normal, RAFT_OFFSET, settings.brushSize);

    if (now - this.lastSave > SAVE_INTERVAL) this.save();
    if (now - this.lastReadout > READOUT_INTERVAL) {
      this.lastReadout = now;
      setReadouts({
        omega: this.state.omega,
        particles: this.state.fluid.n,
        rafts: this.state.rafts.length,
        fps: fps.value,
        simRate: this.simRate,
      });
    }
  };
}

const NO_AIM: Aim = { point: new Vector3(), normal: new Vector3(0, 1, 0) };
