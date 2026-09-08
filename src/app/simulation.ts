import { Vector3 } from 'three';
import { MASS, RAFT_T } from '../physics/constants';
import { injectAt, makeState, spawnRaft, step, type SimState } from '../physics/world';
import type { MarkerKind } from '../render/markers';
import { View } from '../render/view';
import { aimAtDrum, type Aim } from './aim';
import { Controls } from './controls';
import { setReadouts, setSettings, settings } from './settings';

const SUBSTEP = 1 / 120;
const MAX_SUBSTEPS = 3;
const DRAIN_RATE = 240; // particles per second removed along the aim ray
const READOUT_INTERVAL = 100; // ms
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

  constructor(canvas: HTMLCanvasElement) {
    this.view = new View(canvas);
    this.controls = new Controls(canvas, this.view.fly.keys, {
      look: (dx, dy) => this.view.fly.look(dx, dy),
      wheel: (dy) => this.view.fly.adjustSpeed(dy),
      secondary: () => this.placeRaft(),
      lockChange: (locked) => setReadouts('controlsActive', locked),
    });
    window.addEventListener('resize', this.onResize);
  }

  start(): void {
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    this.controls.dispose();
    this.view.renderer.dispose();
  }

  clearWater(): void {
    this.state.fluid.clear();
  }

  clearRafts(): void {
    this.state.rafts.length = 0;
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

  private updateAim(): void {
    const cam = this.view.fly.camera;
    cam.getWorldDirection(this.rayDir);
    this.aim = aimAtDrum(cam.position, this.rayDir);
  }

  private placeRaft(): void {
    if (settings.paused) return;
    this.updateAim();
    if (!this.aim) return;
    const p = this.aim.point.clone().addScaledVector(this.aim.normal, RAFT_OFFSET);
    spawnRaft(
      this.state,
      [p.x, p.y, p.z],
      [this.aim.normal.x, this.aim.normal.y, this.aim.normal.z],
      settings.matchWheel,
    );
  }

  private applyTool(dt: number): void {
    if (!this.controls.primary || !this.aim) return;
    const rate = settings.tool === 'inject' ? settings.flow : DRAIN_RATE;
    this.toolAcc += rate * dt;
    const k = Math.floor(this.toolAcc);
    this.toolAcc -= k;
    if (k <= 0) return;
    if (settings.tool === 'inject') {
      const p = this.aim.point.clone().addScaledVector(this.aim.normal, INJECT_DEPTH);
      injectAt(this.state, p.x, p.y, p.z, k, settings.matchWheel);
    } else {
      this.drain(k);
    }
  }

  /** Remove particles within a tube around the aim ray, nearest to the camera first. */
  private drain(count: number): void {
    const F = this.state.fluid,
      o = this.view.fly.camera.position,
      d = this.rayDir;
    let removed = 0;
    for (let i = F.n - 1; i >= 0 && removed < count; i--) {
      const rx = F.x[i] - o.x,
        ry = F.y[i] - o.y,
        rz = F.z[i] - o.z;
      const t = rx * d.x + ry * d.y + rz * d.z;
      if (t < 0) continue;
      const qx = rx - d.x * t,
        qy = ry - d.y * t,
        qz = rz - d.z * t;
      if (qx * qx + qy * qy + qz * qz < 0.5) {
        F.remove(i);
        removed++;
      }
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
      this.applyTool(dt);
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

    const marker: MarkerKind = this.aim && this.controls.locked ? settings.tool : 'none';
    const aim = this.aim ?? NO_AIM;
    this.view.render(this.state, marker, aim.point, aim.normal, RAFT_OFFSET);

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
