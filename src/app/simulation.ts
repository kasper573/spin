import { MASS, R_OUT } from '../physics/constants';
import { injectAt, makeState, spawnRaft, step, type SimState } from '../physics/world';
import type { MarkerKind } from '../render/markers';
import { View } from '../render/view';
import { Pointer } from './pointer';
import { setReadouts, setSettings, settings } from './settings';

const SUBSTEP = 1 / 120;
const MAX_SUBSTEPS = 3;
const DRAIN_RATE = 240; // particles per second removed under the cursor
const READOUT_INTERVAL = 100; // ms

/** Owns the simulation state, the renderer and the frame loop. */
export class Simulation {
  private state: SimState = makeState();
  private readonly view: View;
  private readonly pointer: Pointer;
  private raf = 0;
  private last = performance.now();
  private acc = 0;
  private toolAcc = 0;
  private fps = { t: 0, n: 0, value: 0 };
  private simRate = 1;
  private lastReadout = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.view = new View(canvas);
    this.pointer = new Pointer(canvas, {
      orbit: (dx, dy) => this.view.orbit.rotate(dx, dy),
      zoom: (dy) => this.view.orbit.zoom(dy),
      click: () => this.onClick(),
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
    this.view.renderer.dispose();
  }

  clearWater(): void {
    this.state.fluid.clear();
  }

  clearRafts(): void {
    this.state.rafts.length = 0;
  }

  addRaft(): void {
    const a = Math.random() * Math.PI * 2,
      r = 0.8 + Math.random() * (R_OUT - 1.6);
    spawnRaft(
      this.state,
      Math.cos(a) * r,
      (Math.random() - 0.5) * 0.4,
      Math.sin(a) * r,
      settings.matchWheel,
    );
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

  private onClick(): void {
    if (settings.tool !== 'raft' || settings.paused) return;
    this.pointer.update(this.view.orbit.camera);
    const p = this.pointer.point;
    spawnRaft(this.state, p.x, 0, p.z, settings.matchWheel);
  }

  private applyTool(dt: number): void {
    const ps = this.pointer.state;
    if (!ps.using || !ps.over) return;
    const rate =
      settings.tool === 'inject' ? settings.flow : settings.tool === 'drain' ? DRAIN_RATE : 0;
    this.toolAcc += rate * dt;
    const k = Math.floor(this.toolAcc);
    this.toolAcc -= k;
    if (k <= 0) return;
    const p = this.pointer.point;
    if (settings.tool === 'inject') injectAt(this.state, p.x, p.y, p.z, k, settings.matchWheel);
    else if (settings.tool === 'drain') this.drain(k);
  }

  /** Remove particles within a tube around the pick ray, nearest to the camera first. */
  private drain(count: number): void {
    const F = this.state.fluid,
      o = this.pointer.rayOrigin,
      d = this.pointer.rayDir;
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
    this.view.orbit.update();
    if (this.pointer.state.over) this.pointer.update(this.view.orbit.camera);

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

    const ps = this.pointer.state;
    const marker: MarkerKind = ps.over && !ps.orbiting ? settings.tool : 'none';
    const p = this.pointer.point;
    this.view.render(this.state, marker, p.x, p.z);

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
