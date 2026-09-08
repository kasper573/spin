import { Raycaster, Vector2, Vector3, type Camera } from 'three';
import { R_OUT } from '../physics/constants';

export interface PointerState {
  /** Screen position and whether the pointer is over the canvas. */
  x: number;
  y: number;
  over: boolean;
  /** Left button held with the active tool. */
  using: boolean;
  /** Camera orbit drag in progress. */
  orbiting: boolean;
}

/**
 * Tracks the pointer over the canvas. Left button drives the tool, right/middle button or Shift
 * orbits the camera, wheel zooms. The cursor is projected onto the drum's mid-plane (y = 0).
 */
export class Pointer {
  readonly state: PointerState = { x: 0, y: 0, over: false, using: false, orbiting: false };
  /** Cursor point on the mid-plane, clamped inside the drum. */
  readonly point = new Vector3();
  /** Pick ray in world space. */
  readonly rayOrigin = new Vector3();
  readonly rayDir = new Vector3();
  private readonly raycaster = new Raycaster();
  private readonly ndc = new Vector2();
  private last = { x: 0, y: 0 };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly handlers: {
      orbit: (dx: number, dy: number) => void;
      zoom: (deltaY: number) => void;
      click: () => void;
    },
  ) {
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    canvas.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('pointercancel', this.onUp);
    canvas.addEventListener('pointerleave', () => {
      this.state.over = false;
    });
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
  }

  private readonly onDown = (e: PointerEvent): void => {
    const s = this.state;
    s.x = e.clientX;
    s.y = e.clientY;
    s.over = true;
    this.canvas.setPointerCapture(e.pointerId);
    if (e.button === 2 || e.button === 1 || e.shiftKey) {
      s.orbiting = true;
      this.last = { x: e.clientX, y: e.clientY };
      this.canvas.classList.add('orbiting');
      return;
    }
    if (e.button !== 0) return;
    s.using = true;
    this.handlers.click();
  };

  private readonly onMove = (e: PointerEvent): void => {
    const s = this.state;
    s.x = e.clientX;
    s.y = e.clientY;
    s.over = true;
    if (s.orbiting) {
      this.handlers.orbit(e.clientX - this.last.x, e.clientY - this.last.y);
      this.last = { x: e.clientX, y: e.clientY };
    }
  };

  private readonly onUp = (): void => {
    this.state.orbiting = false;
    this.state.using = false;
    this.canvas.classList.remove('orbiting');
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.handlers.zoom(e.deltaY);
  };

  /** Recompute the pick ray and mid-plane point for the current camera. */
  update(camera: Camera): void {
    this.ndc.set(
      (this.state.x / window.innerWidth) * 2 - 1,
      -(this.state.y / window.innerHeight) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.ndc, camera);
    const o = this.rayOrigin.copy(this.raycaster.ray.origin),
      d = this.rayDir.copy(this.raycaster.ray.direction);
    let t = Math.abs(d.y) > 1e-4 ? -o.y / d.y : -o.dot(d);
    if (t < 0) t = -o.dot(d);
    let x = o.x + d.x * t,
      z = o.z + d.z * t;
    const r = Math.hypot(x, z),
      rc = R_OUT - 0.3;
    if (r > rc) {
      x *= rc / r;
      z *= rc / r;
    }
    this.point.set(x, 0, z);
  }
}
