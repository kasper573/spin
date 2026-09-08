/**
 * Spacecraft-style input on the canvas. Clicking the canvas takes pointer lock; while locked the
 * mouse steers the camera, the left button drives the primary tool, the right button places a raft
 * and the wheel adjusts fly speed. Escape releases the lock. Keys are reported through `keys`.
 */
export interface ControlHandlers {
  look: (dx: number, dy: number) => void;
  wheel: (deltaY: number) => void;
  secondary: () => void;
  lockChange: (locked: boolean) => void;
}

export class Controls {
  /** Left button held while locked. */
  primary = false;
  locked = false;
  readonly keys: Set<string>;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    keys: Set<string>,
    private readonly handlers: ControlHandlers,
  ) {
    this.keys = keys;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('mousemove', this.onMove);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    document.addEventListener('pointerlockchange', this.onLockChange);
    window.addEventListener('keydown', this.onKey);
    window.addEventListener('keyup', this.onKey);
    window.addEventListener('blur', this.releaseKeys);
  }

  dispose(): void {
    document.removeEventListener('pointerlockchange', this.onLockChange);
    window.removeEventListener('keydown', this.onKey);
    window.removeEventListener('keyup', this.onKey);
    window.removeEventListener('blur', this.releaseKeys);
  }

  private readonly onDown = (e: PointerEvent): void => {
    e.preventDefault();
    if (!this.locked) {
      if (e.button === 0 || e.button === 2) this.canvas.requestPointerLock();
      return;
    }
    if (e.button === 0) this.primary = true;
    else if (e.button === 2) this.handlers.secondary();
  };

  private readonly onUp = (e: PointerEvent): void => {
    if (e.button === 0) this.primary = false;
  };

  private readonly onMove = (e: MouseEvent): void => {
    if (this.locked) this.handlers.look(e.movementX, e.movementY);
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    if (this.locked) this.handlers.wheel(e.deltaY);
  };

  private readonly onLockChange = (): void => {
    this.locked = document.pointerLockElement === this.canvas;
    this.primary = false;
    if (!this.locked) this.releaseKeys();
    this.handlers.lockChange(this.locked);
  };

  private readonly onKey = (e: KeyboardEvent): void => {
    const t = e.target as HTMLElement | null;
    if (!this.locked && t && t !== document.body && t !== this.canvas) return;
    if (e.type === 'keydown') this.keys.add(e.code);
    else this.keys.delete(e.code);
  };

  private readonly releaseKeys = (): void => this.keys.clear();
}
