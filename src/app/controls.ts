/**
 * Spacecraft-style input on the canvas. Clicking the canvas takes pointer lock; while locked the
 * mouse steers the camera, the left button injects water, the right button places a raft, the
 * middle button sculpts the landscape and the wheel adjusts fly speed. Escape releases the lock.
 * Held keys are reported through `keys` by `KeyboardEvent.code`.
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
  /** Middle button held while locked. */
  tertiary = false;
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
    canvas.addEventListener('auxclick', (e) => e.preventDefault());
    canvas.addEventListener('mousemove', this.onMove);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    document.addEventListener('pointerlockchange', this.onLockChange);
    window.addEventListener('keydown', this.onKey);
    window.addEventListener('keyup', this.onKey);
    window.addEventListener('blur', this.releaseAll);
  }

  dispose(): void {
    document.removeEventListener('pointerlockchange', this.onLockChange);
    window.removeEventListener('keydown', this.onKey);
    window.removeEventListener('keyup', this.onKey);
    window.removeEventListener('blur', this.releaseAll);
  }

  /** Control held: the middle button lowers instead of raising. */
  get modifier(): boolean {
    return this.keys.has('ControlLeft') || this.keys.has('ControlRight');
  }

  private readonly onDown = (e: PointerEvent): void => {
    e.preventDefault();
    if (!this.locked) {
      if (e.button === 0 || e.button === 2) this.canvas.requestPointerLock();
      return;
    }
    if (e.button === 0) this.primary = true;
    else if (e.button === 1) this.tertiary = true;
    else if (e.button === 2) this.handlers.secondary();
  };

  private readonly onUp = (e: PointerEvent): void => {
    if (e.button === 0) this.primary = false;
    else if (e.button === 1) this.tertiary = false;
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
    this.releaseAll();
    this.handlers.lockChange(this.locked);
  };

  private readonly onKey = (e: KeyboardEvent): void => {
    const t = e.target as HTMLElement | null;
    if (!this.locked && t && t !== document.body && t !== this.canvas) return;
    if (this.locked && (e.code === 'Space' || e.code === 'Tab')) e.preventDefault();
    if (e.type === 'keydown') this.keys.add(e.code);
    else this.keys.delete(e.code);
  };

  private readonly releaseAll = (): void => {
    this.keys.clear();
    this.primary = false;
    this.tertiary = false;
  };
}
