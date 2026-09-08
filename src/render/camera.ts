import { PerspectiveCamera, Vector3 } from 'three';

const LOOK_RATE = 0.0022; // rad per pixel of mouse movement
const ROLL_RATE = 1.6; // rad/s while Q or E is held
const MIN_SPEED = 0.5;
const MAX_SPEED = 40;

export interface CameraSnapshot {
  position: number[];
  quaternion: number[];
  speed: number;
}

/** Free-flying camera: all motion and rotation are in the camera's own frame, like a spacecraft. */
export class FlyCamera {
  readonly camera = new PerspectiveCamera(42, 1, 0.1, 500);
  /** Translation speed in m/s. */
  speed = 4;
  /** Currently held keys, by `KeyboardEvent.code`. */
  readonly keys = new Set<string>();

  constructor() {
    this.camera.position.set(4.9, 5.5, 7.2);
    this.camera.lookAt(new Vector3(0, 0, 0));
    this.camera.updateMatrixWorld();
  }

  look(dx: number, dy: number): void {
    this.camera.rotateY(-dx * LOOK_RATE);
    this.camera.rotateX(-dy * LOOK_RATE);
  }

  adjustSpeed(deltaY: number): void {
    this.speed = Math.max(MIN_SPEED, Math.min(MAX_SPEED, this.speed * Math.exp(-deltaY * 0.001)));
  }

  setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  /** Pose and fly speed, for persistence. */
  snapshot(): CameraSnapshot {
    return {
      position: this.camera.position.toArray(),
      quaternion: this.camera.quaternion.toArray(),
      speed: this.speed,
    };
  }

  restore(s: CameraSnapshot): void {
    if (
      s.position.length === 3 &&
      s.quaternion.length === 4 &&
      s.position.every(Number.isFinite) &&
      s.quaternion.every(Number.isFinite)
    ) {
      this.camera.position.fromArray(s.position);
      this.camera.quaternion.fromArray(s.quaternion).normalize();
      this.camera.updateMatrixWorld();
    }
    if (Number.isFinite(s.speed)) this.speed = Math.max(MIN_SPEED, Math.min(MAX_SPEED, s.speed));
  }

  /** Apply held keys for this frame. */
  update(dt: number): void {
    const k = this.keys,
      c = this.camera,
      v = this.speed * dt;
    const axis = (neg: string, pos: string) => (k.has(pos) ? 1 : 0) - (k.has(neg) ? 1 : 0);
    c.translateX(axis('KeyA', 'KeyD') * v);
    c.translateY(
      Math.max(
        -1,
        Math.min(
          1,
          axis('KeyF', 'KeyR') + axis('ShiftLeft', 'Space') + axis('ShiftRight', 'Space'),
        ),
      ) * v,
    );
    c.translateZ(-axis('KeyS', 'KeyW') * v);
    c.rotateZ(axis('KeyE', 'KeyQ') * ROLL_RATE * dt);
    c.updateMatrixWorld();
  }
}
