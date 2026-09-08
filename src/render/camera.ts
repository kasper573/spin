import { PerspectiveCamera } from 'three';

/** Orbit camera around the drum centre. */
export class OrbitCamera {
  readonly camera = new PerspectiveCamera(42, 1, 0.1, 500);
  azimuth = 0.6;
  elevation = 0.55;
  distance = 10.5;

  rotate(dx: number, dy: number): void {
    this.azimuth -= dx * 0.006;
    this.elevation = Math.max(-1.45, Math.min(1.45, this.elevation + dy * 0.006));
  }

  zoom(deltaY: number): void {
    this.distance = Math.max(3, Math.min(40, this.distance * Math.exp(deltaY * 0.0012)));
  }

  setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  update(): void {
    const { azimuth: az, elevation: el, distance: d, camera } = this;
    camera.position.set(
      d * Math.cos(el) * Math.sin(az),
      d * Math.sin(el),
      d * Math.cos(el) * Math.cos(az),
    );
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
  }
}
