import {
  BoxGeometry,
  CircleGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  Group,
  Matrix3,
  Mesh,
  MeshStandardMaterial,
  ShaderMaterial,
  TorusGeometry,
  Vector3,
  type Camera,
} from 'three';
import { HALF_W, R_OUT } from '../physics/constants';
import { LandscapeMesh } from './landscape';
import glassFrag from './shaders/glass.frag?raw';
import glassVert from './shaders/glass.vert?raw';

/** The drum's opaque cage (rims, ribs, index mark) and its solid transparent glass shell. */
export class Wheel {
  /** Opaque parts: rendered in the main scene pass. */
  readonly frame = new Group();
  /** Glass shell: rendered last, blended over water and scene. */
  readonly glass = new Group();
  private readonly glassMat: ShaderMaterial;
  readonly landscape = new LandscapeMesh();

  constructor() {
    const R = R_OUT,
      Hh = HALF_W;
    const metal = new MeshStandardMaterial({ color: 0x8e9db5, roughness: 0.45, metalness: 0.7 });
    const dark = new MeshStandardMaterial({ color: 0x2c3442, roughness: 0.6, metalness: 0.5 });

    for (const sy of [-1, 1]) {
      const rim = new Mesh(new TorusGeometry(R + 0.02, 0.05, 12, 160), metal);
      rim.rotation.x = Math.PI / 2;
      rim.position.y = sy * Hh;
      this.frame.add(rim);
    }
    const ribGeo = new BoxGeometry(0.05, 2 * Hh + 0.02, 0.06);
    for (let k = 0; k < 24; k++) {
      const a = (k * Math.PI) / 12;
      const rib = new Mesh(ribGeo, k % 6 === 0 ? metal : dark);
      rib.position.set(Math.cos(a) * (R + 0.03), 0, Math.sin(a) * (R + 0.03));
      rib.rotation.y = -a;
      this.frame.add(rib);
    }
    const mark = new Mesh(
      new BoxGeometry(0.12, 2 * Hh + 0.02, 0.12),
      new MeshStandardMaterial({ color: 0xff7a4a, emissive: 0xff5a2a, emissiveIntensity: 0.5 }),
    );
    mark.position.set(R + 0.05, 0, 0);
    this.frame.add(mark, this.landscape.mesh);

    this.glassMat = new ShaderMaterial({
      uniforms: {
        lightDir: { value: new Vector3() },
        fillDir: { value: new Vector3() },
        tint: { value: new Color(0.62, 0.84, 1.0) },
        uViewToWorld: { value: new Matrix3() },
      },
      vertexShader: glassVert,
      fragmentShader: glassFrag,
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
    });
    const side = new Mesh(new CylinderGeometry(R, R, 2 * Hh, 128, 1, true), this.glassMat);
    this.glass.add(side);
    for (const sy of [-1, 1]) {
      const cap = new Mesh(new CircleGeometry(R, 128), this.glassMat);
      cap.rotation.x = sy > 0 ? -Math.PI / 2 : Math.PI / 2;
      cap.position.y = sy * Hh;
      this.glass.add(cap);
    }
  }

  update(theta: number, camera: Camera, sunDir: Vector3, fillDir: Vector3): void {
    this.frame.rotation.y = theta;
    this.glass.rotation.y = theta;
    const u = this.glassMat.uniforms;
    (u.lightDir.value as Vector3).copy(sunDir).transformDirection(camera.matrixWorldInverse);
    (u.fillDir.value as Vector3).copy(fillDir).transformDirection(camera.matrixWorldInverse);
    (u.uViewToWorld.value as Matrix3).setFromMatrix4(camera.matrixWorld);
  }
}
