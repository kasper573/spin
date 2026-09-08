import { BoxGeometry, Mesh, MeshBasicMaterial, TorusGeometry, Vector3, type Scene } from 'three';
import { RAFT_L, RAFT_T } from '../physics/constants';
import { basisFromNormal } from '../physics/world';

export type MarkerKind = 'none' | 'aim';

const Z = new Vector3(0, 0, 1);
const BRUSH_UNIT = 1;

/** Aim feedback on the drum surface: water ring, raft outline and landscape brush circle. */
export class Markers {
  private readonly ring: Mesh<TorusGeometry, MeshBasicMaterial>;
  private readonly box: Mesh<BoxGeometry, MeshBasicMaterial>;
  private readonly brush: Mesh<TorusGeometry, MeshBasicMaterial>;

  constructor(scene: Scene) {
    this.ring = new Mesh(
      new TorusGeometry(0.28, 0.025, 8, 40),
      new MeshBasicMaterial({
        color: 0x4fb2ff,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
      }),
    );
    this.box = new Mesh(
      new BoxGeometry(RAFT_L, RAFT_T, RAFT_L),
      new MeshBasicMaterial({
        color: 0xc58b48,
        transparent: true,
        opacity: 0.3,
        wireframe: true,
        depthTest: false,
      }),
    );
    this.brush = new Mesh(
      new TorusGeometry(BRUSH_UNIT, 0.012, 6, 64),
      new MeshBasicMaterial({ color: 0xd8c9a0, transparent: true, opacity: 0.5, depthTest: false }),
    );
    this.ring.visible = this.box.visible = this.brush.visible = false;
    scene.add(this.ring, this.box, this.brush);
  }

  update(
    kind: MarkerKind,
    point: Vector3,
    normal: Vector3,
    raftOffset: number,
    brushRadius: number,
  ): void {
    const show = kind !== 'none';
    this.ring.visible = show;
    this.box.visible = show;
    this.brush.visible = show;
    if (!show) return;
    this.ring.position.copy(point);
    this.ring.quaternion.setFromUnitVectors(Z, normal);
    this.brush.position.copy(point);
    this.brush.quaternion.setFromUnitVectors(Z, normal);
    this.brush.scale.setScalar(brushRadius / BRUSH_UNIT);
    this.box.position.copy(point).addScaledVector(normal, raftOffset);
    const q = basisFromNormal([normal.x, normal.y, normal.z]);
    this.box.quaternion.set(q[0], q[1], q[2], q[3]);
  }
}
