import { BoxGeometry, Mesh, MeshBasicMaterial, TorusGeometry, Vector3, type Scene } from 'three';
import { RAFT_L, RAFT_T } from '../physics/constants';
import { basisFromNormal } from '../physics/world';

export type MarkerKind = 'none' | 'inject' | 'drain';

const Z = new Vector3(0, 0, 1);

/** Aim feedback on the drum surface: a ring for the water tool and a raft outline for placement. */
export class Markers {
  private readonly ring: Mesh<TorusGeometry, MeshBasicMaterial>;
  private readonly box: Mesh<BoxGeometry, MeshBasicMaterial>;

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
    this.ring.visible = this.box.visible = false;
    scene.add(this.ring, this.box);
  }

  update(kind: MarkerKind, point: Vector3, normal: Vector3, raftOffset: number): void {
    const show = kind !== 'none';
    this.ring.visible = show;
    this.box.visible = show;
    if (!show) return;
    this.ring.position.copy(point);
    this.ring.quaternion.setFromUnitVectors(Z, normal);
    this.ring.material.color.set(kind === 'drain' ? 0xff8f6a : 0x4fb2ff);
    this.box.position.copy(point).addScaledVector(normal, raftOffset);
    const q = basisFromNormal([normal.x, normal.y, normal.z]);
    this.box.quaternion.set(q[0], q[1], q[2], q[3]);
  }
}
