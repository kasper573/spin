import { BoxGeometry, Mesh, MeshBasicMaterial, TorusGeometry, type Scene } from 'three';
import { RAFT_L, RAFT_T } from '../physics/constants';
import { floorBasis } from '../physics/world';

export type MarkerKind = 'none' | 'inject' | 'drain' | 'raft';

/** Cursor feedback drawn on top of everything. */
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
    this.ring.rotation.x = Math.PI / 2;
    this.box = new Mesh(
      new BoxGeometry(RAFT_L, RAFT_T, RAFT_L),
      new MeshBasicMaterial({
        color: 0xc58b48,
        transparent: true,
        opacity: 0.35,
        wireframe: true,
        depthTest: false,
      }),
    );
    this.ring.visible = this.box.visible = false;
    scene.add(this.ring, this.box);
  }

  update(kind: MarkerKind, x: number, z: number): void {
    this.ring.visible = kind === 'inject' || kind === 'drain';
    this.box.visible = kind === 'raft';
    if (this.ring.visible) {
      this.ring.position.set(x, 0, z);
      this.ring.material.color.set(kind === 'drain' ? 0xff8f6a : 0x4fb2ff);
    }
    if (this.box.visible) {
      this.box.position.set(x, 0, z);
      const q = floorBasis(Math.atan2(z, x));
      this.box.quaternion.set(q[0], q[1], q[2], q[3]);
    }
  }
}
