import {
  BoxGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  type Scene,
} from 'three';
import { RAFT_L, RAFT_T } from '../physics/constants';
import type { Raft } from '../physics/raft';

const wood = new MeshStandardMaterial({ color: 0xb9813f, roughness: 0.85, metalness: 0 });
const woodDark = new MeshStandardMaterial({ color: 0x7a5228, roughness: 0.9, metalness: 0 });
const logRadius = RAFT_T / 2;
const logGeo = new CylinderGeometry(logRadius, logRadius, RAFT_L, 12);
const beamGeo = new BoxGeometry(RAFT_L, RAFT_T * 0.22, RAFT_L * 0.08);

function makeRaftMesh(): Group {
  const g = new Group();
  for (let k = 0; k < 4; k++) {
    const log = new Mesh(logGeo, wood);
    log.rotation.x = Math.PI / 2;
    log.position.set(-RAFT_L / 2 + (k + 0.5) * (RAFT_L / 4), -RAFT_T / 2 + logRadius, 0);
    g.add(log);
  }
  for (const sz of [-1, 1]) {
    const beam = new Mesh(beamGeo, woodDark);
    beam.position.set(0, RAFT_T / 2 - RAFT_T * 0.11, sz * RAFT_L * 0.33);
    g.add(beam);
  }
  return g;
}

/** Keeps one log-raft mesh per rigid body. */
export class RaftMeshes {
  private readonly meshes: Group[] = [];

  constructor(private readonly scene: Scene) {}

  sync(rafts: readonly Raft[]): void {
    while (this.meshes.length < rafts.length) {
      const m = makeRaftMesh();
      this.scene.add(m);
      this.meshes.push(m);
    }
    while (this.meshes.length > rafts.length) {
      const m = this.meshes.pop();
      if (m) this.scene.remove(m);
    }
    for (let i = 0; i < rafts.length; i++) {
      const r = rafts[i],
        g = this.meshes[i];
      g.position.set(r.p[0], r.p[1], r.p[2]);
      g.quaternion.set(r.q[0], r.q[1], r.q[2], r.q[3]);
    }
  }
}
