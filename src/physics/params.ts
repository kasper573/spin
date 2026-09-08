export interface SimParams {
  viscosity: number;
  wallFriction: number;
  raftFriction: number;
  restitution: number;
  raftDrag: number;
  air: boolean;
  /** Time constant (s) for the drum's air to drag free objects into co-rotation. */
  airTau: number;
  /** Maximum spin-up acceleration of the drum (rad/s²). */
  spinAccel: number;
}

export const defaultParams = (): SimParams => ({
  viscosity: 0.15,
  wallFriction: 0.5,
  raftFriction: 0.45,
  restitution: 0.2,
  raftDrag: 0.5,
  air: true,
  airTau: 12,
  spinAccel: 0.6,
});
