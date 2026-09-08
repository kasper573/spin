// Geometry and fluid constants (SI units). The wheel is a solid glass drum spinning about +Y.

export const R_OUT = 3.5; // drum floor radius
export const HALF_W = 0.6; // half width of the drum along its axis

export const D = 0.2; // particle spacing
export const H = 0.4; // SPH kernel radius
export const H2 = H * H;
export const RHO0 = 1000;
export const MASS = RHO0 * D * D * D; // 8 kg per particle
export const RMAX = R_OUT - D * 0.5;
export const YMAX = HALF_W - D * 0.5;

export const MAXP = 4000; // fluid particles
export const MAXN = 56; // fluid neighbours per particle
export const MAXB = 24; // boundary neighbours per particle
export const MAXBP = 1200; // boundary particles across all rafts
export const MAX_RAFTS = 12;

export const POLY6 = 315 / (64 * Math.PI * Math.pow(H, 9));
export const SPIKY = -45 / (Math.PI * Math.pow(H, 6));
export const W0 = POLY6 * H2 * H2 * H2;
export const EPS_LAMBDA = 0.02;
export const ITERS = 3;
export const SCORR_K = 0.001;
export const SCORR_DQ = 0.3 * H;
export const SCORR_WQ = POLY6 * Math.pow(H2 - SCORR_DQ * SCORR_DQ, 3);
export const MAX_SPEED = 15;
export const MAX_DP = 0.5 * D;
export const WET_REF = 300; // fluid density at a raft sample point when fully submerged

export const RAFT_L = 0.3;
export const RAFT_T = 0.09;
export const RAFT_RHO = 500;
export const RAFT_SPACING = 0.075; // boundary sample spacing on the board
