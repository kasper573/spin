//! What the CPU hands the GPU every frame: the solver parameters and body poses of each
//! substep, particles to append, and the bodies' sample tables. Layouts mirror `common.wgsl`.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;

use super::{
    EPS_LAMBDA, FluidParams, Grid, H, H2, MARGIN, MAX_BODIES, MAX_DELTA, MAX_SPEED, PARTICLE_MASS,
    PARTICLE_SPACING, POLY6, REST_DENSITY, SCORR_K, SCORR_WQ, SPIKY, W0, WET_REF,
};
use crate::core::math::Vec3d;
use crate::core::rigid::{Body, BodyShape, Collider, WaterCoupling};

/// Units per unit in the shaders' fixed-point accumulators.
const FIXED: f64 = 65536.0;
pub const ACCUMULATORS_PER_BODY: usize = 32;
/// Two accumulator slots no body uses, stamped with how many substeps the frame ran and its
/// ticket, so a readback tells which frame it reports and whether the water moved in it.
pub const STAMP_SLOT: usize = ACCUMULATORS_PER_BODY - 2;

#[derive(Resource, Clone, Default, ExtractResource)]
pub struct FluidFrame {
    /// Stamped into the accumulators so the readback tells which frame it reports.
    pub ticket: u32,
    pub substeps: Vec<Substep>,
    /// Parameters for appending `pending`: the count before the append and how many join.
    pub inject: Params,
    /// Parameters for re-sorting the final positions before the surface is extracted.
    pub surface: Params,
    /// Two entries per joining particle: position with foam, velocity.
    pub pending: Vec<[f32; 4]>,
    /// The bodies' boundary samples in their local frames (xyz, Ψ), when they changed.
    pub samples: Option<Vec<[f32; 4]>>,
    /// Whether anything moved, so the surface needs rebuilding.
    pub changed: bool,
}

#[derive(Clone)]
pub struct Substep {
    pub params: Params,
    pub bodies: Bodies,
}

#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct Params {
    pub dt: f32,
    pub h: f32,
    pub h_sq: f32,
    pub poly: f32,
    pub spiky: f32,
    pub w_zero: f32,
    pub mass: f32,
    pub rest_density: f32,
    pub scorr_k: f32,
    pub scorr_wq: f32,
    pub eps_lambda: f32,
    pub max_delta: f32,
    pub max_speed: f32,
    pub margin: f32,
    pub air_k: f32,
    pub wall_keep: f32,
    pub viscosity: f32,
    pub body_drag: f32,
    pub wet_ref: f32,
    pub spacing: f32,
    pub count: u32,
    pub body_count: u32,
    pub sample_count: u32,
    pub pending: u32,
    pub grid_min: Vec4,
    pub grid_dims: IVec4,
}

impl Params {
    pub fn new(dt: f32, p: &FluidParams, count: u32, bodies: &Bodies, grid: &Grid) -> Self {
        Params {
            dt,
            h: H,
            h_sq: H2,
            poly: POLY6,
            spiky: SPIKY,
            w_zero: W0,
            mass: PARTICLE_MASS,
            rest_density: REST_DENSITY,
            scorr_k: SCORR_K,
            scorr_wq: SCORR_WQ,
            eps_lambda: EPS_LAMBDA,
            max_delta: MAX_DELTA,
            max_speed: MAX_SPEED,
            margin: MARGIN,
            air_k: if p.air { dt / p.air_tau.0 } else { 0.0 },
            wall_keep: 1.0 - p.wall_friction,
            viscosity: p.viscosity * PARTICLE_MASS / REST_DENSITY,
            body_drag: p.body_drag,
            wet_ref: WET_REF,
            spacing: PARTICLE_SPACING,
            count,
            body_count: bodies.count,
            sample_count: bodies.sample_count,
            pending: 0,
            grid_min: Vec4::new(grid.min[0], grid.min[1], grid.min[2], 1.0 / grid.cell),
            grid_dims: IVec4::new(
                grid.dims[0],
                grid.dims[1],
                grid.dims[2],
                grid.cells() as i32,
            ),
        }
    }

    pub fn with_pending(mut self, pending: u32) -> Self {
        self.pending = pending;
        self
    }
}

#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct GpuBody {
    pub position: Vec4,
    pub row_x: Vec4,
    pub row_y: Vec4,
    pub row_z: Vec4,
    pub velocity: Vec4,
    pub angular: Vec4,
    pub shape: Vec4,
    pub slots: UVec4,
    pub extra: Vec4,
}

#[derive(ShaderType, Clone, Debug)]
pub struct GpuBodies {
    pub items: [GpuBody; MAX_BODIES],
}

impl Default for GpuBodies {
    fn default() -> Self {
        GpuBodies {
            items: [GpuBody::default(); MAX_BODIES],
        }
    }
}

/// The bodies of one substep as the shaders read them, plus how many there are.
#[derive(Clone, Debug, Default)]
pub struct Bodies {
    pub gpu: GpuBodies,
    pub count: u32,
    pub sample_count: u32,
}

/// Where a shape's samples sit in the shared sample table.
#[derive(Clone, Copy, Debug)]
pub struct ShapeSamples {
    pub first: u32,
    pub count: u32,
    pub volume_per_sample: f32,
}

pub fn sample_table(shapes: &[BodyShape]) -> (Vec<ShapeSamples>, Vec<[f32; 4]>) {
    let mut layout = Vec::new();
    let mut table = Vec::new();
    for shape in shapes {
        layout.push(ShapeSamples {
            first: table.len() as u32,
            count: shape.samples.len() as u32,
            volume_per_sample: shape.volume_per_sample as f32,
        });
        table.extend_from_slice(&shape.samples);
    }
    (layout, table)
}

pub fn pack(bodies: &[Body], shapes: &[BodyShape], layout: &[ShapeSamples]) -> Bodies {
    let mut out = Bodies::default();
    let mut boundary = 0u32;
    for (item, body) in out.gpu.items.iter_mut().zip(bodies.iter().take(MAX_BODIES)) {
        let shape = &shapes[body.shape];
        let samples = layout[body.shape];
        let count = if body.solid { samples.count } else { 0 };
        let (shape_v, kind) = match shape.collider {
            Collider::Box { half } => (
                Vec4::new(half[0] as f32, half[1] as f32, half[2] as f32, 0.0),
                0,
            ),
            Collider::Sphere { radius, centre } => (
                Vec4::new(
                    centre[0] as f32,
                    centre[1] as f32,
                    centre[2] as f32,
                    radius as f32,
                ),
                1,
            ),
        };
        let m = body.m.map(|v| v as f32);
        *item = GpuBody {
            position: v4(body.p, if body.solid { 1.0 } else { 0.0 }),
            row_x: Vec4::new(m[0], m[1], m[2], 0.0),
            row_y: Vec4::new(m[3], m[4], m[5], 0.0),
            row_z: Vec4::new(m[6], m[7], m[8], 0.0),
            velocity: v4(body.v, 0.0),
            angular: v4(body.w, 0.0),
            shape: shape_v,
            slots: UVec4::new(samples.first, count, boundary, kind),
            extra: Vec4::new(
                samples.volume_per_sample,
                body.inv_m as f32,
                0.0,
                shape.reach() as f32,
            ),
        };
        boundary += count;
        out.count += 1;
    }
    out.sample_count = boundary;
    out
}

pub fn decode_coupling(raw: &[i32]) -> Vec<WaterCoupling> {
    raw.chunks_exact(ACCUMULATORS_PER_BODY)
        .map(|c| {
            let f = |i: usize| c[i] as f64 / FIXED;
            WaterCoupling {
                buoyancy: [f(0), f(1), f(2)],
                buoyancy_torque: [f(3), f(4), f(5)],
                flow: [f(6), f(7), f(8)],
                flow_moment: [f(9), f(10), f(11)],
                hull: [f(12), f(13), f(14)],
                hull_tensor: [f(15), f(16), f(17), f(18), f(19), f(20)],
                coupling: f(21),
                wet: f(22),
                seconds: 0.0,
                substeps: 0.0,
            }
        })
        .collect()
}

fn v4(v: Vec3d, w: f32) -> Vec4 {
    Vec4::new(v[0] as f32, v[1] as f32, v[2] as f32, w)
}
