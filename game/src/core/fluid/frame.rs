//! What the CPU hands the GPU every frame: the solver parameters and body poses of each
//! substep, particles to append, and the bodies' sample tables. Layouts mirror `common.wgsl`.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;

use super::{
    EPS_LAMBDA, FluidParams, MAX_BODIES, REST_DENSITY, Resolution, SCORR_K, TABLE_CELLS, WET_REF,
};
use crate::core::math::Vec3d;
use crate::core::rigid::{Body, BodyShape, Hull, WaterCoupling};

/// Units per unit in the shaders' fixed-point accumulators.
const FIXED: f64 = 65536.0;
pub const ACCUMULATORS_PER_BODY: usize = 16;
/// An accumulator slot no body uses, stamped with the frame's ticket so a readback tells which
/// frame it reports.
pub const STAMP_SLOT: usize = ACCUMULATORS_PER_BODY - 1;

#[derive(Resource, Clone, Default, ExtractResource)]
pub struct FluidFrame {
    /// Stamped into the accumulators so the readback tells which frame it reports.
    pub ticket: u32,
    /// Parameters for thinning the water to every other particle before anything else: the
    /// count to sort and, as `pending`, the count that remains.
    pub thin: Option<Params>,
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
    /// Whether the substeps couple water and bodies, so the accumulators want reading back.
    pub coupling: bool,
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
    pub inv_cell: f32,
    pub cells: u32,
    pub pad: UVec2,
}

impl Params {
    pub fn new(dt: f32, p: &FluidParams, res: Resolution, count: u32, bodies: &Bodies) -> Self {
        Params {
            dt,
            h: res.h(),
            h_sq: res.h_sq(),
            poly: res.poly6(),
            spiky: res.spiky(),
            w_zero: res.w_zero(),
            mass: res.mass(),
            rest_density: REST_DENSITY,
            scorr_k: SCORR_K,
            scorr_wq: res.scorr_wq(),
            eps_lambda: EPS_LAMBDA,
            max_delta: res.max_delta(),
            max_speed: p.max_speed.0,
            margin: res.margin(),
            air_k: if p.air { dt / p.air_tau.0 } else { 0.0 },
            wall_keep: 1.0 - p.wall_friction,
            viscosity: p.viscosity * res.mass() / REST_DENSITY,
            body_drag: p.body_drag,
            wet_ref: WET_REF,
            spacing: res.spacing.0,
            count,
            body_count: bodies.count,
            sample_count: bodies.sample_count,
            pending: 0,
            inv_cell: 1.0 / res.h(),
            cells: TABLE_CELLS as u32,
            pad: UVec2::ZERO,
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

/// A shape's boundary sample points and the water each displaces.
#[derive(Clone, Debug)]
pub struct ShapePoints {
    pub points: Vec<[f32; 3]>,
    pub volume_per_sample: f32,
}

impl ShapePoints {
    pub fn of(shape: &BodyShape) -> Self {
        ShapePoints {
            points: shape.samples.clone(),
            volume_per_sample: shape.volume_per_sample as f32,
        }
    }
}

/// Where a shape's samples sit in the shared sample table.
#[derive(Clone, Copy, Debug)]
pub struct ShapeSamples {
    pub first: u32,
    pub count: u32,
    pub volume_per_sample: f32,
}

/// Every shape's samples weighted for this resolution, one after the other.
pub fn sample_table(
    shapes: &[ShapePoints],
    resolution: Resolution,
) -> (Vec<ShapeSamples>, Vec<[f32; 4]>) {
    let mut layout = Vec::new();
    let mut table = Vec::new();
    for shape in shapes {
        layout.push(ShapeSamples {
            first: table.len() as u32,
            count: shape.points.len() as u32,
            volume_per_sample: shape.volume_per_sample,
        });
        table.extend(resolution.sample_weights(&shape.points));
    }
    (layout, table)
}

pub fn pack(bodies: &[Body], shapes: &[BodyShape], layout: &[ShapeSamples]) -> Bodies {
    let mut out = Bodies::default();
    let mut boundary = 0u32;
    for (item, body) in out.gpu.items.iter_mut().zip(bodies.iter().take(MAX_BODIES)) {
        let shape = &shapes[body.shape];
        let Some(samples) = layout.get(body.shape) else {
            continue;
        };
        let count = if body.solid { samples.count } else { 0 };
        let Hull { radius, centre } = shape.hull;
        let shape_v = Vec4::new(
            centre[0] as f32,
            centre[1] as f32,
            centre[2] as f32,
            radius as f32,
        );
        let m = body.m.map(|v| v as f32);
        *item = GpuBody {
            position: v4(body.p, if body.solid { 1.0 } else { 0.0 }),
            row_x: Vec4::new(m[0], m[1], m[2], 0.0),
            row_y: Vec4::new(m[3], m[4], m[5], 0.0),
            row_z: Vec4::new(m[6], m[7], m[8], 0.0),
            velocity: v4(body.v, 0.0),
            angular: v4(body.w, 0.0),
            shape: shape_v,
            slots: UVec4::new(samples.first, count, boundary, 0),
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
                coupling: f(9),
                wet: f(10),
                seconds: 0.0,
                substeps: 0.0,
                age: 0.0,
            }
        })
        .collect()
}

fn v4(v: Vec3d, w: f32) -> Vec4 {
    Vec4::new(v[0] as f32, v[1] as f32, v[2] as f32, w)
}
