//! What the CPU hands the GPU every frame: the solver parameters and body poses of each
//! substep, particles to append, and the bodies' sample tables, all in the canonical water's
//! units and the vessel's frame; and what comes back, in the world's units. Layouts mirror
//! `common.wgsl`.
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;

use super::resolution::canonical;
use super::{
    EPS_LAMBDA, FluidParams, MAX_BODIES, REST_DENSITY, Resolution, SCORR_K, TABLE_CELLS, WET_REF,
};
use crate::core::math::{Vec3d, mat3mul, quat_rotate};
use crate::core::rigid::{Body, BodyShape, HullSphere, WaterCoupling};
use crate::core::units::Seconds;
use crate::core::vessel::WaterFrame;

/// Units per unit in the shaders' fixed-point accumulators.
const FIXED: f64 = 65536.0;
pub const ACCUMULATORS_PER_BODY: usize = 16;
/// Frames the accumulators keep apart, each in its own slots: a readback this many frames late
/// still finds every frame it covers.
pub const ACCUMULATOR_FRAMES: usize = 32;
pub const ACCUMULATORS_PER_FRAME: usize = ACCUMULATORS_PER_BODY * MAX_BODIES;
/// The slot after every frame's, stamped with the latest frame's ticket so a readback tells
/// which frames it reports.
pub const STAMP_SLOT: usize = ACCUMULATORS_PER_FRAME * ACCUMULATOR_FRAMES;

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
    /// The first accumulator of this frame's slots.
    pub accumulators: u32,
    pub pad: u32,
    /// What thinning scales the kept particles' positions and velocities by.
    pub thin_scale: f32,
    pub thin_scale_v: f32,
}

impl Params {
    pub fn new(dt: Seconds, p: &FluidParams, res: Resolution, count: u32, bodies: &Bodies) -> Self {
        let (length, time) = (res.length(), res.time());
        Params {
            dt: (dt.0 as f64 / time) as f32,
            h: canonical::H,
            h_sq: canonical::H_SQ,
            poly: canonical::POLY6,
            spiky: canonical::SPIKY,
            w_zero: canonical::W_ZERO,
            mass: canonical::MASS,
            rest_density: REST_DENSITY,
            scorr_k: SCORR_K,
            scorr_wq: canonical::SCORR_WQ,
            eps_lambda: EPS_LAMBDA,
            max_delta: canonical::MAX_DELTA,
            max_speed: (p.max_speed.0 as f64 * time / length) as f32,
            margin: canonical::MARGIN,
            air_k: if p.air {
                (dt.0 as f64 / time) as f32 / p.air_tau.0
            } else {
                0.0
            },
            wall_keep: 1.0 - p.wall_friction,
            viscosity: p.viscosity * canonical::MASS / REST_DENSITY,
            body_drag: p.body_drag,
            wet_ref: WET_REF,
            spacing: canonical::SPACING,
            count,
            body_count: bodies.count,
            sample_count: bodies.sample_count,
            pending: 0,
            inv_cell: 1.0 / canonical::H,
            cells: TABLE_CELLS as u32,
            accumulators: 0,
            pad: 0,
            thin_scale: 1.0,
            thin_scale_v: 1.0,
        }
    }

    pub fn with_pending(mut self, pending: u32) -> Self {
        self.pending = pending;
        self
    }

    /// For thinning water of resolution `from` into this one: what the kept particles'
    /// positions and velocities are scaled by, in the canonical units.
    pub fn thinning(mut self, from: Resolution, to: Resolution) -> Self {
        self.thin_scale = (from.length() / to.length()) as f32;
        self.thin_scale_v = (from.length() / to.length() * to.time() / from.time()) as f32;
        self
    }

    pub fn for_frame(mut self, ticket: u32) -> Self {
        self.accumulators = accumulators_of(ticket);
        self
    }
}

/// The first accumulator of a frame's slots.
pub fn accumulators_of(ticket: u32) -> u32 {
    (ticket as usize % ACCUMULATOR_FRAMES * ACCUMULATORS_PER_FRAME) as u32
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

/// Every shape's samples in the canonical units of this resolution, one after the other.
pub fn sample_table(
    shapes: &[ShapePoints],
    resolution: Resolution,
) -> (Vec<ShapeSamples>, Vec<[f32; 4]>) {
    let length = resolution.length() as f32;
    let mut layout = Vec::new();
    let mut table = Vec::new();
    for shape in shapes {
        layout.push(ShapeSamples {
            first: table.len() as u32,
            count: shape.points.len() as u32,
            volume_per_sample: shape.volume_per_sample / (length * length * length),
        });
        let points: Vec<[f32; 3]> = shape.points.iter().map(|p| p.map(|x| x / length)).collect();
        table.extend(canonical::sample_weights(&points));
    }
    (layout, table)
}

/// The bodies as the water sees them: in the water's frame, in the canonical units.
pub fn pack(
    bodies: &[Body],
    shapes: &[BodyShape],
    layout: &[ShapeSamples],
    frame: &WaterFrame,
    resolution: Resolution,
) -> Bodies {
    let (length, time) = (resolution.length(), resolution.time());
    let mut out = Bodies::default();
    let mut boundary = 0u32;
    for (item, body) in out.gpu.items.iter_mut().zip(bodies.iter().take(MAX_BODIES)) {
        let shape = &shapes[body.shape];
        let Some(samples) = layout.get(body.shape) else {
            continue;
        };
        let count = if body.solid { samples.count } else { 0 };
        let HullSphere { centre, radius } = shape.hull.bulk();
        let p = frame.to_water(body.p).map(|x| x / length);
        let v = frame.vector_to_water(body.v).map(|x| x * time / length);
        let w = frame.vector_to_water(body.w).map(|x| x * time);
        let m = rotated_rows(&frame.rotation, &body.m);
        *item = GpuBody {
            position: v4(p, if body.solid { 1.0 } else { 0.0 }),
            row_x: Vec4::new(m[0], m[1], m[2], 0.0),
            row_y: Vec4::new(m[3], m[4], m[5], 0.0),
            row_z: Vec4::new(m[6], m[7], m[8], 0.0),
            velocity: v4(v, 0.0),
            angular: v4(w, 0.0),
            shape: Vec4::new(
                (centre[0] / length) as f32,
                (centre[1] / length) as f32,
                (centre[2] / length) as f32,
                (radius / length) as f32,
            ),
            slots: UVec4::new(samples.first, count, boundary, 0),
            extra: Vec4::new(
                samples.volume_per_sample,
                (body.inv_m * length * length * length) as f32,
                0.0,
                (shape.reach() / length) as f32,
            ),
        };
        boundary += count;
        out.count += 1;
    }
    out.sample_count = boundary;
    out
}

/// One frame's coupling, from its accumulator slots, in the bodies' frame and units: the
/// accumulators hold the buoyancy as the velocity it gave each body and its torque impulse per
/// unit mass, which keeps every body's sums in the fixed-point range whatever it weighs.
pub fn decode_coupling(
    slots: &[i32],
    resolution: Resolution,
    frame: &WaterFrame,
) -> Vec<WaterCoupling> {
    let (length, time) = (resolution.length(), resolution.time());
    let velocity = length / time;
    slots
        .chunks_exact(ACCUMULATORS_PER_BODY)
        .take(MAX_BODIES)
        .map(|c| {
            let f = |i: usize| c[i] as f64 / FIXED;
            let vector = |i: usize| frame.vector_from_water([f(i), f(i + 1), f(i + 2)]);
            WaterCoupling {
                buoyancy: vector(0).map(|x| x * velocity),
                buoyancy_torque: vector(3).map(|x| x * velocity * length),
                flow: vector(6).map(|x| x * velocity),
                coupling: f(9),
                wet: f(10),
                seconds: 0.0,
                substeps: 0.0,
            }
        })
        .collect()
}

/// The rows of a body's rotation matrix once the whole thing is turned by `q`.
fn rotated_rows(q: &[f64; 4], m: &[f64; 9]) -> [f32; 9] {
    let columns = [
        mat3mul(m, &[1.0, 0.0, 0.0]),
        mat3mul(m, &[0.0, 1.0, 0.0]),
        mat3mul(m, &[0.0, 0.0, 1.0]),
    ]
    .map(|c| quat_rotate(q, &c));
    let mut out = [0.0; 9];
    for (i, column) in columns.iter().enumerate() {
        for row in 0..3 {
            out[row * 3 + i] = column[row] as f32;
        }
    }
    out
}

fn v4(v: Vec3d, w: f32) -> Vec4 {
    Vec4::new(v[0] as f32, v[1] as f32, v[2] as f32, w)
}
