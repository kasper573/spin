//! Water that lies on the ground: the height its face stands at and the flow under it, cell by
//! cell of a chart of the ground, stepped on the GPU (see `shallows.wgsl` for the method). What
//! the ground is, how heavy water is on it and how its frame turns come from a shader module
//! named `ground`, bound through the vessel's group, which whoever owns the ground provides;
//! this module knows only the chart.
mod gpu;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::storage::ShaderBuffer;

use crate::core::units::{Litres, Metres, Seconds};

pub use gpu::{ShallowsBuffers, ShallowsStep};

/// The most cells a chart may have.
pub const MAX_SHALLOWS_CELLS: usize = 1 << 18;
/// The most pourings a frame takes.
pub const MAX_POURED: usize = 256;

/// The GPU water on the ground; the ground's plugin must supply the shaders' `ground` module
/// and the render world's [`VesselLayout`](crate::core::vessel::VesselLayout).
pub struct ShallowsPlugin;

impl Plugin for ShallowsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shallows.wgsl");
        let buffers =
            gpu::create_buffers(&mut app.world_mut().resource_mut::<Assets<ShaderBuffer>>());
        let ready = ShallowsReady::default();
        app.init_resource::<Shallows>()
            .init_resource::<ShallowsFrame>()
            .insert_resource(ready.clone())
            .insert_resource(buffers)
            .add_plugins((
                ExtractResourcePlugin::<ShallowsBuffers>::default(),
                ExtractResourcePlugin::<ShallowsFrame>::default(),
            ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(ready);
        gpu::install(render_app);
    }
}

/// Set by the render world once every kernel has compiled and the ground is bound; a frame
/// made before then would have what it asks lost.
#[derive(Resource, Clone, Default)]
pub struct ShallowsReady(Arc<AtomicBool>);

impl ShallowsReady {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The cells the water is kept in: how many along the chart's two axes, whether the chart
/// closes on itself along each, and a cell's sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShallowsChart {
    pub size: [u32; 2],
    pub wraps: [bool; 2],
    pub cell: [Metres; 2],
}

impl ShallowsChart {
    pub fn cells(&self) -> usize {
        self.size[0] as usize * self.size[1] as usize
    }
}

/// A cell as the GPU keeps it: the height of the water's face, the flow through the cell's two
/// low faces, and how fast the face climbs by the pressure beyond the water's weight.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShallowsCell {
    pub face: f32,
    pub flow: [f32; 2],
    pub climbing: f32,
}

impl ShallowsCell {
    pub const BYTES: usize = 16;

    pub fn read(bytes: &[u8]) -> impl Iterator<Item = ShallowsCell> + '_ {
        bytes.chunks_exact(Self::BYTES).map(|cell| {
            let float =
                |i: usize| f32::from_le_bytes([cell[i], cell[i + 1], cell[i + 2], cell[i + 3]]);
            ShallowsCell {
                face: float(0),
                climbing: float(4),
                flow: [float(8), float(12)],
            }
        })
    }
}

/// What is asked of the water on the ground, until the next frame takes it up.
#[derive(Resource, Default)]
pub struct Shallows {
    chart: Option<ShallowsChart>,
    bed_friction: f32,
    standing: Option<[f32; 4]>,
    poured: Vec<[f32; 4]>,
}

impl Shallows {
    /// Keep the water in these cells, dragged by the bed this much.
    pub fn chart(&mut self, chart: ShallowsChart, bed_friction: f32) {
        assert!(
            chart.cells() <= MAX_SHALLOWS_CELLS,
            "a chart of {} cells, where {MAX_SHALLOWS_CELLS} are kept",
            chart.cells()
        );
        self.chart = Some(chart);
        self.bed_friction = bed_friction;
    }

    pub fn charted(&self) -> Option<ShallowsChart> {
        self.chart
    }

    /// Stand the water at rest to a level over the chart's zero, tilting this much per metre
    /// along the chart's two axes; ground above it is left dry.
    pub fn stand(&mut self, level: Metres, tilt: [f32; 2]) {
        self.standing = Some([level.0, tilt[0], tilt[1], 0.0]);
        self.poured.clear();
    }

    /// Take all the water off the ground.
    pub fn clear(&mut self) {
        self.stand(Metres(f32::MIN), [0.0; 2]);
    }

    /// Pour water onto a place of the chart. More than a frame takes waits for the next.
    pub fn pour(&mut self, at: [f64; 2], volume: Litres) {
        self.poured
            .push([at[0] as f32, at[1] as f32, volume.0 / 1000.0, 0.0]);
    }

    /// What the GPU is to do this frame: the steps to take, with the chart's low corner where
    /// it is now from the zero the ground module counts from.
    pub fn frame(&mut self, steps: &[Seconds], low: [f64; 2]) -> ShallowsFrame {
        let taken = self.poured.len().min(MAX_POURED);
        ShallowsFrame {
            chart: self.chart,
            low: [low[0] as f32, low[1] as f32],
            bed_friction: self.bed_friction,
            steps: steps.to_vec(),
            standing: self.standing.take(),
            poured: self.poured.drain(..taken).collect(),
        }
    }
}

/// A frame's work for the GPU.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct ShallowsFrame {
    pub chart: Option<ShallowsChart>,
    pub low: [f32; 2],
    pub bed_friction: f32,
    pub steps: Vec<Seconds>,
    pub standing: Option<[f32; 4]>,
    pub poured: Vec<[f32; 4]>,
}
