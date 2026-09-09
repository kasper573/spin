//! The drum as the water's vessel on the GPU: its per-substep state as a uniform and the
//! landscape as a height texture, bound at group 1 of every fluid kernel (see `drum.wgsl`).
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{texture_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, DynamicUniformBuffer,
    Extent3d, PipelineCache, ShaderStages, ShaderType, TextureDimension, TextureFormat,
    TextureSampleType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderSystems};
use bevy::shader::Shader;

use super::Drum;
use super::landscape::{ROWS, SEGMENTS};
use crate::core::units::{Radians, RadiansPerSecond};
use crate::core::vessel::{VesselBinding, VesselLayout};

const SHADER: &str = "embedded://game/systems/drum/shaders/drum.wgsl";

#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct DrumUniform {
    spin: f32,
    angle: f32,
    radius: f32,
    half_width: f32,
    landscape: u32,
    segments: u32,
    rows: u32,
    pad: u32,
    dphi: f32,
    dy: f32,
    pad_b: f32,
    pad_c: f32,
}

impl DrumUniform {
    pub fn new(drum: &Drum, spin: RadiansPerSecond, angle: Radians) -> Self {
        DrumUniform {
            spin: spin.0,
            angle: angle.0 as f32,
            radius: drum.ring.radius.0,
            half_width: drum.ring.half_width.0,
            landscape: u32::from(!drum.landscape.is_empty()),
            segments: SEGMENTS as u32,
            rows: ROWS as u32,
            pad: 0,
            dphi: std::f32::consts::TAU / SEGMENTS as f32,
            dy: drum.landscape.row_spacing() as f32,
            pad_b: 0.0,
            pad_c: 0.0,
        }
    }
}

/// The drum's state for each substep of the frame, then as it is after them.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DrumFrame {
    pub states: Vec<DrumUniform>,
    pub heights: Handle<Image>,
}

pub fn install(app: &mut App) {
    let heights = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(height_image(&vec![0.0; SEGMENTS * ROWS]))
    };
    let shader = app.world().resource::<AssetServer>().load::<Shader>(SHADER);
    app.insert_resource(DrumFrame {
        states: Vec::new(),
        heights,
    })
    .insert_resource(VesselShader(shader))
    .add_plugins(ExtractResourcePlugin::<DrumFrame>::default());
    let layout = BindGroupLayoutDescriptor::new(
        "drum vessel",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DrumUniform>(true),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    app.sub_app_mut(RenderApp)
        .insert_resource(VesselLayout(layout))
        .init_resource::<DrumUniforms>()
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups));
}

/// Upload the landscape whenever it changed.
pub fn upload_heights(drum: &Drum, frame: &DrumFrame, images: &mut Assets<Image>) {
    if let Some(mut image) = images.get_mut(&frame.heights) {
        *image = height_image(drum.landscape.heights());
    }
}

#[derive(Resource)]
struct VesselShader(#[allow(dead_code)] Handle<Shader>);

#[derive(Resource, Default)]
struct DrumUniforms(DynamicUniformBuffer<DrumUniform>);

fn height_image(heights: &[f32]) -> Image {
    // rows along x, segments along y: (segment, row) lands at pixel (row, segment)
    Image::new(
        Extent3d {
            width: ROWS as u32,
            height: SEGMENTS as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(heights).to_vec(),
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    frame: Res<DrumFrame>,
    layout: Res<VesselLayout>,
    pipeline_cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut uniforms: ResMut<DrumUniforms>,
) {
    let Some(heights) = images.get(&frame.heights) else {
        return;
    };
    uniforms.0.clear();
    let offsets: Vec<u32> = frame.states.iter().map(|s| uniforms.0.push(s)).collect();
    if offsets.is_empty() {
        return;
    }
    uniforms.0.write_buffer(&device, &queue);
    let Some(binding) = uniforms.0.binding() else {
        return;
    };
    let bind_group = device.create_bind_group(
        "drum vessel",
        &pipeline_cache.get_bind_group_layout(&layout.0),
        &BindGroupEntries::sequential((binding, &heights.texture_view)),
    );
    commands.insert_resource(VesselBinding {
        bind_group,
        offsets,
    });
}
