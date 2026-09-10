//! The drum as the water's vessel on the GPU: its per-substep state as a uniform, in the
//! water's units, and the landscape as a height texture, bound at group 1 of every fluid
//! kernel (see `drum.wgsl`).
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferUsages,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    DynamicUniformBuffer, Extent3d, PipelineCache, ShaderStages, ShaderType, TextureDimension,
    TextureFormat, TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderStartup, RenderSystems};
use bevy::shader::Shader;

use super::Drum;
use super::landscape::{ROWS, SEGMENTS};
use crate::core::fluid::{FluidBuffers, FluidFrame, FluidStep, Resolution};
use crate::core::units::{RadiansPerSecond, RadiansPerSecondSquared};
use crate::core::vessel::{VesselBinding, VesselLayout};

const SHADER: &str = "embedded://game/systems/drum/shaders/drum.wgsl";
const SURVEY_SHADER: &str = "embedded://game/systems/drum/shaders/columns.wgsl";
/// Words per column of the survey: the height the water reaches, the particles counted, and
/// the flow round the ring and along its axis summed over them.
const COLUMN_WORDS: usize = 4;
/// The survey's fixed point: units of the water's length and speed per whole number.
pub const COLUMN_FIXED: f64 = 256.0;
const SURVEY_WORKGROUP: u32 = 64;

#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct DrumUniform {
    spin: f32,
    spin_rate: f32,
    radius: f32,
    half_width: f32,
    per_metre: f32,
    landscape: u32,
    segments: u32,
    rows: u32,
    pad: u32,
    dphi: f32,
    dy: f32,
}

impl DrumUniform {
    pub fn new(
        drum: &Drum,
        spin: RadiansPerSecond,
        spin_rate: RadiansPerSecondSquared,
        resolution: Resolution,
    ) -> Self {
        let (length, time) = (resolution.length(), resolution.time());
        DrumUniform {
            spin: (spin.0 as f64 * time) as f32,
            spin_rate: (spin_rate.0 as f64 * time * time) as f32,
            radius: (drum.ring.radius.0 as f64 / length) as f32,
            half_width: (drum.ring.half_width.0 as f64 / length) as f32,
            per_metre: (1.0 / length) as f32,
            landscape: u32::from(!drum.landscape.is_empty()),
            segments: SEGMENTS as u32,
            rows: ROWS as u32,
            pad: 0,
            dphi: std::f32::consts::TAU / SEGMENTS as f32,
            dy: (drum.landscape.row_spacing() / length) as f32,
        }
    }
}

/// The drum's state for each substep of the frame, then as it is after them, and the water
/// surveyed over every column of the ground once the frame's water has been stepped.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DrumFrame {
    pub states: Vec<DrumUniform>,
    pub heights: Handle<Image>,
    pub columns: Handle<ShaderBuffer>,
}

pub fn install(app: &mut App) {
    let heights = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(height_image(&vec![0.0; SEGMENTS * ROWS]))
    };
    let columns = {
        let mut buffers = app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
        let mut buffer = ShaderBuffer::with_size(SEGMENTS * ROWS * COLUMN_WORDS * 4, default());
        buffer.buffer_description.usage |= BufferUsages::COPY_DST;
        buffers.add(buffer)
    };
    let shader = app.world().resource::<AssetServer>().load::<Shader>(SHADER);
    app.insert_resource(DrumFrame {
        states: Vec::new(),
        heights,
        columns,
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
        .init_resource::<SurveyUniform>()
        .add_systems(RenderStartup, init_survey)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(RenderGraph, survey.after(FluidStep).before(camera_driver));
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

#[derive(ShaderType, Clone, Copy, Default)]
struct Survey {
    count: u32,
}

#[derive(Resource, Default)]
struct SurveyUniform(UniformBuffer<Survey>);

/// The kernel that surveys the water over the ground, and what it binds.
#[derive(Resource)]
struct SurveyPipeline {
    id: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

fn init_survey(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "drum survey",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DrumUniform>(true),
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<Survey>(false),
                storage_buffer_read_only::<Vec4>(false),
                storage_buffer_read_only::<Vec4>(false),
                storage_buffer::<u32>(false),
            ),
        ),
    );
    let empty = BindGroupLayoutDescriptor::new("drum survey none", &[]);
    let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("drum survey".into()),
        layout: vec![empty, layout.clone()],
        shader: assets.load(SURVEY_SHADER),
        entry_point: Some("survey_columns".into()),
        ..default()
    });
    commands.insert_resource(SurveyPipeline { id, layout });
}

/// Survey the water over the ground from the particles as the frame's water left them.
#[allow(clippy::too_many_arguments)]
fn survey(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipeline: Res<SurveyPipeline>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    fluid: Res<FluidBuffers>,
    fluid_frame: Res<FluidFrame>,
    frame: Res<DrumFrame>,
    vessel: Option<Res<VesselBinding>>,
    uniforms: Res<DrumUniforms>,
    mut params: ResMut<SurveyUniform>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(compute), Some(vessel)) = (cache.get_compute_pipeline(pipeline.id), vessel) else {
        return;
    };
    let (Some(&offset), Some(drum)) = (vessel.offsets.last(), uniforms.0.binding()) else {
        return;
    };
    let (Some(heights), Some(position), Some(velocity), Some(columns)) = (
        images.get(&frame.heights),
        buffers.get(&fluid.position),
        buffers.get(&fluid.velocity),
        buffers.get(&frame.columns),
    ) else {
        return;
    };
    params.0.set(Survey {
        count: fluid_frame.surface.count,
    });
    params.0.write_buffer(&device, &queue);
    let Some(survey) = params.0.binding() else {
        return;
    };
    let bind_group = device.create_bind_group(
        "drum survey",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            drum,
            &heights.texture_view,
            survey,
            position.buffer.as_entire_binding(),
            velocity.buffer.as_entire_binding(),
            columns.buffer.as_entire_binding(),
        )),
    );
    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&columns.buffer, 0, None);
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("drum survey"),
        ..default()
    });
    pass.set_pipeline(compute);
    pass.set_bind_group(1, &bind_group, &[offset]);
    pass.dispatch_workgroups(
        fluid_frame.surface.count.div_ceil(SURVEY_WORKGROUP).max(1),
        1,
        1,
    );
}

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
