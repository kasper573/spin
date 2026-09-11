//! The drum as the water's vessel on the GPU: its per-substep state as a uniform, in the
//! water's units and about the water's site, and the sculpted landscape as an atlas of patches
//! and the table they are found by, bound at group 1 of every fluid kernel (see `drum.wgsl`);
//! and the water surveyed over the ground, column by column, for the ground to be lit through.
use bevy::core_pipeline::schedule::camera_driver;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferUsages,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    DynamicUniformBuffer, Extent3d, Origin3d, PipelineCache, ShaderStages, ShaderType,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderStartup, RenderSystems};
use bevy::shader::Shader;

use super::landscape::{Grid, PATCH};
use super::{Drum, Ring};
use crate::core::fluid::{
    Fluid, FluidBuffers, FluidFrame, FluidStep, MAX_PARTICLES, ReadOnce, Resolution,
};
use crate::core::units::{Metres, RadiansPerSecond, RadiansPerSecondSquared};
use crate::core::vessel::{VesselBinding, VesselLayout};

const SHADER: &str = "embedded://game/systems/drum/shaders/drum.wgsl";
const SURVEY_SHADER: &str = "embedded://game/systems/drum/shaders/columns.wgsl";
/// The survey's fixed point: units of the water's length and speed per whole number.
pub const COLUMN_FIXED: f64 = 256.0;
/// Words before the survey's columns, and words per column: its key, the height the water
/// reaches, the particles counted, and the flow round the ring and along its axis summed over
/// them.
pub const SURVEY_HEADER: usize = 4;
pub const COLUMN_WORDS: usize = 5;
/// Every particle stands in one column, so twice as many entries as particles keeps the table
/// at most half full.
pub const SURVEY_SLOTS: usize = 2 * MAX_PARTICLES;
const SURVEY_WORKGROUP: u32 = 64;
/// Patches a row of the atlas holds, and entries a row of the table they are found by holds.
const ATLAS_PATCHES: usize = 64;
const TABLE_WIDTH: usize = 256;
/// The most patches the atlas holds, as far as the largest texture a device must offer lets it:
/// past that, the water sees only the patches nearest its site.
const MOST_PATCHES: usize = ATLAS_PATCHES * (8192 / PATCH as usize);
/// The most things round the ring the water counts in a single precision whole number: a
/// ring with more is too big for the water to reach round, and nothing needs to close on
/// itself.
const MOST_ROUND: i64 = 1 << 30;

#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct DrumUniform {
    spin: f32,
    spin_rate: f32,
    radius: f32,
    half_width: f32,
    per_metre: f32,
    along: f32,
    base: f32,
    cells_round: f32,
    cells_along: f32,
    across: f32,
    first: i32,
    rows: i32,
    patches_round: u32,
    patches: u32,
    table_mask: u32,
    atlas: u32,
}

impl DrumUniform {
    pub fn new(
        drum: &Drum,
        spin: RadiansPerSecond,
        spin_rate: RadiansPerSecondSquared,
        resolution: Resolution,
        ground: &GroundShape,
    ) -> Self {
        let (length, time) = (resolution.length(), resolution.time());
        let grid = drum.landscape.grid();
        let patches_round = grid.patches_round();
        DrumUniform {
            spin: (spin.0 as f64 * time) as f32,
            spin_rate: (spin_rate.0 as f64 * time * time) as f32,
            radius: (drum.ring.radius.0 as f64 / length) as f32,
            half_width: (drum.ring.half_width.0 as f64 / length) as f32,
            per_metre: (1.0 / length) as f32,
            along: (drum.water.y / length) as f32,
            base: (drum.landscape.base() as f64 / length) as f32,
            cells_round: (length / grid.arc) as f32,
            cells_along: (length / grid.along) as f32,
            across: drum.water.round.across as f32,
            first: drum.water.round.cell.rem_euclid(PATCH) as i32,
            rows: grid.rows.min(i32::MAX as i64) as i32,
            patches_round: if patches_round < MOST_ROUND {
                patches_round as u32
            } else {
                0
            },
            patches: ground.patches,
            table_mask: ground.table_mask,
            atlas: ATLAS_PATCHES as u32,
        }
    }
}

/// How the sculpted patches were last laid out for the GPU: how many there are, and the mask
/// of the table they are found by.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroundShape {
    pub patches: u32,
    pub table_mask: u32,
}

/// How the survey's columns lie: how many round the ring, or 0 when there are too many for the
/// water to reach round it, and how many of them a unit of the water's arc spans.
#[derive(ShaderType, Clone, Copy, Debug, Default, PartialEq)]
pub struct Columns {
    pub round: u32,
    pub per_unit: f32,
}

impl Columns {
    /// Columns a spacing of the water wide, as many round the ring as close on themselves.
    pub fn of(ring: Ring, resolution: Resolution) -> Columns {
        let circumference = std::f64::consts::TAU * ring.radius.0 as f64;
        let length = resolution.length();
        let round = (circumference / length).round().max(1.0);
        if round < MOST_ROUND as f64 {
            Columns {
                round: round as u32,
                per_unit: (round * length / circumference) as f32,
            }
        } else {
            Columns {
                round: 0,
                per_unit: 1.0,
            }
        }
    }

    /// The arc one column spans round the ring, in metres.
    pub fn arc(self, resolution: Resolution) -> f64 {
        resolution.length() / self.per_unit as f64
    }
}

/// The drum's state for each substep of the frame, then as it is after them; the sculpted
/// ground as the water sees it; and the water surveyed over every column of the ground once the
/// frame's water has been stepped, and how those columns lie.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DrumFrame {
    pub states: Vec<DrumUniform>,
    pub ground: GroundWrites,
    pub shape: GroundShape,
    pub columns: Handle<ShaderBuffer>,
    pub survey: Columns,
}

pub fn install(app: &mut App) {
    let columns = {
        let mut buffers = app.world_mut().resource_mut::<Assets<ShaderBuffer>>();
        let words = SURVEY_HEADER + SURVEY_SLOTS * COLUMN_WORDS;
        let mut buffer = ShaderBuffer::with_size(words * 4, default());
        buffer.buffer_description.usage |= BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        buffers.add(buffer)
    };
    let shader = app.world().resource::<AssetServer>().load::<Shader>(SHADER);
    app.insert_resource(DrumFrame {
        states: Vec::new(),
        ground: GroundWrites::default(),
        shape: GroundShape::default(),
        columns,
        survey: Columns::default(),
    })
    .insert_resource(VesselShader(shader))
    .add_plugins(ExtractResourcePlugin::<DrumFrame>::default())
    .add_systems(Startup, spawn_reach_watcher)
    .add_systems(Update, watch_reach);
    let layout = BindGroupLayoutDescriptor::new(
        "drum vessel",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DrumUniform>(true),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Sint),
            ),
        ),
    );
    app.sub_app_mut(RenderApp)
        .insert_resource(VesselLayout(layout))
        .init_resource::<DrumUniforms>()
        .init_resource::<SurveyUniform>()
        .add_systems(RenderStartup, (init_survey, init_ground))
        .add_systems(
            Render,
            (
                write_ground.in_set(RenderSystems::PrepareResources),
                prepare.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        .add_systems(RenderGraph, survey.after(FluidStep).before(camera_driver));
}

/// Where the sculpted patches lie in the atlas the water finds them in, kept from frame to frame
/// so that only the patches that changed are written to it again. The table they are found by
/// is keyed by which patch round the ring each is from the one the water's site is in, the short
/// way, and which along the axis, so all of it is laid out afresh, nearest first, when the ring
/// or the water's site changes or the atlas outgrows its texture. Past the most the atlas holds,
/// patches sculpted since it was laid out are left unseen by the water.
#[derive(Default)]
pub struct GroundLayout {
    /// The grid and the water's site's patch it was laid out for, and the landscape's version it
    /// has been brought up to.
    laid: Option<(Grid, i64)>,
    version: u64,
    slots: HashMap<(i64, i64), u32>,
    free: Vec<u32>,
    /// The slots handed out so far, freed or not, and how many the atlas holds.
    used: u32,
    capacity: u32,
}

impl GroundLayout {
    /// Bring the atlas up to the landscape, handing the render world what changed.
    pub fn update(&mut self, drum: &Drum, frame: &mut DrumFrame) {
        frame.ground.patches.clear();
        frame.ground.table = None;
        let landscape = &drum.landscape;
        let laid = (landscape.grid(), drum.water.round.cell.div_euclid(PATCH));
        if self.laid != Some(laid) {
            return self.lay(drum, frame);
        }
        if landscape.version() == self.version {
            return;
        }
        let mut reshaped = false;
        let free = &mut self.free;
        self.slots.retain(|&(round, along), slot| {
            let kept = landscape.patch(round, along).is_some();
            if !kept {
                free.push(*slot);
                reshaped = true;
            }
            kept
        });
        for (key, heights) in landscape.changed_since(self.version) {
            let slot = match self.slots.get(&key) {
                Some(slot) => *slot,
                None if keyed(laid, key).is_none() => continue,
                None => {
                    let slot = match self.free.pop() {
                        Some(slot) => slot,
                        None if self.used < self.capacity => {
                            self.used += 1;
                            self.used - 1
                        }
                        None if (self.capacity as usize) < MOST_PATCHES => {
                            return self.lay(drum, frame);
                        }
                        None => continue,
                    };
                    self.slots.insert(key, slot);
                    reshaped = true;
                    slot
                }
            };
            frame.ground.patches.push((slot, heights.to_vec()));
        }
        self.version = landscape.version();
        if reshaped {
            self.hand_table(frame);
        }
    }

    fn lay(&mut self, drum: &Drum, frame: &mut DrumFrame) {
        let landscape = &drum.landscape;
        let laid = (landscape.grid(), drum.water.round.cell.div_euclid(PATCH));
        let mut nearest: Vec<(i64, (i64, i64))> = landscape
            .patches()
            .filter_map(|key| {
                let (apart, along) = keyed(laid, key)?;
                Some(((apart as i64).abs() + (along as i64).abs(), key))
            })
            .collect();
        nearest.sort_unstable();
        nearest.truncate(MOST_PATCHES);
        let rows = nearest
            .len()
            .div_ceil(ATLAS_PATCHES)
            .max(1)
            .next_power_of_two();
        *self = GroundLayout {
            laid: Some(laid),
            version: landscape.version(),
            slots: HashMap::default(),
            free: Vec::new(),
            used: nearest.len() as u32,
            capacity: (rows * ATLAS_PATCHES).min(MOST_PATCHES) as u32,
        };
        frame.ground.patches.clear();
        for (slot, (_, key)) in nearest.into_iter().enumerate() {
            if let Some(heights) = landscape.patch(key.0, key.1) {
                self.slots.insert(key, slot as u32);
                frame.ground.patches.push((slot as u32, heights.to_vec()));
            }
        }
        frame.ground.capacity = self.capacity;
        self.hand_table(frame);
    }

    fn hand_table(&self, frame: &mut DrumFrame) {
        let capacity = (2 * self.slots.len()).max(TABLE_WIDTH).next_power_of_two();
        let mut table = vec![[0i32; 4]; capacity];
        for (&key, &slot) in &self.slots {
            let Some((apart, along)) = self.laid.and_then(|laid| keyed(laid, key)) else {
                continue;
            };
            let mut at = table_slot(apart, along) & (capacity - 1);
            while table[at][2] != 0 {
                at = (at + 1) & (capacity - 1);
            }
            table[at] = [apart, along, slot as i32 + 1, 0];
        }
        frame.ground.table = Some(table);
        frame.shape = GroundShape {
            patches: self.slots.len() as u32,
            table_mask: capacity as u32 - 1,
        };
    }
}

/// What the render world is to write of the ground this frame: how many patches the atlas holds;
/// every patch whose heights changed and the square of the atlas it goes in; and the whole table
/// the patches are found by, when that changed.
#[derive(Clone, Default)]
pub struct GroundWrites {
    pub capacity: u32,
    pub patches: Vec<(u32, Vec<f32>)>,
    pub table: Option<Vec<[i32; 4]>>,
}

/// Which patch round the ring a patch is from the one the water's site is in, the short way, and
/// which along the axis, as the table keys it, unless that is too far for it to count.
fn keyed((grid, site): (Grid, i64), (round, along): (i64, i64)) -> Option<(i32, i32)> {
    let patches = grid.patches_round() as i128;
    let apart = (round as i128 - site as i128 + patches / 2).rem_euclid(patches) - patches / 2;
    Some((i32::try_from(apart).ok()?, i32::try_from(along).ok()?))
}

/// Where a patch starts looking in the table: mirrors `patch_slot` in `drum.wgsl`.
fn table_slot(round: i32, along: i32) -> usize {
    let mut h = (round as u32).wrapping_mul(0x9e3779b1) ^ (along as u32).wrapping_mul(0x85ebca77);
    h ^= h >> 15;
    h as usize
}

/// The sculpted patches' heights, each a square of the atlas, along the axis across and round
/// the ring down, and the table they are found by, kept by the render world and written only
/// where they changed.
#[derive(Resource)]
struct GroundTextures {
    atlas: Texture,
    atlas_view: TextureView,
    table: Texture,
    table_view: TextureView,
}

fn init_ground(mut commands: Commands, device: Res<RenderDevice>) {
    let (atlas, atlas_view) = atlas_texture(&device, ATLAS_PATCHES as u32);
    let (table, table_view) = table_texture(&device, TABLE_WIDTH);
    commands.insert_resource(GroundTextures {
        atlas,
        atlas_view,
        table,
        table_view,
    });
}

fn write_ground(
    frame: Res<DrumFrame>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut textures: ResMut<GroundTextures>,
) {
    let writes = &frame.ground;
    let side = PATCH as u32;
    let rows = writes.capacity.div_ceil(ATLAS_PATCHES as u32).max(1);
    if textures.atlas.height() != rows * side {
        (textures.atlas, textures.atlas_view) = atlas_texture(&device, rows * ATLAS_PATCHES as u32);
    }
    for (slot, heights) in &writes.patches {
        let atlas = ATLAS_PATCHES as u32;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &textures.atlas,
                mip_level: 0,
                origin: Origin3d {
                    x: slot % atlas * side,
                    y: slot / atlas * side,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(heights),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
    if let Some(table) = &writes.table {
        let rows = (table.len() / TABLE_WIDTH) as u32;
        if textures.table.height() != rows {
            (textures.table, textures.table_view) = table_texture(&device, table.len());
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &textures.table,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(table),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TABLE_WIDTH as u32 * 16),
                rows_per_image: None,
            },
            Extent3d {
                width: TABLE_WIDTH as u32,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn atlas_texture(device: &RenderDevice, patches: u32) -> (Texture, TextureView) {
    let side = PATCH as u32;
    let atlas = ATLAS_PATCHES as u32;
    texture(
        device,
        "drum ground",
        Extent3d {
            width: atlas * side,
            height: patches.div_ceil(atlas) * side,
            depth_or_array_layers: 1,
        },
        TextureFormat::R32Float,
    )
}

fn table_texture(device: &RenderDevice, entries: usize) -> (Texture, TextureView) {
    texture(
        device,
        "drum ground table",
        Extent3d {
            width: TABLE_WIDTH as u32,
            height: entries.div_ceil(TABLE_WIDTH) as u32,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Sint,
    )
}

fn texture(
    device: &RenderDevice,
    label: &'static str,
    size: Extent3d,
    format: TextureFormat,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

#[derive(Resource)]
struct VesselShader(#[allow(dead_code)] Handle<Shader>);

#[derive(Resource, Default)]
struct DrumUniforms(DynamicUniformBuffer<DrumUniform>);

#[derive(ShaderType, Clone, Copy, Default)]
struct Survey {
    count: u32,
    columns_round: u32,
    per_unit: f32,
    mask: u32,
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
                texture_2d(TextureSampleType::Sint),
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
    ground: Res<GroundTextures>,
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
    let (Some(position), Some(velocity), Some(columns)) = (
        buffers.get(&fluid.position),
        buffers.get(&fluid.velocity),
        buffers.get(&frame.columns),
    ) else {
        return;
    };
    params.0.set(Survey {
        count: fluid_frame.surface.count,
        columns_round: frame.survey.round,
        per_unit: frame.survey.per_unit,
        mask: SURVEY_SLOTS as u32 - 1,
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
            &ground.atlas_view,
            &ground.table_view,
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

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    frame: Res<DrumFrame>,
    layout: Res<VesselLayout>,
    pipeline_cache: Res<PipelineCache>,
    ground: Res<GroundTextures>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut uniforms: ResMut<DrumUniforms>,
) {
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
        &BindGroupEntries::sequential((binding, &ground.atlas_view, &ground.table_view)),
    );
    commands.insert_resource(VesselBinding {
        bind_group,
        offsets,
    });
}

/// The entity whose readback brings back how far the water reaches, and whether one is in
/// flight, asked of which water: water emptied since is not the water it reports.
#[derive(Component)]
struct ReachWatcher {
    awaiting: Option<u32>,
}

fn spawn_reach_watcher(mut commands: Commands) {
    commands
        .spawn(ReachWatcher { awaiting: None })
        .observe(receive_reach);
}

/// Ask how far the water reaches from its site while there is water, one readback at a time.
fn watch_reach(
    mut commands: Commands,
    fluid: Res<Fluid>,
    frame: Res<DrumFrame>,
    watcher: Single<(Entity, &mut ReachWatcher)>,
) {
    let (entity, mut watcher) = watcher.into_inner();
    if watcher.awaiting.is_none() && !fluid.is_empty() {
        commands.entity(entity).insert((
            Readback::buffer_range(frame.columns.clone(), 0, 4),
            ReadOnce,
        ));
        watcher.awaiting = Some(fluid.emptied());
    }
}

fn receive_reach(
    event: On<ReadbackComplete>,
    mut fluid: ResMut<Fluid>,
    mut watchers: Query<&mut ReachWatcher>,
) {
    let Ok(mut watcher) = watchers.get_mut(event.entity) else {
        return;
    };
    let asked = watcher.awaiting.take();
    let bits: Vec<u32> = event.to_shader_type();
    if let Some(reach) = bits.first().map(|b| f32::from_bits(*b))
        && reach.is_finite()
        && asked == Some(fluid.emptied())
    {
        fluid.reached(Metres(reach));
    }
}
