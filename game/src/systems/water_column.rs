//! How much water lies along each line of sight. The water's surface closes round all of the
//! water, so a line of sight that goes into water through a face of it comes out again through
//! another, and the water between them is as long as the second face is further than the first.
//! A camera of its own for every vantage draws the far sides of all the faces it sees, adding
//! how far each is, and the near sides, taking it away: what is left at a pixel is the length
//! of all the water the line of sight through it crosses, which is what the water's shading
//! needs to know to dim and colour what is seen through it. An eye under water is past the
//! face it would have gone in through, and is left with how far the way out is, as it should be.
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, Hdr, ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::light::NotShadowCaster;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, CompareFunction,
    Extent3d, RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureFormat,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;
use bevy::transform::TransformSystems;

use crate::core::fluid::{FluidBuffers, MAX_INDICES};
use crate::systems::player::PlayerCamera;
use crate::systems::scene::{SeenFrom, VANTAGES};
use crate::systems::water::{WaterMesh, numbered_mesh, water_bounds};

const SHADER: &str = "embedded://game/systems/shaders/water_column.wgsl";

/// The picture of how much water each vantage's lines of sight cross; see `water_column.wgsl`
/// for what a pixel of one holds.
#[derive(Resource)]
pub struct WaterColumns([Handle<Image>; VANTAGES]);

impl WaterColumns {
    pub fn seen_from(&self, vantage: usize) -> Handle<Image> {
        self.0[vantage].clone()
    }
}

/// The pictures are made in this set, at startup.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MakeWaterColumns;

pub struct WaterColumnPlugin;

impl Plugin for WaterColumnPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterColumnMaterial>::default())
            .add_systems(Startup, spawn.in_set(MakeWaterColumns))
            .add_systems(
                PostUpdate,
                (size, follow).before(TransformSystems::Propagate),
            );
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WaterColumnMaterial {
    /// The plane nothing short of is seen by the vantage's eye, in the eye's own frame: water
    /// short of it is left out of the count.
    #[uniform(0)]
    seen_past: Vec4,
    #[storage(1, read_only)]
    vertices: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    indices: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    counters: Handle<ShaderBuffer>,
}

impl Material for WaterColumnMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // every face counts, whatever is before it
        descriptor.primitive.cull_mode = None;
        let sum = BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        };
        if let Some(fragment) = descriptor.fragment.as_mut() {
            for target in fragment.targets.iter_mut().flatten() {
                target.blend = Some(BlendState {
                    color: sum,
                    alpha: sum,
                });
            }
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_compare = Some(CompareFunction::Always);
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

/// The camera that counts the water for a vantage, and its material.
#[derive(Component)]
struct WaterColumnEye(Handle<WaterColumnMaterial>);

fn layers(seen: SeenFrom) -> RenderLayers {
    RenderLayers::layer(VANTAGES + seen.0)
}

fn spawn(
    mut commands: Commands,
    buffers: Res<FluidBuffers>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterColumnMaterial>>,
) {
    let faces = meshes.add(numbered_mesh(MAX_INDICES));
    let pictures = [(); VANTAGES].map(|()| { let mut im = Image::new_target_texture(1, 1, TextureFormat::Rgba16Float, None); im.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC; images.add(im) });
    for (seen, picture) in (0..VANTAGES).map(SeenFrom).zip(&pictures) {
        let material = materials.add(WaterColumnMaterial {
            seen_past: Vec4::ZERO,
            vertices: buffers.surface.polished.clone(),
            indices: buffers.surface.indices.clone(),
            counters: buffers.surface.counters.clone(),
        });
        commands.spawn((
            WaterMesh,
            seen,
            layers(seen),
            NotShadowCaster,
            Mesh3d(faces.clone()),
            MeshMaterial3d(material.clone()),
            water_bounds(),
            Transform::default(),
        ));
        commands.spawn((
            WaterColumnEye(material),
            seen,
            layers(seen),
            Camera3d::default(),
            Camera {
                order: seen.0 as isize - 2 - VANTAGES as isize,
                is_active: false,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            RenderTarget::Image(ImageRenderTarget {
                handle: picture.clone(),
                scale_factor: 1.0,
            }),
            Hdr,
            Tonemapping::None,
            DebandDither::Disabled,
            Msaa::Off,
        ));
    }
    commands.insert_resource(WaterColumns(pictures));
}

/// The pictures are as large as the view they are read for.
fn size(
    columns: Res<WaterColumns>,
    mut images: ResMut<Assets<Image>>,
    player: Query<&Camera, With<PlayerCamera>>,
) {
    let Some(wanted) = player.iter().find_map(Camera::physical_target_size) else {
        return;
    };
    for handle in &columns.0 {
        if images
            .get(handle)
            .is_some_and(|image| image.size() != wanted)
            && let Some(mut image) = images.get_mut(handle)
        {
            image.resize(Extent3d {
                width: wanted.x,
                height: wanted.y,
                depth_or_array_layers: 1,
            });
        }
    }
}

/// Each counting camera looks as its vantage's eye does, but cuts nothing off short of the
/// plane that eye sees past: a face short of it is counted as lying in it, so that water the
/// plane cuts through is counted from the plane on.
#[allow(clippy::type_complexity)]
fn follow(
    player: Query<(&Camera, &Transform, &Projection), With<PlayerCamera>>,
    eyes: Query<
        (&SeenFrom, &Camera, &Transform, &Projection),
        (Without<PlayerCamera>, Without<WaterColumnEye>),
    >,
    mut counting: Query<
        (&WaterColumnEye, &SeenFrom, &mut Camera, &mut Transform, &mut Projection),
        Without<PlayerCamera>,
    >,
    mut materials: ResMut<Assets<WaterColumnMaterial>>,
) {
    for (WaterColumnEye(material), seen, mut camera, mut transform, mut projection) in &mut counting
    {
        let eye = match seen.0 {
            0 => player.iter().next(),
            _ => eyes
                .iter()
                .find(|(of, ..)| *of == seen)
                .map(|(_, camera, transform, projection)| (camera, transform, projection)),
        };
        let looking = eye.filter(|(eye, ..)| eye.is_active);
        if camera.is_active != looking.is_some() {
            camera.is_active = looking.is_some();
        }
        let Some((_, pose, Projection::Perspective(lens))) = looking else {
            continue;
        };
        *transform = *pose;
        let plain = PerspectiveProjection {
            near_clip_plane: Vec4::new(0.0, 0.0, -1.0, -lens.near),
            ..lens.clone()
        };
        if !matches!(&*projection, Projection::Perspective(had) if had.near == plain.near
            && had.fov == plain.fov
            && had.aspect_ratio == plain.aspect_ratio
            && had.near_clip_plane == plain.near_clip_plane)
        {
            *projection = Projection::Perspective(plain);
        }
        if let Some(mut material) = materials.get_mut(material)
            && material.seen_past != lens.near_clip_plane
        {
            material.seen_past = lens.near_clip_plane;
        }
    }
}
