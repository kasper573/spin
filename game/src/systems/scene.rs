//! Camera, lights and the star field.
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::core::fly_camera::FlyCamera;

pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5145, 0.7717, 0.3430);
const FILL_DIRECTION: Vec3 = Vec3::new(-0.5976, -0.7171, -0.3586);
const SPACE: Color = Color::srgb(0.02, 0.027, 0.05);

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .insert_resource(ClearColor(SPACE))
            .insert_resource(GlobalAmbientLight {
                color: Color::srgb(0.75, 0.82, 1.0),
                brightness: 160.0,
                ..default()
            })
            .add_systems(Startup, spawn);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    background: LinearRgba,
}

impl Material for StarsMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/stars.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stars: ResMut<Assets<StarsMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 42f32.to_radians(),
            near: 0.1,
            far: 600.0,
            ..default()
        }),
        Tonemapping::None,
        Msaa::Sample4,
        Transform::from_xyz(4.9, 5.5, 7.2).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCamera::default(),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 1400.0,
            ..default()
        },
        Transform::default().looking_to(-SUN_DIRECTION, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.42, 0.55, 1.0),
            illuminance: 450.0,
            ..default()
        },
        Transform::default().looking_to(-FILL_DIRECTION, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(400.0).mesh().uv(48, 24))),
        MeshMaterial3d(stars.add(StarsMaterial {
            background: SPACE.to_linear(),
        })),
    ));
}
