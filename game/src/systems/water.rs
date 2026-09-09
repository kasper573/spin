//! Water rendering: the isosurface the GPU extracts is drawn straight from its buffers by a
//! cel-shaded material, through a placeholder mesh whose vertex shader looks the geometry up.
//! The surface comes out in the drum's own frame, so the mesh is turned with the drum and water
//! at rest in it rides round without being re-extracted.
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::PrimitiveTopology;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::core::fluid::{FluidBuffers, MAX_INDICES};
use crate::systems::scene::SUN_DIRECTION;
use crate::systems::sim::{SimSet, Simulation};

const SHADER: &str = "embedded://game/systems/shaders/water.wgsl";

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(Startup, spawn)
            .add_systems(Update, tick.in_set(SimSet::Observe));
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WaterMaterial {
    #[uniform(0)]
    sun: Vec4,
    #[uniform(0)]
    deep: LinearRgba,
    #[uniform(0)]
    shallow: LinearRgba,
    #[uniform(0)]
    clock: Vec4,
    #[storage(1, read_only)]
    vertices: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    indices: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    counters: Handle<ShaderBuffer>,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
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

#[derive(Resource)]
struct Water(Handle<WaterMaterial>);

/// The mesh the surface is drawn through, turned with the drum.
#[derive(Component)]
struct WaterMesh;

fn spawn(
    mut commands: Commands,
    buffers: Res<FluidBuffers>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let material = materials.add(WaterMaterial {
        sun: SUN_DIRECTION.extend(0.0),
        deep: LinearRgba::new(0.02, 0.16, 0.36, 1.0),
        shallow: LinearRgba::new(0.18, 0.58, 0.82, 1.0),
        clock: Vec4::ZERO,
        vertices: buffers.surface.vertices.clone(),
        indices: buffers.surface.indices.clone(),
        counters: buffers.surface.counters.clone(),
    });
    // the mesh only fixes how many vertices are drawn; the vertex shader fetches each one
    let placeholder = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; MAX_INDICES]);
    commands.insert_resource(Water(material.clone()));
    commands.spawn((
        WaterMesh,
        Mesh3d(meshes.add(placeholder)),
        MeshMaterial3d(material),
        NoFrustumCulling,
        Transform::default(),
    ));
}

fn tick(
    sim: Res<Simulation>,
    water: Res<Water>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut meshes: Query<&mut Transform, With<WaterMesh>>,
) {
    if let Some(mut material) = materials.get_mut(&water.0) {
        material.clock = Vec4::new(sim.time.0, 0.0, 0.0, 0.0);
    }
    for mut transform in &mut meshes {
        transform.rotation = Quat::from_rotation_y(sim.drum.angle.0 as f32);
    }
}
