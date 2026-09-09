//! Water rendering: the isosurface the GPU extracts is drawn straight from its buffers by a
//! cel-shaded material, through a placeholder mesh whose vertex shader looks the geometry up.
//! The surface comes out in the water's frame, about the drum's centre, relative to an anchor
//! cell near the viewer and in the water's own units, so the mesh is scaled to metres, turned
//! and placed into the bodies' frame about the viewer.
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

use crate::core::fluid::Fluid;
use crate::core::fluid::{FluidBuffers, MAX_INDICES};
use crate::core::vessel::Vessel;
use crate::systems::scene::{self, SUN_DIRECTION, Sky, Viewpoint};
use crate::systems::sim::{SimSet, Simulation};

const SHADER: &str = "embedded://game/systems/shaders/water.wgsl";

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(Startup, spawn)
            .add_systems(
                Update,
                (
                    anchor.after(scene::locate).before(SimSet::Command),
                    tick.in_set(SimSet::Observe),
                ),
            );
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

/// The surface is extracted about the site, so its vertices stay small near the viewer.
fn anchor(sim: Res<Simulation>, mut fluid: ResMut<Fluid>) {
    fluid.set_anchor(sim.drum.to_water([0.0; 3]));
}

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
    fluid: Res<Fluid>,
    viewpoint: Res<Viewpoint>,
    sky: Res<Sky>,
    water: Res<Water>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut meshes: Query<&mut Transform, With<WaterMesh>>,
) {
    if let Some(mut material) = materials.get_mut(&water.0) {
        material.clock = Vec4::new(sim.time.0, 0.0, 0.0, 0.0);
        material.sun = sky.sun.extend(0.0);
    }
    let (origin, metres_per_unit) = fluid.surface_origin();
    let frame = sim.drum.water_frame();
    let rotation = {
        let [x, y, z, w] = frame.rotation;
        Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32).inverse()
    };
    for mut transform in &mut meshes {
        *transform = viewpoint
            .place(sim.drum.from_water(origin))
            .with_rotation(rotation)
            .with_scale(Vec3::splat(metres_per_unit as f32));
    }
}
