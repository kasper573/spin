//! Water rendering: the particle cloud becomes an isosurface mesh every frame, drawn with a
//! cel-shaded material whose foam comes from the particles' agitation.
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::core::fluid::PARTICLE_SPACING;
use crate::core::surface::SurfaceGrid;
use crate::systems::drum::{HALF_WIDTH, RADIUS};
use crate::systems::scene::SUN_DIRECTION;
use crate::systems::sim::{SimSet, Simulation};

const SURFACE_CELL: f32 = 0.8 * PARTICLE_SPACING;
const SPLAT_RADIUS: f32 = 1.6 * PARTICLE_SPACING;
const ISO_LEVEL: f32 = 0.9;

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        let pad = SPLAT_RADIUS + SURFACE_CELL;
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .insert_resource(Surface {
                grid: SurfaceGrid::new(
                    [-RADIUS - pad, -HALF_WIDTH - pad, -RADIUS - pad],
                    [RADIUS + pad, HALF_WIDTH + pad, RADIUS + pad],
                    SURFACE_CELL,
                ),
                material: None,
            })
            .add_systems(Startup, spawn)
            .add_systems(Update, rebuild.in_set(SimSet::Observe));
    }
}

#[derive(Resource)]
struct Surface {
    grid: SurfaceGrid,
    material: Option<Handle<WaterMaterial>>,
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
}

impl Material for WaterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/water.wgsl".into()
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

#[derive(Component)]
struct WaterSurface;

fn spawn(
    mut commands: Commands,
    mut surface: ResMut<Surface>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let material = materials.add(WaterMaterial {
        sun: SUN_DIRECTION.extend(0.0),
        deep: LinearRgba::new(0.02, 0.16, 0.36, 1.0),
        shallow: LinearRgba::new(0.18, 0.58, 0.82, 1.0),
        clock: Vec4::ZERO,
    });
    surface.material = Some(material.clone());
    commands.spawn((
        WaterSurface,
        MeshMaterial3d(material),
        NoFrustumCulling,
        Transform::default(),
        Visibility::Hidden,
    ));
}

fn rebuild(
    mut commands: Commands,
    sim: Res<Simulation>,
    mut surface: ResMut<Surface>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut waters: Query<(Entity, Option<&Mesh3d>, &mut Visibility), With<WaterSurface>>,
) {
    let Ok((entity, mesh, mut visibility)) = waters.single_mut() else {
        return;
    };
    if let Some(mut material) = surface
        .material
        .as_ref()
        .and_then(|handle| materials.get_mut(handle))
    {
        material.clock = Vec4::new(sim.time.0, sim.drum.angle.0 as f32, 0.0, 0.0);
    }
    if sim.fluid.is_empty() {
        *visibility = Visibility::Hidden;
        return;
    }
    let grid = &mut surface.grid;
    grid.clear();
    for particle in sim.fluid.particles() {
        grid.splat(particle.position, particle.foam, SPLAT_RADIUS);
    }
    let extracted = grid.extract(ISO_LEVEL);
    if extracted.indices.is_empty() {
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Inherited;
    let mut out = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let colours: Vec<[f32; 4]> = extracted.foam.iter().map(|f| [*f, 0.0, 0.0, 1.0]).collect();
    out.insert_attribute(Mesh::ATTRIBUTE_POSITION, extracted.positions);
    out.insert_attribute(Mesh::ATTRIBUTE_NORMAL, extracted.normals);
    out.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    out.insert_indices(Indices::U32(extracted.indices));
    match mesh {
        Some(mesh) => {
            let _ = meshes.insert(mesh.id(), out);
        }
        None => {
            commands.entity(entity).insert(Mesh3d(meshes.add(out)));
        }
    }
}
