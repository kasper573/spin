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

use super::landscape::{ROWS, SEGMENTS};
use super::{HALF_WIDTH, RADIUS};
use crate::systems::scene::SUN_DIRECTION;
use crate::systems::sim::{SimSet, Simulation};

pub struct DrumPlugin;

impl Plugin for DrumPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<GlassMaterial>::default())
            .add_systems(Startup, spawn)
            .add_systems(Update, (turn, rebuild_terrain).in_set(SimSet::Observe));
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GlassMaterial {
    #[uniform(0)]
    tint: LinearRgba,
    #[uniform(0)]
    sun: Vec4,
}

impl Material for GlassMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/glass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
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

/// Everything fixed to the glass, so it turns with the drum.
#[derive(Component)]
struct WheelFrame;

#[derive(Component)]
struct Terrain {
    version: Option<u64>,
}

fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    mut standard: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(
            meshes.add(
                Cylinder::new(RADIUS, 2.0 * HALF_WIDTH)
                    .mesh()
                    .resolution(128),
            ),
        ),
        MeshMaterial3d(glass.add(GlassMaterial {
            tint: LinearRgba::new(0.55, 0.75, 0.95, 0.09),
            sun: SUN_DIRECTION.extend(0.0),
        })),
    ));

    let metal = standard.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.17, 0.2),
        metallic: 0.8,
        perceptual_roughness: 0.45,
        ..default()
    });
    let ring = meshes.add(
        Torus::new(RADIUS + 0.02, RADIUS + 0.1)
            .mesh()
            .major_resolution(128),
    );
    let strut = meshes.add(Cuboid::new(0.07, 2.0 * HALF_WIDTH + 0.16, 0.07));
    let terrain_material = standard.add(StandardMaterial {
        perceptual_roughness: 0.95,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands
        .spawn((WheelFrame, Transform::default(), Visibility::default()))
        .with_children(|frame| {
            for side in [-1.0, 1.0] {
                frame.spawn((
                    Mesh3d(ring.clone()),
                    MeshMaterial3d(metal.clone()),
                    Transform::from_xyz(0.0, side * (HALF_WIDTH + 0.04), 0.0),
                ));
            }
            for k in 0..12 {
                let a = k as f32 / 12.0 * std::f32::consts::TAU;
                let r = RADIUS + 0.06;
                frame.spawn((
                    Mesh3d(strut.clone()),
                    MeshMaterial3d(metal.clone()),
                    Transform::from_xyz(r * a.cos(), 0.0, r * a.sin()),
                ));
            }
            frame.spawn((
                Terrain { version: None },
                MeshMaterial3d(terrain_material),
                NoFrustumCulling,
                Transform::default(),
                Visibility::Hidden,
            ));
        });
}

fn turn(sim: Res<Simulation>, mut frames: Query<&mut Transform, With<WheelFrame>>) {
    for mut transform in &mut frames {
        transform.rotation = Quat::from_rotation_y(sim.drum.angle.0 as f32);
    }
}

fn rebuild_terrain(
    mut commands: Commands,
    sim: Res<Simulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrains: Query<(Entity, &mut Terrain, Option<&Mesh3d>, &mut Visibility)>,
) {
    let landscape = &sim.drum.landscape;
    for (entity, mut terrain, mesh, mut visibility) in &mut terrains {
        if terrain.version == Some(landscape.version()) {
            continue;
        }
        terrain.version = Some(landscape.version());
        *visibility = if landscape.is_empty() {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if landscape.is_empty() {
            continue;
        }
        match mesh {
            Some(mesh) => {
                let _ = meshes.insert(mesh.id(), terrain_mesh(landscape));
            }
            None => {
                commands
                    .entity(entity)
                    .insert(Mesh3d(meshes.add(terrain_mesh(landscape))));
            }
        }
    }
}

/// The raised parts of the landscape in the wheel's frame, plus skirts down to the glass along
/// both caps so raised ground reads as solid from the side. Bare glass gets no triangles.
fn terrain_mesh(landscape: &super::Landscape) -> Mesh {
    let dphi = std::f32::consts::TAU / SEGMENTS as f32;
    let dy = 2.0 * HALF_WIDTH / (ROWS as f32 - 1.0);
    let rows = ROWS + 2;
    let mut positions = Vec::with_capacity(SEGMENTS * rows);
    let mut colours = Vec::with_capacity(SEGMENTS * rows);
    for i in 0..SEGMENTS {
        let phi = i as f32 * dphi;
        let (s, c) = phi.sin_cos();
        let mut push = |height: f32, y: f32| {
            let r = RADIUS - height;
            positions.push([r * c, y, r * s]);
            colours.push(ground_colour(height));
        };
        push(0.0, -HALF_WIDTH);
        for j in 0..ROWS {
            push(landscape.height_at(i, j), -HALF_WIDTH + j as f32 * dy);
        }
        push(0.0, HALF_WIDTH);
    }
    let raised = |i: usize, j: usize| {
        let row = j.clamp(1, ROWS) - 1;
        landscape.height_at(i, row) > 0.0
    };
    let mut indices = Vec::with_capacity(SEGMENTS * (rows - 1) * 6);
    for i in 0..SEGMENTS {
        let i1 = (i + 1) % SEGMENTS;
        for j in 0..rows - 1 {
            if !(raised(i, j) || raised(i1, j) || raised(i, j + 1) || raised(i1, j + 1)) {
                continue;
            }
            let a = (i * rows + j) as u32;
            let b = (i1 * rows + j) as u32;
            let c = (i1 * rows + j + 1) as u32;
            let d = (i * rows + j + 1) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

/// Bare dirt at the glass, grass once the ground has risen a little.
fn ground_colour(height: f32) -> [f32; 4] {
    let dirt = LinearRgba::from(Color::srgb(0.45, 0.32, 0.2));
    let grass = LinearRgba::from(Color::srgb(0.32, 0.58, 0.22));
    let t = ((height - 0.05) / 0.45).clamp(0.0, 1.0);
    let t = t * t * (3.0 - 2.0 * t);
    let mix = |a: f32, b: f32| a + (b - a) * t;
    [
        mix(dirt.red, grass.red),
        mix(dirt.green, grass.green),
        mix(dirt.blue, grass.blue),
        1.0,
    ]
}
