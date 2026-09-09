use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use super::landscape::{ROWS, SEGMENTS};
use super::{DrumFrame, DrumUniform, Landscape, Ring};
use crate::systems::scene::SUN_DIRECTION;
use crate::systems::sim::{SimSet, Simulation};

/// Ground never touches the glass; it stops this far short of it.
const GLASS_INSET: f32 = 0.02;
const STRUTS: usize = 24;
/// The ground is tiled in squares about this size, alternately tinted.
const TILE: f32 = 2.0;
/// The glass is made of square panes about this size, gridded together round the ring, with
/// seams this wide between them.
const PANE: f32 = 1.0;
const SEAM: f32 = 0.03;

pub struct DrumPlugin;

impl Plugin for DrumPlugin {
    fn build(&self, app: &mut App) {
        super::gpu::install(app);
        app.add_plugins((
            MaterialPlugin::<GlassMaterial>::default(),
            MaterialPlugin::<TerrainMaterial>::default(),
        ))
        .add_systems(Startup, spawn)
        .add_systems(
            Update,
            (rebuild_structure, turn, rebuild_terrain, feed_water).in_set(SimSet::Observe),
        );
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GlassMaterial {
    #[uniform(0)]
    tint: LinearRgba,
    #[uniform(0)]
    sun: Vec4,
    /// The drum's angle, the pane size round and along, and the seam width.
    #[uniform(0)]
    panes: Vec4,
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

/// The ground's colouring on top of the standard material; see `terrain.wgsl`.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct TerrainExtension {
    /// The drum's angle, the glass radius, and the tile size round the ring and along it.
    #[uniform(100)]
    tiling: Vec4,
    #[uniform(100)]
    dirt: LinearRgba,
    #[uniform(100)]
    grass: LinearRgba,
    #[uniform(100)]
    grass_dark: LinearRgba,
}

impl MaterialExtension for TerrainExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/terrain.wgsl".into()
    }
}

type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// Everything fixed to the glass, so it turns with the drum.
#[derive(Component)]
struct WheelFrame;

/// The glass, its rim rings and struts: built for a ring of one size, rebuilt for another.
#[derive(Component)]
struct Structure(Ring);

#[derive(Component)]
struct Terrain {
    version: Option<u64>,
}

/// The materials the structure is rebuilt with, and the ground's.
#[derive(Resource)]
struct StructureMaterials {
    glass: Handle<GlassMaterial>,
    metal: Handle<StandardMaterial>,
    terrain: Handle<TerrainMaterial>,
}

#[allow(clippy::too_many_arguments)]
fn spawn(
    mut commands: Commands,
    sim: Res<Simulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    mut standard: ResMut<Assets<StandardMaterial>>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
) {
    let ring = sim.drum.ring;
    let materials = StructureMaterials {
        glass: glass.add(GlassMaterial {
            tint: LinearRgba::new(0.55, 0.75, 0.95, 0.09),
            sun: SUN_DIRECTION.extend(0.0),
            panes: pane_layout(ring, 0.0),
        }),
        metal: standard.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.17, 0.2),
            metallic: 0.8,
            perceptual_roughness: 0.45,
            ..default()
        }),
        terrain: terrain.add(ExtendedMaterial {
            base: StandardMaterial {
                perceptual_roughness: 0.95,
                double_sided: true,
                cull_mode: None,
                ..default()
            },
            extension: TerrainExtension {
                tiling: tile_layout(ring, 0.0),
                dirt: Color::srgb(0.45, 0.32, 0.2).into(),
                grass: Color::srgb(0.36, 0.62, 0.24).into(),
                grass_dark: Color::srgb(0.3, 0.54, 0.2).into(),
            },
        }),
    };
    let terrain_material = materials.terrain.clone();
    let frame = commands
        .spawn((WheelFrame, Transform::default(), Visibility::default()))
        .with_children(|frame| {
            frame.spawn((
                Terrain { version: None },
                MeshMaterial3d(terrain_material),
                NoFrustumCulling,
                Transform::default(),
                Visibility::Hidden,
            ));
        })
        .id();
    build_structure(&mut commands, &mut meshes, frame, ring, &materials);
    commands.insert_resource(materials);
}

/// The glass cylinder and, fixed to the wheel, the rim rings and struts, all sized to `ring`.
fn build_structure(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    frame: Entity,
    ring: Ring,
    materials: &StructureMaterials,
) {
    let (radius, half_width) = (ring.radius.0, ring.half_width.0);
    let metal = &materials.metal;
    commands.spawn((
        Structure(ring),
        Mesh3d(
            meshes.add(
                Cylinder::new(radius, 2.0 * half_width)
                    .mesh()
                    .resolution(256),
            ),
        ),
        MeshMaterial3d(materials.glass.clone()),
    ));
    let rim = meshes.add(
        Torus::new(radius + 0.05, radius + 0.35)
            .mesh()
            .major_resolution(256),
    );
    let strut = meshes.add(Cuboid::new(0.25, 2.0 * half_width + 0.6, 0.25));
    commands.entity(frame).with_children(|frame| {
        for side in [-1.0, 1.0] {
            frame.spawn((
                Structure(ring),
                Mesh3d(rim.clone()),
                MeshMaterial3d(metal.clone()),
                Transform::from_xyz(0.0, side * (half_width + 0.15), 0.0),
            ));
        }
        for k in 0..STRUTS {
            let a = k as f32 / STRUTS as f32 * std::f32::consts::TAU;
            let r = radius + 0.2;
            frame.spawn((
                Structure(ring),
                Mesh3d(strut.clone()),
                MeshMaterial3d(metal.clone()),
                Transform::from_xyz(r * a.cos(), 0.0, r * a.sin()),
            ));
        }
    });
}

/// The pane grid for a ring of this size at this angle: a whole number of panes round the glass,
/// so the grid closes on itself.
fn pane_layout(ring: Ring, angle: f32) -> Vec4 {
    let circumference = std::f32::consts::TAU * ring.radius.0;
    let round = circumference / (circumference / PANE).round().max(1.0);
    Vec4::new(angle, round, PANE, SEAM)
}

/// The ground's tiles for a ring of this size at this angle: a whole number of tiles round the
/// ring so the checker closes on itself, and square ones along its axis.
fn tile_layout(ring: Ring, angle: f32) -> Vec4 {
    let circumference = std::f32::consts::TAU * ring.radius.0;
    let round = circumference / (circumference / TILE).round().max(1.0);
    Vec4::new(angle, ring.radius.0, round, TILE)
}

/// Tear down and rebuild the glass and its frame when the ring changes size.
fn rebuild_structure(
    mut commands: Commands,
    sim: Res<Simulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<StructureMaterials>,
    structures: Query<(Entity, &Structure)>,
    frames: Query<Entity, With<WheelFrame>>,
) {
    let ring = sim.drum.ring;
    if structures.iter().all(|(_, built)| built.0 == ring) {
        return;
    }
    for (entity, _) in &structures {
        commands.entity(entity).despawn();
    }
    for frame in &frames {
        build_structure(&mut commands, &mut meshes, frame, ring, &materials);
    }
}

fn turn(
    sim: Res<Simulation>,
    materials: Res<StructureMaterials>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
    mut frames: Query<&mut Transform, With<WheelFrame>>,
) {
    let angle = sim.drum.angle.0 as f32;
    for mut transform in &mut frames {
        transform.rotation = Quat::from_rotation_y(angle);
    }
    if let Some(mut material) = glass.get_mut(&materials.glass) {
        material.panes = pane_layout(sim.drum.ring, angle);
    }
    if let Some(mut material) = terrain.get_mut(&materials.terrain) {
        material.extension.tiling = tile_layout(sim.drum.ring, angle);
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
/// both caps so raised ground reads as solid from the side. Bare glass gets no triangles, and
/// nothing is placed on the glass itself, which would fight it for depth.
fn terrain_mesh(landscape: &Landscape) -> Mesh {
    let ring = landscape.ring();
    let (radius, half_width) = (ring.radius.0, ring.half_width.0);
    let dphi = std::f32::consts::TAU / SEGMENTS as f32;
    let dy = landscape.row_spacing() as f32;
    let rows = ROWS + 2;
    let mut positions = Vec::with_capacity(SEGMENTS * rows);
    for i in 0..SEGMENTS {
        let phi = i as f32 * dphi;
        let (s, c) = phi.sin_cos();
        let mut push = |height: f32, y: f32| {
            let r = radius - height.max(GLASS_INSET);
            let y = y.clamp(-half_width + GLASS_INSET, half_width - GLASS_INSET);
            positions.push([r * c, y, r * s]);
        };
        push(0.0, -half_width);
        for j in 0..ROWS {
            push(landscape.height_at(i, j), -half_width + j as f32 * dy);
        }
        push(0.0, half_width);
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
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

/// Hand the water the drum's state for every substep taken this frame and the landscape when it
/// changed.
fn feed_water(
    sim: Res<Simulation>,
    mut frame: ResMut<DrumFrame>,
    mut images: ResMut<Assets<Image>>,
    mut uploaded: Local<Option<u64>>,
) {
    frame.states.clear();
    for record in &sim.substeps {
        frame
            .states
            .push(DrumUniform::new(&sim.drum, record.spin, record.angle));
    }
    frame
        .states
        .push(DrumUniform::new(&sim.drum, sim.drum.spin, sim.drum.angle));
    let version = sim.drum.landscape.version();
    if *uploaded != Some(version) {
        *uploaded = Some(version);
        super::gpu::upload_heights(&sim.drum, &frame, &mut images);
    }
}
