//! Wooden boards: their shared shape and the meshes that follow the rigid bodies.
use bevy::prelude::*;

use crate::core::rigid::BodyShape;
use crate::core::units::Metres;
use crate::systems::sim::{SimSet, Simulation};

pub const LENGTH: f64 = 0.3;
pub const THICKNESS: f64 = 0.09;
const DENSITY: f64 = 500.0;
const SAMPLE_SPACING: f64 = 0.075;

/// Gap between the surface a raft is placed on and the raft's centre.
pub const PLACEMENT_OFFSET: Metres = Metres((THICKNESS / 2.0) as f32 + 0.04);

pub fn shape() -> BodyShape {
    BodyShape::board([LENGTH, THICKNESS, LENGTH], DENSITY, SAMPLE_SPACING)
}

pub struct RaftsPlugin;

impl Plugin for RaftsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, prepare)
            .add_systems(Update, sync.in_set(SimSet::Observe));
    }
}

#[derive(Resource)]
struct RaftAssets {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

#[derive(Component)]
struct RaftMesh;

fn prepare(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(RaftAssets {
        mesh: meshes.add(Cuboid::new(LENGTH as f32, THICKNESS as f32, LENGTH as f32)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.62, 0.42, 0.22),
            perceptual_roughness: 0.85,
            ..default()
        }),
    });
}

fn sync(
    mut commands: Commands,
    sim: Res<Simulation>,
    assets: Res<RaftAssets>,
    mut rafts: Query<(Entity, &mut Transform), With<RaftMesh>>,
) {
    let mut existing: Vec<(Entity, Mut<Transform>)> = rafts.iter_mut().collect();
    existing.sort_by_key(|(entity, _)| *entity);
    for (body, slot) in sim.rafts().iter().zip(existing.iter_mut()) {
        slot.1.translation = Vec3::new(body.p[0] as f32, body.p[1] as f32, body.p[2] as f32);
        slot.1.rotation = Quat::from_xyzw(
            body.q[0] as f32,
            body.q[1] as f32,
            body.q[2] as f32,
            body.q[3] as f32,
        );
    }
    for (entity, _) in existing.iter().skip(sim.rafts().len()) {
        commands.entity(*entity).despawn();
    }
    for body in sim.rafts().iter().skip(existing.len()) {
        commands.spawn((
            RaftMesh,
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(assets.material.clone()),
            Transform::from_xyz(body.p[0] as f32, body.p[1] as f32, body.p[2] as f32)
                .with_rotation(Quat::from_xyzw(
                    body.q[0] as f32,
                    body.q[1] as f32,
                    body.q[2] as f32,
                    body.q[3] as f32,
                )),
        ));
    }
}
