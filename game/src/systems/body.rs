//! The avatar's body as the world sees it: the hull the simulation collides, each of its
//! spheres drawn as it is, and a neck between the bulk and the head so that they read as one
//! body. It is lit, shaded and mirrored like anything else in the ring. The eye sits inside
//! the head and the whole body turns with it, so the viewer never sees it directly: it shows
//! in its shadow, and in the water and the glass.
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::core::avatar;
use crate::systems::figure::Mirrored;
use crate::systems::scene::Viewpoint;
use crate::systems::sim::{SimSet, Simulation};

pub struct BodyPlugin;

impl Plugin for BodyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, pose.in_set(SimSet::Observe));
    }
}

/// The neck runs up the avatar's own upright from the middle of the bulk, and stops this far
/// short of the eye.
const NECK_RADIUS: f32 = 0.07;
const NECK_SHORT_OF_EYE: f32 = 0.2;

const ENAMEL: Color = Color::srgb(0.86, 0.87, 0.88);
const DARK_GLASS: Color = Color::srgb(0.02, 0.02, 0.025);
const GRAPHITE: Color = Color::srgb(0.12, 0.125, 0.135);

#[derive(Component)]
struct AvatarBody;

fn spawn(
    mut commands: Commands,
    sim: Res<Simulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let enamel = materials.add(StandardMaterial {
        base_color: ENAMEL,
        perceptual_roughness: 0.45,
        ..default()
    });
    let dark_glass = materials.add(StandardMaterial {
        base_color: DARK_GLASS,
        perceptual_roughness: 0.08,
        reflectance: 0.6,
        ..default()
    });
    let graphite = materials.add(StandardMaterial {
        base_color: GRAPHITE,
        metallic: 0.85,
        perceptual_roughness: 0.35,
        ..default()
    });

    let body = commands
        .spawn((AvatarBody, Transform::default(), Visibility::default()))
        .id();
    let spheres = &sim.shapes()[sim.avatar().shape].hull.spheres;
    for (k, sphere) in spheres.iter().enumerate() {
        let (material, colour) = if k == 0 {
            (&enamel, ENAMEL)
        } else {
            (&dark_glass, DARK_GLASS)
        };
        let radius = sphere.radius as f32;
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(radius))),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(DVec3::from_array(sphere.centre).as_vec3()),
            Mirrored::matte(Vec3::ZERO, Vec3::ZERO, radius, colour),
            ChildOf(body),
        ));
    }
    let foot = DVec3::from_array(spheres[0].centre).as_vec3();
    let top = DVec3::from_array(avatar::eye_offset()).as_vec3() - Vec3::Y * NECK_SHORT_OF_EYE;
    let length = foot.distance(top);
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(NECK_RADIUS, length))),
        MeshMaterial3d(graphite),
        Transform::from_translation((foot + top) / 2.0)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, (top - foot) / length)),
        Mirrored::matte(
            Vec3::NEG_Y * length / 2.0,
            Vec3::Y * length / 2.0,
            NECK_RADIUS,
            GRAPHITE,
        ),
        ChildOf(body),
    ));
}

fn pose(
    sim: Res<Simulation>,
    viewpoint: Res<Viewpoint>,
    mut bodies: Query<&mut Transform, With<AvatarBody>>,
) {
    let hull = sim.avatar();
    let q = hull.q.map(|c| c as f32);
    for mut body in &mut bodies {
        *body = viewpoint
            .place(hull.p)
            .with_rotation(Quat::from_xyzw(q[0], q[1], q[2], q[3]));
    }
}
