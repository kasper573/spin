//! Camera, lights, the star field, and the viewpoint everything is drawn about. The simulation
//! works in the drum's own frame about a site on its wall, which this moves along under the
//! viewer: everything near the viewer then has small coordinates however far the wheel
//! reaches, and the wheel stands still while the sky turns.
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::core::avatar;
use crate::core::math::Vec3d;
use crate::core::rigid::Body;
use crate::systems::drum::slack;
use crate::systems::player::PlayerCamera;
use crate::systems::sim::{SimSet, Simulation};

/// Where the sun is among the stars.
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5145, 0.7717, 0.3430);
const FILL_DIRECTION: Vec3 = Vec3::new(-0.5976, -0.7171, -0.3586);
const SPACE: Color = Color::srgb(0.02, 0.027, 0.05);
/// The projection itself has no far plane; this only bounds what is worth culling.
const FAR: f32 = 1e15;
/// The viewer is never taken to be closer to the wheel's wall than this when choosing how
/// finely to draw it.
const NEAREST: f64 = 1.0;

/// What the scene is drawn about: the avatar's centre in the drum's frame, how far the viewer
/// was from the wall when the site was last moved, and where the site is in the patterns
/// fixed to the wheel, which the site's moves are added up in exactly so that they never
/// jump.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Viewpoint {
    /// The avatar's centre; the renderer's origin.
    pub origin: Vec3d,
    /// How far the viewer was from the wall when the site last moved.
    pub standoff: f64,
    /// How far spinward round the ring the site has moved since the ring was set, within one
    /// turn of it.
    pub arc: f64,
    /// The site's place along the axis.
    pub axial: f64,
    /// How far the site has turned round the ring since the ring was set, within one turn.
    pub turn: f64,
}

impl Default for Viewpoint {
    fn default() -> Self {
        Viewpoint {
            origin: [0.0; 3],
            standoff: avatar::EYE_HEIGHT.0 as f64,
            arc: 0.0,
            axial: 0.0,
            turn: 0.0,
        }
    }
}

impl Viewpoint {
    /// A point of the drum's frame as the renderer sees it.
    pub fn local(&self, p: Vec3d) -> Vec3 {
        Vec3::new(
            (p[0] - self.origin[0]) as f32,
            (p[1] - self.origin[1]) as f32,
            (p[2] - self.origin[2]) as f32,
        )
    }

    /// Where something at a point of the drum's frame is drawn.
    pub fn place(&self, p: Vec3d) -> Transform {
        Transform::from_translation(self.local(p))
    }

    /// The camera at the hull's eye, looking straight out of it.
    pub fn view(&self, hull: &Body) -> Transform {
        let q = hull.q;
        Transform {
            translation: self.local(avatar::eye(hull)),
            rotation: Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32),
            scale: Vec3::ONE,
        }
    }
}

/// Which way the sky faces: the turn that takes a direction among the stars into the drum's
/// frame, and the sun's direction there.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    pub rotation: Quat,
    pub sun: Vec3,
}

impl Default for Sky {
    fn default() -> Self {
        Sky {
            rotation: Quat::IDENTITY,
            sun: SUN_DIRECTION,
        }
    }
}

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .init_resource::<Viewpoint>()
            .init_resource::<Sky>()
            .insert_resource(ClearColor(SPACE))
            .insert_resource(GlobalAmbientLight {
                color: Color::srgb(0.75, 0.82, 1.0),
                brightness: 700.0,
                ..default()
            })
            .add_systems(Startup, spawn)
            .add_systems(Update, locate.before(SimSet::Command))
            .add_systems(Update, turn_sky.in_set(SimSet::Observe));
    }
}

/// Follow the avatar, and move the site when the viewer has gone far enough from it, come
/// much closer to the wall than the wheel was drawn for, or gone much farther from it.
pub fn locate(mut sim: ResMut<Simulation>, mut viewpoint: ResMut<Viewpoint>) {
    let ring = sim.drum.ring;
    let p = sim.avatar().p;
    let standoff = sim.drum.height_above_glass(p).abs().max(NEAREST);
    let slack = slack(ring, viewpoint.standoff) / 2.0;
    let moved = p[2].abs() > slack || p[1].abs() > slack;
    let closer = standoff < viewpoint.standoff / 2.0;
    let farther = standoff > viewpoint.standoff * 4.0;
    if moved || closer || farther {
        let shift = sim.resite(p);
        let circumference = std::f64::consts::TAU * ring.radius.0 as f64;
        viewpoint.arc = (viewpoint.arc + shift.arc).rem_euclid(circumference);
        viewpoint.turn =
            (viewpoint.turn + shift.arc / ring.radius.0 as f64).rem_euclid(std::f64::consts::TAU);
        viewpoint.axial = sim.drum.site.y;
        viewpoint.standoff = standoff;
    }
    viewpoint.origin = sim.avatar().p;
}

/// The stars and the sun turn the other way from the drum.
fn turn_sky(
    sim: Res<Simulation>,
    mut sky: ResMut<Sky>,
    mut stars: Query<&mut Transform, (With<StarSphere>, Without<SunLight>)>,
    mut suns: Query<(&SunLight, &mut Transform), Without<StarSphere>>,
) {
    let angle = sim.drum.site.phi - sim.drum.angle.0;
    sky.rotation = Quat::from_rotation_y(angle.rem_euclid(std::f64::consts::TAU) as f32);
    sky.sun = sky.rotation * SUN_DIRECTION;
    for mut transform in &mut stars {
        transform.rotation = sky.rotation;
    }
    for (SunLight(direction), mut transform) in &mut suns {
        *transform = Transform::default().looking_to(-(sky.rotation * *direction), Vec3::Y);
    }
}

#[derive(Component)]
struct StarSphere;

/// A light from a fixed direction among the stars.
#[derive(Component)]
struct SunLight(Vec3);

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    background: LinearRgba,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://game/systems/shaders/stars.wgsl".into()
    }

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
            fov: 60f32.to_radians(),
            near: 0.1,
            far: FAR,
            ..default()
        }),
        Tonemapping::None,
        Msaa::Sample4,
        PlayerCamera,
    ));
    commands.spawn((
        SunLight(SUN_DIRECTION),
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 3200.0,
            ..default()
        },
        Transform::default().looking_to(-SUN_DIRECTION, Vec3::Y),
    ));
    commands.spawn((
        SunLight(FILL_DIRECTION),
        DirectionalLight {
            color: Color::srgb(0.42, 0.55, 1.0),
            illuminance: 1100.0,
            ..default()
        },
        Transform::default().looking_to(-FILL_DIRECTION, Vec3::Y),
    ));
    commands.spawn((
        StarSphere,
        Mesh3d(meshes.add(Sphere::new(400.0).mesh().uv(48, 24))),
        MeshMaterial3d(stars.add(StarsMaterial {
            background: SPACE.to_linear(),
        })),
    ));
}
