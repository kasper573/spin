//! Camera, lights, the star field, and the viewpoint everything is drawn about. The simulation
//! works in the drum's own frame about a site on its wall, which this moves along under the
//! viewer: everything near the viewer then has small coordinates however far the wheel
//! reaches, and the wheel stands still while the sky turns.
use bevy::asset::RenderAssetUsages;
use bevy::camera::{Exposure, Hdr};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{
    CascadeShadowConfig, CascadeShadowConfigBuilder, DirectionalLightShadowMap, NotShadowCaster,
    NotShadowReceiver, ShadowFilteringMethod, light_consts,
};
use bevy::math::cubic_splines::LinearSpline;
use bevy::pbr::ScreenSpaceTransmission;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::post_process::auto_exposure::{
    AutoExposure, AutoExposureCompensationCurve, AutoExposurePlugin,
};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    TextureDimension, TextureFormat,
};
use bevy::shader::{Shader, ShaderRef};

use crate::core::avatar;
use crate::core::math::Vec3d;
use crate::core::rigid::Body;
use crate::systems::drum::{Ring, slack};
use crate::systems::player::PlayerCamera;
use crate::systems::sim::{SimSet, Simulation};

/// Where the sun is among the stars.
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5145, 0.7717, 0.3430);
pub const SPACE: Color = Color::srgb(0.02, 0.027, 0.05);
/// There is no sky in space: what light falls on the shaded side of anything is the sun's,
/// bounced off the sunlit ground across the ring, dimmed and greened by it.
const BOUNCE: Color = Color::srgb(0.62, 0.7, 0.55);
const BOUNCE_BRIGHTNESS: f32 = 15000.0;
const SUN: Color = Color::srgb(1.0, 0.98, 0.95);
/// The light bounced round the ring as a lit surface shows it, at the exposure the sun is
/// seen at.
pub fn bounce_light() -> Vec3 {
    BOUNCE.to_linear().to_vec3() * BOUNCE_BRIGHTNESS * Exposure::SUNLIGHT.exposure()
}

/// The sun's light as a surface facing it shows it, at that same exposure.
pub fn sunlight() -> Vec3 {
    SUN.to_linear().to_vec3() * light_consts::lux::DIRECT_SUNLIGHT / std::f32::consts::PI
        * Exposure::SUNLIGHT.exposure()
}

/// How much of the light the eye's own lens scatters about what is bright.
const BLOOM: f32 = 0.06;
/// Each cascade of the sun's shadow map is this many texels across.
const SHADOW_MAP: usize = 2048;
/// The projection itself has no far plane; this only bounds what is worth culling.
const FAR: f32 = 1e15;
/// The viewer is never taken to be closer to the wheel's wall than this when choosing how
/// finely to draw it.
const NEAREST: f64 = 1.0;

/// What the scene is drawn about: the avatar's centre in the drum's frame, and how far the
/// viewer was from the wall when the site was last moved.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Viewpoint {
    /// The avatar's centre; the renderer's origin.
    pub origin: Vec3d,
    /// How far the viewer was from the wall when the site last moved.
    pub standoff: f64,
}

impl Default for Viewpoint {
    fn default() -> Self {
        Viewpoint {
            origin: [0.0; 3],
            standoff: avatar::EYE_HEIGHT.0 as f64,
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
        app.add_plugins((
            MaterialPlugin::<StarsMaterial>::default(),
            AutoExposurePlugin,
        ))
        .init_resource::<Viewpoint>()
        .init_resource::<Sky>()
        .insert_resource(ClearColor(SPACE))
        .insert_resource(GlobalAmbientLight {
            color: BOUNCE,
            brightness: BOUNCE_BRIGHTNESS,
            ..default()
        })
        .insert_resource(DirectionalLightShadowMap { size: SHADOW_MAP })
        .add_systems(Startup, (load_shared, spawn))
        .add_systems(Update, locate.before(SimSet::Command))
        .add_systems(Update, (turn_sky, shade).in_set(SimSet::Observe));
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
        sim.resite(p);
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

/// The sun's shadows reach across the ring, up to this far, and are sharpest close by.
const SHADOW_REACH: f32 = 400.0;

/// The sun's shadows reach across the ring, as far as they are worth drawing, and are
/// sharpest close by.
fn shade(
    sim: Res<Simulation>,
    mut shaded: Local<Option<Ring>>,
    mut suns: Query<&mut CascadeShadowConfig, With<SunLight>>,
) {
    let ring = sim.drum.ring;
    if *shaded == Some(ring) {
        return;
    }
    *shaded = Some(ring);
    let (radius, half_width) = (ring.radius.0, ring.half_width.0);
    for mut cascades in &mut suns {
        let reach = (2.0 * radius + 2.0 * half_width).min(SHADOW_REACH);
        *cascades = CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.05,
            maximum_distance: reach,
            first_cascade_far_bound: 5.0 + reach / 40.0,
            overlap_proportion: 0.2,
        }
        .build();
    }
}

/// How the eye adapts: the compensation, in stops, for each average log luminance of the
/// view, such that the sunlit ring is seen as it is, the shaded side is brightened by up to
/// three stops, and glare darkens the view by at most a stop and a half.
const ADAPTATION: [Vec2; 4] = [
    Vec2::new(-8.0, -5.0),
    Vec2::new(-4.75, -1.75),
    Vec2::new(-0.25, -1.75),
    Vec2::new(8.0, 6.5),
];
/// How fast the eye adapts, in stops per second.
const ADAPTATION_SPEED: f32 = 6.0;
const METERING: u32 = 64;

/// The eye adapts to what it looks at: the middle of the view counts most.
fn metering_mask() -> Image {
    let mut data = Vec::with_capacity((METERING * METERING) as usize);
    for y in 0..METERING {
        for x in 0..METERING {
            let u = (x as f32 + 0.5) / METERING as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / METERING as f32 * 2.0 - 1.0;
            let weight = 1.0 - 0.8 * (u * u + v * v).sqrt().min(1.0);
            data.push((weight * 255.0) as u8);
        }
    }
    Image::new(
        Extent3d {
            width: METERING,
            height: METERING,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

#[derive(Component)]
struct StarSphere;

/// The shader modules the materials share, kept loaded: space, for the stars and whatever
/// reflects them; the optics of smooth surfaces, for the water and the glass; and the
/// ripples, for the water and the ground they cast their light on.
#[derive(Resource)]
struct SharedShaders(#[allow(dead_code)] [Handle<Shader>; 3]);

fn load_shared(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(SharedShaders([
        assets.load("embedded://game/systems/shaders/space.wgsl"),
        assets.load("embedded://game/systems/shaders/optics.wgsl"),
        assets.load("embedded://game/systems/shaders/ripples.wgsl"),
    ]));
}

/// A light from a fixed direction among the stars.
#[derive(Component)]
struct SunLight(Vec3);

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    background: LinearRgba,
    #[uniform(0)]
    sun: Vec4,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://game/systems/shaders/stars.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/stars.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "embedded://game/systems/shaders/stars_prepass.wgsl".into()
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
    mut images: ResMut<Assets<Image>>,
    mut curves: ResMut<Assets<AutoExposureCompensationCurve>>,
) {
    let metering = images.add(metering_mask());
    let adaptation = curves.add(
        AutoExposureCompensationCurve::from_curve(LinearSpline::new(ADAPTATION))
            .expect("the adaptation curve is continuous"),
    );
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Projection::Perspective(PerspectiveProjection {
            fov: 60f32.to_radians(),
            near: 0.1,
            far: FAR,
            ..default()
        }),
        Exposure::SUNLIGHT,
        AutoExposure {
            metering_mask: metering.clone(),
            compensation_curve: adaptation.clone(),
            speed_brighten: ADAPTATION_SPEED,
            speed_darken: ADAPTATION_SPEED,
            ..default()
        },
        Tonemapping::AcesFitted,
        Bloom {
            intensity: BLOOM,
            ..Bloom::NATURAL
        },
        ScreenSpaceTransmission {
            steps: 2,
            ..default()
        },
        Msaa::Sample4,
        DepthPrepass,
        ShadowFilteringMethod::Gaussian,
        PlayerCamera,
    ));
    commands.spawn((
        SunLight(SUN_DIRECTION),
        DirectionalLight {
            color: SUN,
            illuminance: light_consts::lux::DIRECT_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-SUN_DIRECTION, Vec3::Y),
    ));
    commands.spawn((
        StarSphere,
        NotShadowCaster,
        NotShadowReceiver,
        Mesh3d(meshes.add(Sphere::new(400.0).mesh().uv(48, 24))),
        MeshMaterial3d(stars.add(StarsMaterial {
            background: SPACE.to_linear(),
            sun: SUN_DIRECTION.extend(0.0),
        })),
    ));
}
