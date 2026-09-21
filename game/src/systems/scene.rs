//! Camera, lights, the star field, and the viewpoint everything is drawn about. The simulation
//! works in the drum's own frame about a site on its wall, which this moves along under the
//! viewer: everything near the viewer then has small coordinates however far the wheel
//! reaches, and the wheel stands still while the sky turns.
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
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
use crate::core::fluid::Fluid;
use crate::core::math::{Quatd, Vec3d};
use crate::systems::air::{Air, AirUniform};
use crate::systems::drum::{Drum, Ring, Site, SiteFrame, bed_albedo, slack};
use crate::systems::player::PlayerCamera;
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::water;

/// How many of the things that let the scene through, the water and the glass among them, are
/// each drawn over a picture of all that is behind it: every one of them, however many a view
/// has. Drawn fewer pictures than that, they share them by their count alone, and whichever
/// shares one with what is behind it is drawn as if that were not there.
pub const SEEN_THROUGH_LAYERS: usize = 64;

/// Where the sun is among the stars.
pub const SUN_DIRECTION: Vec3 = Vec3::new(0.5145, 0.7717, 0.3430);
pub const SPACE: Color = Color::srgb(0.02, 0.027, 0.05);
/// There is no sky in space: what light falls on the shaded side of anything is the sun's,
/// bounced off the sunlit ground across the ring, dimmed and coloured by whatever covers it.
const BOUNCE: Color = Color::srgb(0.62, 0.7, 0.55);
const BOUNCE_BRIGHTNESS: f32 = 15000.0;
const SUN: Color = Color::srgb(1.0, 0.98, 0.95);
/// The light bounced round the ring as a lit surface shows it, at the exposure the sun is
/// seen at.
pub fn bounce_light(ambient: &GlobalAmbientLight) -> Vec3 {
    ambient.color.to_linear().to_vec3() * ambient.brightness * Exposure::SUNLIGHT.exposure()
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
/// Nothing nearer the eye than this is seen, and the projection itself has no far plane: this
/// only bounds what is worth culling.
pub const NEAR: f32 = 0.1;
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

    /// The camera at an eye turned so, looking straight out of it.
    pub fn view(&self, (eye, q): (Vec3d, Quatd)) -> Transform {
        Transform {
            translation: self.local(eye),
            rotation: Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32),
            scale: Vec3::ONE,
        }
    }
}

/// How many places the ring is seen from at once: the viewer's own, and the far side of each
/// of a pair of portals.
pub const VANTAGES: usize = 3;

/// A place the ring is seen from. Everything drawn for it is drawn in the frame about a site
/// of its own on the wall and about a point of its own in that frame, so that what is near it
/// is drawn exactly however far it is from the viewer, and is seen by its own camera only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vantage {
    pub frame: SiteFrame,
    pub viewpoint: Viewpoint,
    pub sky: Sky,
    /// Whether what looks from here looks from inside the drum, and from under its water.
    pub enclosed: bool,
    pub submerged: bool,
}

impl Vantage {
    /// A vantage about a site, with `origin` a point of the drum's frame.
    pub fn about(drum: &Drum, site: Site, origin: Vec3d, standoff: f64) -> Vantage {
        let frame = drum.frame_at(site);
        let angle = site.phi - drum.angle.0;
        let rotation = Quat::from_rotation_y(angle.rem_euclid(std::f64::consts::TAU) as f32);
        Vantage {
            frame,
            viewpoint: Viewpoint {
                origin: frame.point(origin),
                standoff,
            },
            sky: Sky {
                rotation,
                sun: rotation * SUN_DIRECTION,
            },
            enclosed: drum.encloses(origin),
            submerged: false,
        }
    }

    /// A point of the drum's frame as this vantage's renderer sees it.
    pub fn local(&self, p: Vec3d) -> Vec3 {
        self.viewpoint.local(self.frame.point(p))
    }

    /// The same exactly, and the point of the drum's frame that the renderer sees so.
    pub fn drawn(&self, p: Vec3d) -> Vec3d {
        let (p, o) = (self.frame.point(p), self.viewpoint.origin);
        [p[0] - o[0], p[1] - o[1], p[2] - o[2]]
    }

    pub fn drawn_back(&self, x: Vec3d) -> Vec3d {
        let o = self.viewpoint.origin;
        self.frame
            .point_back([x[0] + o[0], x[1] + o[1], x[2] + o[2]])
    }

    /// Where something at a point of the drum's frame, turned so, is drawn.
    pub fn pose(&self, p: Vec3d, q: Quatd) -> Transform {
        let q = self.frame.attitude(q);
        Transform {
            translation: self.local(p),
            rotation: Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32),
            scale: Vec3::ONE,
        }
    }
}

/// The places the ring is seen from this frame: the viewer's own first, which is always there.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Vantages(pub [Option<Vantage>; VANTAGES]);

/// Which vantage something is drawn for, which is the render layer it is drawn on.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeenFrom(pub usize);

impl SeenFrom {
    pub fn layers(self) -> RenderLayers {
        RenderLayers::layer(self.0)
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

/// The lines drawn for what is seen beyond each mouth, since a group of lines is drawn on one
/// set of layers: the default group is the viewer's own.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct LinesBeyondBlue;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct LinesBeyondOrange;

/// Whatever settles the frame's vantages runs in this set, after everything that reads the
/// stepped simulation may, and what tells the scenery where it is seen from runs after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SettleVantages;

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<StarsMaterial>::default(),
            AutoExposurePlugin,
        ))
        .init_resource::<Viewpoint>()
        .init_resource::<Vantages>()
        .insert_gizmo_config(LinesBeyondBlue, lines_seen_from(SeenFrom(1)))
        .insert_gizmo_config(LinesBeyondOrange, lines_seen_from(SeenFrom(2)))
        .init_resource::<Sky>()
        .init_resource::<Air>()
        .insert_resource(ClearColor(SPACE))
        .insert_resource(GlobalAmbientLight {
            color: BOUNCE,
            brightness: BOUNCE_BRIGHTNESS,
            ..default()
        })
        .insert_resource(DirectionalLightShadowMap { size: SHADOW_MAP })
        .add_systems(Startup, (load_shared, spawn))
        .add_systems(Update, locate.before(SimSet::Command))
        .configure_sets(Update, SettleVantages.in_set(SimSet::Observe))
        .add_systems(
            Update,
            (turn_sky.after(SettleVantages), shade, bounce).in_set(SimSet::Observe),
        );
    }
}

fn lines_seen_from(seen: SeenFrom) -> GizmoConfig {
    GizmoConfig {
        render_layers: seen.layers(),
        ..default()
    }
}

/// What the ring bounces round to itself is the sunlit ground across it, so it takes the
/// colour of whatever covers that ground: the grass where the ring is dry, and where a sea
/// stands over it the share of a ray that the water turns back rather than swallows, which is
/// the colour deep water settles at.
fn bounce(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut last: Local<Option<(u64, f32)>>,
) {
    let litres = fluid.litres().0;
    let asked = (sim.drum.landscape.version(), litres);
    if *last == Some(asked) {
        return;
    }
    *last = Some(asked);
    let flood = sim.drum.landscape.flooded(litres as f64 / 1000.0);
    // what comes back off flooded ground is its bed through the water, which the water has
    // crossed twice, and the water's own colour in place of what it swallowed
    let left = (-2.0 * water::extinction() * flood.depth.0).exp();
    let sea = bed_albedo().to_vec3() * left
        + water::SCATTERING / water::extinction() * (Vec3::ONE - left);
    let colour = BOUNCE.to_linear().to_vec3().lerp(sea, flood.covered);
    ambient.color = Color::linear_rgb(colour.x, colour.y, colour.z);
}

/// Follow the viewer, and move the site when the viewer has gone far enough from it, come
/// much closer to the wall than the wheel was drawn for, or gone much farther from it.
pub fn locate(
    mut sim: ResMut<Simulation>,
    mut viewpoint: ResMut<Viewpoint>,
    mut vantages: ResMut<Vantages>,
) {
    let ring = sim.drum.ring;
    let p = sim.viewer();
    let standoff = sim.drum.height_above_glass(p).abs().max(NEAREST);
    let slack = slack(ring, viewpoint.standoff) / 2.0;
    let moved = p[2].abs() > slack || p[1].abs() > slack;
    let closer = standoff < viewpoint.standoff / 2.0;
    let farther = standoff > viewpoint.standoff * 4.0;
    if moved || closer || farther {
        sim.resite(p);
        viewpoint.standoff = standoff;
    }
    viewpoint.origin = sim.viewer();
    let drum = &sim.drum;
    vantages.0[0] = Some(Vantage {
        enclosed: drum.encloses(sim.avatar().p),
        submerged: sim.submerged(),
        ..Vantage::about(drum, drum.site, viewpoint.origin, viewpoint.standoff)
    });
}

/// The stars and the sun turn the other way from the drum.
#[allow(clippy::type_complexity)]
fn turn_sky(
    sim: Res<Simulation>,
    air: Res<Air>,
    vantages: Res<Vantages>,
    mut sky: ResMut<Sky>,
    mut materials: ResMut<Assets<StarsMaterial>>,
    mut stars: Query<
        (&SeenFrom, &MeshMaterial3d<StarsMaterial>, &mut Transform),
        (With<StarSphere>, Without<SunLight>),
    >,
    mut suns: Query<(&SeenFrom, &SunLight, &mut Transform), Without<StarSphere>>,
) {
    if let Some(own) = vantages.0[0] {
        *sky = own.sky;
    }
    let ring = sim.drum.ring;
    for (seen, material, mut transform) in &mut stars {
        let Some(vantage) = vantages.0[seen.0] else {
            continue;
        };
        transform.rotation = vantage.sky.rotation;
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.air = air.uniform(sim.drum.spin);
            let to_stars = vantage.sky.rotation.inverse();
            material.to_stars = Vec4::new(to_stars.x, to_stars.y, to_stars.z, to_stars.w);
            let [x, y, z] = vantage.viewpoint.origin;
            let along = y + vantage.frame.site.y;
            material.origin = Vec4::new(x as f32, along as f32, z as f32, 0.0);
            material.ring = Vec4::new(ring.radius.0, ring.half_width.0, 0.0, 0.0);
        }
    }
    for (seen, SunLight(direction), mut transform) in &mut suns {
        if let Some(vantage) = vantages.0[seen.0] {
            *transform =
                Transform::default().looking_to(-(vantage.sky.rotation * *direction), Vec3::Y);
        }
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
/// reflects them; the ring as a shape rays are cast against; the optics of smooth surfaces,
/// for the water and the glass; the air between the eye and everything in the ring; the
/// ripples, for the water and the ground they cast their light on; the viewer's own figure,
/// for whatever mirrors it; and the portals, for the surfaces they are let into.
#[derive(Resource)]
struct SharedShaders(#[allow(dead_code)] [Handle<Shader>; 7]);

fn load_shared(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(SharedShaders([
        assets.load("embedded://game/systems/shaders/space.wgsl"),
        assets.load("embedded://game/systems/shaders/ring.wgsl"),
        assets.load("embedded://game/systems/shaders/optics.wgsl"),
        assets.load("embedded://game/systems/shaders/air.wgsl"),
        assets.load("embedded://game/systems/shaders/ripples.wgsl"),
        assets.load("embedded://game/systems/shaders/figure.wgsl"),
        assets.load("embedded://game/systems/shaders/portals.wgsl"),
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
    /// Turns a direction of the ring's frame into one among the stars.
    #[uniform(0)]
    to_stars: Vec4,
    /// Where the viewpoint lies in the ring's frame, in metres, and the ring's radius and half
    /// width: space is seen through whatever of the ring's air stands between.
    #[uniform(0)]
    origin: Vec4,
    #[uniform(0)]
    ring: Vec4,
    /// The air itself; see `systems/air.rs`.
    #[uniform(0)]
    air: AirUniform,
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
            near: NEAR,
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
            steps: SEEN_THROUGH_LAYERS,
            ..default()
        },
        Msaa::Sample4,
        DepthPrepass,
        ShadowFilteringMethod::Gaussian,
        IsDefaultUiCamera,
        PlayerCamera,
    ));
    let sphere = meshes.add(Sphere::new(400.0).mesh().uv(48, 24));
    for seen in (0..VANTAGES).map(SeenFrom) {
        commands.spawn((
            seen,
            seen.layers(),
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
            seen,
            seen.layers(),
            StarSphere,
            NotShadowCaster,
            NotShadowReceiver,
            Mesh3d(sphere.clone()),
            MeshMaterial3d(stars.add(StarsMaterial {
                background: SPACE.to_linear(),
                sun: SUN_DIRECTION.extend(0.0),
                to_stars: Vec4::W,
                origin: Vec4::ZERO,
                ring: Vec4::ZERO,
                air: AirUniform::default(),
            })),
        ));
    }
}
