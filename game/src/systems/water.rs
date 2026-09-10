//! Water rendering: the isosurface the GPU extracts is drawn straight from its buffers through a
//! placeholder mesh whose vertex shader looks the geometry up, and shaded as water is: what is
//! behind it seen through it, bent by refraction and dimmed by the depth of water the light
//! crossed; what is around it mirrored in it, the ring found by marching the reflected ray
//! across the screen and space beyond that; the sun glinting off it; foam where the water
//! churns; and ripples riding on the flow. The droplets the extraction leaves out of the
//! surface are drawn the same way from their own list, each as a sphere of its volume. Both
//! come out in the water's frame, about the drum's centre, relative to an anchor cell near the
//! viewer and in the water's own units, so the meshes are scaled to metres, turned and placed
//! into the bodies' frame about the viewer.
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::PrimitiveTopology;
use bevy::pbr::{DistanceFog, FogFalloff, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::core::fluid::{Fluid, Resolution};
use crate::core::fluid::{FluidBuffers, MAX_DROPLETS, MAX_INDICES};
use crate::core::vessel::Vessel;
use crate::systems::drum::ground_albedo;
use crate::systems::player::PlayerCamera;
use crate::systems::scene::{self, SPACE, Sky, Viewpoint};
use crate::systems::sim::{SimSet, Simulation};

const SHADER: &str = "embedded://game/systems/shaders/water.wgsl";

/// How much of each colour a metre of water takes out of light crossing it, and how much of
/// its own colour a metre of water adds by scattering light within it.
pub const ABSORPTION: Vec3 = Vec3::new(0.24, 0.04, 0.012);
pub const SCATTER: Vec3 = Vec3::new(0.012, 0.11, 0.15);
pub const SCATTER_PER_METRE: f32 = 0.14;
/// Water's refractive index, which what is seen through water from within it bends by.
pub const WATER_IOR: f32 = 1.333;

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(Startup, spawn)
            .add_systems(
                Update,
                (
                    anchor.after(scene::locate).before(SimSet::Command),
                    (tick, submerge).in_set(SimSet::Observe),
                ),
            );
    }
}

/// What the water is like and where it is drawn, shared by the surface and the droplets.
#[derive(ShaderType, Clone)]
struct WaterUniform {
    /// Turns a direction of the drum's frame into one among the stars.
    to_stars: Vec4,
    /// Turns a vector of the water's frame into the drum's.
    from_water: Vec4,
    /// Where the viewpoint lies in the ring's frame, in metres: round the ring from the site,
    /// and along the axis from the ring's middle.
    origin: Vec4,
    /// The ring's radius and half width, in metres, and whether the eye is inside the drum,
    /// where what the screen shows can be mirrored.
    ring: Vec4,
    /// The ground's colour as seen from across the ring.
    ground: Vec4,
    background: Vec4,
    /// xyz: the anchor cell the vertices are relative to, in the water's units; w: metres per
    /// unit.
    anchor: Vec4,
    /// x: simulated seconds; y: metres per second per unit of the water's velocity; z: a
    /// droplet's radius in metres.
    clock: Vec4,
    absorption: Vec4,
    scatter: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WaterMaterial {
    #[uniform(0)]
    water: WaterUniform,
    /// Where the water is drawn among what else lets the scene through: the glass and the
    /// water each show what was drawn before them, so what lies beyond the other must be
    /// drawn first: the water when the eye is outside the drum, the glass when it is inside.
    /// Added to the water's sorting distance, which grows toward the eye, so positive puts
    /// it last.
    order: f32,
    #[storage(1, read_only)]
    vertices: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    indices: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    counters: Handle<ShaderBuffer>,
    #[storage(4, read_only)]
    droplets: Handle<ShaderBuffer>,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn depth_bias(&self) -> f32 {
        self.order
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

/// The mesh the water is drawn through, turned with the drum.
#[derive(Component)]
pub struct WaterMesh;

/// Far enough along the sorting distance to put the water before or after anything else in
/// the scene that lets it through.
const ORDER: f32 = 1.0e6;

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
    let water = WaterUniform {
        to_stars: Vec4::new(0.0, 0.0, 0.0, 1.0),
        from_water: Vec4::new(0.0, 0.0, 0.0, 1.0),
        origin: Vec4::ZERO,
        ring: Vec4::ZERO,
        ground: ground_albedo().to_vec4(),
        background: SPACE.to_linear().to_vec4(),
        anchor: Vec4::new(0.0, 0.0, 0.0, 1.0),
        clock: Vec4::new(0.0, 1.0, 0.0, 0.0),
        absorption: ABSORPTION.extend(0.0),
        scatter: SCATTER.extend(SCATTER_PER_METRE),
    };
    let material = materials.add(WaterMaterial {
        water,
        order: ORDER,
        vertices: buffers.surface.polished.clone(),
        indices: buffers.surface.indices.clone(),
        counters: buffers.surface.counters.clone(),
        droplets: buffers.surface.droplets.clone(),
    });
    // the mesh only fixes how many vertices are drawn, the surface's and then the droplets'
    // squares; the vertex shader fetches each one
    let placeholder = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0f32; 3]; MAX_INDICES + MAX_DROPLETS * 6],
    );
    commands.insert_resource(Water(material.clone()));
    commands.spawn((
        WaterMesh,
        NotShadowCaster,
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
    let resolution = fluid.resolution();
    let metres_per_unit = resolution.length();
    let frame = sim.drum.water_frame();
    let rotation = {
        let [x, y, z, w] = frame.rotation;
        Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32).inverse()
    };
    if let Some(mut material) = materials.get_mut(&water.0) {
        let enclosed = sim.drum.encloses(sim.avatar().p);
        material.order = if enclosed { ORDER } else { -ORDER };
        let uniform = &mut material.water;
        let stars = sky.rotation.inverse();
        uniform.to_stars = Vec4::new(stars.x, stars.y, stars.z, stars.w);
        uniform.from_water = Vec4::new(rotation.x, rotation.y, rotation.z, rotation.w);
        let [x, y, z] = viewpoint.origin;
        uniform.origin = Vec4::new(x as f32, (y + sim.drum.site.y) as f32, z as f32, 0.0);
        let ring = sim.drum.ring;
        uniform.ring = Vec4::new(
            ring.radius.0,
            ring.half_width.0,
            if enclosed { 1.0 } else { 0.0 },
            0.0,
        );
        let anchor = fluid.surface().anchor.as_vec3() * fluid.surface().cell;
        uniform.anchor = anchor.extend(metres_per_unit as f32);
        uniform.clock = Vec4::new(
            sim.time.0,
            (metres_per_unit / resolution.time()) as f32,
            droplet_radius(resolution) as f32,
            0.0,
        );
    }
    let (origin, _) = fluid.surface_origin();
    for mut transform in &mut meshes {
        *transform = viewpoint
            .place(sim.drum.from_water(origin))
            .with_rotation(rotation)
            .with_scale(Vec3::splat(metres_per_unit as f32));
    }
}

/// The radius of a drop of one particle's water.
fn droplet_radius(resolution: Resolution) -> f64 {
    let cubic_metres = resolution.litres_per_particle().0 as f64 / 1000.0;
    (cubic_metres * 3.0 / (4.0 * std::f64::consts::PI)).cbrt()
}

/// With the eye under water, everything seen is seen through water: dimmed and coloured by
/// how far off it is.
fn submerge(
    mut commands: Commands,
    sim: Res<Simulation>,
    cameras: Query<(Entity, Has<DistanceFog>), With<PlayerCamera>>,
) {
    let under = sim.submerged();
    // what the water scatters is lit by the light bounced round the ring, as the eye sees it
    let glow = SCATTER * scene::bounce_light();
    for (camera, fogged) in &cameras {
        if under && !fogged {
            commands.entity(camera).insert(DistanceFog {
                color: Color::linear_rgb(glow.x, glow.y, glow.z),
                falloff: FogFalloff::Atmospheric {
                    extinction: ABSORPTION,
                    inscattering: Vec3::splat(SCATTER_PER_METRE),
                },
                ..default()
            });
        } else if !under && fogged {
            commands.entity(camera).remove::<DistanceFog>();
        }
    }
}
