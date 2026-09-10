//! Water rendering: the isosurface the GPU extracts is drawn straight from its buffers through a
//! placeholder mesh whose vertex shader looks the geometry up, and shaded as water is: what is
//! behind it seen through it, bent by refraction and dimmed by the depth of water the light
//! crossed; what is around it mirrored in it, the ring found by marching the reflected ray
//! across the screen and space beyond that; the sun glinting off it; foam where the water
//! churns; and ripples riding on the flow. The surface comes out in the water's frame, about
//! the drum's centre, relative to an anchor cell near the viewer and in the water's own units,
//! so the mesh is scaled to metres, turned and placed into the bodies' frame about the viewer.
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::PrimitiveTopology;
use bevy::pbr::{DistanceFog, FogFalloff, MaterialPipeline, MaterialPipelineKey};
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
/// The hull is this wet when the eye is under water.
const SUBMERGED: f64 = 0.9;

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

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WaterMaterial {
    /// Turns a direction of the drum's frame into one among the stars.
    #[uniform(0)]
    to_stars: Vec4,
    /// Turns a vector of the water's frame into the drum's.
    #[uniform(0)]
    from_water: Vec4,
    /// Where the viewpoint lies in the site's frame, in metres.
    #[uniform(0)]
    origin: Vec4,
    /// The ring's radius and half width, in metres.
    #[uniform(0)]
    ring: Vec4,
    /// The ground's colour as seen from across the ring.
    #[uniform(0)]
    ground: LinearRgba,
    #[uniform(0)]
    background: LinearRgba,
    /// xyz: the anchor cell the vertices are relative to, in the water's units; w: metres per
    /// unit.
    #[uniform(0)]
    anchor: Vec4,
    /// x: simulated seconds; y: metres per second per unit of the water's velocity.
    #[uniform(0)]
    clock: Vec4,
    #[uniform(0)]
    absorption: Vec4,
    #[uniform(0)]
    scatter: Vec4,
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

    fn reads_view_transmission_texture(&self) -> bool {
        true
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
        to_stars: Vec4::new(0.0, 0.0, 0.0, 1.0),
        from_water: Vec4::new(0.0, 0.0, 0.0, 1.0),
        origin: Vec4::ZERO,
        ring: Vec4::ZERO,
        ground: ground_albedo(),
        background: SPACE.to_linear(),
        anchor: Vec4::new(0.0, 0.0, 0.0, 1.0),
        clock: Vec4::new(0.0, 1.0, 0.0, 0.0),
        absorption: ABSORPTION.extend(0.0),
        scatter: SCATTER.extend(SCATTER_PER_METRE),
        vertices: buffers.surface.polished.clone(),
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
        let stars = sky.rotation.inverse();
        material.to_stars = Vec4::new(stars.x, stars.y, stars.z, stars.w);
        material.from_water = Vec4::new(rotation.x, rotation.y, rotation.z, rotation.w);
        let [x, y, z] = viewpoint.origin;
        material.origin = Vec4::new(x as f32, y as f32, z as f32, 0.0);
        let ring = sim.drum.ring;
        material.ring = Vec4::new(ring.radius.0, ring.half_width.0, 0.0, 0.0);
        let anchor = fluid.surface().anchor.as_vec3() * fluid.surface().cell;
        material.anchor = anchor.extend(metres_per_unit as f32);
        material.clock = Vec4::new(
            sim.time.0,
            (metres_per_unit / resolution.time()) as f32,
            0.0,
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

/// With the eye under water, everything seen is seen through water: dimmed and coloured by
/// how far off it is.
fn submerge(
    mut commands: Commands,
    sim: Res<Simulation>,
    cameras: Query<(Entity, Has<DistanceFog>), With<PlayerCamera>>,
) {
    let under = sim.avatar().wet > SUBMERGED;
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
