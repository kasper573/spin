//! Water rendering: the isosurface the GPU extracts is drawn straight from its buffers through a
//! placeholder mesh whose vertex shader looks the geometry up, and shaded as water is: what is
//! behind it seen through it, bent by refraction and dimmed by the depth of water the light
//! crossed; what is around it mirrored in it, the ring found by marching the reflected ray
//! across the screen and space beyond that; the sun glinting off it; foam where the water
//! churns; and ripples riding on the flow. The droplets the extraction leaves out of the
//! surface are drawn the same way from their own list, each as a sphere of its volume. Both
//! come out in the water's frame and in the water's own units, so the meshes are scaled to
//! metres, turned and placed into the bodies' frame about the viewer.
use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;
use bevy::light::NotShadowCaster;
use bevy::mesh::PrimitiveTopology;
use bevy::pbr::{DistanceFog, FogFalloff, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, BlendState, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::core::fluid::{Fluid, Resolution};
use crate::core::fluid::{
    FluidBuffers, GRID_REACH, MAX_DROPLETS, MAX_INDICES, MAX_MOTES, surface_cell,
};
use crate::core::math::quat_conjugate;
use crate::core::sheet::{Sheet, SheetBuffers};
use crate::core::vessel::Vessel;
use crate::systems::air::{Air, AirUniform};
use crate::systems::drum::bed_albedo;
use crate::systems::figure::{Figure, FigureGathered, FigureUniform};
use crate::systems::player::PlayerCamera;
use crate::systems::portal::{MouthsUniform, Pictures};
use crate::systems::scene::{
    self, SPACE, SeenFrom, SettleVantages, Sky, VANTAGES, Vantage, Vantages,
};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::water_column::{MakeWaterColumns, WaterColumnPlugin, WaterColumns};

const SHADER: &str = "embedded://game/systems/shaders/water.wgsl";
const SPRAY_SHADER: &str = "embedded://game/systems/shaders/spray.wgsl";

/// How much of each colour a metre of water takes out of light crossing it, and how much of
/// that light a metre turns back toward the eye. Absorption is clear water's own: it lets blue
/// through and stops red. The scattering is the carbonate the water carries in suspension,
/// ground off its own bed and near enough white that what comes back out is coloured by the
/// water it crossed rather than by the grains that turned it.
pub const ABSORPTION: Vec3 = Vec3::new(0.38, 0.062, 0.012);
pub const SCATTERING: Vec3 = Vec3::new(0.010, 0.013, 0.016);

/// What a ray of light loses to a metre of water, whether it is swallowed or turned aside.
pub fn extinction() -> Vec3 {
    ABSORPTION + SCATTERING
}

/// Air's refractive index, and water's against it rather than against vacuum, which is what
/// anything seen through water from within it bends by.
pub const AIR_IOR: f32 = 1.000293;
pub const WATER_IOR: f32 = 1.333 / AIR_IOR;

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<WaterMaterial>::default(),
            MaterialPlugin::<SprayMaterial>::default(),
            WaterColumnPlugin,
        ))
        .add_systems(Startup, spawn.after(MakeWaterColumns))
        .add_systems(
            Update,
            (tick.after(SettleVantages), size_lying, submerge).in_set(SimSet::Observe),
        )
        .add_systems(PostUpdate, mirror.after(FigureGathered));
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
    /// x: metres per unit of the water's length; y: a cell of the surface's grid in metres;
    /// z: 1 where the surface is a sheet's, whose vertices say how deep the water stands.
    units: Vec4,
    /// x: simulated seconds; y: metres per second per unit of the water's velocity; z: a
    /// droplet's radius in metres.
    clock: Vec4,
    absorption: Vec4,
    scatter: Vec4,
    /// The air between the eye and the water; see `systems/air.rs`.
    air: AirUniform,
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
    /// The viewer's own figure, for the water to mirror; see `systems/figure.rs`.
    #[uniform(10)]
    figure: FigureUniform,
    /// The portals, for the water to mirror; see `systems/portal`.
    #[uniform(11)]
    mouths: MouthsUniform,
    /// What is seen through each portal, for where it is open.
    #[texture(12)]
    #[sampler(14)]
    through_blue: Option<Handle<Image>>,
    #[texture(13)]
    through_orange: Option<Handle<Image>>,
    /// How much water each line of sight crosses; see `water_column.rs`.
    #[texture(15)]
    columns: Handle<Image>,
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
        // the rim of a parcel of water covers only a share of a pixel, which its samples show
        descriptor.multisample.alpha_to_coverage_enabled = true;
        Ok(())
    }
}

/// The spray, drawn from the motes the solver flies; see `spray.wgsl`. It lets the scene
/// through as the water does, so it is drawn among the water and the glass, in the water's
/// place among them.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct SprayMaterial {
    #[uniform(0)]
    water: WaterUniform,
    order: f32,
    #[storage(1, read_only)]
    motes: Handle<ShaderBuffer>,
    #[uniform(11)]
    mouths: MouthsUniform,
    #[texture(12)]
    #[sampler(14)]
    through_blue: Option<Handle<Image>>,
    #[texture(13)]
    through_orange: Option<Handle<Image>>,
}

impl Material for SprayMaterial {
    fn vertex_shader() -> ShaderRef {
        SPRAY_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SPRAY_SHADER.into()
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
        // a mote hides only a share of what is behind it, and nothing of what is drawn after it
        if let Some(fragment) = descriptor.fragment.as_mut() {
            for target in fragment.targets.iter_mut().flatten() {
                target.blend = Some(BlendState::ALPHA_BLENDING);
            }
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

/// The water's material as each vantage sees it, the material of the sheet of it lying on the
/// floor, and its spray's.
#[derive(Resource)]
struct Water([Handle<WaterMaterial>; VANTAGES]);

#[derive(Resource)]
struct LyingWater([Handle<WaterMaterial>; VANTAGES]);

#[derive(Resource)]
struct Spray([Handle<SprayMaterial>; VANTAGES]);

/// The mesh the water is drawn through, turned with the drum.
#[derive(Component)]
pub struct WaterMesh;

/// The mesh the sheet of water lying on the floor is drawn through, which is measured in
/// metres rather than in the water's units, and how many of the sheet's cells it has the
/// vertices to draw.
#[derive(Component)]
struct LyingWaterMesh {
    cells: u32,
}

/// A mesh that only says how many vertices are drawn: the vertex shader fetches each one from
/// the water's buffers by its number, which the vertex carries as its place along x. The
/// number a shader is told a vertex has is no use for that: meshes are packed many to a
/// buffer, and the count runs on from one to the next.
pub fn numbered_mesh(vertices: usize) -> Mesh {
    assert!(
        vertices < 1 << f32::MANTISSA_DIGITS,
        "a float counts no higher exactly"
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        (0..vertices)
            .map(|n| [n as f32, 0.0, 0.0])
            .collect::<Vec<_>>(),
    )
}

/// Where all that a water mesh draws lies, in the mesh's own frame: the vertices come out of
/// the water's buffers rather than the mesh, so its bounds are told, as far as the grid the
/// water is kept on reaches. Left out of culling altogether instead, a mesh is only ever handed
/// to the cameras that are looking in the frame it first shows, and one that opens later, as
/// a portal's does, never draws it.
pub fn water_bounds() -> (Aabb, NoAutoAabb) {
    let reach = Vec3::splat(GRID_REACH);
    (Aabb::from_min_max(-reach, reach), NoAutoAabb)
}

/// Far enough along the sorting distance to put the water before or after anything else in
/// the scene that lets it through.
const ORDER: f32 = 1.0e6;

fn spawn(
    mut commands: Commands,
    buffers: Res<FluidBuffers>,
    sheet: Res<SheetBuffers>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut sprays: ResMut<Assets<SprayMaterial>>,
    columns: Res<WaterColumns>,
) {
    let water = WaterUniform {
        to_stars: Vec4::new(0.0, 0.0, 0.0, 1.0),
        from_water: Vec4::new(0.0, 0.0, 0.0, 1.0),
        origin: Vec4::ZERO,
        ring: Vec4::ZERO,
        ground: bed_albedo().to_vec4(),
        background: SPACE.to_linear().to_vec4(),
        units: Vec4::new(1.0, 0.0, 0.0, 0.0),
        clock: Vec4::new(0.0, 1.0, 0.0, 0.0),
        absorption: ABSORPTION.extend(0.0),
        scatter: SCATTERING.extend(0.0),
        air: AirUniform::default(),
    };
    let seen = std::array::from_fn(|vantage| {
        materials.add(WaterMaterial {
            water: water.clone(),
            order: ORDER,
            vertices: buffers.surface.polished.clone(),
            indices: buffers.surface.indices.clone(),
            counters: buffers.surface.counters.clone(),
            droplets: buffers.surface.droplets.clone(),
            figure: FigureUniform::default(),
            mouths: MouthsUniform::default(),
            through_blue: None,
            through_orange: None,
            columns: columns.seen_from(vantage),
        })
    });
    // the surface's vertices and then the droplets' squares
    let placeholder = meshes.add(numbered_mesh(MAX_INDICES + MAX_DROPLETS * 6));
    for (seen, material) in (0..VANTAGES).map(SeenFrom).zip(&seen) {
        commands.spawn((
            WaterMesh,
            seen,
            seen.layers(),
            NotShadowCaster,
            Mesh3d(placeholder.clone()),
            MeshMaterial3d(material.clone()),
            water_bounds(),
            Transform::default(),
        ));
    }
    commands.insert_resource(Water(seen));

    // the sheet has no parcels, and its count of them is always none
    let lying = std::array::from_fn(|vantage| {
        materials.add(WaterMaterial {
            water: water.clone(),
            order: ORDER,
            vertices: sheet.vertices.clone(),
            indices: sheet.indices.clone(),
            counters: sheet.counters.clone(),
            droplets: buffers.surface.droplets.clone(),
            figure: FigureUniform::default(),
            mouths: MouthsUniform::default(),
            through_blue: None,
            through_orange: None,
            columns: columns.seen_from(vantage),
        })
    });
    let yet_to_fit = meshes.add(numbered_mesh(3));
    for (seen, material) in (0..VANTAGES).map(SeenFrom).zip(&lying) {
        commands.spawn((
            LyingWaterMesh { cells: 0 },
            Visibility::Hidden,
            seen,
            seen.layers(),
            NotShadowCaster,
            Mesh3d(yet_to_fit.clone()),
            MeshMaterial3d(material.clone()),
            water_bounds(),
            Transform::default(),
        ));
    }
    commands.insert_resource(LyingWater(lying));

    let spray = [(); VANTAGES].map(|()| {
        sprays.add(SprayMaterial {
            water: water.clone(),
            order: ORDER,
            motes: buffers.surface.motes.clone(),
            mouths: MouthsUniform::default(),
            through_blue: None,
            through_orange: None,
        })
    });
    let squares = meshes.add(numbered_mesh(MAX_MOTES * 6));
    for (seen, material) in (0..VANTAGES).map(SeenFrom).zip(&spray) {
        commands.spawn((
            WaterMesh,
            seen,
            seen.layers(),
            NotShadowCaster,
            Mesh3d(squares.clone()),
            MeshMaterial3d(material.clone()),
            water_bounds(),
            Transform::default(),
        ));
    }
    commands.insert_resource(Spray(spray));
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn tick(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    vantages: Res<Vantages>,
    pictures: Res<Pictures>,
    air: Res<Air>,
    (water, lying, spray): (Res<Water>, Res<LyingWater>, Res<Spray>),
    (mut materials, mut sprays): (ResMut<Assets<WaterMaterial>>, ResMut<Assets<SprayMaterial>>),
    mut meshes: Query<(&SeenFrom, &mut Transform), (With<WaterMesh>, Without<LyingWaterMesh>)>,
    mut lying_meshes: Query<(&SeenFrom, &mut Transform), With<LyingWaterMesh>>,
) {
    let resolution = fluid.resolution();
    let metres_per_unit = resolution.length();
    let frame = sim.drum.water_frame();
    let pose = |vantage: &Vantage| {
        vantage.pose(
            sim.drum.from_water([0.0; 3]),
            quat_conjugate(&frame.rotation),
        )
    };
    for (seen, mut transform) in &mut meshes {
        if let Some(vantage) = &vantages.0[seen.0] {
            *transform = pose(vantage).with_scale(Vec3::splat(metres_per_unit as f32));
        }
    }
    for (seen, mut transform) in &mut lying_meshes {
        if let Some(vantage) = &vantages.0[seen.0] {
            *transform = pose(vantage);
        }
    }
    for (k, vantage) in vantages.0.iter().enumerate() {
        let (Some(vantage), Some(mut material)) = (vantage, materials.get_mut(&water.0[k])) else {
            continue;
        };
        let seen = SeenFrom(k);
        let rotation = pose(vantage).rotation;
        material.order = if vantage.enclosed { ORDER } else { -ORDER };
        material.mouths =
            MouthsUniform::of(&sim.drum, vantage, pictures.found_from(seen.0), sim.time);
        [material.through_blue, material.through_orange] = pictures.read_from(seen.0);
        let uniform = &mut material.water;
        uniform.air = air.uniform(sim.drum.spin);
        let stars = vantage.sky.rotation.inverse();
        uniform.to_stars = Vec4::new(stars.x, stars.y, stars.z, stars.w);
        uniform.from_water = Vec4::new(rotation.x, rotation.y, rotation.z, rotation.w);
        let [x, y, z] = vantage.viewpoint.origin;
        uniform.origin = Vec4::new(x as f32, (y + vantage.frame.site.y) as f32, z as f32, 0.0);
        let ring = sim.drum.ring;
        uniform.ring = Vec4::new(
            ring.radius.0,
            ring.half_width.0,
            if vantage.enclosed { 1.0 } else { 0.0 },
            0.0,
        );
        uniform.units = Vec4::new(metres_per_unit as f32, surface_cell(resolution).0, 0.0, 0.0);
        uniform.clock = Vec4::new(
            sim.time.0,
            (metres_per_unit / resolution.time()) as f32,
            droplet_radius(resolution) as f32,
            0.0,
        );
        let shared = WaterMaterial::clone(&material);
        drop(material);
        if let Some(mut material) = materials.get_mut(&lying.0[k]) {
            material.order = shared.order;
            material.mouths = shared.mouths.clone();
            material.through_blue = shared.through_blue.clone();
            material.through_orange = shared.through_orange.clone();
            material.water = WaterUniform {
                units: Vec4::new(1.0, 0.0, 1.0, 0.0),
                clock: Vec4::new(sim.time.0, 1.0, 0.0, 0.0),
                ..shared.water.clone()
            };
        }
        if let Some(mut mist) = sprays.get_mut(&spray.0[k]) {
            let material = &shared;
            mist.water = material.water.clone();
            mist.order = material.order;
            mist.mouths = material.mouths.clone();
            mist.through_blue = material.through_blue.clone();
            mist.through_orange = material.through_orange.clone();
        }
    }
}

/// The sheet is drawn only while there is water on it, through a mesh with no more vertices
/// than its cells' triangles have corners: every vertex of a mesh is run whether the sheet has
/// a triangle for it or not.
fn size_lying(
    sheet: Res<Sheet>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut lying: Query<(&mut LyingWaterMesh, &mut Mesh3d, &mut Visibility)>,
) {
    let cells = sheet.lie().map_or(0, |lie| lie.cells[0] * lie.cells[1]);
    let shown = if sheet.is_empty() {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    let fitting = lying
        .iter()
        .any(|(mesh, ..)| mesh.cells != cells)
        .then(|| meshes.add(numbered_mesh(6 * cells as usize)));
    for (mut mesh, mut drawn, mut visibility) in &mut lying {
        if let Some(fitting) = &fitting {
            meshes.remove(&drawn.0);
            *drawn = Mesh3d(fitting.clone());
            mesh.cells = cells;
        }
        if *visibility != shown {
            *visibility = shown;
        }
    }
}

fn mirror(figure: Res<Figure>, water: Res<Water>, mut materials: ResMut<Assets<WaterMaterial>>) {
    if let Some(mut material) = materials.get_mut(&water.0[0]) {
        material.figure = figure.0.clone();
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
    sky: Res<Sky>,
    ambient: Res<GlobalAmbientLight>,
    mut cameras: Query<(Entity, Option<&mut DistanceFog>), With<PlayerCamera>>,
) {
    let under = sim.submerged();
    // what the water turns back is lit by what comes down to it: the sun, where it stands over
    // the water rather than behind the ring, on top of the light bounced round the ring. Deep
    // water settles at the share of a ray that scattering rather than absorption takes.
    let (_, outward) = sim.drum.depth_and_outward(sim.avatar().p);
    let up = Vec3::new(-outward[0] as f32, -outward[1] as f32, -outward[2] as f32);
    let downwelling = scene::bounce_light(&ambient) + scene::sunlight() * sky.sun.dot(up).max(0.0);
    let glow = SCATTERING / extinction() * downwelling;
    let colour = Color::linear_rgb(glow.x, glow.y, glow.z);
    for (camera, fog) in &mut cameras {
        match (under, fog) {
            // the light coming down turns with the ring, so the water it lights turns with it
            (true, Some(mut fog)) => fog.color = colour,
            (true, None) => {
                commands.entity(camera).insert(DistanceFog {
                    color: colour,
                    falloff: FogFalloff::Atmospheric {
                        extinction: extinction(),
                        inscattering: extinction(),
                    },
                    ..default()
                });
            }
            (false, Some(_)) => {
                commands.entity(camera).remove::<DistanceFog>();
            }
            (false, None) => {}
        }
    }
}
