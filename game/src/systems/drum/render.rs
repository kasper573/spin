//! Drawing the drum: the glass, its rims and struts, and the landscape. The wheel is drawn in
//! the drum's frame about the site, gridded round the ring and along its axis with steps that
//! grow with the distance from the site, so that the curve is never visibly faceted where the
//! viewer is, no triangle is large next to its distance from the viewer, which single precision
//! rasterization could not place, and the whole wheel costs about the same number of triangles
//! at any size. The meshes are rebuilt when the site moves, when the ring changes size, or
//! when the landscape changes; the patterns on the glass and the ground are placed from where
//! the site is in them, which is kept exactly.
use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use super::gpu::{COLUMN_FIXED, Columns, GroundLayout, SURVEY_SLOTS};
use super::landscape::{Landscape, PATCH};
use super::sheet::{SheetWindow, feed_sheet};
use super::{DrumFrame, DrumUniform, GLASS_THICKNESS, PANE, Place, Ring, Round, Site, TILE};
use crate::core::fluid::Fluid;
use crate::core::math::Vec3d;
use crate::core::sheet::{SHEET_CELLS, Sheet, SheetBuffers};
use crate::systems::air::{Air, AirUniform};
use crate::systems::figure::{Figure, FigureGathered, FigureUniform};
use crate::systems::portal::{MouthsUniform, Pictures, SolidCopies, SolidMaterial, solid};
use crate::systems::scene::{SPACE, SeenFrom, SettleVantages, VANTAGES, Vantages};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::water;
use crate::systems::water::WATER_IOR;

/// The two tints the grass is tiled in, and the dirt under it.
const GRASS: Color = Color::srgb(0.36, 0.62, 0.24);
const GRASS_DARK: Color = Color::srgb(0.3, 0.54, 0.2);
const DIRT: Color = Color::srgb(0.45, 0.32, 0.2);
/// The bed under standing water: the carbonate sand that settles out of it, pale enough that
/// what the water leaves of the light bounced off it is the colour the water is seen by.
const BED: Color = Color::srgb(0.68, 0.66, 0.58);

/// The ground's colour as seen from across the ring, where its tiles blur together.
/// What the water lies on, which is what light crossing it falls on and comes back from.
pub fn bed_albedo() -> LinearRgba {
    BED.to_linear()
}

pub fn ground_albedo() -> LinearRgba {
    let (a, b) = (GRASS.to_linear(), GRASS_DARK.to_linear());
    LinearRgba::new(
        (a.red + b.red) / 2.0,
        (a.green + b.green) / 2.0,
        (a.blue + b.blue) / 2.0,
        1.0,
    )
}

/// Ground never touches the glass; it stops this far short of it.
const GLASS_INSET: f64 = 0.02;
/// The ground is this thick where it has been dug away and ends, which keeps its rim clear of
/// the glass under it.
const GROUND_ENDS: f32 = 0.03;
const STRUTS: usize = 24;
/// The grooves bevelled into the glass along its pane grid are this wide.
const BEVEL: f32 = 0.08;
/// A chord's sagitta over its distance from the viewer, as a fraction: a tenth of a pixel or so.
const CHORD: f64 = 0.03;
/// How many chords of the finest size lie on either side of the anchor before they grow.
const FINE_CHORDS: f64 = 25.0;
/// A step of the grid over its distance from the site, at most, and the finest step over the
/// viewer's distance from the wall.
const GROWTH: f64 = 0.5;
/// The same for sculpted ground, down to its own cells: it is sampled this much finer than
/// bare ground, so what was sculpted keeps its shape out to some way off.
const DETAIL: f64 = GROWTH / 4.0;
const RIM_SIDES: usize = 8;
const RIM_RADIUS: f64 = 0.15;
const RIM_OFFSET: f64 = 0.2;
const STRUT_HALF: f64 = 0.125;
/// How far the struts reach past the caps.
const STRUT_OVERHANG: f64 = 0.3;

pub struct DrumPlugin;

impl Plugin for DrumPlugin {
    fn build(&self, app: &mut App) {
        super::gpu::install(app);
        app.add_plugins((
            MaterialPlugin::<GlassMaterial>::default(),
            MaterialPlugin::<TerrainMaterial>::default(),
        ))
        .init_resource::<SheetWindow>()
        .add_systems(Startup, spawn)
        .add_systems(
            Update,
            (rebuild, place, light, wet, feed_water, feed_sheet)
                .chain()
                .in_set(SimSet::Observe)
                .after(SettleVantages),
        )
        .add_systems(PostUpdate, mirror.after(FigureGathered));
    }
}

/// The chord the wheel is drawn with this far along its arc from the site, for a viewer this
/// far from its wall: the finest chord out to a slack around the site, then growing as the
/// square root of the distance, so the sagitta stays a fixed fraction of the distance and the
/// chords round the whole ring stay a fixed number whatever its size.
pub fn chord(ring: Ring, standoff: f64, arc: f64) -> f64 {
    let radius = ring.radius.0 as f64;
    let slack = FINE_CHORDS * CHORD * (radius * standoff).sqrt();
    CHORD * (radius * standoff.max(arc - slack)).sqrt()
}

/// How far the viewer may move from the site before the wheel is drawn about a new one.
pub fn slack(ring: Ring, standoff: f64) -> f64 {
    FINE_CHORDS * chord(ring, standoff, 0.0)
}

/// The glass panes: what they let through, and what they mirror; see `glass.wgsl`.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GlassMaterial {
    /// How much of each colour a pane lets through.
    #[uniform(0)]
    tint: LinearRgba,
    /// Turns a direction of the drum's frame into one among the stars.
    #[uniform(0)]
    to_stars: Vec4,
    #[uniform(0)]
    background: LinearRgba,
    /// The pane size round the wall and along it, the groove width and the glass thickness.
    #[uniform(0)]
    panes: Vec4,
    /// Where the viewpoint lies in the ring's frame, in metres: round the ring from the site,
    /// and along the axis from the ring's middle.
    #[uniform(0)]
    origin: Vec4,
    /// The ring's radius and half width, in metres; whether the eye is inside the drum,
    /// where what the screen shows can be mirrored; and the refractive index of what the
    /// eye is in, air or water.
    #[uniform(0)]
    ring: Vec4,
    /// The ground's colour as seen from across the ring.
    #[uniform(0)]
    ground: LinearRgba,
    /// The air between the eye and the glass; see `systems/air.rs`.
    #[uniform(0)]
    air: AirUniform,
    /// The viewer's own figure, for the glass to mirror; see `systems/figure.rs`.
    #[uniform(10)]
    figure: FigureUniform,
    /// The portals let into the caps; see `systems/portal`.
    #[uniform(11)]
    mouths: MouthsUniform,
    /// What is seen through each portal, for where it is open.
    #[texture(12)]
    #[sampler(14)]
    through_blue: Option<Handle<Image>>,
    #[texture(13)]
    through_orange: Option<Handle<Image>>,
}

impl Material for GlassMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/glass.wgsl".into()
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

/// The ground, lit by the sun and by what comes down to it through the water; see
/// `terrain.wgsl`.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct TerrainMaterial {
    #[uniform(0)]
    dirt: LinearRgba,
    #[uniform(0)]
    grass: LinearRgba,
    #[uniform(0)]
    grass_dark: LinearRgba,
    #[uniform(0)]
    bed: LinearRgba,
    /// The site everything is drawn about, from the water's: how far round the ring and along
    /// the axis, in metres; the glass radius; and the mask of the survey's table.
    #[uniform(0)]
    site: Vec4,
    /// Where the point everything is drawn about lies in the site's frame, in metres.
    #[uniform(0)]
    origin: Vec4,
    /// A surveyed column's arc round the ring and width along the axis, and the drum's half
    /// width, in metres, and the depth of water each particle surveyed over a column adds.
    #[uniform(0)]
    grid: Vec4,
    /// x: seconds; y: metres per unit of a surveyed column's height; z: metres per second per
    /// unit of its flow; w: how many columns there are round the ring, or 0 when there are too
    /// many for the water to reach round it.
    #[uniform(0)]
    clock: Vec4,
    #[uniform(0)]
    absorption: Vec4,
    #[uniform(0)]
    scatter: Vec4,
    /// The air between the eye and the ground; see `systems/air.rs`.
    #[uniform(0)]
    air: AirUniform,
    /// The sheet of water lying on the floor: where its first corner lies from the water's site,
    /// round the ring and along the axis, and a cell's arc and width, in metres; then how many
    /// cells it has each way, how many corners a row of its vertices holds, and whether its
    /// cells close on themselves round the ring.
    #[uniform(0)]
    lying: Vec4,
    #[uniform(0)]
    lying_cells: Vec4,
    #[storage(1, read_only)]
    columns: Handle<ShaderBuffer>,
    /// The corners of the sheet's surface, which say how deep it lies at each.
    #[storage(2, read_only)]
    lying_corners: Handle<ShaderBuffer>,
    /// The portals let into the ground; see `systems/portal`.
    #[uniform(11)]
    mouths: MouthsUniform,
    /// What is seen through each portal, for where it is open.
    #[texture(12)]
    #[sampler(14)]
    through_blue: Option<Handle<Image>>,
    #[texture(13)]
    through_orange: Option<Handle<Image>>,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/terrain.wgsl".into()
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

/// The wheel as it is drawn about a site of its wall, which is the viewer's own or, through a
/// portal, the one a far mouth stands at: its meshes are built in the frame about that site.
struct Wheel<'a> {
    ring: Ring,
    site: Site,
    landscape: &'a Landscape,
}

impl Wheel<'_> {
    fn wall_point(&self, turn: f64, axial: f64) -> Vec3d {
        self.ring.wall_point(turn, axial)
    }

    fn depth_and_outward(&self, p: Vec3d) -> (f64, Vec3d) {
        self.ring.depth_and_outward(p)
    }
}

/// Something fixed to the wheel: where it sits in the frame of the vantage it is seen from.
#[derive(Component)]
struct Placed(Vec3d);

/// The glass, rims and struts, built for one size of ring about one site.
#[derive(Component)]
struct Structure;

#[derive(Component)]
struct Terrain;

/// What the wheel's meshes were last built for, for each vantage.
#[derive(Resource, Default)]
struct Built([BuiltFor; VANTAGES]);

#[derive(Default)]
struct BuiltFor {
    structure: Option<(Ring, Site, f64)>,
    terrain: Option<(Ring, u64, Site, f64)>,
}

/// The materials the wheel is built with: the metal is the same from wherever it is seen, the
/// glass and the ground are told where they are seen from.
#[derive(Resource)]
struct WheelMaterials {
    metal: [Handle<SolidMaterial>; VANTAGES],
    seen: [SeenMaterials; VANTAGES],
}

struct SeenMaterials {
    glass: Handle<GlassMaterial>,
    terrain: Handle<TerrainMaterial>,
}

fn spawn(
    mut commands: Commands,
    frame: Res<DrumFrame>,
    sheet: Res<SheetBuffers>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    mut standard: ResMut<Assets<SolidMaterial>>,
    mut copies: ResMut<SolidCopies>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
) {
    let seen = [(); VANTAGES].map(|()| SeenMaterials {
        glass: glass.add(GlassMaterial {
            tint: LinearRgba::new(0.9, 0.96, 0.98, 1.0),
            to_stars: Vec4::new(0.0, 0.0, 0.0, 1.0),
            background: SPACE.to_linear(),
            panes: Vec4::ZERO,
            origin: Vec4::ZERO,
            ring: Vec4::new(0.0, 0.0, 1.0, 1.0),
            ground: ground_albedo(),
            air: AirUniform::default(),
            figure: FigureUniform::default(),
            mouths: MouthsUniform::default(),
            through_blue: None,
            through_orange: None,
        }),
        terrain: terrain.add(TerrainMaterial {
            dirt: DIRT.into(),
            grass: GRASS.into(),
            grass_dark: GRASS_DARK.into(),
            bed: BED.into(),
            site: Vec4::ZERO,
            origin: Vec4::ZERO,
            grid: Vec4::ONE,
            clock: Vec4::ZERO,
            absorption: water::ABSORPTION.extend(0.0),
            scatter: water::SCATTERING.extend(0.0),
            air: AirUniform::default(),
            lying: Vec4::ZERO,
            lying_cells: Vec4::ZERO,
            columns: frame.columns.clone(),
            lying_corners: sheet.vertices.clone(),
            mouths: MouthsUniform::default(),
            through_blue: None,
            through_orange: None,
        }),
    });
    let metal = standard.add(solid(StandardMaterial {
        base_color: Color::srgb(0.16, 0.17, 0.2),
        metallic: 0.8,
        perceptual_roughness: 0.45,
        ..default()
    }));
    commands.insert_resource(WheelMaterials {
        metal: std::array::from_fn(|k| copies.seen_from(&metal, k, &mut standard)),
        seen,
    });
    commands.init_resource::<Built>();
}

/// Rebuild whatever the ring, a vantage's site or the landscape has outdated, and take down
/// what was built for a vantage that nothing is seen from any more.
#[allow(clippy::too_many_arguments)]
fn rebuild(
    mut commands: Commands,
    sim: Res<Simulation>,
    vantages: Res<Vantages>,
    materials: Res<WheelMaterials>,
    mut built: ResMut<Built>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    structures: Query<(Entity, &SeenFrom), With<Structure>>,
    terrains: Query<(Entity, &SeenFrom), With<Terrain>>,
) {
    let (ring, landscape) = (sim.drum.ring, &sim.drum.landscape);
    for (k, vantage) in vantages.0.iter().enumerate() {
        let seen = SeenFrom(k);
        let about = vantage.map(|v| (v.frame.site, v.viewpoint.standoff));
        let structure = about.map(|(site, standoff)| (ring, site, standoff));
        if built.0[k].structure != structure {
            built.0[k].structure = structure;
            for (entity, _) in structures.iter().filter(|(_, of)| **of == seen) {
                commands.entity(entity).despawn();
            }
            if let Some((site, standoff)) = about {
                let wheel = Wheel {
                    ring,
                    site,
                    landscape,
                };
                let columns = columns(ring, standoff);
                let spans = spans(&wheel, standoff);
                if let Some(mut material) = glass.get_mut(&materials.seen[k].glass) {
                    material.panes = Vec4::new(
                        pane_round(ring) as f32,
                        PANE as f32,
                        BEVEL,
                        GLASS_THICKNESS as f32,
                    );
                }
                let mut structure = |mesh: Mesh| {
                    (
                        Structure,
                        seen,
                        seen.layers(),
                        Placed([0.0; 3]),
                        Mesh3d(meshes.add(mesh)),
                        NoFrustumCulling,
                        Transform::default(),
                    )
                };
                let glass_wall = glass_mesh(&wheel, &columns, &spans, &depths(ring, standoff));
                commands.spawn((
                    structure(glass_wall),
                    MeshMaterial3d(materials.seen[k].glass.clone()),
                    NotShadowCaster,
                ));
                for side in [-1.0, 1.0] {
                    let rim = rim_mesh(&wheel, &columns, side);
                    commands.spawn((structure(rim), MeshMaterial3d(materials.metal[k].clone())));
                }
                let struts = strut_mesh(&wheel, &spans);
                commands.spawn((
                    structure(struts),
                    MeshMaterial3d(materials.metal[k].clone()),
                ));
            }
        }
        let version = landscape.version();
        let terrain = about.map(|(site, standoff)| (ring, version, site, standoff));
        if built.0[k].terrain != terrain {
            built.0[k].terrain = terrain;
            for (entity, _) in terrains.iter().filter(|(_, of)| **of == seen) {
                commands.entity(entity).despawn();
            }
            if let Some((site, standoff)) = about.filter(|_| !landscape.is_empty()) {
                let wheel = Wheel {
                    ring,
                    site,
                    landscape,
                };
                let columns = ground_columns(&wheel, standoff);
                let rows = ground_rows(&wheel, standoff);
                commands.spawn((
                    Terrain,
                    seen,
                    seen.layers(),
                    Placed([0.0; 3]),
                    Mesh3d(meshes.add(terrain_mesh(&wheel, &columns, &rows))),
                    MeshMaterial3d(materials.seen[k].terrain.clone()),
                    NoFrustumCulling,
                    Transform::default(),
                ));
            }
        }
    }
}

/// Everything fixed to the wheel is placed about the viewpoint of the vantage it is seen from.
fn place(vantages: Res<Vantages>, mut placed: Query<(&Placed, &SeenFrom, &mut Transform)>) {
    for (Placed(at), seen, mut transform) in &mut placed {
        if let Some(vantage) = vantages.0[seen.0] {
            *transform = vantage.viewpoint.place(*at);
        }
    }
}

/// The glass mirrors space wherever the sky has turned it.
fn light(
    sim: Res<Simulation>,
    vantages: Res<Vantages>,
    pictures: Res<Pictures>,
    air: Res<Air>,
    materials: Res<WheelMaterials>,
    mut glass: ResMut<Assets<GlassMaterial>>,
) {
    for (k, (vantage, seen)) in vantages.0.iter().zip(&materials.seen).enumerate() {
        let (Some(vantage), Some(mut material)) = (vantage, glass.get_mut(&seen.glass)) else {
            continue;
        };
        material.air = air.uniform(sim.drum.spin);
        material.mouths = MouthsUniform::of(&sim.drum, vantage, pictures.found_from(k), sim.time);
        [material.through_blue, material.through_orange] = pictures.read_from(k);
        let stars = vantage.sky.rotation.inverse();
        material.to_stars = Vec4::new(stars.x, stars.y, stars.z, stars.w);
        let [x, y, z] = vantage.viewpoint.origin;
        material.origin = Vec4::new(x as f32, (y + vantage.frame.site.y) as f32, z as f32, 0.0);
        let ring = sim.drum.ring;
        let medium = if vantage.submerged { WATER_IOR } else { 1.0 };
        material.ring = Vec4::new(
            ring.radius.0,
            ring.half_width.0,
            if vantage.enclosed { 1.0 } else { 0.0 },
            medium,
        );
    }
}

fn mirror(
    figure: Res<Figure>,
    materials: Res<WheelMaterials>,
    mut glass: ResMut<Assets<GlassMaterial>>,
) {
    if let Some(mut material) = glass.get_mut(&materials.seen[0].glass) {
        material.figure = figure.0.clone();
    }
}

/// The ground is told where each vantage's site lies on the ring and where its viewpoint lies
/// about the site, so it can look up the water surveyed over each of its points.
#[allow(clippy::too_many_arguments)]
fn wet(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    sheet: Res<Sheet>,
    vantages: Res<Vantages>,
    pictures: Res<Pictures>,
    air: Res<Air>,
    materials: Res<WheelMaterials>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
) {
    let drum = &sim.drum;
    let resolution = fluid.resolution();
    let columns = Columns::of(drum.ring, resolution);
    let (across, along) = (columns.arc(resolution), resolution.length());
    let particle = resolution.length().powi(3);
    for (k, (vantage, seen)) in vantages.0.iter().zip(&materials.seen).enumerate() {
        let (Some(vantage), Some(mut material)) = (vantage, terrain.get_mut(&seen.terrain)) else {
            continue;
        };
        material.air = air.uniform(drum.spin);
        material.mouths = MouthsUniform::of(drum, vantage, pictures.found_from(k), sim.time);
        [material.through_blue, material.through_orange] = pictures.read_from(k);
        let site = vantage.frame.site;
        let arc = drum.water.round.arc_to(site.round, drum.landscape.grid());
        material.site = Vec4::new(
            arc as f32,
            (site.y - drum.water.y) as f32,
            drum.ring.radius.0,
            (SURVEY_SLOTS - 1) as f32,
        );
        let [x, y, z] = vantage.viewpoint.origin;
        material.origin = Vec4::new(x as f32, y as f32, z as f32, 0.0);
        material.lying_cells = Vec4::ZERO;
        if let Some(lie) = sheet.lie().filter(|_| !sheet.is_empty()) {
            let [round, along] = sheet.origin();
            material.lying = Vec4::new(round.0, along.0, lie.cell[0].0, lie.cell[1].0);
            material.lying_cells = Vec4::new(
                lie.cells[0] as f32,
                lie.cells[1] as f32,
                (SHEET_CELLS[0] + 1) as f32,
                f32::from(u8::from(lie.closed)),
            );
        }
        material.grid = Vec4::new(
            across as f32,
            along as f32,
            drum.ring.half_width.0,
            (particle / (across * along)) as f32,
        );
        material.clock = Vec4::new(
            sim.time.0,
            (resolution.length() / COLUMN_FIXED) as f32,
            (resolution.length() / resolution.time() / COLUMN_FIXED) as f32,
            columns.round as f32,
        );
    }
}

/// The turns round the ring the meshes are sampled at, from the site round to it again: the
/// last column repeats the first a full turn on, so the seams and tiles wrap without a jump.
fn columns(ring: Ring, standoff: f64) -> Vec<f64> {
    let radius = ring.radius.0 as f64;
    let half = std::f64::consts::PI * radius;
    let arcs = samples(standoff, half, |arc| chord(ring, standoff, arc));
    let mut turns: Vec<f64> = arcs.iter().rev().map(|a| -a / radius).collect();
    turns.extend(arcs.iter().skip(1).map(|a| a / radius));
    turns
}

/// The places along the axis, from the site, the meshes are sampled at, cap to cap.
fn spans(drum: &Wheel, standoff: f64) -> Vec<f64> {
    let half_width = drum.ring.half_width.0 as f64;
    let flat = |_: f64| f64::INFINITY;
    let below = samples(standoff, half_width + drum.site.y, flat);
    let above = samples(standoff, half_width - drum.site.y, flat);
    let mut rows: Vec<f64> = below.iter().rev().map(|d| -d).collect();
    rows.extend(above.iter().skip(1));
    rows
}

/// The depths in from the glass a cap is sampled at, to the axis.
fn depths(ring: Ring, standoff: f64) -> Vec<f64> {
    samples(standoff, ring.radius.0 as f64, |_| f64::INFINITY)
}

/// Distances from the site out to `reach` in steps that start at a fraction of the viewer's
/// distance from the wall and grow with the distance, within whatever `limit` allows at each.
fn samples(standoff: f64, reach: f64, limit: impl Fn(f64) -> f64) -> Vec<f64> {
    let fine = GROWTH * standoff;
    let mut out = vec![0.0];
    let mut d = 0.0;
    while d < reach {
        d = (d + fine.max(GROWTH * d).min(limit(d))).min(reach);
        out.push(d);
    }
    out
}

/// The turns round the ring the ground is sampled at: as the glass is, and finer over sculpted
/// ground.
fn ground_columns(drum: &Wheel, standoff: f64) -> Vec<f64> {
    let ring = drum.ring;
    let radius = ring.radius.0 as f64;
    let grid = drum.landscape.grid();
    let reach = PATCH as f64 * grid.arc;
    let starts = drum.landscape.patches().map(|(round, _)| {
        let start = Round {
            cell: round * PATCH,
            across: 0.0,
        };
        drum.site.round.arc_to(start, grid)
    });
    let Sculpted { behind, ahead } = sculpted(starts, reach);
    let half = std::f64::consts::PI * radius;
    let side = |spans: &[(f64, f64)]| {
        samples(standoff, half, |arc| {
            chord(ring, standoff, arc).min(finer(spans, arc, grid.arc))
        })
    };
    let mut turns: Vec<f64> = side(&behind).iter().rev().map(|a| -a / radius).collect();
    turns.extend(side(&ahead).iter().skip(1).map(|a| a / radius));
    turns
}

/// The places along the axis, from the site, the ground is sampled at, cap to cap: as the glass
/// is, and finer over sculpted ground.
fn ground_rows(drum: &Wheel, standoff: f64) -> Vec<f64> {
    let half_width = drum.ring.half_width.0 as f64;
    let grid = drum.landscape.grid();
    let reach = PATCH as f64 * grid.along;
    let starts = drum
        .landscape
        .patches()
        .map(|(_, along)| (along * PATCH) as f64 * grid.along - drum.site.y);
    let Sculpted {
        behind: below,
        ahead: above,
    } = sculpted(starts, reach);
    let side =
        |room: f64, spans: &[(f64, f64)]| samples(standoff, room, |d| finer(spans, d, grid.along));
    let mut rows: Vec<f64> = side(half_width + drum.site.y, &below)
        .iter()
        .rev()
        .map(|d| -d)
        .collect();
    rows.extend(side(half_width - drum.site.y, &above).iter().skip(1));
    rows
}

/// Stretches of sculpted ground on either side of the site, as distances from it where each
/// starts and ends.
struct Sculpted {
    behind: Vec<(f64, f64)>,
    ahead: Vec<(f64, f64)>,
}

/// Stretches of sculpted ground, each `reach` long from where it starts.
fn sculpted(starts: impl Iterator<Item = f64>, reach: f64) -> Sculpted {
    let mut starts: Vec<f64> = starts.collect();
    starts.sort_by(f64::total_cmp);
    starts.dedup();
    let (mut behind, mut ahead) = (Vec::new(), Vec::new());
    for start in starts {
        let end = start + reach;
        if end > 0.0 {
            ahead.push((start.max(0.0), end));
        }
        if start < 0.0 {
            behind.push(((-end).max(0.0), -start));
        }
    }
    Sculpted { behind, ahead }
}

/// The longest step from `d` that neither misses the start of a stretch of sculpted ground nor
/// crosses one faster than its cells allow, which are sampled down to one `cell` apart close by
/// and a share of their distance farther off.
fn finer(spans: &[(f64, f64)], d: f64, cell: f64) -> f64 {
    let mut limit = f64::INFINITY;
    for &(start, end) in spans {
        if d + cell >= start && d <= end + cell {
            limit = limit.min(cell.max(DETAIL * d));
        } else if start > d {
            limit = limit.min(start - d);
        }
    }
    limit
}

/// A whole number of panes round the glass, so the grid closes on itself.
fn pane_round(ring: Ring) -> f64 {
    let circumference = std::f64::consts::TAU * ring.radius.0 as f64;
    circumference / (circumference / PANE).round().max(1.0)
}

/// A whole number of tiles round the ring, so the checker closes on itself.
fn tile_round(ring: Ring) -> f64 {
    let circumference = std::f64::consts::TAU * ring.radius.0 as f64;
    circumference / (circumference / TILE).round().max(1.0)
}

fn single(p: Vec3d) -> [f32; 3] {
    p.map(|x| x as f32)
}

/// A distance along an axis in units of `size`, from where the site is in the pattern, so it
/// stays small near the site and the pattern still lines up round the wheel.
fn cells(along: f64, phase: f64, size: f64) -> f32 {
    ((along + phase.rem_euclid(size)) / size) as f32
}

/// The glass: its wall over the columns and spans, and a cap at each end over the columns and
/// the depths in from the wall.
fn glass_mesh(drum: &Wheel, columns: &[f64], spans: &[f64], depths: &[f64]) -> Mesh {
    let (ring, phase) = (drum.ring, drum.site);
    let (radius, half_width) = (ring.radius.0 as f64, ring.half_width.0 as f64);
    let pane = pane_round(ring);
    let phase_arc = phase.arc(ring);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for &turn in columns {
        for &y in spans {
            let at = drum.wall_point(turn, y);
            let (_, outward) = drum.depth_and_outward(at);
            positions.push(single(at));
            normals.push(single(outward));
            uvs.push([
                cells(turn * radius, phase_arc, pane),
                cells(y, phase.y, PANE),
            ]);
        }
    }
    grid_indices(&mut indices, columns.len(), spans.len(), 0, |_, _| true);
    let (sin_turn, cos_turn) = phase.phi.sin_cos();
    for (side, y) in [(-1.0f32, -half_width), (1.0, half_width)] {
        let first = positions.len() as u32;
        let y = y - drum.site.y;
        for &turn in columns {
            let at = drum.wall_point(turn, y);
            let (_, outward) = drum.depth_and_outward(at);
            for &depth in depths {
                let x = at[0] - outward[0] * depth;
                let z = at[2] - outward[2] * depth;
                positions.push(single([x, y, z]));
                normals.push([0.0, side, 0.0]);
                // the cap's grid is fixed to the wheel: turned by the site's angle and offset
                // by where the site is in it
                uvs.push([
                    cells(x * cos_turn - z * sin_turn, phase.cap[0], PANE),
                    cells(x * sin_turn + z * cos_turn, phase.cap[1], PANE),
                ]);
            }
        }
        grid_indices(&mut indices, columns.len(), depths.len(), first, |_, _| {
            true
        });
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Two triangles for every cell of a grid of `along` by `across` vertices laid out across
/// first, from vertex `first`, where `keep` says the cell is wanted.
fn grid_indices(
    indices: &mut Vec<u32>,
    along: usize,
    across: usize,
    first: u32,
    keep: impl Fn(usize, usize) -> bool,
) {
    for i in 0..along - 1 {
        for j in 0..across - 1 {
            if !keep(i, j) {
                continue;
            }
            let a = first + (i * across + j) as u32;
            let b = first + ((i + 1) * across + j) as u32;
            indices.extend_from_slice(&[a, b, b + 1, a, b + 1, a + 1]);
        }
    }
}

/// The struts: square beams along the axis just outside the glass, evenly round the ring,
/// each gridded along the spans so that the one the viewer stands by is drawn finely.
fn strut_mesh(drum: &Wheel, spans: &[f64]) -> Mesh {
    let mut rows = Vec::with_capacity(spans.len() + 2);
    rows.push(spans[0] - STRUT_OVERHANG);
    rows.extend_from_slice(spans);
    rows.push(spans[spans.len() - 1] + STRUT_OVERHANG);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for k in 0..STRUTS {
        let a = k as f64 / STRUTS as f64 * std::f64::consts::TAU;
        let turn = (a - drum.site.phi + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        let at = drum.wall_point(turn, 0.0);
        let (_, out) = drum.depth_and_outward(at);
        let along = [-out[2], 0.0, out[0]];
        let axes = [
            out,
            along,
            [-out[0], 0.0, -out[2]],
            [-along[0], 0.0, -along[2]],
        ];
        for f in 0..4 {
            let (n, t) = (axes[f], axes[(f + 1) % 4]);
            let first = positions.len() as u32;
            for &y in &rows {
                for sign in [-1.0, 1.0] {
                    positions.push(single([
                        at[0] + out[0] * RIM_OFFSET + n[0] * STRUT_HALF + t[0] * sign * STRUT_HALF,
                        y,
                        at[2] + out[2] * RIM_OFFSET + n[2] * STRUT_HALF + t[2] * sign * STRUT_HALF,
                    ]));
                    normals.push(single(n));
                }
            }
            grid_indices(&mut indices, rows.len(), 2, first, |_, _| true);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// A rim: a tube round one edge of the glass, along the columns.
fn rim_mesh(drum: &Wheel, columns: &[f64], side: f64) -> Mesh {
    let y = side * (drum.ring.half_width.0 as f64 + RIM_RADIUS) - drum.site.y;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for &turn in columns {
        let at = drum.wall_point(turn, y);
        let (_, outward) = drum.depth_and_outward(at);
        for k in 0..RIM_SIDES {
            let a = k as f64 / RIM_SIDES as f64 * std::f64::consts::TAU;
            let (out, up) = (a.cos(), a.sin());
            let r = RIM_OFFSET + RIM_RADIUS * out;
            positions.push(single([
                at[0] + outward[0] * r,
                at[1] + RIM_RADIUS * up,
                at[2] + outward[2] * r,
            ]));
            normals.push(single([outward[0] * out, up, outward[2] * out]));
        }
    }
    let sides = RIM_SIDES as u32;
    for i in 0..columns.len() as u32 - 1 {
        for k in 0..sides {
            let a = i * sides + k;
            let b = i * sides + (k + 1) % sides;
            indices.extend_from_slice(&[a, a + sides, b + sides, a, b + sides, b]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// The raised parts of the landscape over the columns and rows, plus skirts down to the glass
/// along both caps so raised ground reads as solid from the side. A skirt hangs from an edge of
/// its own rather than from the ground's, so the ground is shaded to its very edge by its own
/// slope and not by the drop beside it. Bare glass gets no triangles, and nothing is placed on
/// the glass itself, which would fight it for depth. The tiles round and along and the height
/// above the glass ride along as attributes.
fn terrain_mesh(drum: &Wheel, columns: &[f64], rows: &[f64]) -> Mesh {
    let phase = drum.site;
    let landscape = &drum.landscape;
    let ring = drum.ring;
    let radius = ring.radius.0 as f64;
    let tile = tile_round(ring);
    let across = rows.len() + 4;
    let mut positions = Vec::with_capacity(columns.len() * across);
    let mut uvs = Vec::with_capacity(columns.len() * across);
    let mut heights = Vec::with_capacity(columns.len() * across);
    let mut raised = Vec::with_capacity(columns.len() * across);
    let grid = landscape.grid();
    let phase_arc = phase.arc(ring);
    for &turn in columns {
        let round = drum.site.round.on(turn * radius, grid);
        let mut push = |height: f64, y: f64| {
            let room = drum.ring.room_along(GLASS_INSET);
            let y = y.clamp(-room - drum.site.y, room - drum.site.y);
            let at = drum.wall_point(turn, y);
            let (_, outward) = drum.depth_and_outward(at);
            let lift = height.max(GLASS_INSET);
            positions.push(single([
                at[0] - outward[0] * lift,
                at[1],
                at[2] - outward[2] * lift,
            ]));
            uvs.push([
                cells(turn * radius, phase_arc, tile),
                cells(y, phase.y, TILE),
            ]);
            heights.push([height as f32, 0.0]);
            raised.push(height > 0.0);
        };
        let ground = |y: f64| {
            let at = Place {
                round,
                along: y + drum.site.y,
            };
            landscape.reach(at)
        };
        let (first, last) = (rows[0], rows[rows.len() - 1]);
        push(0.0, first);
        push(ground(first), first);
        for &y in rows {
            push(ground(y), y);
        }
        push(ground(last), last);
        push(0.0, last);
    }
    let raised = |i: usize, j: usize| raised[i * across + j.clamp(1, across - 2)];
    let mut indices = Vec::with_capacity(columns.len() * (across - 1) * 6);
    grid_indices(&mut indices, columns.len(), across, 0, |i, j| {
        let skirt = j == 0 || j == across - 2;
        skirt && (raised(i, j) || raised(i + 1, j) || raised(i, j + 1) || raised(i + 1, j + 1))
    });
    // the ground itself ends along the line its height runs out at, which crosses the cells
    // it runs out in, rather than along their edges
    let mut rim = HashMap::new();
    for i in 0..columns.len() - 1 {
        for j in 2..across - 3 {
            let a = (i * across + j) as u32;
            let b = ((i + 1) * across + j) as u32;
            let corners = [(a, b), (b, b + 1), (b + 1, a + 1), (a + 1, a)];
            let standing = corners
                .iter()
                .filter(|(corner, _)| heights[*corner as usize][0] >= GROUND_ENDS)
                .count();
            if standing == corners.len() {
                indices.extend_from_slice(&[a, b, b + 1, a, b + 1, a + 1]);
                continue;
            }
            if standing == 0 {
                continue;
            }
            let mut ground = Vec::with_capacity(6);
            for (corner, next) in corners {
                let [here, there] = [corner, next].map(|v| heights[v as usize][0]);
                if here >= GROUND_ENDS {
                    ground.push(corner);
                }
                if (here >= GROUND_ENDS) != (there >= GROUND_ENDS) {
                    let edge = (corner.min(next), corner.max(next));
                    ground.push(*rim.entry(edge).or_insert_with(|| {
                        let t = (GROUND_ENDS - here) / (there - here);
                        let [from, to] = [corner, next].map(|v| v as usize);
                        let between = |a: f32, b: f32| a + (b - a) * t;
                        positions
                            .push([0, 1, 2].map(|k| between(positions[from][k], positions[to][k])));
                        uvs.push([0, 1].map(|k| between(uvs[from][k], uvs[to][k])));
                        heights.push([GROUND_ENDS, 0.0]);
                        (positions.len() - 1) as u32
                    }));
                }
            }
            for k in 1..ground.len().saturating_sub(1) {
                indices.extend_from_slice(&[ground[0], ground[k], ground[k + 1]]);
            }
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, heights);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

/// Hand the water the drum's state for every substep taken this frame, what changed of the
/// sculpted ground, and how the survey's columns lie.
fn feed_water(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    mut frame: ResMut<DrumFrame>,
    mut layout: Local<GroundLayout>,
) {
    let drum = &sim.drum;
    layout.update(drum, &mut frame);
    let resolution = fluid.resolution();
    frame.survey = Columns::of(drum.ring, resolution);
    let shape = frame.shape;
    frame.states.clear();
    for record in &sim.substeps {
        frame.states.push(DrumUniform::new(
            drum,
            record.spin,
            record.spin_rate,
            resolution,
            &shape,
        ));
    }
    frame.states.push(DrumUniform::new(
        drum,
        drum.spin,
        drum.spin_rate,
        resolution,
        &shape,
    ));
}
