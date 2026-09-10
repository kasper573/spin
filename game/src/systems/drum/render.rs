//! Drawing the drum: the glass, its rims and struts, and the landscape. The wheel is drawn in
//! the drum's frame about the site, gridded round the ring and along its axis with steps that
//! grow with the distance from the site, so that the curve is never visibly faceted where the
//! viewer is, no triangle is large next to its distance from the viewer, which single precision
//! rasterization could not place, and the whole wheel costs about the same number of triangles
//! at any size. The meshes are rebuilt when the site moves, when the ring changes size, or
//! when the landscape changes; the patterns on the glass and the ground are placed from where
//! the site is in them, which is kept exactly.
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

use super::gpu::COLUMN_FIXED;
use super::landscape::{ROWS, SEGMENTS};
use super::{Drum, DrumFrame, DrumUniform, GLASS_THICKNESS, PANE, Ring, Site, TILE};
use crate::core::fluid::Fluid;
use crate::core::math::Vec3d;
use crate::systems::scene::{SPACE, Sky, Viewpoint};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::water;

/// The two tints the grass is tiled in, and the dirt under it.
const GRASS: Color = Color::srgb(0.36, 0.62, 0.24);
const GRASS_DARK: Color = Color::srgb(0.3, 0.54, 0.2);
const DIRT: Color = Color::srgb(0.45, 0.32, 0.2);

/// The ground's colour as seen from across the ring, where its tiles blur together.
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
        .add_systems(Startup, spawn)
        .add_systems(
            Update,
            (rebuild, place, light, wet, feed_water)
                .chain()
                .in_set(SimSet::Observe),
        );
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
    /// Where the viewpoint lies in the site's frame, in metres.
    #[uniform(0)]
    origin: Vec4,
    /// The ring's radius and half width, in metres.
    #[uniform(0)]
    ring: Vec4,
    /// The ground's colour as seen from across the ring.
    #[uniform(0)]
    ground: LinearRgba,
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
    /// The site everything is drawn about: its place round the ring in segments of the grid,
    /// its place along the axis and the glass radius, in metres.
    #[uniform(0)]
    site: Vec4,
    /// Where the point everything is drawn about lies in the site's frame, in metres.
    #[uniform(0)]
    origin: Vec4,
    /// The landscape grid's angle per segment, its row spacing and the drum's half width, in
    /// metres, and the depth of water each particle surveyed over a column adds to it.
    #[uniform(0)]
    grid: Vec4,
    /// x: seconds; z: metres per second per unit of a surveyed column's flow; w: the grid's
    /// rows and segments, packed.
    #[uniform(0)]
    clock: Vec4,
    #[uniform(0)]
    absorption: Vec4,
    #[uniform(0)]
    scatter: Vec4,
    #[storage(1, read_only)]
    columns: Handle<ShaderBuffer>,
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

/// Something fixed to the wheel: where it sits in the drum's frame.
#[derive(Component)]
struct Placed(Vec3d);

/// The glass, rims and struts, built for one size of ring about one site.
#[derive(Component)]
struct Structure;

#[derive(Component)]
struct Terrain;

/// What the wheel's meshes were last built for.
#[derive(Resource, Default)]
struct Built {
    structure: Option<(Ring, Site, f64)>,
    terrain: Option<(Ring, u64, Site, f64)>,
}

/// The materials the wheel is built with.
#[derive(Resource)]
struct WheelMaterials {
    glass: Handle<GlassMaterial>,
    metal: Handle<StandardMaterial>,
    terrain: Handle<TerrainMaterial>,
}

fn spawn(
    mut commands: Commands,
    frame: Res<DrumFrame>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    mut standard: ResMut<Assets<StandardMaterial>>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
) {
    commands.insert_resource(WheelMaterials {
        glass: glass.add(GlassMaterial {
            tint: LinearRgba::new(0.9, 0.96, 0.98, 1.0),
            to_stars: Vec4::new(0.0, 0.0, 0.0, 1.0),
            background: SPACE.to_linear(),
            panes: Vec4::ZERO,
            origin: Vec4::ZERO,
            ring: Vec4::ZERO,
            ground: ground_albedo(),
        }),
        metal: standard.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.17, 0.2),
            metallic: 0.8,
            perceptual_roughness: 0.45,
            ..default()
        }),
        terrain: terrain.add(TerrainMaterial {
            dirt: DIRT.into(),
            grass: GRASS.into(),
            grass_dark: GRASS_DARK.into(),
            site: Vec4::ZERO,
            origin: Vec4::ZERO,
            grid: Vec4::ONE,
            clock: Vec4::ZERO,
            absorption: water::ABSORPTION.extend(0.0),
            scatter: water::SCATTER.extend(water::SCATTER_PER_METRE),
            columns: frame.columns.clone(),
        }),
    });
    commands.init_resource::<Built>();
}

/// Rebuild whatever the ring, the site or the landscape has outdated.
#[allow(clippy::too_many_arguments)]
fn rebuild(
    mut commands: Commands,
    sim: Res<Simulation>,
    viewpoint: Res<Viewpoint>,
    materials: Res<WheelMaterials>,
    mut built: ResMut<Built>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut glass: ResMut<Assets<GlassMaterial>>,
    structures: Query<Entity, With<Structure>>,
    terrains: Query<Entity, With<Terrain>>,
) {
    let drum = &sim.drum;
    let (ring, site, standoff) = (drum.ring, drum.site, viewpoint.standoff);
    let phase = site;
    if built.structure != Some((ring, site, standoff)) {
        built.structure = Some((ring, site, standoff));
        for entity in &structures {
            commands.entity(entity).despawn();
        }
        let columns = columns(ring, standoff);
        let spans = spans(drum, standoff);
        if let Some(mut material) = glass.get_mut(&materials.glass) {
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
                Placed([0.0; 3]),
                Mesh3d(meshes.add(mesh)),
                NoFrustumCulling,
                Transform::default(),
            )
        };
        let glass_wall = glass_mesh(drum, phase, &columns, &spans, &depths(ring, standoff));
        commands.spawn((
            structure(glass_wall),
            MeshMaterial3d(materials.glass.clone()),
            NotShadowCaster,
        ));
        for side in [-1.0, 1.0] {
            let rim = rim_mesh(drum, &columns, side);
            commands.spawn((structure(rim), MeshMaterial3d(materials.metal.clone())));
        }
        let struts = strut_mesh(drum, &spans);
        commands.spawn((structure(struts), MeshMaterial3d(materials.metal.clone())));
    }
    let landscape = &drum.landscape;
    let version = landscape.version();
    if built.terrain != Some((ring, version, site, standoff)) {
        built.terrain = Some((ring, version, site, standoff));
        for entity in &terrains {
            commands.entity(entity).despawn();
        }
        if !landscape.is_empty() {
            let columns = columns(ring, standoff);
            let rows = ground_rows(drum, &spans(drum, standoff));
            commands.spawn((
                Terrain,
                Placed([0.0; 3]),
                Mesh3d(meshes.add(terrain_mesh(drum, phase, &columns, &rows))),
                MeshMaterial3d(materials.terrain.clone()),
                NoFrustumCulling,
                Transform::default(),
            ));
        }
    }
}

/// Everything fixed to the wheel is placed about the viewpoint.
fn place(viewpoint: Res<Viewpoint>, mut placed: Query<(&Placed, &mut Transform)>) {
    for (Placed(at), mut transform) in &mut placed {
        *transform = viewpoint.place(*at);
    }
}

/// The glass mirrors space wherever the sky has turned it.
fn light(
    sim: Res<Simulation>,
    viewpoint: Res<Viewpoint>,
    sky: Res<Sky>,
    materials: Res<WheelMaterials>,
    mut glass: ResMut<Assets<GlassMaterial>>,
) {
    if let Some(mut material) = glass.get_mut(&materials.glass) {
        let stars = sky.rotation.inverse();
        material.to_stars = Vec4::new(stars.x, stars.y, stars.z, stars.w);
        let [x, y, z] = viewpoint.origin;
        material.origin = Vec4::new(x as f32, y as f32, z as f32, 0.0);
        let ring = sim.drum.ring;
        material.ring = Vec4::new(ring.radius.0, ring.half_width.0, 0.0, 0.0);
    }
}

/// The ground is told where the site lies on the ring and where the viewpoint lies about the
/// site, so it can look up the water surveyed over each of its points.
fn wet(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    viewpoint: Res<Viewpoint>,
    materials: Res<WheelMaterials>,
    mut terrain: ResMut<Assets<TerrainMaterial>>,
) {
    let Some(mut material) = terrain.get_mut(&materials.terrain) else {
        return;
    };
    let drum = &sim.drum;
    let resolution = fluid.resolution();
    let dphi = std::f64::consts::TAU / SEGMENTS as f64;
    material.site = Vec4::new(
        (drum.site.phi / dphi).rem_euclid(SEGMENTS as f64) as f32,
        drum.site.y as f32,
        drum.ring.radius.0,
        0.0,
    );
    let [x, y, z] = viewpoint.origin;
    material.origin = Vec4::new(x as f32, y as f32, z as f32, 0.0);
    let footprint = drum.landscape.segment_arc() * drum.landscape.row_spacing();
    let particle = resolution.length().powi(3);
    material.grid = Vec4::new(
        std::f32::consts::TAU / SEGMENTS as f32,
        drum.landscape.row_spacing() as f32,
        drum.ring.half_width.0,
        (particle / footprint) as f32,
    );
    material.clock = Vec4::new(
        sim.time.0,
        (resolution.length() / COLUMN_FIXED) as f32,
        (resolution.length() / resolution.time() / COLUMN_FIXED) as f32,
        (ROWS * 4096 + SEGMENTS) as f32,
    );
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
fn spans(drum: &Drum, standoff: f64) -> Vec<f64> {
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

/// The rows the ground is sampled at: the spans, and every row of the landscape, so nothing
/// sculpted is skipped.
fn ground_rows(drum: &Drum, spans: &[f64]) -> Vec<f64> {
    let half_width = drum.ring.half_width.0 as f64;
    let dy = drum.landscape.row_spacing();
    let mut rows = spans.to_vec();
    rows.extend((0..ROWS).map(|j| -half_width + j as f64 * dy - drum.site.y));
    rows.sort_by(|a, b| a.total_cmp(b));
    rows.dedup_by(|a, b| (*a - *b).abs() < 1e-9 * dy);
    rows
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
fn glass_mesh(drum: &Drum, phase: Site, columns: &[f64], spans: &[f64], depths: &[f64]) -> Mesh {
    let ring = drum.ring;
    let (radius, half_width) = (ring.radius.0 as f64, ring.half_width.0 as f64);
    let pane = pane_round(ring);
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
                cells(turn * radius, phase.arc, pane),
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
fn strut_mesh(drum: &Drum, spans: &[f64]) -> Mesh {
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
fn rim_mesh(drum: &Drum, columns: &[f64], side: f64) -> Mesh {
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
/// along both caps so raised ground reads as solid from the side. Bare glass gets no
/// triangles, and nothing is placed on the glass itself, which would fight it for depth. The
/// tiles round and along and the height above the glass ride along as attributes.
fn terrain_mesh(drum: &Drum, phase: Site, columns: &[f64], rows: &[f64]) -> Mesh {
    let landscape = &drum.landscape;
    let ring = drum.ring;
    let (radius, half_width) = (ring.radius.0 as f64, ring.half_width.0 as f64);
    let tile = tile_round(ring);
    let across = rows.len() + 2;
    let mut positions = Vec::with_capacity(columns.len() * across);
    let mut uvs = Vec::with_capacity(columns.len() * across);
    let mut heights = Vec::with_capacity(columns.len() * across);
    let mut raised = Vec::with_capacity(columns.len() * across);
    for &turn in columns {
        let phi = drum.site.phi + turn;
        let mut push = |height: f64, y: f64| {
            let y = y.clamp(
                -half_width + GLASS_INSET - drum.site.y,
                half_width - GLASS_INSET - drum.site.y,
            );
            let at = drum.wall_point(turn, y);
            let (_, outward) = drum.depth_and_outward(at);
            let lift = height.max(GLASS_INSET);
            positions.push(single([
                at[0] - outward[0] * lift,
                at[1],
                at[2] - outward[2] * lift,
            ]));
            uvs.push([
                cells(turn * radius, phase.arc, tile),
                cells(y, phase.y, TILE),
            ]);
            heights.push([height as f32, 0.0]);
            raised.push(height > 0.0);
        };
        push(0.0, rows[0]);
        for &y in rows {
            push(landscape.sample(phi, y + drum.site.y).0, y);
        }
        push(0.0, rows[rows.len() - 1]);
    }
    let raised = |i: usize, j: usize| raised[i * across + j.clamp(1, rows.len())];
    let mut indices = Vec::with_capacity(columns.len() * (across - 1) * 6);
    grid_indices(&mut indices, columns.len(), across, 0, |i, j| {
        raised(i, j) || raised(i + 1, j) || raised(i, j + 1) || raised(i + 1, j + 1)
    });
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

/// Hand the water the drum's state for every substep taken this frame and the landscape when it
/// changed.
fn feed_water(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    mut frame: ResMut<DrumFrame>,
    mut images: ResMut<Assets<Image>>,
    mut uploaded: Local<Option<u64>>,
) {
    let resolution = fluid.resolution();
    frame.states.clear();
    for record in &sim.substeps {
        frame.states.push(DrumUniform::new(
            &sim.drum,
            record.spin,
            record.spin_rate,
            resolution,
        ));
    }
    frame.states.push(DrumUniform::new(
        &sim.drum,
        sim.drum.spin,
        sim.drum.spin_rate,
        resolution,
    ));
    let version = sim.drum.landscape.version();
    if *uploaded != Some(version) {
        *uploaded = Some(version);
        super::gpu::upload_heights(&sim.drum, &frame, &mut images);
    }
}
