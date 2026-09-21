//! Water lying on the floor of a spun ring as a sheet, solved on the GPU: see
//! `shaders/sheet.wgsl` for how. Water that lies on a floor is no thicker than it is deep, and
//! a sheet says how deep it is over every cell of the floor however thin it lies there, which
//! no count of parcels of water does.
//!
//! The sheet knows nothing of the floor but what it is told: how its cells lie on the ring,
//! and how high the floor stands at every corner of them. It is drawn as the water is, from
//! buffers of the water's own vertices.
mod gpu;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::Shader;

pub use gpu::{DRAINS as SHEET_DRAINS, JET_INDICES as SHEET_JET_INDICES, SheetBuffers, SheetStep};

use crate::core::fluid::ReadOnce;
use crate::core::units::{Litres, Metres, MetresPerSecond, RadiansPerSecond, Seconds};

/// The most cells the sheet has round the ring and along its axis.
pub const SHEET_CELLS: [u32; 2] = [1024, 256];
/// The corners of those cells, and the two triangles over each.
pub const SHEET_CORNERS: usize = (SHEET_CELLS[0] as usize + 1) * (SHEET_CELLS[1] as usize + 1);
pub const SHEET_INDICES: usize = 6 * SHEET_CELLS[0] as usize * SHEET_CELLS[1] as usize;

/// How many cells a side the block is whose water is reported back, and its corners.
pub const SHEET_WATCHED: u32 = 16;
pub const SHEET_WATCHED_CORNERS: usize = ((SHEET_WATCHED + 1) * (SHEET_WATCHED + 1)) as usize;

/// The most grains each way a cell is made of: see [`SheetCarried`].
pub const SHEET_MOST_GRAINS: u32 = 32;

/// The most steps a frame's water takes; time it has no steps for is owed to the next frame.
const MOST_STEPS: u32 = 8;
/// `COURANT` in `sheet.wgsl`: the share of a cell the quickest wave may cross in a step.
const COURANT: f32 = 0.2;

const SHADER: &str = "embedded://game/core/sheet/shaders/sheet.wgsl";
const JETS_SHADER: &str = "embedded://game/core/sheet/shaders/jets.wgsl";

/// Set by the render world once every kernel has compiled; until then the sheet takes no water,
/// so none is lost.
#[derive(Resource, Clone, Default)]
pub struct SheetReady(Arc<AtomicBool>);

impl SheetReady {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// How the sheet's cells lie on the ring: how many of them are in use round it and along it,
/// whether those round it close on themselves, how long a cell's arc on the glass is and how
/// wide it is along the axis, and the glass's radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetLie {
    pub cells: [u32; 2],
    pub closed: bool,
    pub cell: [Metres; 2],
    pub radius: Metres,
}

/// How water that is on the sheet lies on it once the sheet is laid anew. The cells it lay on
/// and those it lies on now are both made of whole grains of the floor, so many each way to a
/// cell then and now, and each of the sheet's grains as it is now, round the ring and along
/// its axis, was one of its grains as it was or none of them. A grain keeps the water it had,
/// which stands the deeper on it the smaller the grain has become: `spread` is how many times
/// its area now its area was.
#[derive(Clone, Debug, PartialEq)]
pub struct SheetCarried {
    pub grains: [u32; 2],
    pub was: [Vec<Option<u32>>; 2],
    pub spread: f32,
}

/// An opening in the sheet's floor that its water runs out of, as fast as its head drives it
/// against whatever is on the far side: the block of cells it lies among, by the first of them
/// and how many each way, its middle in cells from that first one, how far from its middle it is
/// open, which other drain's water presses back on it from the far side, if any does, and
/// where its water comes out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetDrain {
    pub first: [u32; 2],
    pub cells: [u32; 2],
    pub middle: [f32; 2],
    pub reach: Metres,
    pub far: Option<usize>,
    pub outlet: SheetOutlet,
}

/// Where a drain's water comes out, in the frame the sheet's surface is drawn in: the middle of
/// the opening, the way out of it, and two ways across it, all three of unit length and square
/// to one another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetOutlet {
    pub middle: Vec3,
    pub way: Vec3,
    pub across: [Vec3; 2],
}

/// How poured water lands: how fast it runs over the floor as it does, round the ring and
/// along its axis, and how fast it came down, which is how fast it spreads from where it lands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheetLanding {
    pub run: [MetresPerSecond; 2],
    pub spread: MetresPerSecond,
}

/// The sheet's water at a place: how deep it is, the level it stands at over the glass, and how
/// fast it runs round the ring and along its axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheetWater {
    pub depth: Metres,
    pub level: Metres,
    pub run: [MetresPerSecond; 2],
}

/// The water round a block of the sheet's cells, as the GPU last reported it: the cell the block
/// begins at, and the water at each of its corners.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SheetWatched {
    first: [u32; 2],
    /// How quickly the sheet's quickest wave crosses a cell, per second, once that is known.
    quickest: Option<f32>,
    corners: Vec<[f32; 4]>,
}

impl SheetWatched {
    /// The water at a place among the sheet's cells, counted in cells from the sheet's first
    /// corner, if the block reaches there.
    pub fn at(&self, lie: SheetLie, place: [f64; 2]) -> Option<SheetWater> {
        let side = SHEET_WATCHED as usize + 1;
        if self.corners.len() < side * side {
            return None;
        }
        let mut round = place[0] - self.first[0] as f64;
        if lie.closed {
            round = round.rem_euclid(lie.cells[0] as f64);
        }
        let along = place[1] - self.first[1] as f64;
        let reach = SHEET_WATCHED as f64;
        if !(0.0..reach).contains(&round) || !(0.0..reach).contains(&along) {
            return None;
        }
        let (i, j) = (round.floor() as usize, along.floor() as usize);
        let (fu, fv) = ((round - i as f64) as f32, (along - j as f64) as f32);
        let corner = |i: usize, j: usize| self.corners[j * side + i];
        let mixed: [f32; 4] = std::array::from_fn(|k| {
            (corner(i, j)[k] * (1.0 - fu) + corner(i + 1, j)[k] * fu) * (1.0 - fv)
                + (corner(i, j + 1)[k] * (1.0 - fu) + corner(i + 1, j + 1)[k] * fu) * fv
        });
        Some(SheetWater {
            depth: Metres(mixed[0]),
            level: Metres(mixed[1]),
            run: [MetresPerSecond(mixed[2]), MetresPerSecond(mixed[3])],
        })
    }
}

/// The sheet as the game has it: what it is told of the floor, what is poured on it, and what
/// it was last found to hold.
#[derive(Resource, Default)]
pub struct Sheet {
    ready: SheetReady,
    lie: Option<SheetLie>,
    frame: SheetFrame,
    held: Litres,
    flying: Litres,
    watched: SheetWatched,
}

impl Sheet {
    pub fn lie(&self) -> Option<SheetLie> {
        self.lie
    }

    /// Lay the sheet's cells on a floor, whose height over the glass `floor` gives at every
    /// corner of them, by which corner round the ring and along it. Water on the sheet is
    /// carried as `carried` says, and left in the cells it is in if nothing says.
    pub fn lay(
        &mut self,
        lie: SheetLie,
        carried: Option<SheetCarried>,
        floor: impl Fn(u32, u32) -> f32,
    ) {
        assert!(
            lie.cells[0] <= SHEET_CELLS[0] && lie.cells[1] <= SHEET_CELLS[1],
            "the sheet has no more cells than its buffers hold"
        );
        if let (Some(carried), Some(was)) = (carried, self.lie) {
            self.frame.carried = carried.packed(was, lie);
            self.frame.carries += 1;
        }
        let stride = SHEET_CELLS[0] as usize + 1;
        let mut bed = vec![0.0; SHEET_CORNERS];
        for along in 0..=lie.cells[1] {
            for round in 0..=lie.cells[0] {
                bed[along as usize * stride + round as usize] = floor(round, along);
            }
        }
        self.lie = Some(lie);
        self.frame.bed = Arc::from(bed);
        self.frame.floors += 1;
    }

    /// Where the sheet's first corner lies from the site its surface is drawn about: how far
    /// round the glass, and how far along the axis.
    pub fn place(&mut self, origin: [Metres; 2]) {
        self.frame.params.origin = Vec2::new(origin[0].0, origin[1].0);
    }

    pub fn origin(&self) -> [Metres; 2] {
        let origin = self.frame.params.origin;
        [Metres(origin.x), Metres(origin.y)]
    }

    /// Pour water over a block of cells, which must lie within those in use, most of it in the
    /// middle of the block and none at its rim, landing as `landing` says. Says whether the
    /// sheet took it, which it does not before it is ready to.
    pub fn pour(
        &mut self,
        first: [u32; 2],
        cells: [u32; 2],
        water: Litres,
        landing: SheetLanding,
    ) -> bool {
        let Some(lie) = self.lie.filter(|_| self.ready.get()) else {
            return false;
        };
        let area = lie.cell[0].0 * lie.cell[1].0;
        let middle = Vec2::new(cells[0] as f32, cells[1] as f32) / 2.0;
        let cell = Vec2::new(lie.cell[0].0, lie.cell[1].0);
        let reach = (middle * cell).max_element();
        // `falling` in `sheet.wgsl`
        let falling = |round: u32, along: u32| {
            let from_middle = (Vec2::new(round as f32, along as f32) + 0.5 - middle) * cell;
            let within = (1.0 - (from_middle / reach).length_squared()).max(0.0);
            (within * within) as f64
        };
        let shares: f64 = (0..cells[1])
            .flat_map(|along| (0..cells[0]).map(move |round| (round, along)))
            .map(|(round, along)| falling(round, along))
            .sum();
        self.frame.params.pour_first = first.into();
        self.frame.params.pour_cells = cells.into();
        self.frame.params.pour = (water.0 as f64 / 1000.0 / (shares * area as f64)) as f32;
        self.frame.params.pour_run = Vec2::new(landing.run[0].0, landing.run[1].0);
        self.frame.params.pour_middle = middle;
        self.frame.params.pour_spread = landing.spread.0;
        self.frame.params.pour_reach = reach;
        self.frame.wetted = true;
        true
    }

    /// Whether no water has been poured on the sheet since it was last emptied.
    pub fn is_empty(&self) -> bool {
        !self.frame.wetted
    }

    /// Let the sheet's water run for this long, in a ring turning this fast.
    pub fn advance(&mut self, by: Seconds, spin: RadiansPerSecond) {
        self.frame.params.advance += by.0;
        self.frame.params.spin = spin.0;
    }

    /// The openings the sheet's water runs out of from now on.
    pub fn drain(&mut self, drains: [Option<SheetDrain>; SHEET_DRAINS]) {
        self.frame.jets.drains = drains.map(|drain| {
            drain.map_or_else(DrainParams::default, |drain| DrainParams {
                first: drain.first.into(),
                cells: drain.cells.into(),
                middle: drain.middle.into(),
                reach: drain.reach.0,
                far: drain.far.map_or(SHEET_DRAINS as u32, |far| far as u32),
                out_middle: drain.outlet.middle.extend(0.0),
                out_way: drain.outlet.way.extend(0.0),
                out_across: drain.outlet.across[0].extend(0.0),
                out_up: drain.outlet.across[1].extend(0.0),
            })
        });
    }

    /// Report the water round the block of cells that begins at this one from now on.
    pub fn watch(&mut self, first: [u32; 2]) {
        self.frame.params.watch_first = first.into();
    }

    /// The water round the watched block as it was last reported.
    pub fn watched(&self) -> &SheetWatched {
        &self.watched
    }

    /// Take all the water off the sheet.
    pub fn empty(&mut self) {
        self.frame.emptied += 1;
        self.frame.wetted = false;
        self.held = Litres(0.0);
        self.flying = Litres(0.0);
        self.watched = SheetWatched::default();
    }

    /// The water the sheet was last found to hold, that of it in the air with the rest.
    pub fn held(&self) -> Litres {
        self.held
    }

    /// Whether water that ran out of the sheet's drains was last found not to have come down.
    pub fn has_jets(&self) -> bool {
        self.flying.0 > 0.0
    }

    /// The water that had run out of the sheet's drains and not yet come down, when last found.
    pub fn flying(&self) -> Litres {
        self.flying
    }
}

impl SheetCarried {
    /// `Carried` in `sheet.wgsl`.
    fn packed(&self, was: SheetLie, now: SheetLie) -> Arc<[u32]> {
        assert!(
            self.grains.iter().all(|grains| (1..=SHEET_MOST_GRAINS).contains(grains))
                && (0..2).all(|axis| self.was[axis].len() == (now.cells[axis] * self.grains[1]) as usize),
            "every grain of the sheet's cells is carried, and a cell has no more than the most"
        );
        let grain = |was: &Option<u32>| was.map_or(u32::MAX, |was| was);
        [
            was.cells[0],
            was.cells[1],
            self.grains[0],
            self.grains[1],
            self.spread.to_bits(),
            self.was[0].len() as u32,
        ]
        .into_iter()
        .chain(self.was.iter().flatten().map(grain))
        .collect()
    }
}

/// What the render world is handed of the sheet for a frame.
#[derive(Resource, Clone, ExtractResource)]
pub struct SheetFrame {
    params: SheetParams,
    jets: JetsParams,
    /// The floor's height at every corner, and how many floors there have been.
    bed: Arc<[f32]>,
    floors: u32,
    /// How the water was last carried, and how many times it has been.
    carried: Arc<[u32]>,
    carries: u32,
    /// How many times the sheet has been emptied, and whether water has been poured since.
    emptied: u32,
    wetted: bool,
    /// How many steps the frame's kernels are sent out for. The GPU takes no more of them than
    /// the frame's time asks for, but each costs its sending out whether it is taken or not.
    steps: u32,
}

impl Default for SheetFrame {
    fn default() -> Self {
        SheetFrame {
            params: SheetParams::default(),
            jets: JetsParams::default(),
            bed: Arc::from(Vec::new()),
            floors: 0,
            carried: Arc::from(Vec::new()),
            carries: 0,
            emptied: 0,
            wetted: false,
            steps: MOST_STEPS,
        }
    }
}

/// `Sheet` in `sheet.wgsl`.
#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct SheetParams {
    stored: UVec2,
    used: UVec2,
    cell: Vec2,
    origin: Vec2,
    radius: f32,
    spin: f32,
    advance: f32,
    closed: u32,
    pour_first: UVec2,
    pour_cells: UVec2,
    pour_run: Vec2,
    pour_middle: Vec2,
    pour_spread: f32,
    pour_reach: f32,
    pour: f32,
    watch_first: UVec2,
}

/// `drains` in `jets.wgsl`.
#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct JetsParams {
    drains: [DrainParams; SHEET_DRAINS],
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct DrainParams {
    first: UVec2,
    cells: UVec2,
    middle: Vec2,
    reach: f32,
    far: u32,
    out_middle: Vec4,
    out_way: Vec4,
    out_across: Vec4,
    out_up: Vec4,
}

pub struct SheetPlugin;

impl Plugin for SheetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/sheet.wgsl");
        embedded_asset!(app, "shaders/jets.wgsl");
        let shader = app.world().resource::<AssetServer>().load::<Shader>(SHADER);
        let buffers =
            gpu::create_buffers(&mut app.world_mut().resource_mut::<Assets<ShaderBuffer>>());
        let ready = SheetReady::default();
        app.insert_resource(buffers)
            .insert_resource(ready.clone())
            .insert_resource(SheetShader(shader))
            .insert_resource(Sheet {
                ready: ready.clone(),
                ..default()
            })
            .init_resource::<SheetFrame>()
            .add_plugins((
                ExtractResourcePlugin::<SheetBuffers>::default(),
                ExtractResourcePlugin::<SheetFrame>::default(),
            ))
            .add_systems(Startup, spawn_watchers)
            .add_systems(PostUpdate, (hand_over, watch_held, watch_watched));
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(ready);
        gpu::install(render_app);
    }
}

#[derive(Resource)]
struct SheetShader(#[allow(dead_code)] Handle<Shader>);

/// Hand the render world the frame as it has been asked for, and start the next.
fn hand_over(mut sheet: ResMut<Sheet>, mut frame: ResMut<SheetFrame>) {
    let Some(lie) = sheet.lie else {
        return;
    };
    let params = &mut sheet.frame.params;
    params.stored = SHEET_CELLS.into();
    params.used = lie.cells.into();
    params.cell = Vec2::new(lie.cell[0].0, lie.cell[1].0);
    params.radius = lie.radius.0;
    params.closed = u32::from(lie.closed);
    // as many steps as the quickest wave last reported asks of the time owed, and as many
    // again in case it has quickened since; all of them until any has been reported
    let owed = params.advance;
    sheet.frame.steps = sheet.watched.quickest.map_or(MOST_STEPS, |quickest| {
        (2 * (owed * quickest / COURANT).ceil() as u32).clamp(2, MOST_STEPS)
    });
    *frame = sheet.frame.clone();
    let params = &mut sheet.frame.params;
    params.advance = 0.0;
    params.pour = 0.0;
    params.pour_cells = UVec2::ZERO;
}

/// The entity whose readback brings back the water the sheet holds, and whether one is in
/// flight, asked of which water: a sheet emptied since does not hold what it reports.
#[derive(Component)]
struct HeldWatcher {
    awaiting: Option<u32>,
}

/// The entity whose readback brings back the water round the watched block.
#[derive(Component)]
struct WatchedWatcher {
    awaiting: Option<u32>,
}

fn spawn_watchers(mut commands: Commands) {
    commands
        .spawn(HeldWatcher { awaiting: None })
        .observe(receive_held);
    commands
        .spawn(WatchedWatcher { awaiting: None })
        .observe(receive_watched);
}

fn watch_watched(
    mut commands: Commands,
    sheet: Res<Sheet>,
    buffers: Res<SheetBuffers>,
    watcher: Single<(Entity, &mut WatchedWatcher)>,
) {
    let (entity, mut watcher) = watcher.into_inner();
    if watcher.awaiting.is_none() && !sheet.is_empty() {
        commands
            .entity(entity)
            .insert((Readback::buffer(buffers.watched.clone()), ReadOnce));
        watcher.awaiting = Some(sheet.frame.emptied);
    }
}

fn receive_watched(
    event: On<ReadbackComplete>,
    mut sheet: ResMut<Sheet>,
    mut watchers: Query<&mut WatchedWatcher>,
) {
    let Ok(mut watcher) = watchers.get_mut(event.entity) else {
        return;
    };
    let asked = watcher.awaiting.take();
    let reported: Vec<Vec4> = event.to_shader_type();
    if let (true, Some((first, corners))) =
        (asked == Some(sheet.frame.emptied), reported.split_first())
    {
        sheet.watched = SheetWatched {
            first: [first.x as u32, first.y as u32],
            quickest: Some(first.z),
            corners: corners.iter().map(Vec4::to_array).collect(),
        };
    }
}

fn watch_held(
    mut commands: Commands,
    sheet: Res<Sheet>,
    buffers: Res<SheetBuffers>,
    watcher: Single<(Entity, &mut HeldWatcher)>,
) {
    let (entity, mut watcher) = watcher.into_inner();
    if watcher.awaiting.is_none() && sheet.lie.is_some() {
        commands
            .entity(entity)
            .insert((Readback::buffer(buffers.held.clone()), ReadOnce));
        watcher.awaiting = Some(sheet.frame.emptied);
    }
}

fn receive_held(
    event: On<ReadbackComplete>,
    mut sheet: ResMut<Sheet>,
    mut watchers: Query<&mut HeldWatcher>,
) {
    let Ok(mut watcher) = watchers.get_mut(event.entity) else {
        return;
    };
    let asked = watcher.awaiting.take();
    let rows: Vec<f32> = event.to_shader_type();
    if asked == Some(sheet.frame.emptied) {
        let cubic_metres: f64 = rows.iter().map(|row| *row as f64).sum();
        sheet.held = Litres((cubic_metres * 1000.0) as f32);
        sheet.flying = Litres(rows.last().map_or(0.0, |flying| flying * 1000.0));
    }
}
