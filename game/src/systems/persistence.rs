//! Everything worth keeping across a reload — settings, avatar, drum, water, landscape — in
//! the browser's storage, written every couple of seconds: one JSON snapshot of all of it but
//! the sculpted ground, which is kept patch by patch, so that a save writes only the patches
//! that changed since the last. The water lives on the GPU, so a save first asks for a copy and
//! writes when it arrives. The world waits to run until what was kept has been read back.
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::avatar::Gyros;
use crate::core::codec;
use crate::core::fluid::{Fluid, FluidBuffers, Particle, Resolution};
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::web;
use crate::systems::drum::{Grid, Ground, Landscape, Patch, Ring, Round, Site};
use crate::systems::settings::Settings;
use crate::systems::sim::{SimSet, Simulation};

/// The key the snapshot is kept under, and what the key of every patch starts with.
const SNAPSHOT: &str = "v8";
const PATCHES: &str = "v8/patch/";
const AUTOSAVE_INTERVAL: Seconds = Seconds(2.0);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub settings: Settings,
    pub avatar: AvatarPose,
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    /// The points of the wall the avatar and the water are measured from.
    pub site: Site,
    pub water_site: Site,
    pub time: Seconds,
    /// How much water each particle stands for.
    pub water: Resolution,
    /// Seven floats per particle, in the water's frame: position, velocity, foam.
    #[serde(with = "codec::f32s")]
    pub fluid: Vec<f32>,
    /// The ground, whose patches storage keeps apart from the rest, each on its own.
    pub landscape: Ground,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct AvatarPose {
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
}

/// Saves written to storage so far, the one waiting for the water, and whether what was kept
/// has been read back and laid down, or found to be nothing.
#[derive(Resource, Default)]
pub struct Saves {
    pub completed: u32,
    pending: Option<u32>,
    restored: bool,
}

/// Whether what was kept has been read back: the world does not run before, and nothing is
/// saved over it.
pub fn restored(saves: Res<Saves>) -> bool {
    saves.restored
}

pub struct PersistencePlugin;

impl Plugin for PersistencePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Autosave(Timer::from_seconds(
            AUTOSAVE_INTERVAL.0,
            TimerMode::Repeating,
        )))
        .init_resource::<Saves>()
        .init_resource::<Stored>()
        .configure_sets(Update, SimSet::Step.run_if(restored))
        .add_systems(Startup, open)
        .add_systems(PreUpdate, restore.run_if(not(restored)))
        .add_systems(
            Update,
            (tally, autosave, flush)
                .chain()
                .in_set(SimSet::Observe)
                .run_if(restored),
        );
    }
}

pub fn snapshot(settings: &Settings, sim: &Simulation, fluid: &Fluid) -> Snapshot {
    Snapshot {
        landscape: sim.drum.landscape.ground(),
        ..kept(settings, sim, fluid)
    }
}

pub fn apply(
    snapshot: &Snapshot,
    settings: &mut Settings,
    sim: &mut Simulation,
    fluid: &mut Fluid,
) {
    *settings = snapshot.settings.clone().sanitized();
    sim.reset();
    sim.resize(settings.ring());
    fluid.restore(water_resolution(snapshot.water));
    sim.drum.spin = RadiansPerSecond(finite(snapshot.spin.0).clamp(-999.0, 999.0));
    sim.drum.target_spin = settings.spin;
    sim.drum.angle = Radians(finite_f64(snapshot.angle.0));
    let ring = sim.drum.ring;
    sim.drum.site = site_on(snapshot.site, ring);
    sim.drum.water = site_on(snapshot.water_site, ring);
    sim.time = Seconds(finite(snapshot.time.0));
    sim.drum.landscape.load(&snapshot.landscape);
    let margin = fluid.resolution().margin().0 as f64;
    for chunk in snapshot.fluid.chunks_exact(7) {
        if chunk.iter().all(|v| v.is_finite()) {
            let position = [chunk[0], chunk[1], chunk[2]].map(|x| x as f64);
            fluid.add(Particle {
                position: sim.drum.place_inside(position, margin),
                velocity: [chunk[3], chunk[4], chunk[5]].map(|v| v as f64),
                foam: chunk[6].clamp(0.0, 1.0),
            });
        }
    }
    pose_avatar(&snapshot.avatar, sim);
    sim.avatar_mut().solid = settings.collisions;
}

/// A saved site on this ring, unless it is nonsense, in which case it is made a place on it.
fn site_on(saved: Site, ring: Ring) -> Site {
    let grid = Grid::of(ring);
    let half_width = ring.half_width.0 as f64;
    let round = Round {
        cell: saved.round.cell.rem_euclid(grid.round),
        across: finite_f64(saved.round.across).clamp(0.0, 1.0),
    };
    Site::on(
        round,
        finite_f64(saved.y).clamp(-half_width, half_width),
        ring,
    )
}

/// Ask for the water and save once it has arrived.
pub fn save_soon(world: &mut World) {
    let buffers = world.resource::<FluidBuffers>().clone();
    let ticket = world.resource_scope(|world, mut fluid: Mut<Fluid>| {
        let mut commands = world.commands();
        fluid.request_snapshot(&mut commands, &buffers)
    });
    world.flush();
    world.resource_mut::<Saves>().pending = Some(ticket);
}

#[derive(Resource)]
struct Autosave(Timer);

/// A saved resolution, unless it is nonsense, in which case the water starts out fine.
fn water_resolution(saved: Resolution) -> Resolution {
    if saved.spacing.0.is_finite() && saved.spacing.0 >= Resolution::FINEST.spacing.0 {
        saved
    } else {
        Resolution::FINEST
    }
}

fn pose_avatar(pose: &AvatarPose, sim: &mut Simulation) {
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    let q_len = pose.rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(finite(&pose.position)
        && finite(&pose.rotation)
        && finite(&pose.velocity)
        && finite(&pose.angular_velocity)
        && q_len > 0.5)
    {
        return;
    }
    let avatar = sim.avatar_mut();
    avatar.place(pose.position, pose.rotation.map(|x| x / q_len));
    avatar.v = pose.velocity;
    avatar.w = pose.angular_velocity;
    sim.gyros = Gyros::holding(sim.avatar());
}

/// Everything but the sculpted patches, which storage keeps one by one.
fn kept(settings: &Settings, sim: &Simulation, fluid: &Fluid) -> Snapshot {
    let avatar = sim.avatar();
    Snapshot {
        settings: settings.clone(),
        avatar: AvatarPose {
            position: avatar.p,
            rotation: avatar.q,
            velocity: avatar.v,
            angular_velocity: avatar.w,
        },
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        site: sim.drum.site,
        water_site: sim.drum.water,
        time: sim.time,
        water: fluid.resolution(),
        fluid: fluid
            .particles()
            .flat_map(|p| {
                let [x, y, z] = p.position.map(|x| x as f32);
                let [vx, vy, vz] = p.velocity.map(|v| v as f32);
                [x, y, z, vx, vy, vz, p.foam]
            })
            .collect(),
        landscape: Ground {
            base: sim.drum.landscape.base(),
            patches: Vec::new(),
        },
    }
}

/// What storage holds of the ground: the grid its patches lie on, or none when it is to be
/// written afresh; the version of the landscape it has been brought up to; which patches it
/// holds; and how many writes had failed when last told.
#[derive(Resource, Default)]
struct Stored {
    grid: Option<Grid>,
    version: u64,
    patches: HashSet<(i64, i64)>,
    failed: u32,
}

impl Stored {
    /// Bring what storage holds of the ground up to the landscape: whether to clear all it holds
    /// first, the patches to put, each under its key, and the keys to delete.
    fn bring_up(&mut self, landscape: &Landscape) -> (bool, Vec<(String, String)>, Vec<String>) {
        let afresh = self.grid != Some(landscape.grid());
        if afresh {
            self.patches.clear();
        }
        let mut deletes = Vec::new();
        self.patches.retain(|&(round, along)| {
            let kept = landscape.patch(round, along).is_some();
            if !kept {
                deletes.push(patch_key(round, along));
            }
            kept
        });
        let since = if afresh { 0 } else { self.version };
        let mut puts = Vec::new();
        for ((round, along), heights) in landscape.changed_since(since) {
            let patch = Patch {
                round,
                along,
                heights: heights.to_vec(),
            };
            if let Ok(text) = serde_json::to_string(&patch) {
                puts.push((patch_key(round, along), text));
                self.patches.insert((round, along));
            }
        }
        self.grid = Some(landscape.grid());
        self.version = landscape.version();
        (afresh, puts, deletes)
    }
}

fn open() {
    web::storage_open();
}

/// Lay what was kept back down once it has been read. Storage is written afresh at the first
/// save unless all of it was read and laid.
fn restore(world: &mut World) {
    let Some(records) = web::storage_opened() else {
        return;
    };
    world.resource_mut::<Saves>().restored = true;
    let mut snapshot: Option<Snapshot> = None;
    let mut patches = Vec::new();
    let mut whole = true;
    for (key, value) in records {
        if key == SNAPSHOT {
            snapshot = serde_json::from_str(&value).ok();
        } else if let Some(patch) = key
            .starts_with(PATCHES)
            .then(|| serde_json::from_str::<Patch>(&value).ok())
            .flatten()
            .filter(|patch| key == patch_key(patch.round, patch.along))
        {
            patches.push(patch);
        } else {
            whole = false;
        }
    }
    let Some(mut snapshot) = snapshot else {
        return;
    };
    let read = patches.len();
    snapshot.landscape.patches = patches;
    world.resource_scope(|world, mut settings: Mut<Settings>| {
        world.resource_scope(|world, mut sim: Mut<Simulation>| {
            world.resource_scope(|_, mut fluid: Mut<Fluid>| {
                apply(&snapshot, &mut settings, &mut sim, &mut fluid)
            })
        })
    });
    let landscape = &world.resource::<Simulation>().drum.landscape;
    let laid: HashSet<(i64, i64)> = landscape.patches().collect();
    let stored = Stored {
        grid: (whole && laid.len() == read).then_some(landscape.grid()),
        version: landscape.version(),
        patches: laid,
        failed: web::storage_failed(),
    };
    world.insert_resource(stored);
}

/// Count the saves storage has kept, and write it afresh at the next save once one has failed.
fn tally(mut saves: ResMut<Saves>, mut stored: ResMut<Stored>) {
    saves.completed = web::storage_committed();
    let failed = web::storage_failed();
    if failed != stored.failed {
        stored.failed = failed;
        stored.grid = None;
    }
}

fn autosave(world: &mut World) {
    let delta = world.resource::<Time>().delta();
    if world
        .resource_mut::<Autosave>()
        .0
        .tick(delta)
        .just_finished()
    {
        save_soon(world);
    }
}

fn flush(world: &mut World) {
    let Some(ticket) = world.resource::<Saves>().pending else {
        return;
    };
    if !world.resource::<Fluid>().snapshot_ready(ticket) {
        return;
    }
    world.resource_mut::<Saves>().pending = None;
    let kept = kept(
        world.resource::<Settings>(),
        world.resource::<Simulation>(),
        world.resource::<Fluid>(),
    );
    let Ok(text) = serde_json::to_string(&kept) else {
        return;
    };
    world.resource_scope(|world, mut stored: Mut<Stored>| {
        let landscape = &world.resource::<Simulation>().drum.landscape;
        let (clear, mut puts, deletes) = stored.bring_up(landscape);
        puts.push((SNAPSHOT.to_owned(), text));
        web::storage_write(clear, &puts, &deletes);
    });
}

fn patch_key(round: i64, along: i64) -> String {
    format!("{PATCHES}{round},{along}")
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn finite_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}
