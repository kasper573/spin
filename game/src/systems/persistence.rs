//! Everything worth keeping across a reload — settings, avatar, drum, water, landscape —
//! as one JSON snapshot in the browser's storage, written every couple of seconds. The water
//! lives on the GPU, so a save first asks for a copy and writes when it arrives.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::avatar::Gyros;
use crate::core::codec;
use crate::core::fluid::{Fluid, FluidBuffers, Particle, Resolution};
use crate::core::math::{cross, rotate_y};
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::web;
use crate::systems::drum::Site;
use crate::systems::settings::Settings;
use crate::systems::sim::{SimSet, Simulation};

const KEY: &str = "spin-gravity-wheel/v7";
const AUTOSAVE_INTERVAL: Seconds = Seconds(2.0);

/// Snapshots before this version kept the water and the avatar in the world's frame rather
/// than the drum's; the avatar of such a snapshot is stood back on the ground.
const DRUM_FRAME: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Snapshot {
    #[serde(default)]
    pub version: u32,
    pub settings: Settings,
    pub avatar: AvatarPose,
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    /// The point of the wall the avatar is measured from.
    #[serde(default)]
    pub site: Site,
    pub time: Seconds,
    /// How much water each particle stands for.
    pub water: Resolution,
    /// Seven floats per particle, in the drum's frame: position, velocity, foam.
    #[serde(with = "codec::f32s")]
    pub fluid: Vec<f32>,
    #[serde(with = "codec::f32s")]
    pub landscape: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct AvatarPose {
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
}

/// Saves requested and completed so far.
#[derive(Resource, Default)]
pub struct Saves {
    pub completed: u32,
    pending: Option<u32>,
}

pub struct PersistencePlugin;

impl Plugin for PersistencePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Autosave(Timer::from_seconds(
            AUTOSAVE_INTERVAL.0,
            TimerMode::Repeating,
        )))
        .init_resource::<Saves>()
        .add_systems(PostStartup, restore)
        .add_systems(Update, (autosave, flush).chain().in_set(SimSet::Observe));
    }
}

pub fn snapshot(settings: &Settings, sim: &Simulation, fluid: &Fluid) -> Snapshot {
    let avatar = sim.avatar();
    Snapshot {
        version: DRUM_FRAME,
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
        landscape: sim.drum.landscape.heights().to_vec(),
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
    let half_width = sim.drum.ring.half_width.0 as f64;
    sim.drum.site = Site {
        phi: finite_f64(snapshot.site.phi).rem_euclid(std::f64::consts::TAU),
        y: finite_f64(snapshot.site.y).clamp(-half_width, half_width),
    };
    sim.time = Seconds(finite(snapshot.time.0));
    sim.drum.landscape.load(&snapshot.landscape);
    let margin = fluid.resolution().margin().0 as f64;
    let legacy = snapshot.version < DRUM_FRAME;
    let angle = sim.drum.angle.0;
    let spin = sim.drum.spin.0 as f64;
    for chunk in snapshot.fluid.chunks_exact(7) {
        if chunk.iter().all(|v| v.is_finite()) {
            let mut position = [chunk[0], chunk[1], chunk[2]].map(|x| x as f64);
            let mut velocity = [chunk[3], chunk[4], chunk[5]].map(|v| v as f64);
            if legacy {
                let carried = cross(&[0.0, spin, 0.0], &position);
                velocity = rotate_y(
                    &[
                        velocity[0] - carried[0],
                        velocity[1] - carried[1],
                        velocity[2] - carried[2],
                    ],
                    -angle,
                );
                position = rotate_y(&position, -angle);
            }
            fluid.add(Particle {
                position: sim.drum.place_inside(position, margin),
                velocity,
                foam: chunk[6].clamp(0.0, 1.0),
            });
        }
    }
    if !legacy {
        pose_avatar(&snapshot.avatar, sim);
    }
    sim.avatar_mut().solid = settings.collisions;
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

fn restore(world: &mut World) {
    let Some(snapshot) =
        web::storage_load(KEY).and_then(|text| serde_json::from_str::<Snapshot>(&text).ok())
    else {
        return;
    };
    world.resource_scope(|world, mut settings: Mut<Settings>| {
        world.resource_scope(|world, mut sim: Mut<Simulation>| {
            world.resource_scope(|_, mut fluid: Mut<Fluid>| {
                apply(&snapshot, &mut settings, &mut sim, &mut fluid)
            })
        })
    });
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
    let snapshot = snapshot(
        world.resource::<Settings>(),
        world.resource::<Simulation>(),
        world.resource::<Fluid>(),
    );
    if let Ok(text) = serde_json::to_string(&snapshot) {
        web::storage_save(KEY, &text);
    }
    let mut saves = world.resource_mut::<Saves>();
    saves.pending = None;
    saves.completed += 1;
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn finite_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}
