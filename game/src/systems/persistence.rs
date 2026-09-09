//! Everything worth keeping across a reload — settings, avatar, drum, water, rafts, landscape —
//! as one JSON snapshot in the browser's storage, written every couple of seconds. The water
//! lives on the GPU, so a save first asks for a copy and writes when it arrives.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::avatar::Gyros;
use crate::core::codec;
use crate::core::fluid::{Fluid, FluidBuffers, Particle, Resolution};
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::vessel::Vessel;
use crate::core::web;
use crate::systems::settings::Settings;
use crate::systems::sim::{SimSet, Simulation};

const KEY: &str = "spin-gravity-wheel/v7";
const AUTOSAVE_INTERVAL: Seconds = Seconds(2.0);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub settings: Settings,
    pub avatar: AvatarPose,
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    pub time: Seconds,
    /// How much water each particle stands for.
    pub water: Resolution,
    /// Seven floats per particle: position, velocity, foam.
    #[serde(with = "codec::f32s")]
    pub fluid: Vec<f32>,
    /// Thirteen floats per raft: position, orientation (x y z w), velocity, angular velocity.
    #[serde(with = "codec::f32s")]
    pub rafts: Vec<f32>,
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
        settings: settings.clone(),
        avatar: AvatarPose {
            position: avatar.p,
            rotation: avatar.q,
            velocity: avatar.v,
            angular_velocity: avatar.w,
        },
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        time: sim.time,
        water: fluid.resolution(),
        fluid: fluid
            .particles()
            .flat_map(|p| {
                let [x, y, z] = p.position;
                let [vx, vy, vz] = p.velocity;
                [x, y, z, vx, vy, vz, p.foam]
            })
            .collect(),
        rafts: sim
            .rafts()
            .iter()
            .flat_map(|b| {
                [
                    b.p[0], b.p[1], b.p[2], b.q[0], b.q[1], b.q[2], b.q[3], b.v[0], b.v[1], b.v[2],
                    b.w[0], b.w[1], b.w[2],
                ]
                .map(|v| v as f32)
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
    sim.time = Seconds(finite(snapshot.time.0));
    sim.drum.landscape.load(&snapshot.landscape);
    let margin = fluid.resolution().margin();
    for chunk in snapshot.fluid.chunks_exact(7) {
        if chunk.iter().all(|v| v.is_finite()) {
            let mut position = [chunk[0], chunk[1], chunk[2]];
            sim.drum.confine(&mut position, margin);
            fluid.add(Particle {
                position,
                velocity: [chunk[3], chunk[4], chunk[5]],
                foam: chunk[6].clamp(0.0, 1.0),
            });
        }
    }
    for chunk in snapshot.rafts.chunks_exact(13) {
        if chunk.iter().all(|v| v.is_finite()) {
            let f = |i: usize| chunk[i] as f64;
            sim.spawn_raft([f(0), f(1), f(2)], [0.0, 1.0, 0.0]);
            if let Some(raft) = sim.rafts_mut().last_mut() {
                raft.place([f(0), f(1), f(2)], [f(3), f(4), f(5), f(6)]);
                raft.v = [f(7), f(8), f(9)];
                raft.w = [f(10), f(11), f(12)];
            }
        }
    }
    pose_avatar(&snapshot.avatar, sim);
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
    let finest = Resolution::FINEST.spacing.0;
    if saved.spacing.0.is_finite() && saved.spacing.0 >= finest && saved.spacing.0 < finest * 64.0 {
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
