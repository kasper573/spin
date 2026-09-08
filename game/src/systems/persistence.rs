//! Everything worth keeping across a reload — settings, shuttle, drum, water, rafts, landscape —
//! as one JSON snapshot in the browser's storage, written every couple of seconds.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::codec;
use crate::core::fluid::{PARTICLE_SPACING, Particle};
use crate::core::rigid::Body;
use crate::core::shuttle::Look;
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::vessel::Vessel;
use crate::core::web;
use crate::systems::player::Player;
use crate::systems::settings::Settings;
use crate::systems::sim::{SimSet, Simulation};

const KEY: &str = "spin-gravity-wheel/v5";
const AUTOSAVE_INTERVAL: Seconds = Seconds(2.0);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub settings: Settings,
    pub shuttle: ShuttlePose,
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    pub time: Seconds,
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
pub struct ShuttlePose {
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub look: Look,
}

pub struct PersistencePlugin;

impl Plugin for PersistencePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Autosave(Timer::from_seconds(
            AUTOSAVE_INTERVAL.0,
            TimerMode::Repeating,
        )))
        .add_systems(PostStartup, restore)
        .add_systems(Update, autosave.in_set(SimSet::Observe));
    }
}

pub fn snapshot(settings: &Settings, sim: &Simulation, player: &Player) -> Snapshot {
    let shuttle = sim.shuttle();
    Snapshot {
        settings: settings.clone(),
        shuttle: ShuttlePose {
            position: shuttle.p,
            rotation: shuttle.q,
            velocity: shuttle.v,
            angular_velocity: shuttle.w,
            look: player.look,
        },
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        time: sim.time,
        fluid: sim
            .fluid
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
    player: &mut Player,
) {
    *settings = snapshot.settings.clone().sanitized();
    sim.reset();
    sim.drum.spin = RadiansPerSecond(finite(snapshot.spin.0).clamp(-10.0, 10.0));
    sim.drum.target_spin = settings.spin;
    sim.drum.angle = Radians(finite_f64(snapshot.angle.0));
    sim.time = Seconds(finite(snapshot.time.0));
    sim.drum.landscape.load(&snapshot.landscape);
    for chunk in snapshot.fluid.chunks_exact(7) {
        if chunk.iter().all(|v| v.is_finite()) {
            let mut position = [chunk[0], chunk[1], chunk[2]];
            sim.drum.confine(&mut position, PARTICLE_SPACING * 0.5);
            sim.fluid.add(Particle {
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
    pose_shuttle(&snapshot.shuttle, sim.shuttle_mut(), player);
    sim.shuttle_mut().solid = settings.collisions;
}

pub fn save_now(world: &mut World) {
    let snapshot = snapshot(
        world.resource::<Settings>(),
        world.resource::<Simulation>(),
        world.resource::<Player>(),
    );
    if let Ok(text) = serde_json::to_string(&snapshot) {
        web::storage_save(KEY, &text);
    }
}

#[derive(Resource)]
struct Autosave(Timer);

fn pose_shuttle(pose: &ShuttlePose, shuttle: &mut Body, player: &mut Player) {
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    let q_len = pose.rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(finite(&pose.position)
        && finite(&pose.rotation)
        && finite(&pose.velocity)
        && finite(&pose.angular_velocity)
        && pose.look.yaw.is_finite()
        && pose.look.pitch.is_finite()
        && q_len > 0.5)
    {
        return;
    }
    shuttle.place(pose.position, pose.rotation.map(|x| x / q_len));
    shuttle.v = pose.velocity;
    shuttle.w = pose.angular_velocity;
    player.look = pose.look;
}

fn restore(world: &mut World) {
    let Some(snapshot) =
        web::storage_load(KEY).and_then(|text| serde_json::from_str::<Snapshot>(&text).ok())
    else {
        return;
    };
    world.resource_scope(|world, mut settings: Mut<Settings>| {
        world.resource_scope(|world, mut sim: Mut<Simulation>| {
            world.resource_scope(|_, mut player: Mut<Player>| {
                apply(&snapshot, &mut settings, &mut sim, &mut player)
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
        save_now(world);
    }
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn finite_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}
