//! Automation hooks: scripts push JSON commands through the platform and read back a status line.
use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::web;
use crate::systems::hud::FrameRate;
use crate::systems::persistence;
use crate::systems::player::Player;
use crate::systems::settings::Settings;
use crate::systems::sim::{SimSet, Simulation};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum ScriptCommand {
    Inject {
        x: f32,
        y: f32,
        z: f32,
        count: u32,
    },
    Raft {
        x: f64,
        y: f64,
        z: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    },
    Sculpt {
        phi: f64,
        y: f64,
        radius: f64,
        amount: f64,
    },
    Spin {
        value: f32,
    },
    Camera {
        x: f32,
        y: f32,
        z: f32,
        look_x: f32,
        look_y: f32,
        look_z: f32,
    },
    Advance {
        seconds: f32,
    },
    Reset,
    Save,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ScriptStatus {
    /// Frames rendered so far; a command queued during one frame has run by the second after it.
    pub frame: u32,
    pub particles: usize,
    pub litres: f32,
    pub rafts: usize,
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    pub time: Seconds,
    pub fps: f32,
    pub sim_rate: f32,
    pub landscape_max: f32,
    /// Each raft's distance from the axis and its tangential speed relative to the glass.
    pub raft_slip: Vec<[f32; 2]>,
}

pub struct TestingPlugin;

impl Plugin for TestingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, drain.in_set(SimSet::Command))
            .add_systems(Update, publish.in_set(SimSet::Observe));
    }
}

fn drain(world: &mut World) {
    let commands = web::take_script_commands();
    for text in commands {
        match serde_json::from_str::<ScriptCommand>(&text) {
            Ok(command) => run(world, command),
            Err(error) => warn!("ignored script command {text}: {error}"),
        }
    }
}

fn run(world: &mut World, command: ScriptCommand) {
    match command {
        ScriptCommand::Inject { x, y, z, count } => {
            world.resource_mut::<Simulation>().inject([x, y, z], count);
        }
        ScriptCommand::Raft {
            x,
            y,
            z,
            nx,
            ny,
            nz,
        } => {
            world
                .resource_mut::<Simulation>()
                .spawn_raft([x, y, z], [nx, ny, nz]);
        }
        ScriptCommand::Sculpt {
            phi,
            y,
            radius,
            amount,
        } => world
            .resource_mut::<Simulation>()
            .drum
            .landscape
            .sculpt(phi, y, radius, amount),
        ScriptCommand::Spin { value } => {
            world.resource_mut::<Settings>().spin = RadiansPerSecond(value);
            world.resource_mut::<Simulation>().drum.target_spin = RadiansPerSecond(value);
        }
        ScriptCommand::Camera {
            x,
            y,
            z,
            look_x,
            look_y,
            look_z,
        } => {
            world.resource_scope(|world, mut player: Mut<Player>| {
                player.teleport(
                    world.resource_mut::<Simulation>().shuttle_mut(),
                    [x as f64, y as f64, z as f64],
                    [look_x as f64, look_y as f64, look_z as f64],
                );
            });
        }
        ScriptCommand::Advance { seconds } => world
            .resource_mut::<Simulation>()
            .advance_exact(Seconds(seconds)),
        ScriptCommand::Reset => world.resource_mut::<Simulation>().reset(),
        ScriptCommand::Save => persistence::save_now(world),
    }
}

fn publish(sim: Res<Simulation>, fps: Res<FrameRate>, frame: Res<FrameCount>) {
    let status = ScriptStatus {
        frame: frame.0,
        particles: sim.fluid.len(),
        litres: sim.water().0,
        rafts: sim.rafts().len(),
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        time: sim.time,
        fps: fps.0,
        sim_rate: sim.rate,
        landscape_max: sim.drum.landscape.max_height(),
        raft_slip: sim
            .rafts()
            .iter()
            .map(|b| {
                let r = (b.p[0] * b.p[0] + b.p[2] * b.p[2]).sqrt().max(1e-9);
                let tangential = (b.v[0] * b.p[2] - b.v[2] * b.p[0]) / r;
                [r as f32, (tangential - sim.drum.spin.0 as f64 * r) as f32]
            })
            .collect(),
    };
    if let Ok(text) = serde_json::to_string(&status) {
        web::publish_status(&text);
    }
}
