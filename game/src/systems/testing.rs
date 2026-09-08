//! Automation hooks: scripts push JSON commands through the platform and read back a status line;
//! plus the headless app the bench and the tests drive frame by frame.
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::storage::ShaderBuffer;
use serde::{Deserialize, Serialize};

use crate::core::avatar::Look;
use crate::core::fluid::{Fluid, FluidBuffers, FluidReady};
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::web;
use crate::systems::app;
use crate::systems::controls::pilot;
use crate::systems::hud::FrameRate;
use crate::systems::persistence::{self, Saves};
use crate::systems::player::{PilotAxes, Player, PlayerCamera};
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
    /// Make the avatar a ghost and put its eye somewhere, looking at a point.
    Camera {
        x: f32,
        y: f32,
        z: f32,
        look_x: f32,
        look_y: f32,
        look_z: f32,
    },
    /// Hold the movement keys for a while: `x` right and `z` backward in the head's level frame,
    /// each in [-1, 1].
    Walk {
        x: f32,
        z: f32,
        run: bool,
        jump: bool,
        seconds: f32,
    },
    /// Turn the head to an absolute yaw and pitch relative to the hull.
    Look {
        yaw: f64,
        pitch: f64,
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
    /// Saves written to storage so far.
    pub saves: u32,
    /// Each raft's distance from the axis and its tangential speed relative to the glass.
    pub raft_slip: Vec<[f32; 2]>,
    /// The avatar's weight in g as the ground pushes back, zero when nothing does.
    pub weight: f32,
    /// The avatar's speed over the ground it stands on, or through the air.
    pub ground_speed: f32,
    pub airborne: bool,
}

/// Movement keys a script holds down until a simulated time.
#[derive(Resource, Default)]
struct ScriptedWalk {
    axes: PilotAxes,
    until: Seconds,
}

pub struct TestingPlugin;

impl Plugin for TestingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScriptedWalk>()
            .add_systems(
                Update,
                (drain, steer).chain().in_set(SimSet::Command).after(pilot),
            )
            .add_systems(Update, publish.in_set(SimSet::Observe));
    }
}

fn drain(world: &mut World) {
    let commands = web::take_script_commands();
    for text in commands {
        match serde_json::from_str::<ScriptCommand>(&text) {
            Ok(command) => execute(world, command),
            Err(error) => warn!("ignored script command {text}: {error}"),
        }
    }
}

fn execute(world: &mut World, command: ScriptCommand) {
    match command {
        ScriptCommand::Inject { x, y, z, count } => {
            world.resource_scope(|world, mut fluid: Mut<Fluid>| {
                world
                    .resource::<Simulation>()
                    .inject(&mut fluid, [x, y, z], count)
            });
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
            world.resource_mut::<Settings>().collisions = false;
            world.resource_scope(|world, mut player: Mut<Player>| {
                let mut sim = world.resource_mut::<Simulation>();
                sim.avatar_mut().solid = false;
                player.teleport(
                    sim.avatar_mut(),
                    [x as f64, y as f64, z as f64],
                    [look_x as f64, look_y as f64, look_z as f64],
                );
            });
        }
        ScriptCommand::Walk {
            x,
            z,
            run,
            jump,
            seconds,
        } => {
            let now = world.resource::<Simulation>().time;
            *world.resource_mut::<ScriptedWalk>() = ScriptedWalk {
                axes: PilotAxes {
                    motion: Vec3::new(x, 0.0, z),
                    roll: 0.0,
                    run,
                    jump,
                },
                until: Seconds(now.0 + seconds),
            };
        }
        ScriptCommand::Look { yaw, pitch } => {
            world.resource_mut::<Player>().look = Look { yaw, pitch };
        }
        ScriptCommand::Advance { seconds } => {
            world.resource_mut::<Simulation>().request(Seconds(seconds))
        }
        ScriptCommand::Reset => {
            world.resource_mut::<Simulation>().reset();
            world.resource_mut::<Fluid>().clear();
        }
        ScriptCommand::Save => persistence::save_soon(world),
    }
}

/// Keep the scripted keys held until their time is up.
fn steer(walk: Res<ScriptedWalk>, player: Res<Player>, mut sim: ResMut<Simulation>) {
    if sim.time < walk.until {
        sim.avatar_input = player.input(sim.avatar(), walk.axes);
    }
}

fn publish(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    saves: Res<Saves>,
    fps: Res<FrameRate>,
    frame: Res<FrameCount>,
) {
    let footing = sim.footing();
    let status = ScriptStatus {
        frame: frame.0,
        particles: fluid.len(),
        litres: Simulation::water(&fluid).0,
        rafts: sim.rafts().len(),
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        time: sim.time,
        fps: fps.0,
        sim_rate: sim.rate,
        landscape_max: sim.drum.landscape.max_height(),
        saves: saves.completed,
        raft_slip: sim
            .rafts()
            .iter()
            .map(|b| {
                let r = (b.p[0] * b.p[0] + b.p[2] * b.p[2]).sqrt().max(1e-9);
                let tangential = (b.v[0] * b.p[2] - b.v[2] * b.p[0]) / r;
                [r as f32, (tangential - sim.drum.spin.0 as f64 * r) as f32]
            })
            .collect(),
        weight: footing.weight,
        ground_speed: footing.ground_speed,
        airborne: footing.airborne,
    };
    if let Ok(text) = serde_json::to_string(&status) {
        web::publish_status(&text);
    }
}

/// The simulation with rendering into nothing, stepped by hand.
pub fn headless() -> App {
    let mut app = app::build_headless();
    app.finish();
    app.cleanup();
    app
}

/// Run frames until this much simulated time has passed.
pub fn run(app: &mut App, seconds: Seconds) {
    app.world_mut()
        .resource_mut::<Simulation>()
        .request(seconds);
    let mut idle = 0;
    while app.world().resource::<Simulation>().queued().0 > 0.0 {
        app.update();
        if app.world().resource::<FluidReady>().get() {
            idle = 0;
        } else {
            idle += 1;
            assert!(idle < 2000, "the water's shaders never became ready");
        }
    }
}

/// Run frames until a fresh copy of the water has been read back, then return it.
pub fn particles(app: &mut App) -> Vec<crate::core::fluid::Particle> {
    let buffers = app.world().resource::<FluidBuffers>().clone();
    let ticket = app
        .world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut commands = world.commands();
            fluid.request_snapshot(&mut commands, &buffers)
        });
    for _ in 0..600 {
        app.update();
        if app.world().resource::<Fluid>().snapshot_ready(ticket) {
            break;
        }
    }
    app.world().resource::<Fluid>().particles().collect()
}

/// Point the player's camera at an offscreen image of this size, for reading frames back.
pub fn render_to_image(app: &mut App, width: u32, height: u32) -> Handle<Image> {
    app.update();
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    let handle = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let mut cameras = app
        .world_mut()
        .query_filtered::<&mut RenderTarget, With<PlayerCamera>>();
    for mut target in cameras.iter_mut(app.world_mut()) {
        *target = RenderTarget::from(handle.clone());
    }
    handle
}

/// Run frames until the image has been rendered and read back: RGBA bytes, row by row.
pub fn capture(app: &mut App, image: &Handle<Image>) -> Vec<u8> {
    read_back(app, Readback::texture(image.clone()))
}

/// How many triangles the water's surface currently has, read back from the GPU.
pub fn surface_triangles(app: &mut App) -> u32 {
    let counters = app
        .world()
        .resource::<FluidBuffers>()
        .surface
        .counters
        .clone();
    read_u32s(app, counters).get(1).copied().unwrap_or(0) / 3
}

#[derive(Resource, Default)]
struct ReadResult(Option<Vec<u8>>);

fn read_u32s(app: &mut App, buffer: Handle<ShaderBuffer>) -> Vec<u32> {
    read_back(app, Readback::buffer(buffer))
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_back(app: &mut App, readback: Readback) -> Vec<u8> {
    app.world_mut().insert_resource(ReadResult::default());
    app.world_mut().spawn(readback).observe(
        |event: On<ReadbackComplete>, mut result: ResMut<ReadResult>, mut commands: Commands| {
            result.0 = Some(event.data.clone());
            commands.entity(event.entity).try_despawn();
        },
    );
    for _ in 0..600 {
        app.update();
        if app.world().resource::<ReadResult>().0.is_some() {
            break;
        }
    }
    app.world_mut()
        .remove_resource::<ReadResult>()
        .and_then(|r| r.0)
        .unwrap_or_default()
}
