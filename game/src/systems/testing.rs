//! Automation hooks: scripts push JSON commands through the platform and read back a status line;
//! plus the headless app the bench and the tests drive frame by frame.
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, OnceLock};

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::diagnostic::{DiagnosticsStore, FrameCount};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::initialize_renderer;
use bevy::render::settings::{Backends, RenderCreation, RenderResources, WgpuSettings};
use bevy::render::storage::ShaderBuffer;
use serde::{Deserialize, Serialize};

use crate::core::fluid::{Fluid, FluidBuffers, FluidReady, MAX_SUBSTEPS_PER_FRAME, ReadOnce};
use crate::core::units::{Radians, RadiansPerSecond, Seconds};
use crate::core::web;
use crate::systems::app;
use crate::systems::controls::pilot;
use crate::systems::hud::FrameRate;
use crate::systems::persistence::{self, Saves};
use crate::systems::player::{PilotInput, Player, PlayerCamera};
use crate::systems::settings::{Dial, Settings};
use crate::systems::sim::{SUBSTEP_RATE, SimSet, Simulation};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum ScriptCommand {
    Inject {
        x: f32,
        y: f32,
        z: f32,
        count: u32,
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
    /// Make the ring another size.
    Ring {
        diameter: f32,
        width: f32,
    },
    /// Set the thrusters' power to what the standing gravity calls for.
    Equalize,
    /// Make the avatar a ghost and put its eye somewhere, looking at a point.
    Camera {
        x: f32,
        y: f32,
        z: f32,
        look_x: f32,
        look_y: f32,
        look_z: f32,
    },
    /// Hold the thrusters at these levels (0 to 1, idle when left out) for a while.
    Thrust {
        #[serde(default)]
        forward: f32,
        #[serde(default)]
        back: f32,
        #[serde(default)]
        left: f32,
        #[serde(default)]
        right: f32,
        #[serde(default)]
        up: f32,
        #[serde(default)]
        down: f32,
        #[serde(default)]
        roll_left: f32,
        #[serde(default)]
        roll_right: f32,
        #[serde(default)]
        pitch_up: f32,
        #[serde(default)]
        pitch_down: f32,
        #[serde(default)]
        yaw_left: f32,
        #[serde(default)]
        yaw_right: f32,
        seconds: f32,
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
    pub spin: RadiansPerSecond,
    pub angle: Radians,
    pub time: Seconds,
    pub fps: f32,
    /// The longest frame of the last second.
    pub worst_frame: Seconds,
    pub sim_rate: f32,
    pub landscape_max: f32,
    /// Saves written to storage so far.
    pub saves: u32,
    /// The avatar's weight in g as the ground pushes back, zero when nothing does.
    pub weight: f32,
    /// The avatar's speed over the ground it stands on, or through the air.
    pub ground_speed: f32,
    pub airborne: bool,
    /// How hard each thruster is firing, in the order of `Thruster::ALL`.
    pub thrust: [f32; 12],
    /// Peak acceleration of a thruster.
    pub thrust_power: f32,
    pub diameter: f32,
    pub width: f32,
    pub avatar: [f32; 3],
    /// Frames whose water coupling the GPU has yet to report, and frames so far in which the
    /// bodies waited for it.
    pub outstanding: u32,
    /// The water's substeps in the last frame.
    pub substeps: usize,
    /// The render passes' smoothed times in ms, on the GPU and the CPU, where the device can
    /// time them.
    pub render: Vec<(String, f32)>,
}

/// Thruster keys a script holds down until a simulated time.
#[derive(Resource, Default)]
struct ScriptedThrust {
    pilot: PilotInput,
    until: Seconds,
}

pub struct TestingPlugin;

impl Plugin for TestingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScriptedThrust>()
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
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [x as f64, y as f64, z as f64],
                    count,
                )
            });
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
        ScriptCommand::Ring { diameter, width } => {
            let mut settings = world.resource_mut::<Settings>();
            Dial::Diameter.set(&mut settings, diameter);
            Dial::Width.set(&mut settings, width);
        }
        ScriptCommand::Equalize => world.resource_mut::<Settings>().equalize_thrust(),
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
                    &mut sim,
                    [x as f64, y as f64, z as f64],
                    [look_x as f64, look_y as f64, look_z as f64],
                );
            });
        }
        ScriptCommand::Thrust {
            forward,
            back,
            left,
            right,
            up,
            down,
            roll_left,
            roll_right,
            pitch_up,
            pitch_down,
            yaw_left,
            yaw_right,
            seconds,
        } => {
            let now = world.resource::<Simulation>().time;
            *world.resource_mut::<ScriptedThrust>() = ScriptedThrust {
                pilot: PilotInput {
                    levels: [
                        forward, back, left, right, up, down, roll_left, roll_right, pitch_up,
                        pitch_down, yaw_left, yaw_right,
                    ],
                },
                until: Seconds(now.0 + seconds),
            };
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
fn steer(thrust: Res<ScriptedThrust>, player: Res<Player>, mut sim: ResMut<Simulation>) {
    if sim.time < thrust.until {
        sim.avatar_input = player.input(thrust.pilot);
    }
}

fn publish(
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    saves: Res<Saves>,
    fps: Res<FrameRate>,
    frame: Res<FrameCount>,
    diagnostics: Option<Res<DiagnosticsStore>>,
) {
    let footing = sim.footing();
    let status = ScriptStatus {
        frame: frame.0,
        particles: fluid.len(),
        litres: fluid.litres().0,
        spin: sim.drum.spin,
        angle: sim.drum.angle,
        time: sim.time,
        fps: fps.fps,
        worst_frame: fps.worst,
        sim_rate: sim.rate,
        landscape_max: sim.drum.landscape.max_height(),
        saves: saves.completed,
        weight: footing.weight,
        ground_speed: footing.ground_speed,
        airborne: footing.airborne,
        thrust: sim.thrusters.levels().map(|l| l as f32),
        thrust_power: sim.thrusters.power.0 as f32,
        diameter: sim.drum.ring.radius.0 * 2.0,
        width: sim.drum.ring.half_width.0 * 2.0,
        avatar: sim.avatar().p.map(|c| c as f32),
        outstanding: fluid.outstanding(),
        substeps: sim.substeps.len(),
        render: diagnostics
            .iter()
            .flat_map(|store| store.iter())
            .filter(|d| d.path().as_str().starts_with("render/"))
            .filter_map(|d| Some((d.path().as_str().to_owned(), d.smoothed()? as f32)))
            .collect(),
    };
    if let Ok(text) = serde_json::to_string(&status) {
        web::publish_status(&text);
    }
}

/// The simulation with rendering into nothing, stepped by hand. No wall-clock time passes in it:
/// the simulation advances only by `run`, so what a test observes never depends on how fast the
/// machine is. One such app exists at a time in a process, and they all draw with the process's
/// one GPU device.
pub fn headless() -> Headless {
    let turn = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut app = app::build_headless(RenderCreation::Manual(device()));
    app.finish();
    app.cleanup();
    app.world_mut().resource_mut::<Time<Virtual>>().pause();
    Headless { app, _turn: turn }
}

/// A headless app and its turn on the GPU, which it keeps until it is dropped.
pub struct Headless {
    app: App,
    _turn: MutexGuard<'static, ()>,
}

impl Deref for Headless {
    type Target = App;

    fn deref(&self) -> &App {
        &self.app
    }
}

impl DerefMut for Headless {
    fn deref_mut(&mut self) -> &mut App {
        &mut self.app
    }
}

static GPU: Mutex<()> = Mutex::new(());

/// What a texture copy aligns the start of every row to.
const ROW_ALIGNMENT: usize = 256;

/// The process's one GPU device, made on first use and kept for the life of the process. A
/// process that makes a device for every app it runs faults the GPU sooner or later: the driver
/// corrupts a later device's command streams once earlier ones have been torn down.
fn device() -> RenderResources {
    static DEVICE: OnceLock<RenderResources> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let settings = WgpuSettings::default();
            let backends = settings.backends.unwrap_or(Backends::all());
            bevy::tasks::block_on(initialize_renderer(backends, None, &settings))
        })
        .clone()
}

/// Run frames until this much simulated time has passed, a frame's worth at a time, each frame
/// waiting for the water's report on the last: the bodies never run ahead of the water, however
/// fast or slow the machine.
pub fn run(app: &mut App, seconds: Seconds) {
    let longest = SUBSTEP_RATE.period().0 * MAX_SUBSTEPS_PER_FRAME as f32;
    let mut left = seconds.0;
    while left > 0.0 {
        let step = left.min(longest);
        left -= step;
        frame(app, Seconds(step));
        settle(app);
    }
}

/// Run one frame of this much simulated time without waiting for the water's report on it.
pub fn frame(app: &mut App, seconds: Seconds) {
    app.world_mut()
        .resource_mut::<Simulation>()
        .request(seconds);
    app.update();
}

/// Run frames that simulate nothing until the water has reported on every frame issued.
pub fn settle(app: &mut App) {
    let mut idle = 0;
    loop {
        let ready = app.world().resource::<FluidReady>().get();
        let queued = app.world().resource::<Simulation>().queued().0 > 0.0;
        let outstanding = app.world().resource::<Fluid>().outstanding() > 0;
        if ready && !queued && !outstanding {
            return;
        }
        app.update();
        idle += 1;
        assert!(idle < 2000, "the water never reported back");
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

/// Run frames until the image has been rendered and read back: RGBA bytes, row by row, with
/// the padding the copy aligned each row to taken out.
pub fn capture(app: &mut App, image: &Handle<Image>) -> Vec<u8> {
    let padded = read_back(app, Readback::texture(image.clone()));
    let width = app
        .world()
        .resource::<Assets<Image>>()
        .get(image)
        .map_or(0, |image| image.width() as usize);
    let row = width * 4;
    let stride = row.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
    padded
        .chunks(stride)
        .flat_map(|line| line[..row.min(line.len())].iter().copied())
        .collect()
}

/// The water's surface as last extracted: each vertex's position with its foam, read back from
/// the GPU.
pub fn surface_vertices(app: &mut App) -> Vec<[f32; 4]> {
    let surface = app.world().resource::<FluidBuffers>().surface.clone();
    let metres = app.world().resource::<Fluid>().resolution().length() as f32;
    let count = read_u32s(app, surface.counters.clone())
        .first()
        .copied()
        .unwrap_or(0) as usize;
    read_back(app, Readback::buffer(surface.vertices))
        .chunks_exact(32)
        .take(count)
        .map(|v| {
            let f = |i: usize| f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [f(0) * metres, f(4) * metres, f(8) * metres, f(12)]
        })
        .collect()
}

/// How many triangles the water's surface currently has, read back from the GPU.
pub fn surface_triangles(app: &mut App) -> u32 {
    surface_demand(app).indices / 3
}

/// What the last extraction of the water's surface asked for, whether or not it fit.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceDemand {
    pub vertices: u32,
    pub indices: u32,
    pub blocks: u32,
}

pub fn surface_demand(app: &mut App) -> SurfaceDemand {
    let counters = app
        .world()
        .resource::<FluidBuffers>()
        .surface
        .counters
        .clone();
    let counts = read_u32s(app, counters);
    let at = |i: usize| counts.get(i).copied().unwrap_or(0);
    SurfaceDemand {
        vertices: at(0),
        indices: at(1),
        blocks: at(2),
    }
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
    app.world_mut().spawn((readback, ReadOnce)).observe(
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
