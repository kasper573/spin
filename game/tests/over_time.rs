//! How the world behaves over time, played as a player plays it, through the thrusters' keys:
//! what happens does not depend on how fast the machine draws it.
use bevy::prelude::*;
use game::core::avatar::Thruster;
use game::core::fluid::Fluid;
use game::core::math::{Vec3d, quat_rotate};
use game::core::sheet::Sheet;
use game::core::units::Seconds;
use game::systems::drum::{CELL, Drum, Ring, Site};
use game::systems::player::{PilotInput, Player};
use game::systems::sim::Simulation;
use game::systems::testing;
use game::systems::tools::Toolbelt;

/// Frame rates a player's machine may draw at, from a slow one to a fast display's.
const RATES: [f64; 3] = [60.0, 144.0, 240.0];
/// A flight: turning on the spot, walking, thrusting up off the ground and coming down again,
/// each for a whole number of frames at every rate.
const FLIGHT: [(Option<Thruster>, f64); 4] = [
    (Some(Thruster::YawLeft), 0.5),
    (Some(Thruster::Forward), 1.5),
    (Some(Thruster::Up), 0.75),
    (None, 1.0),
];
/// How far apart the same flight may end at two rates: what a single step of the bodies at a
/// walk carries the eye, and turns it by.
const SAME_PLACE_M: f64 = 0.01;
const SAME_HEADING_RAD: f64 = 0.01;
/// A display faster than the bodies are stepped: most of its frames fall between two substeps.
const FAST_DISPLAY: f64 = 240.0;
/// How much the eye's move from one frame to the next may change, as a share of its mean move,
/// while walking on: a hitch would show as whole frames without a move.
const EVEN_MOVE: f64 = 0.25;
/// A machine too slow for the world to keep up with real time: each of its frames lasts longer
/// than the few substeps a frame may take.
const SLOW_MACHINE: f64 = 10.0;
/// The tools that put something out while held: which slot of the belt each is in, and how much
/// of what it puts out the world holds.
const PUTTING_OUT: [Tool; 2] = [
    Tool {
        name: "the water tool",
        slot: 0,
        amount: water_m3,
    },
    Tool {
        name: "the land tool",
        slot: 1,
        amount: ground_m3,
    },
];
/// How long the player looks down at their feet before a tool is worked there.
const LOOKING_DOWN: f64 = 0.5;
/// How long a tool is held, in the world's time.
const HELD: f64 = 2.0;
/// How long the world is left after a tool is put down for what it put out to be counted.
const SETTLING: Seconds = Seconds(0.25);
/// How far what a tool puts out may differ at two frame rates, as a share of it: a frame's worth
/// of the slow machine's.
const SAME_AMOUNT: f64 = 0.02;

#[test]
#[ignore = "wants a GPU"]
fn a_flight_ends_where_it_would_at_any_frame_rate() {
    let ends: Vec<Flown> = RATES.iter().map(|&fps| flown_at(fps)).collect();
    let first = &ends[0];
    for (fps, other) in RATES.iter().zip(&ends).skip(1) {
        let (eye, ahead) = other.seen_from(first);
        let apart = (0..3)
            .map(|k| (eye[k] - first.eye[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let turned = (0..3)
            .map(|k| ahead[k] * first.ahead[k])
            .sum::<f64>()
            .clamp(-1.0, 1.0)
            .acos();
        eprintln!(
            "at {fps} fps the flight ends {apart:.4} m and {turned:.4} rad from where it does at {} fps",
            RATES[0]
        );
        assert!(
            apart < SAME_PLACE_M && turned < SAME_HEADING_RAD,
            "at {fps} fps the flight ends {apart:.3} m and {turned:.3} rad from where it does at {} fps",
            RATES[0]
        );
    }
}

#[test]
#[ignore = "wants a GPU"]
fn a_fast_display_sees_the_eye_move_evenly() {
    let mut app = testing::headless();
    testing::watch(&mut app, Seconds(0.5));
    let walk = [Thruster::Forward];
    let frames = (FLIGHT[1].1 * FAST_DISPLAY).round() as usize;
    let mut seen = Vec::with_capacity(frames);
    for _ in 0..frames {
        let player = *app.world().resource::<Player>();
        app.world_mut().resource_mut::<Simulation>().avatar_input =
            player.input(PilotInput::firing(&walk));
        testing::frame_as_played(&mut app, Seconds((1.0 / FAST_DISPLAY) as f32));
        let sim = app.world().resource::<Simulation>();
        seen.push((sim.drum.site, sim.eye().0));
    }
    let (first, _) = seen[0];
    let mut drum = Drum::new(app.world().resource::<Simulation>().drum.ring);
    let path: Vec<Vec3d> = seen
        .iter()
        .map(|&(site, eye)| {
            drum.site = first;
            drum.frame_at(site).point_back(eye)
        })
        .collect();
    let moves: Vec<f64> = path
        .windows(2)
        .map(|w| {
            (0..3)
                .map(|k| (w[1][k] - w[0][k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .skip(frames / 2)
        .collect();
    let mean = moves.iter().sum::<f64>() / moves.len() as f64;
    let uneven = moves
        .windows(2)
        .map(|w| (w[1] - w[0]).abs() / mean)
        .fold(0.0, f64::max);
    eprintln!(
        "walking at {FAST_DISPLAY} fps the eye moves {mean:.5} m a frame, changing by up to {uneven:.3} of it"
    );
    assert!(
        uneven < EVEN_MOVE,
        "walking at {FAST_DISPLAY} fps the eye's move changes by {uneven:.3} of its mean from one frame to the next"
    );
}

#[test]
#[ignore = "wants a GPU"]
fn a_tool_puts_out_as_much_at_any_frame_rate() {
    for tool in PUTTING_OUT {
        let name = tool.name;
        let fast = held_at(RATES[0], &tool);
        let slow = held_at(SLOW_MACHINE, &tool);
        let off = (slow - fast).abs() / fast.abs().max(1e-9);
        eprintln!(
            "{name} held for {HELD} s of the world's time puts out {fast:.4} m3 at {} fps and {slow:.4} m3 at {SLOW_MACHINE} fps",
            RATES[0]
        );
        assert!(
            fast.abs() > 0.0 && off < SAME_AMOUNT,
            "{name} held for {HELD} s of the world's time puts out {fast:.4} m3 at {} fps and {slow:.4} m3 at {SLOW_MACHINE} fps",
            RATES[0]
        );
    }
}

struct Tool {
    name: &'static str,
    slot: usize,
    amount: fn(&App) -> f64,
}

/// What a tool puts out, held for [`HELD`] of the world's time with frames drawn this many times
/// a second, after the player has looked down at their feet.
fn held_at(fps: f64, tool: &Tool) -> f64 {
    let mut app = testing::headless();
    testing::watch(&mut app, Seconds(0.5));
    testing::tap(&mut app, Toolbelt::key(tool.slot));
    let look_down = [Thruster::PitchDown];
    for _ in 0..(LOOKING_DOWN * RATES[0]).round() as u32 {
        let player = *app.world().resource::<Player>();
        app.world_mut().resource_mut::<Simulation>().avatar_input =
            player.input(PilotInput::firing(&look_down));
        testing::frame_as_played(&mut app, Seconds((1.0 / RATES[0]) as f32));
    }
    let player = *app.world().resource::<Player>();
    app.world_mut().resource_mut::<Simulation>().avatar_input =
        player.input(PilotInput::firing(&[]));
    let before = (tool.amount)(&app);
    let start = app.world().resource::<Simulation>().time.0 as f64;
    testing::button(&mut app, MouseButton::Left, true);
    while (app.world().resource::<Simulation>().time.0 as f64) - start < HELD - 1e-3 {
        testing::frame_as_played(&mut app, Seconds((1.0 / fps) as f32));
    }
    testing::button(&mut app, MouseButton::Left, false);
    testing::watch(&mut app, SETTLING);
    (tool.amount)(&app) - before
}

fn water_m3(app: &App) -> f64 {
    let flying = app.world().resource::<Fluid>().litres().0;
    let lying = app.world().resource::<Sheet>().held().0;
    (flying + lying) as f64 / 1000.0
}

/// How much ground stands above the flat ground the ring starts with.
fn ground_m3(app: &App) -> f64 {
    let landscape = &app.world().resource::<Simulation>().drum.landscape;
    let base = landscape.base() as f64;
    landscape
        .patches()
        .filter_map(|(round, along)| landscape.patch(round, along))
        .flatten()
        .map(|&height| (height as f64 - base) * CELL * CELL)
        .sum()
}

/// Where a flight ended: the eye and the way it looks, in the frame about the site of the wall
/// the simulation had moved to.
struct Flown {
    ring: Ring,
    site: Site,
    eye: Vec3d,
    ahead: Vec3d,
}

impl Flown {
    /// The eye and the way it looks in the frame the other flight ended in.
    fn seen_from(&self, other: &Flown) -> (Vec3d, Vec3d) {
        let mut drum = Drum::new(self.ring);
        drum.site = other.site;
        let frame = drum.frame_at(self.site);
        (frame.point_back(self.eye), frame.vector_back(self.ahead))
    }
}

/// Fly the flight with frames drawn this many times a second.
fn flown_at(fps: f64) -> Flown {
    let mut app = testing::headless();
    testing::watch(&mut app, Seconds(0.5));
    for (thruster, seconds) in FLIGHT {
        let held: Vec<Thruster> = thruster.into_iter().collect();
        for _ in 0..(seconds * fps).round() as u32 {
            let player = *app.world().resource::<Player>();
            app.world_mut().resource_mut::<Simulation>().avatar_input =
                player.input(PilotInput::firing(&held));
            testing::frame_as_played(&mut app, Seconds((1.0 / fps) as f32));
        }
    }
    let sim = app.world().resource::<Simulation>();
    let (eye, attitude) = sim.eye();
    Flown {
        ring: sim.drum.ring,
        site: sim.drum.site,
        eye,
        ahead: quat_rotate(&attitude, &[0.0, 0.0, -1.0]),
    }
}
