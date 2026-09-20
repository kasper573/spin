//! The play gate: the game played as a player plays it — the default ring, the avatar in its
//! body, the tools worked by their keys and buttons, every frame drawn — and held to what a
//! player would hold it to. Each scene leaves the frames it drew and what it measured in
//! `target/playgate/<scene>/`, for whoever changed the game to look at before saying it works.
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use game::core::avatar::Thruster;
use game::core::fluid::Fluid;
use game::core::units::Seconds;
use game::systems::drum::DEFAULT_RING;
use game::systems::player::{PilotInput, Player};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;
use game::systems::testing;
use game::systems::tools::Toolbelt;

const FPS: u32 = 60;
/// The game is held to its frame budget drawn as large as a player may have it.
const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
/// Frames are timed so many at a time, and the last of them is kept and the water measured.
const KEPT_EVERY: u32 = 10;
/// The slowest the game may run.
const FRAME_BUDGET_MS: f64 = 1000.0 / 120.0;
const WATER_TOOL: usize = 0;

/// What a player does for a stretch of a scene.
#[derive(Clone, Copy)]
struct Stretch {
    seconds: f32,
    pilot: [Option<Thruster>; 2],
    pouring: bool,
}

impl Stretch {
    fn idle(seconds: f32) -> Stretch {
        Stretch {
            seconds,
            pilot: [None; 2],
            pouring: false,
        }
    }

    fn pouring(seconds: f32) -> Stretch {
        Stretch {
            pouring: true,
            ..Stretch::idle(seconds)
        }
    }

    fn flying(seconds: f32, thruster: Thruster) -> Stretch {
        Stretch {
            pilot: [Some(thruster), None],
            ..Stretch::idle(seconds)
        }
    }
}

/// Where the player looks from, when not from the body standing where the game starts: outside
/// the ring, through the glass of a cap, as a ghost.
#[derive(Clone, Copy, PartialEq)]
enum Seat {
    Body,
    OutsideTheCap,
}

struct Measured {
    frame_ms: Vec<f64>,
    broken_vertices: usize,
    stray_vertices: usize,
    most_vertices: usize,
    poured_m3: f64,
    water_m3: Vec<f64>,
    /// What the GPU spent on each of its passes, in milliseconds a pass, the most first. A
    /// pass that runs for every substep of the water runs more than once a frame.
    gpu_ms: Vec<(String, f64)>,
}

fn play(scene: &str, seat: Seat, litres_per_second: f32, stretches: &[Stretch]) -> Measured {
    let out = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/playgate")).join(scene);
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).expect("create the scene's folder");
    let mut app = testing::headless();
    let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
    Dial::Flow.set(
        &mut app.world_mut().resource_mut::<Settings>(),
        litres_per_second,
    );
    if seat == Seat::OutsideTheCap {
        app.world_mut().resource_mut::<Settings>().collisions = false;
    }
    testing::watch(&mut app, Seconds(0.5));
    if seat == Seat::OutsideTheCap {
        let (up, along) = (
            DEFAULT_RING.radius.0 as f64,
            DEFAULT_RING.half_width.0 as f64,
        );
        let mut player = *app.world().resource::<Player>();
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        player.teleport(&mut sim, [2.0 - up, along + 12.0, 3.0], [-up, along, 0.0]);
    }
    testing::tap(&mut app, Toolbelt::key(WATER_TOOL));

    let mut measured = Measured {
        frame_ms: Vec::new(),
        broken_vertices: 0,
        stray_vertices: 0,
        most_vertices: 0,
        poured_m3: 0.0,
        water_m3: Vec::new(),
        gpu_ms: Vec::new(),
    };
    let mut pouring = false;
    let mut frame = 0u32;
    let mut started = Instant::now();
    for stretch in stretches {
        if stretch.pouring != pouring {
            testing::button(&mut app, MouseButton::Left, stretch.pouring);
            pouring = stretch.pouring;
        }
        let held: Vec<Thruster> = stretch.pilot.iter().flatten().copied().collect();
        for _ in 0..(stretch.seconds * FPS as f32).round() as u32 {
            let player = *app.world().resource::<Player>();
            app.world_mut().resource_mut::<Simulation>().avatar_input =
                player.input(PilotInput::firing(&held));
            if frame.is_multiple_of(KEPT_EVERY) {
                started = Instant::now();
            }
            testing::frame(&mut app, Seconds(1.0 / FPS as f32));
            if pouring {
                measured.poured_m3 += litres_per_second as f64 / 1000.0 / FPS as f64;
            }
            frame += 1;
            if frame.is_multiple_of(KEPT_EVERY) {
                // the frames run as the game runs them, one issued after the other, and the
                // GPU is waited for only here, so that what they cost it is in their time
                testing::settle(&mut app);
                testing::wait_for_gpu(&mut app);
                let each = started.elapsed().as_secs_f64() * 1000.0 / KEPT_EVERY as f64;
                measured.frame_ms.push(each);
                let pixels = testing::capture(&mut app, &image);
                keep(&mut app, &pixels, &out, frame, &mut measured);
            }
        }
    }
    measured.gpu_ms = app
        .world()
        .resource::<DiagnosticsStore>()
        .iter()
        .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
        .filter_map(|d| Some((d.path().as_str().to_owned(), d.average()?)))
        .collect();
    measured.gpu_ms.sort_by(|a, b| b.1.total_cmp(&a.1));
    fs::write(out.join("measured.txt"), measured.report(scene)).expect("write what was measured");
    eprintln!("{}", measured.report(scene));
    measured
}

fn keep(app: &mut App, pixels: &[u8], out: &std::path::Path, frame: u32, measured: &mut Measured) {
    image::save_buffer(
        out.join(format!("frame_{frame:04}.png")),
        pixels,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
    )
    .expect("write a frame");
    let vertices = testing::surface_vertices(app);
    let sim = app.world().resource::<Simulation>();
    let (radius, half_width) = (
        sim.drum.ring.radius.0 as f64,
        sim.drum.ring.half_width.0 as f64,
    );
    measured.most_vertices = measured.most_vertices.max(vertices.len());
    for v in &vertices {
        if !v.iter().all(|c| c.is_finite()) {
            measured.broken_vertices += 1;
            continue;
        }
        let [x, y, z] = sim.drum.from_water([v[0] as f64, v[1] as f64, v[2] as f64]);
        let from_axis = (x + radius).hypot(z);
        if from_axis > radius + 0.5 || (y + sim.drum.site.y).abs() > half_width + 0.5 {
            measured.stray_vertices += 1;
        }
    }
    measured.water_m3.push(water_m3(app));
}

fn water_m3(app: &App) -> f64 {
    app.world().resource::<Fluid>().litres().0 as f64 / 1000.0
}

impl Measured {
    fn percentile(&self, share: f64) -> f64 {
        let mut sorted = self.frame_ms.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() - 1) as f64 * share) as usize]
    }

    fn report(&self, scene: &str) -> String {
        let water: Vec<String> = self
            .water_m3
            .iter()
            .step_by(6)
            .map(|m3| format!("{m3:.1}"))
            .collect();
        format!(
            "GATE {scene}: frame ms p50 {:.1} p95 {:.1} worst {:.1} | poured {:.1} m3, water by the second: {} | vertices at most {}, broken {}, stray {}",
            self.percentile(0.5),
            self.percentile(0.95),
            self.percentile(1.0),
            self.poured_m3,
            water.join(" "),
            self.most_vertices,
            self.broken_vertices,
            self.stray_vertices,
        ) + &self
            .gpu_ms
            .iter()
            .take(12)
            .map(|(path, ms)| format!("\n    {ms:6.2} ms  {path}"))
            .collect::<String>()
    }

    /// What a player would hold the scene to, beside what the frames show.
    fn hold(&self) {
        let last = self.water_m3.last().copied().unwrap_or(0.0);
        let mut failures = Vec::new();
        if self.percentile(0.95) > FRAME_BUDGET_MS {
            failures.push(format!(
                "frames take {:.1} ms, over the {FRAME_BUDGET_MS:.1} ms a frame has",
                self.percentile(0.95)
            ));
        }
        if self.broken_vertices + self.stray_vertices > 0 {
            failures.push(format!(
                "{} vertices of the surface are no numbers and {} lie outside the ring",
                self.broken_vertices, self.stray_vertices
            ));
        }
        if (last - self.poured_m3).abs() > 0.02 * self.poured_m3 + 0.05 {
            failures.push(format!(
                "{:.1} m3 were poured and {last:.1} m3 are left",
                self.poured_m3
            ));
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }
}

#[test]
#[ignore = "wants a GPU"]
fn a_trickle_poured_from_the_body() {
    let scene = [
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::idle(5.0),
    ];
    play("trickle", Seat::Body, 5_000.0, &scene).hold();
}

#[test]
#[ignore = "wants a GPU"]
fn a_stream_poured_from_the_body() {
    let scene = [
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::idle(5.0),
    ];
    play("stream", Seat::Body, 50_000.0, &scene).hold();
}

#[test]
#[ignore = "wants a GPU"]
fn a_flood_poured_from_the_body() {
    let scene = [
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::idle(5.0),
    ];
    play("flood", Seat::Body, 175_000.0, &scene).hold();
}

#[test]
#[ignore = "wants a GPU"]
fn a_flood_poured_from_outside_the_cap() {
    let scene = [
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::idle(5.0),
    ];
    play("flood_from_outside", Seat::OutsideTheCap, 175_000.0, &scene).hold();
}

#[test]
#[ignore = "wants a GPU"]
fn a_pool_poured_at_the_feet_and_waded_through() {
    let scene = [
        Stretch::flying(1.0, Thruster::PitchDown),
        Stretch::pouring(6.0),
        Stretch::flying(1.0, Thruster::PitchUp),
        Stretch::flying(6.0, Thruster::Forward),
    ];
    play("wade", Seat::Body, 50_000.0, &scene).hold();
}
