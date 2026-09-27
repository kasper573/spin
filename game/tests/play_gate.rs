//! The play gate: the game played as a player plays it — the default ring, the avatar in its
//! body, the tools worked by their keys and buttons, every frame drawn — and held to what a
//! player would hold it to. Each scene leaves the frames it drew and what it measured in
//! `target/playgate/<scene>/`, for whoever changed the game to look at before saying it works.
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use bevy::diagnostic::{Diagnostic, DiagnosticsStore};
use bevy::prelude::*;
use game::core::avatar::Thruster;
use game::core::fluid::Fluid;
use game::core::sheet::Sheet;
use game::core::units::Metres;
use game::core::units::Seconds;
use game::systems::drum::{DEFAULT_RING, Ring};
use game::systems::player::{PilotInput, Player};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;
use game::systems::tools::Toolbelt;

const FPS: u32 = 60;
/// The game is held to its frame budget drawn as large as a player may have it.
const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
/// Frames are timed so many at a time, and the last of them is kept and the water measured.
const KEPT_EVERY: u32 = 10;
/// The empty ring is timed over this many frames before a scene is played.
const CALIBRATION_FRAMES: u32 = 30;
/// How many runs of passes a frame bevy's render diagnostics keep the GPU's time of.
const TIMED_RUNS_KEPT: u32 = 128;
/// The slowest the game may run.
const FRAME_BUDGET_MS: f64 = 1000.0 / 120.0;
const WATER_TOOL: usize = 0;
const LAND_TOOL: usize = 1;

const PORTAL_TOOL: usize = 2;

/// What a player does for a stretch of a scene: which tool is out, which button of it is held,
/// which thruster, and a dial turned to a new setting as the stretch begins.
#[derive(Clone, Copy, Debug)]
struct Stretch {
    seconds: f32,
    pilot: Option<Thruster>,
    /// A second thruster held all the while, as a player holds two keys.
    held_too: Option<Thruster>,
    tool: usize,
    button: Option<MouseButton>,
    dial: Option<(Dial, f32)>,
}

impl Stretch {
    fn idle(seconds: f32) -> Stretch {
        Stretch {
            seconds,
            pilot: None,
            held_too: None,
            tool: WATER_TOOL,
            button: None,
            dial: None,
        }
    }

    /// A dial turned to a setting, and what comes of it watched.
    fn dialling(seconds: f32, dial: Dial, to: f32) -> Stretch {
        Stretch {
            dial: Some((dial, to)),
            ..Stretch::idle(seconds)
        }
    }

    /// One of the land tool's buttons held: the left raises the ground, the right lowers it.
    fn sculpting(seconds: f32, button: MouseButton) -> Stretch {
        Stretch {
            tool: LAND_TOOL,
            button: Some(button),
            ..Stretch::idle(seconds)
        }
    }

    fn pouring(seconds: f32) -> Stretch {
        Stretch {
            button: Some(MouseButton::Left),
            ..Stretch::idle(seconds)
        }
    }

    fn flying(seconds: f32, thruster: Thruster) -> Stretch {
        Stretch {
            pilot: Some(thruster),
            ..Stretch::idle(seconds)
        }
    }

    /// A press of one of the portal tool's buttons, which lets a portal into what it is aimed at.
    fn shooting(button: MouseButton) -> Stretch {
        Stretch {
            tool: PORTAL_TOOL,
            button: Some(button),
            ..Stretch::idle(0.2)
        }
    }

    fn with(self, tool: usize) -> Stretch {
        Stretch { tool, ..self }
    }

    fn holding(self, thruster: Thruster) -> Stretch {
        Stretch {
            held_too: Some(thruster),
            ..self
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
    /// What a frame of the empty default ring cost just before the scene was played.
    empty_ring_ms: f64,
    frame_ms: Vec<f64>,
    /// How much of a frame's time the CPU took to issue it, the GPU being waited for after.
    issuing_ms: Vec<f64>,
    broken_vertices: usize,
    stray_vertices: usize,
    /// How many of the kept frames found the avatar, solid, outside the drum.
    escapes: usize,
    most_vertices: usize,
    poured_m3: f64,
    water_m3: Vec<f64>,
    /// How much of the water was in the air, having run out of the sheet's drains.
    in_the_air_m3: Vec<f64>,
    /// What the GPU spent on each of its passes, in milliseconds a frame and in so many runs
    /// of the pass a frame, the most first.
    gpu_ms: Vec<(String, f64, f64)>,
    /// What the CPU spent recording each of those passes, as the GPU's are kept.
    cpu_ms: Vec<(String, f64, f64)>,
}

/// A scene as a player plays it: what they do, from where, in a ring of what size, with the
/// water tool set to pour so much.
struct Scene {
    name: String,
    ring: Ring,
    seat: Seat,
    litres_per_second: f32,
    stretches: Vec<Stretch>,
}

impl Scene {
    fn new(name: &str, seat: Seat, litres_per_second: f32, stretches: &[Stretch]) -> Scene {
        Scene {
            name: name.to_owned(),
            ring: DEFAULT_RING,
            seat,
            litres_per_second,
            stretches: stretches.to_vec(),
        }
    }

    /// The scene played in a ring the player has dialled to this size, and spun and equalized
    /// for it.
    fn in_ring(self, ring: Ring) -> Scene {
        Scene { ring, ..self }
    }
}

/// Play a scene a frame of simulated time at a time, measuring what the frames cost and keeping
/// every tenth.
fn play(scene: &Scene) -> Measured {
    let out =
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/playgate")).join(&scene.name);
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).expect("create the scene's folder");
    fs::write(out.join("plan.txt"), plan(&scene.stretches)).expect("write what the scene plays");
    let mut app = testing::headless();
    let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
    let empty_ring_ms = empty_ring_ms(&mut app);
    set_up(&mut app, scene);

    let mut measured = Measured {
        empty_ring_ms,
        frame_ms: Vec::new(),
        issuing_ms: Vec::new(),
        broken_vertices: 0,
        stray_vertices: 0,
        escapes: 0,
        most_vertices: 0,
        poured_m3: 0.0,
        water_m3: Vec::new(),
        in_the_air_m3: Vec::new(),
        gpu_ms: Vec::new(),
        cpu_ms: Vec::new(),
    };
    let mut hands = Hands::default();
    let mut frame = 0u32;
    let mut started = Instant::now();
    for stretch in &scene.stretches {
        hands.take_up(&mut app, stretch);
        let pouring = stretch.tool == WATER_TOOL && stretch.button == Some(MouseButton::Left);
        for _ in 0..(stretch.seconds * FPS as f32).round() as u32 {
            fly(&mut app, stretch);
            if frame.is_multiple_of(KEPT_EVERY) {
                // what the frames run to settle the water and capture the picture is no frame
                // the game plays, and whatever of it the GPU has yet to do is not theirs
                testing::wait_for_gpu(&app);
                started = Instant::now();
            }
            testing::frame(&mut app, Seconds(1.0 / FPS as f32));
            if pouring {
                let flow = Dial::Flow.get(app.world().resource::<Settings>());
                measured.poured_m3 += flow as f64 / 1000.0 / FPS as f64;
            }
            frame += 1;
            if frame.is_multiple_of(KEPT_EVERY) {
                // the frames run as the game runs them, one issued after the other, and the
                // GPU is waited for only here, so that what they cost it is in their time
                let issued = started.elapsed().as_secs_f64() * 1000.0 / KEPT_EVERY as f64;
                testing::wait_for_gpu(&app);
                let each = started.elapsed().as_secs_f64() * 1000.0 / KEPT_EVERY as f64;
                measured.frame_ms.push(each);
                measured.issuing_ms.push(issued);
                testing::settle(&mut app);
                let pixels = testing::capture(&mut app, &image);
                keep(&mut app, &pixels, &out, frame, &mut measured);
            }
        }
    }
    measured.gpu_ms = passes(&app, "elapsed_gpu");
    measured.cpu_ms = passes(&app, "elapsed_cpu");
    fs::write(out.join("measured.txt"), measured.report(&scene.name))
        .expect("write what was measured");
    eprintln!("{}", measured.report(&scene.name));
    measured
}

/// A live scene is kept as a video of this many frames a second, this large.
const VIDEO_FPS: f64 = 60.0;
const VIDEO_WIDTH: u32 = 1920;
const VIDEO_HEIGHT: u32 = 1080;
/// How many frames the last pictures kept of a live scene are given to come back from the GPU.
const DRAINING_FRAMES: u32 = 10;

/// Play a scene as the game runs it: each frame simulates the time the one before it took, and
/// begins as soon as that one has, as fast as the machine goes. What is on screen is kept every
/// sixtieth of a second, a picture being kept again for every sixtieth it stood, so that the
/// video runs as the scene ran and a slow frame shows as the stall it was.
fn play_live(scene: &Scene) {
    let out = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/playlive"));
    fs::create_dir_all(&out).expect("create the videos' folder");
    let mut app = testing::headless();
    let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
    let kept = testing::scaled_copy(&mut app, &image, VIDEO_WIDTH, VIDEO_HEIGHT);
    set_up(&mut app, scene);
    let video = Video::start(&out.join(format!("{}.mp4", scene.name)));
    let mut frame_ms = Vec::new();
    let mut hands = Hands::default();
    let mut ends = 0.0;
    let mut kept_to = 0;
    let started = Instant::now();
    let mut last = started;
    for stretch in &scene.stretches {
        hands.take_up(&mut app, stretch);
        ends += stretch.seconds as f64;
        while started.elapsed().as_secs_f64() < ends {
            fly(&mut app, stretch);
            let now = Instant::now();
            let took = now - last;
            last = now;
            frame_ms.push(took.as_secs_f64() * 1000.0);
            testing::frame_as_played(&mut app, Seconds(took.as_secs_f32()));
            let due = (started.elapsed().as_secs_f64() * VIDEO_FPS) as u32;
            if due > kept_to {
                let frames = video.frames.clone();
                testing::read_back_later(&mut app, &kept, move |pixels| {
                    let _ = frames.send((due, pixels));
                });
                kept_to = due;
            }
        }
    }
    for _ in 0..DRAINING_FRAMES {
        app.update();
    }
    // the pictures still on their way hold on to the video until the app is gone
    drop(app);
    video.finish();
    let report = live_report(&scene.name, &frame_ms);
    fs::write(out.join(format!("{}.txt", scene.name)), &report).expect("write how it ran");
    eprintln!("{report}");
}

/// A video of a live scene, made by ffmpeg from the pictures kept, each handed over with the
/// sixtieth of a second it was kept at.
struct Video {
    frames: Sender<(u32, Vec<u8>)>,
    made: JoinHandle<()>,
}

impl Video {
    fn start(path: &Path) -> Video {
        // the encoder is kept out of the way of the game the video is of
        let mut ffmpeg = Command::new("nice")
            .args(["-n", "19", "ffmpeg", "-y", "-loglevel", "error"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{VIDEO_WIDTH}x{VIDEO_HEIGHT}"))
            .args(["-r", &VIDEO_FPS.to_string(), "-i", "-", "-c:v", "libx264"])
            .args(["-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .expect("run ffmpeg");
        let mut into = ffmpeg.stdin.take().expect("ffmpeg takes pictures in");
        let (frames, kept) = mpsc::channel::<(u32, Vec<u8>)>();
        let made = thread::spawn(move || {
            let mut shown = 0;
            let mut on_screen: Option<Vec<u8>> = None;
            for (due, pixels) in kept {
                if due <= shown {
                    continue;
                }
                if let Some(stood) = &on_screen {
                    for _ in shown + 1..due {
                        into.write_all(stood).expect("hand ffmpeg a picture");
                    }
                }
                into.write_all(&pixels).expect("hand ffmpeg a picture");
                shown = due;
                on_screen = Some(pixels);
            }
            drop(into);
            ffmpeg.wait().expect("ffmpeg finishes the video");
        });
        Video { frames, made }
    }

    fn finish(self) {
        drop(self.frames);
        self.made.join().expect("the video is made");
    }
}

/// How fast a live scene ran: every frame's time, as the player would have had them.
fn live_report(scene: &str, frame_ms: &[f64]) -> String {
    let mut sorted = frame_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |share: f64| sorted[((sorted.len() - 1) as f64 * share) as usize];
    let (mut rates, mut worsts) = (Vec::new(), Vec::new());
    let (mut frames, mut passed, mut worst) = (0, 0.0, 0.0f64);
    for &ms in frame_ms {
        frames += 1;
        passed += ms;
        worst = worst.max(ms);
        if passed >= 1000.0 {
            rates.push(format!("{:.0}", frames as f64 * 1000.0 / passed));
            worsts.push(format!("{worst:.0}"));
            (frames, passed, worst) = (0, 0.0, 0.0);
        }
    }
    format!(
        "LIVE {scene}: {} frames in {:.1} s | frame ms p50 {:.1} p95 {:.1} worst {:.1} | fps by the second: {} | worst ms by the second: {}",
        frame_ms.len(),
        frame_ms.iter().sum::<f64>() / 1000.0,
        at(0.5),
        at(0.95),
        at(1.0),
        rates.join(" "),
        worsts.join(" ")
    )
}

/// What the player has in hand: the tool out, and the button of it held.
#[derive(Default)]
struct Hands {
    tool: Option<usize>,
    button: Option<MouseButton>,
}

impl Hands {
    /// Take up what a stretch has the player hold, letting go of what it does not, and turn
    /// its dial.
    fn take_up(&mut self, app: &mut App, stretch: &Stretch) {
        if (self.button != stretch.button || self.tool != Some(stretch.tool))
            && let Some(held) = self.button.take()
        {
            testing::button(app, held, false);
        }
        if self.tool != Some(stretch.tool) {
            testing::tap(app, Toolbelt::key(stretch.tool));
            self.tool = Some(stretch.tool);
        }
        if self.button != stretch.button {
            if let Some(pressed) = stretch.button {
                testing::button(app, pressed, true);
            }
            self.button = stretch.button;
        }
        if let Some((dial, to)) = stretch.dial {
            dial.set(&mut app.world_mut().resource_mut::<Settings>(), to);
        }
    }
}

/// Fire the thrusters a stretch holds for the coming frame.
fn fly(app: &mut App, stretch: &Stretch) {
    let held: Vec<Thruster> = stretch
        .pilot
        .iter()
        .chain(&stretch.held_too)
        .copied()
        .collect();
    let player = *app.world().resource::<Player>();
    app.world_mut().resource_mut::<Simulation>().avatar_input =
        player.input(PilotInput::firing(&held));
}

/// The game as a scene finds it: the water tool's flow set, the ring dialled, the player seated,
/// and a moment passed for the eye to settle.
fn set_up(app: &mut App, scene: &Scene) {
    Dial::Flow.set(
        &mut app.world_mut().resource_mut::<Settings>(),
        scene.litres_per_second,
    );
    if scene.seat == Seat::OutsideTheCap {
        app.world_mut().resource_mut::<Settings>().collisions = false;
    }
    if scene.ring != DEFAULT_RING {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, scene.ring.radius.0 * 2.0);
        Dial::Width.set(&mut settings, scene.ring.half_width.0 * 2.0);
        Dial::Spin.set(&mut settings, standing_spin(scene.ring).0);
        settings.equalize_thrust();
    }
    testing::watch(app, Seconds(0.5));
    if scene.seat == Seat::OutsideTheCap {
        let (up, along) = (
            DEFAULT_RING.radius.0 as f64,
            DEFAULT_RING.half_width.0 as f64,
        );
        let mut player = *app.world().resource::<Player>();
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        player.teleport(&mut sim, [2.0 - up, along + 12.0, 3.0], [-up, along, 0.0]);
    }
}

/// Each stretch of a scene with the second it starts at.
fn plan(stretches: &[Stretch]) -> String {
    let mut at = 0.0;
    stretches
        .iter()
        .map(|stretch| {
            let line = format!("{at:6.2} s  {stretch:?}\n");
            at += stretch.seconds;
            line
        })
        .collect()
}

/// What a frame of the empty default ring costs this machine right now. It is the same before
/// every scene of every run while the machine has nothing else to do, so a run in which it
/// reads otherwise was not alone on the machine, and nothing it measured is to be kept.
fn empty_ring_ms(app: &mut App) -> f64 {
    testing::watch(app, Seconds(0.5));
    testing::wait_for_gpu(app);
    let started = Instant::now();
    for _ in 0..CALIBRATION_FRAMES {
        testing::frame(app, Seconds(1.0 / FPS as f32));
    }
    testing::wait_for_gpu(app);
    started.elapsed().as_secs_f64() * 1000.0 / CALIBRATION_FRAMES as f64
}

fn by_the_second(kept: &[f64]) -> String {
    let every = (FPS / KEPT_EVERY) as usize;
    let each: Vec<String> = kept
        .iter()
        .step_by(every)
        .map(|v| format!("{v:.1}"))
        .collect();
    each.join(" ")
}

/// What each pass cost a frame by the render diagnostic of this name, the most first.
fn passes(app: &App, diagnostic: &str) -> Vec<(String, f64, f64)> {
    let mut passes: Vec<_> = app
        .world()
        .resource::<DiagnosticsStore>()
        .iter()
        .filter(|d| d.path().as_str().ends_with(diagnostic))
        .filter_map(|d| {
            let (ms, runs) = each_frame(d)?;
            Some((d.path().as_str().to_owned(), ms, runs))
        })
        .collect();
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    passes
}

/// What a pass cost the GPU a frame, in milliseconds. A pass that runs more than once a frame,
/// for every substep of the water or for every camera, reports each run on its own, all of a
/// frame's within a moment of each other: they are summed frame by frame.
fn each_frame(pass: &Diagnostic) -> Option<(f64, f64)> {
    let mut frames = 0u32;
    let mut runs = 0u32;
    let mut total = 0.0;
    let mut last = None;
    for run in pass.measurements() {
        runs += 1;
        if last.is_none_or(|last| run.time.duration_since(last).as_secs_f64() > 5e-4) {
            frames += 1;
        }
        last = Some(run.time);
        total += run.value;
    }
    (frames > 0).then(|| (total / frames as f64, runs as f64 / frames as f64))
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
    if app.world().resource::<Settings>().collisions && !sim.drum.holds(sim.avatar().p) {
        measured.escapes += 1;
    }
    measured.water_m3.push(water_m3(app));
    let in_the_air = app.world().resource::<Sheet>().flying().0 as f64 / 1000.0;
    measured.in_the_air_m3.push(in_the_air);
}

/// All the water there is: what flies as parcels, and what the GPU last found lying on the floor.
fn water_m3(app: &App) -> f64 {
    let flying = app.world().resource::<Fluid>().litres().0;
    let lying = app.world().resource::<Sheet>().held().0;
    (flying + lying) as f64 / 1000.0
}

impl Measured {
    fn percentile(&self, share: f64) -> f64 {
        let mut sorted = self.frame_ms.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() - 1) as f64 * share) as usize]
    }

    /// When the frames that took longest ended, in seconds of the scene.
    fn worst_at(&self) -> f64 {
        let worst = (0..self.frame_ms.len())
            .max_by(|&a, &b| self.frame_ms[a].total_cmp(&self.frame_ms[b]))
            .unwrap_or(0);
        ((worst + 1) * KEPT_EVERY as usize) as f64 / FPS as f64
    }

    fn report(&self, scene: &str) -> String {
        format!(
            "GATE {scene}: empty ring {:.1} ms a frame | frame ms p50 {:.1} p95 {:.1} worst {:.1} at {:.1} s, by the second: {}, of which the CPU issuing them: {} | poured {:.1} m3, water by the second: {}, of which in the air: {} | vertices at most {}, broken {}, stray {}",
            self.empty_ring_ms,
            self.percentile(0.5),
            self.percentile(0.95),
            self.percentile(1.0),
            self.worst_at(),
            by_the_second(&self.frame_ms),
            by_the_second(&self.issuing_ms),
            self.poured_m3,
            by_the_second(&self.water_m3),
            by_the_second(&self.in_the_air_m3),
            self.most_vertices,
            self.broken_vertices,
            self.stray_vertices,
        ) + &self
            .gpu_ms
            .iter()
            .filter(|(_, ms, _)| *ms >= 0.1)
            .map(|(path, ms, runs)| format!("\n    {ms:6.2} ms in {runs:4.1} runs  {path}"))
            .collect::<String>()
            + &format!(
                "\n    {:.0} timed runs a frame, of the {TIMED_RUNS_KEPT} the GPU's clock keeps: past that, runs go untimed",
                self.gpu_ms.iter().map(|(_, _, runs)| runs).sum::<f64>()
            )
            + &format!(
                "\n  recorded by the CPU, {:.2} ms a frame in all:",
                self.cpu_ms.iter().map(|(_, ms, _)| ms).sum::<f64>()
            )
            + &self
                .cpu_ms
                .iter()
                .filter(|(_, ms, _)| *ms >= 0.1)
                .map(|(path, ms, runs)| format!("\n    {ms:6.2} ms in {runs:4.1} runs  {path}"))
                .collect::<String>()
    }

    /// What a player would hold the scene to, beside what the frames show.
    fn hold(&self) {
        let mut failures = self.unsound();
        if self.percentile(0.95) > FRAME_BUDGET_MS {
            failures.push(format!(
                "frames take {:.1} ms, over the {FRAME_BUDGET_MS:.1} ms a frame has",
                self.percentile(0.95)
            ));
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }

    /// The world the scene left is sound, however fast it ran.
    fn sound(&self) {
        let failures = self.unsound();
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }

    /// What is wrong with the world the scene left, however fast it ran.
    fn unsound(&self) -> Vec<String> {
        let last = self.water_m3.last().copied().unwrap_or(0.0);
        let mut failures = Vec::new();
        if self.escapes > 0 {
            failures.push(format!(
                "the avatar was outside the drum in {} of the kept frames",
                self.escapes
            ));
        }
        if self.broken_vertices + self.stray_vertices > 0 {
            failures.push(format!(
                "{} vertices of the surface are no numbers and {} lie outside the ring",
                self.broken_vertices, self.stray_vertices
            ));
        }
        if let Some(second) = self.water_m3.iter().position(|m3| !m3.is_finite()) {
            failures.push(format!(
                "the water came to no number {:.1} s in",
                (second as u32 + 1) as f32 * KEPT_EVERY as f32 / FPS as f32
            ));
        } else if (last - self.poured_m3).abs() > 0.02 * self.poured_m3 + 0.05 {
            failures.push(format!(
                "{:.1} m3 were poured and {last:.1} m3 are left",
                self.poured_m3
            ));
        }
        failures
    }
}

/// Every scene, played two ways: by the gate, stepped a frame at a time and measured, and live,
/// as the game runs it, into a video. The gate holds a scene to its frame budget as well as to
/// the world staying sound, or to the world staying sound alone.
macro_rules! scenes {
    ($($test:ident: $scene:expr, $judged:ident;)*) => {
        mod gate {
            use super::*;
            $(
                #[test]
                #[ignore = "wants a GPU"]
                fn $test() {
                    play(&$scene).$judged();
                }
            )*
        }
        mod live {
            use super::*;
            $(
                #[test]
                #[ignore = "wants a GPU"]
                fn $test() {
                    play_live(&$scene);
                }
            )*
        }
    };
}

scenes! {
    a_trickle_poured_from_the_body: poured("trickle", Seat::Body, 5_000.0), hold;
    a_stream_poured_from_the_body: poured("stream", Seat::Body, 50_000.0), hold;
    a_flood_poured_from_the_body: poured("flood", Seat::Body, 175_000.0), hold;
    a_flood_poured_from_outside_the_cap: poured("flood_from_outside", Seat::OutsideTheCap, 175_000.0), hold;
    a_pool_poured_at_the_feet_and_waded_through: wade(), hold;
    a_stream_poured_into_a_portal_in_the_ground: portals(), hold;
    a_pool_falling_through_a_portal_in_its_bed_out_of_one_in_the_cap_over_it: waterfall(), hold;
    a_portal_pair_opened_in_a_dry_ring: dry_portals(), hold;
    a_hill_raised_and_a_pit_dug_and_water_poured_between_them: land(), hold;
    a_mound_raised_at_the_feet_and_the_horizon_looked_at_again: mound(), hold;
    a_ring_with_water_in_it_widened_and_grown_and_spun_down: dials(), hold;
    a_flood_poured_in_a_ring_two_kilometres_across: big_ring(), hold;
    a_scene_played_at_random_1: random(1), sound;
    a_scene_played_at_random_2: random(2), sound;
    a_scene_played_at_random_3: random(3), sound;
    a_scene_played_at_random_4: random(4), sound;
    a_scene_played_at_random_5: random(5), sound;
    a_scene_played_at_random_6: random(6), sound;
    a_scene_played_at_random_7: random(7), sound;
    a_scene_played_at_random_8: random(8), sound;
}

/// Water poured straight ahead for a while, and watched as it settles.
fn poured(name: &str, seat: Seat, litres_per_second: f32) -> Scene {
    let stretches = [
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::idle(5.0),
    ];
    Scene::new(name, seat, litres_per_second, &stretches)
}

fn wade() -> Scene {
    let stretches = [
        Stretch::flying(1.0, Thruster::PitchDown),
        Stretch::pouring(6.0),
        Stretch::flying(1.0, Thruster::PitchUp),
        Stretch::flying(6.0, Thruster::Forward),
    ];
    Scene::new("wade", Seat::Body, 50_000.0, &stretches)
}

fn portals() -> Scene {
    let stretches = [
        Stretch::flying(0.35, Thruster::PitchDown).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Left),
        Stretch::flying(0.7, Thruster::YawLeft).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Right),
        Stretch::idle(1.0),
        Stretch::pouring(8.0),
        Stretch::flying(0.35, Thruster::YawRight),
        Stretch::idle(4.0),
    ];
    Scene::new("portals", Seat::Body, 50_000.0, &stretches)
}

fn waterfall() -> Scene {
    let braced = |stretch: Stretch| stretch.holding(Thruster::Down);
    let stretches = [
        // facing a cap from well back, a small mouth let into the ground ahead and the other
        // into the cap straight over it, a metre clear of where the pool will stand
        Stretch::flying(0.98, Thruster::YawLeft).with(PORTAL_TOOL),
        Stretch::flying(2.6, Thruster::Back).with(PORTAL_TOOL),
        Stretch::dialling(0.5, Dial::Portal, 0.8).with(PORTAL_TOOL),
        Stretch::flying(0.3, Thruster::PitchDown).with(PORTAL_TOOL),
        Stretch::idle(0.3).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Left),
        Stretch::flying(0.337, Thruster::PitchUp).with(PORTAL_TOOL),
        Stretch::idle(0.3).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Right),
        Stretch::flying(0.116, Thruster::PitchDown).with(PORTAL_TOOL),
        Stretch::idle(0.3),
        // round the ring and up to the cap, to where the fall is seen from its side
        Stretch::flying(4.0, Thruster::Right),
        Stretch::idle(0.4),
        Stretch::flying(4.8, Thruster::Forward),
        Stretch::idle(0.4),
        // a pool a metre deep poured away from the fall, the body held down in it, and the
        // fall watched
        braced(Stretch::flying(0.98, Thruster::YawRight)),
        braced(Stretch::pouring(39.4)),
        braced(Stretch::flying(1.75, Thruster::YawLeft)),
        braced(Stretch::flying(0.12, Thruster::PitchUp)),
        braced(Stretch::idle(12.0)),
    ];
    Scene::new("waterfall", Seat::Body, 20_000.0, &stretches)
}

fn dry_portals() -> Scene {
    let stretches = [
        Stretch::flying(0.35, Thruster::PitchDown).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Left),
        Stretch::flying(0.7, Thruster::YawLeft).with(PORTAL_TOOL),
        Stretch::shooting(MouseButton::Right),
        Stretch::idle(4.0),
    ];
    Scene::new("dry_portals", Seat::Body, 0.0, &stretches)
}

fn land() -> Scene {
    let stretches = [
        Stretch::flying(0.2, Thruster::PitchDown).with(LAND_TOOL),
        Stretch::sculpting(2.5, MouseButton::Left),
        Stretch::flying(0.5, Thruster::YawLeft).with(LAND_TOOL),
        Stretch::sculpting(3.0, MouseButton::Right),
        Stretch::flying(0.25, Thruster::YawRight),
        Stretch::pouring(6.0),
        Stretch::idle(4.0),
    ];
    Scene::new("land", Seat::Body, 20_000.0, &stretches)
}

fn mound() -> Scene {
    let stretches = [
        Stretch::idle(1.0).with(LAND_TOOL),
        Stretch::flying(0.5, Thruster::PitchDown).with(LAND_TOOL),
        Stretch::sculpting(1.5, MouseButton::Left),
        Stretch::flying(0.5, Thruster::PitchUp).with(LAND_TOOL),
        Stretch::idle(2.0).with(LAND_TOOL),
    ];
    Scene::new("mound", Seat::Body, 0.0, &stretches)
}

fn dials() -> Scene {
    let stretches = [
        Stretch::flying(0.3, Thruster::PitchDown).with(LAND_TOOL),
        Stretch::sculpting(3.0, MouseButton::Right),
        Stretch::pouring(3.0),
        Stretch::idle(1.5),
        Stretch::dialling(3.0, Dial::Width, 20.0),
        Stretch::dialling(4.0, Dial::Diameter, 30.0),
        Stretch::dialling(4.0, Dial::Spin, 0.4),
    ];
    Scene::new("dials", Seat::Body, 5_000.0, &stretches)
}

fn big_ring() -> Scene {
    let stretches = [
        Stretch::idle(1.0),
        Stretch::pouring(10.0),
        Stretch::flying(1.0, Thruster::PitchUp),
        Stretch::idle(4.0),
    ];
    Scene::new("big_ring", Seat::Body, 999_000.0, &stretches).in_ring(Ring {
        radius: Metres(1000.0),
        half_width: Metres(100.0),
    })
}

/// A scene no one wrote: a player turning any dial to anything, pouring, digging, raising and
/// letting portals into whatever is before them, from wherever they have flown to, drawn from
/// its seed and then left to settle. What is asked of it is only that the world stays sound;
/// what it looks like is for its sheet to show.
fn random(seed: u64) -> Scene {
    Scene::new(
        &format!("random_{seed}"),
        Seat::Body,
        20_000.0,
        &random_scene(seed),
    )
}

const RANDOM_SECONDS: f32 = 20.0;
/// Long enough for water still leaving the tool to land.
const SETTLING_SECONDS: f32 = 3.0;

fn random_scene(seed: u64) -> Vec<Stretch> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let thrusters = Thruster::ALL;
    let mut stretches = Vec::new();
    let mut seconds = 0.0;
    while seconds < RANDOM_SECONDS {
        let long = rng.random_range(0.3..2.5);
        let stretch = match rng.random_range(0..10) {
            0 => Stretch::dialling(
                long,
                Dial::Spin,
                standing_spin(DEFAULT_RING).0 * rng.random_range(0.0..2.5),
            ),
            1 => Stretch::dialling(long, Dial::Diameter, rng.random_range(6.0..120.0)),
            2 => Stretch::dialling(long, Dial::Width, rng.random_range(2.0..40.0)),
            3 => Stretch::dialling(long, Dial::Flow, rng.random_range(1_000.0..200_000.0)),
            4 => Stretch::pouring(long),
            5 => Stretch::sculpting(
                long,
                if rng.random_bool(0.5) {
                    MouseButton::Left
                } else {
                    MouseButton::Right
                },
            ),
            6 => Stretch::shooting(if rng.random_bool(0.5) {
                MouseButton::Left
            } else {
                MouseButton::Right
            }),
            7 => Stretch::dialling(0.2, Dial::Portal, rng.random_range(0.5..4.0)).with(PORTAL_TOOL),
            _ => Stretch::flying(long, thrusters[rng.random_range(0..thrusters.len())]),
        };
        seconds += stretch.seconds;
        stretches.push(stretch);
    }
    stretches.push(Stretch::idle(SETTLING_SECONDS));
    stretches
}
