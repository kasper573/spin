//! A first-person video of the avatar on the ring: standing, walking, running, jumping, turning
//! round and walking against the spin, rendered headless frame by frame into `target/record/`
//! as PNGs beside an SRT with the avatar's readouts, for ffmpeg to stitch (see `just record`).
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use bevy::prelude::*;
use game::core::units::Seconds;
use game::systems::player::{PilotAxes, Player};
use game::systems::sim::Simulation;
use game::systems::testing;

const FPS: u32 = 30;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// A stretch of the script: how long it lasts, the keys held and where the head looks.
struct Phase {
    seconds: f32,
    axes: PilotAxes,
    yaw: f64,
    pitch: f64,
    caption: &'static str,
}

fn script() -> Vec<Phase> {
    let walk = Vec3::new(0.0, 0.0, -1.0);
    let phase = |seconds, motion, run, jump, yaw, pitch, caption| Phase {
        seconds,
        axes: PilotAxes {
            motion,
            roll: 0.0,
            run,
            jump,
        },
        yaw,
        pitch,
        caption,
    };
    let level = 0.3;
    let pi = std::f64::consts::PI;
    vec![
        phase(
            2.0,
            Vec3::ZERO,
            false,
            false,
            0.0,
            1.0,
            "standing still, looking up round the ring",
        ),
        phase(2.0, Vec3::ZERO, false, false, 0.0, level, "standing still"),
        phase(4.0, walk, false, false, 0.0, level, "walking spinward (W)"),
        phase(
            3.0,
            walk,
            true,
            false,
            0.0,
            level,
            "running spinward (Shift+W)",
        ),
        phase(1.5, Vec3::ZERO, false, false, 0.0, level, "stopping"),
        phase(0.1, Vec3::ZERO, false, true, 0.0, level, "jump (Space)"),
        phase(1.9, Vec3::ZERO, false, false, 0.0, level, "jump (Space)"),
        phase(
            0.1,
            Vec3::ZERO,
            false,
            true,
            0.0,
            -0.7,
            "jump, looking down (Space)",
        ),
        phase(
            1.9,
            Vec3::ZERO,
            false,
            false,
            0.0,
            -0.7,
            "jump, looking down (Space)",
        ),
        phase(1.5, Vec3::ZERO, false, false, pi, level, "turning round"),
        phase(
            4.0,
            walk,
            false,
            false,
            pi,
            level,
            "walking against the spin (W)",
        ),
        phase(
            1.0,
            walk,
            true,
            false,
            pi,
            level,
            "running against the spin (Shift+W)",
        ),
        phase(
            0.1,
            walk,
            true,
            true,
            pi,
            level,
            "running jump (Shift+W+Space)",
        ),
        phase(
            2.4,
            Vec3::ZERO,
            false,
            false,
            pi,
            level,
            "running jump (Shift+W+Space)",
        ),
        phase(1.5, Vec3::ZERO, false, false, pi, level, "standing still"),
    ]
}

fn main() {
    let out = Path::new("target/record");
    let _ = fs::remove_dir_all(out);
    fs::create_dir_all(out).expect("create target/record");
    let mut app = testing::headless();
    let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
    testing::run(&mut app, Seconds(0.5));

    let frame_time = Seconds(1.0 / FPS as f32);
    let mut srt = String::new();
    let mut frame = 0u32;
    let mut look = (0.0, 0.0);
    for phase in script() {
        let frames = (phase.seconds * FPS as f32).round() as u32;
        for _ in 0..frames {
            look.0 += (phase.yaw - look.0) * 0.1;
            look.1 += (phase.pitch - look.1) * 0.1;
            {
                let mut player = app.world_mut().resource_mut::<Player>();
                player.look.yaw = look.0;
                player.look.pitch = look.1;
                let player = *player;
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                sim.avatar_input = player.input(sim.avatar(), phase.axes);
            }
            testing::run(&mut app, frame_time);
            let readout = {
                let sim = app.world().resource::<Simulation>();
                let footing = sim.footing();
                let state = if footing.airborne {
                    "airborne".to_owned()
                } else {
                    format!(
                        "weight {:.2} g | ground speed {:.2} m/s",
                        footing.weight, footing.ground_speed
                    )
                };
                format!(
                    "{}\\N{state}\\Nspin {:.4} rad/s | t {:.1} s",
                    phase.caption, sim.drum.spin.0, sim.time.0
                )
            };
            let pixels = testing::capture(&mut app, &image);
            image::save_buffer(
                out.join(format!("frame_{frame:04}.png")),
                &pixels,
                WIDTH,
                HEIGHT,
                image::ColorType::Rgba8,
            )
            .expect("write frame");
            let start = frame as f64 / FPS as f64;
            let end = (frame + 1) as f64 / FPS as f64;
            let _ = writeln!(
                srt,
                "{}\n{} --> {}\n{readout}\n",
                frame + 1,
                timestamp(start),
                timestamp(end)
            );
            frame += 1;
        }
    }
    fs::write(out.join("readout.srt"), srt).expect("write readout.srt");
    println!("{frame} frames in {}", out.display());
}

fn timestamp(seconds: f64) -> String {
    let ms = (seconds * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}
