//! A first-person video of the avatar on the ring: each thruster firing on its own for the widget
//! to show, then walking, a hop, flying in bursts, turning round and walking against the spin,
//! rendered headless frame by frame into `target/record/` as PNGs beside the thrusters' voices
//! as a WAV and an SRT with the avatar's readouts, for ffmpeg to stitch (see `just record`).
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use game::core::audio::{self, Placement, Voice};
use game::core::avatar::Thruster;
use game::core::units::Seconds;
use game::systems::player::{PilotInput, Player};
use game::systems::sim::Simulation;
use game::systems::testing;

const FPS: u32 = 30;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// A stretch of the script: how long it lasts, the thrusters held and where the head looks.
struct Phase {
    seconds: f32,
    pilot: PilotInput,
    yaw: f64,
    pitch: f64,
    caption: &'static str,
}

fn script() -> Vec<Phase> {
    use Thruster::*;
    let level = 0.3;
    let pi = std::f64::consts::PI;
    let phase = |seconds, held: &[Thruster], yaw, pitch, caption| Phase {
        seconds,
        pilot: PilotInput::firing(held),
        yaw,
        pitch,
        caption,
    };
    let mut phases = vec![
        phase(
            2.0,
            &[],
            0.0,
            1.0,
            "standing still, looking up round the ring",
        ),
        phase(1.5, &[], 0.0, level, "standing still"),
    ];
    let showcase = [
        (Forward, "forward thruster (W)"),
        (Back, "back thruster (S)"),
        (Left, "left thruster (A)"),
        (Right, "right thruster (D)"),
        (Up, "up thruster (Space)"),
        (Down, "down thruster (Shift)"),
        (RollLeft, "roll left thruster (Q)"),
        (RollRight, "roll right thruster (E)"),
    ];
    for (thruster, caption) in showcase {
        phases.push(phase(0.8, &[thruster], 0.0, level, caption));
        let rest = if thruster == Up { 2.6 } else { 1.0 };
        phases.push(phase(rest, &[], 0.0, level, caption));
    }
    phases.extend([
        phase(4.0, &[Forward], 0.0, level, "walking spinward (W)"),
        phase(1.5, &[], 0.0, level, "stopping"),
        phase(
            0.8,
            &[Up],
            0.0,
            level,
            "a hop: a burst of up thrust (Space)",
        ),
        phase(
            2.2,
            &[],
            0.0,
            -0.5,
            "a hop: the ground comes back up to meet you",
        ),
        phase(
            0.4,
            &[Forward, Up],
            0.0,
            level,
            "flying forward in bursts (W + Space)",
        ),
        phase(0.2, &[Up], 0.0, level, "flying forward in bursts (Space)"),
        phase(0.4, &[], 0.0, level, "flying forward in bursts"),
        phase(0.6, &[Up], 0.0, level, "flying forward in bursts (Space)"),
        phase(0.4, &[], 0.0, level, "flying forward in bursts"),
        phase(0.6, &[Up], 0.0, level, "flying forward in bursts (Space)"),
        phase(3.0, &[], 0.0, level, "landing"),
        phase(1.5, &[], pi, level, "turning round"),
        phase(4.0, &[Forward], pi, level, "walking against the spin (W)"),
        phase(1.5, &[], pi, level, "standing still"),
    ]);
    phases
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
    let mut soundtrack = Soundtrack::new(out.join("thrusters.wav"));
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
                sim.avatar_input = player.input(sim.avatar(), phase.pilot);
            }
            testing::run(&mut app, frame_time);
            soundtrack.frame(app.world().resource::<Simulation>().thrusters.levels());
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
    soundtrack.finish();
    println!("{frame} frames in {}", out.display());
}

/// The thrusters' voices, each heard from where it sits around the head, mixed down a frame at a
/// time, fading between the loudness its level asked for at the last frame and at this one.
struct Soundtrack {
    writer: hound::WavWriter<std::io::BufWriter<fs::File>>,
    voices: Vec<(Voice, Placement)>,
    gains: [f32; 8],
}

impl Soundtrack {
    fn new(path: std::path::PathBuf) -> Soundtrack {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: audio::SAMPLE_RATE.get(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        Soundtrack {
            writer: hound::WavWriter::create(path, spec).expect("create thrusters.wav"),
            voices: Thruster::ALL
                .map(|t| (Voice::new(t as u64), Placement::around(t.mount())))
                .to_vec(),
            gains: [0.0; 8],
        }
    }

    fn frame(&mut self, levels: [f64; 8]) {
        let gains = levels.map(audio::gain);
        let samples = audio::SAMPLE_RATE.get() / FPS;
        for k in 0..samples {
            let t = k as f32 / samples as f32;
            let mut mix = [0.0; 2];
            for (((voice, placement), from), to) in
                self.voices.iter_mut().zip(self.gains).zip(gains)
            {
                let [left, right] = placement.hear(voice.sample());
                let gain = from + (to - from) * t;
                mix[0] += left * gain;
                mix[1] += right * gain;
            }
            for ear in mix {
                let sample = (ear.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                self.writer
                    .write_sample(sample)
                    .expect("write thrusters.wav");
            }
        }
        self.gains = gains;
    }

    fn finish(self) {
        self.writer.finalize().expect("finish thrusters.wav");
    }
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
