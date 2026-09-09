//! A first-person video of the avatar on the ring: each thruster firing on its own for the widget
//! to show, then walking, a hop, flying in bursts, turning round and walking against the spin,
//! wading through water and out of it, and the ring made bigger with the thrusters equalized to
//! it, rendered headless frame by frame into `target/record/` as PNGs beside the thrusters'
//! voices as a WAV and an SRT with the avatar's readouts, for ffmpeg to stitch (see `just record`).
//! The body turns to look, so looking round is a matter of the turning thrusters. Run with
//! `marker` as its argument, it records the crosshair's marker wrapping the ground instead.
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use bevy::prelude::*;
use game::core::audio::{self, Fader, Placement, Voice};
use game::core::avatar::{TURN_RATE, Thruster};
use game::core::fluid::Fluid;
use game::core::units::Seconds;
use game::systems::aim::Aim;
use game::systems::controls::{BRUSH_RATE, BRUSH_SIZE};
use game::systems::player::{PilotInput, Player};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;
use game::systems::testing;

const FPS: u32 = 30;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Something the script does to the world as a phase begins.
#[derive(Clone, Copy, PartialEq)]
enum Cue {
    None,
    /// Pour water in all round the ring.
    Flood,
    /// Make the ring bigger.
    Enlarge,
    /// Equalize the thruster power to the standing gravity.
    Equalize,
    /// Raise a few hills on the ground ahead.
    Hills,
}

/// A stretch of the script: how long it lasts and the thrusters held, at what level.
struct Phase {
    seconds: f32,
    pilot: PilotInput,
    caption: &'static str,
    cue: Cue,
    /// Whether the sculpting brush is held on the crosshair throughout.
    sculpting: bool,
}

/// The time the turning thrusters at `level` take to turn the body by `radians`.
fn turn_time(radians: f64, level: f32) -> f32 {
    (radians / (TURN_RATE * level as f64)) as f32
}

fn script() -> Vec<Phase> {
    use Thruster::*;
    let pi = std::f64::consts::PI;
    let phase = |seconds, held: &[Thruster], caption| Phase {
        seconds,
        pilot: PilotInput::firing(held),
        caption,
        cue: Cue::None,
        sculpting: false,
    };
    let eased = |seconds, held: &[Thruster], level: f32, caption| {
        let mut pilot = PilotInput::default();
        for thruster in held {
            pilot.levels[*thruster as usize] = level;
        }
        Phase {
            pilot,
            ..phase(seconds, &[], caption)
        }
    };
    let cued = |cue, seconds, held: &[Thruster], caption| Phase {
        cue,
        ..phase(seconds, held, caption)
    };
    let glance = 0.5;
    let mut phases = vec![
        eased(
            turn_time(0.7, glance),
            &[PitchUp],
            glance,
            "standing still, looking up round the ring (mouse up)",
        ),
        phase(1.5, &[], "standing still, looking up round the ring"),
        eased(
            turn_time(0.7, glance),
            &[PitchDown],
            glance,
            "looking down again (mouse down)",
        ),
        phase(1.0, &[], "standing still"),
    ];
    let showcase = [
        (Forward, "W: the back thruster pushes forward"),
        (Back, "S: the front thruster pushes back"),
        (Left, "A: the right thruster pushes left"),
        (Right, "D: the left thruster pushes right"),
        (Up, "Space: the bottom thruster pushes up"),
        (Down, "Shift: the top thruster pushes down"),
        (RollLeft, "Q: the right shoulder thruster rolls left"),
        (RollRight, "E: the left shoulder thruster rolls right"),
        (PitchUp, "mouse up: the chin thruster pitches the face up"),
        (
            PitchDown,
            "mouse down: the forehead thruster pitches it down",
        ),
        (YawLeft, "mouse left: the right cheek thruster yaws it left"),
        (
            YawRight,
            "mouse right: the left cheek thruster yaws it right",
        ),
    ];
    for (thruster, caption) in showcase {
        let level = if thruster.turns() { glance } else { 1.0 };
        phases.push(eased(0.8, &[thruster], level, caption));
        let rest = if thruster == Up { 2.6 } else { 1.0 };
        phases.push(phase(rest, &[], caption));
    }
    phases.extend([
        phase(4.0, &[Forward], "walking spinward (W)"),
        phase(1.5, &[], "stopping"),
        phase(0.8, &[Up], "a hop: a burst of up thrust (Space)"),
        phase(2.2, &[], "a hop: the ground comes back up to meet you"),
        phase(0.4, &[Forward, Up], "flying forward in bursts (W + Space)"),
        phase(0.2, &[Up], "flying forward in bursts (Space)"),
        phase(0.4, &[], "flying forward in bursts"),
        phase(0.6, &[Up], "flying forward in bursts (Space)"),
        phase(0.4, &[], "flying forward in bursts"),
        phase(0.6, &[Up], "flying forward in bursts (Space)"),
        phase(3.0, &[], "landing"),
        phase(turn_time(pi, 1.0), &[YawLeft], "turning round (mouse left)"),
        phase(0.5, &[], "turning round"),
        phase(4.0, &[Forward], "walking against the spin (W)"),
        phase(1.5, &[], "standing still"),
        cued(Cue::Flood, 4.0, &[], "flooding the ring with water"),
        phase(4.0, &[Forward], "wading forward through the water (W)"),
        phase(2.0, &[Up], "thrusting up out of the water (Space)"),
        phase(3.5, &[], "floating"),
        cued(
            Cue::Enlarge,
            5.0,
            &[],
            "ring diameter 21 to 30 m, width 12 to 16 m (F6, F7): the spin stays, so it pulls harder",
        ),
        phase(0.8, &[Up], "a hop with the old thruster power (Space)"),
        phase(2.0, &[], "a hop with the old thruster power"),
        cued(
            Cue::Equalize,
            1.0,
            &[],
            "T: equalize thruster power to gravity",
        ),
        phase(0.8, &[Up], "a hop with the equalized power (Space)"),
        phase(3.0, &[], "a hop with the equalized power"),
        eased(
            turn_time(1.2, glance),
            &[YawLeft, PitchUp],
            glance,
            "the glass panes round the ring (mouse up and left)",
        ),
        phase(2.0, &[], "the glass panes round the ring"),
    ]);
    phases
}

/// The crosshair's marker: looking down at flat ground, panning across it, raising a ridge
/// with the brush while its outline wraps what it raises, and panning over hills and hollows
/// with the brush and with the marker at rest.
fn marker_script() -> Vec<Phase> {
    use Thruster::*;
    let eased = |seconds, held: &[Thruster], level: f32, sculpting, caption| {
        let mut pilot = PilotInput::default();
        for thruster in held {
            pilot.levels[*thruster as usize] = level;
        }
        Phase {
            seconds,
            pilot,
            caption,
            cue: Cue::None,
            sculpting,
        }
    };
    let (pan, sweep) = (0.3, 0.55);
    vec![
        eased(
            turn_time(0.55, 0.5),
            &[PitchDown],
            0.5,
            false,
            "the crosshair's marker: looking down at flat ground",
        ),
        eased(1.0, &[], 0.0, false, "the marker lies flat on flat ground"),
        eased(
            turn_time(0.9, pan),
            &[YawLeft],
            pan,
            false,
            "panning across flat ground: the marker lies on it",
        ),
        eased(
            turn_time(1.8, sweep),
            &[YawRight],
            sweep,
            true,
            "right mouse held: the brush raises a ridge, and its outline wraps what it raises",
        ),
        eased(
            turn_time(1.8, pan),
            &[YawLeft],
            pan,
            false,
            "the marker at rest follows the ridge exactly, up its sides and over its top",
        ),
        eased(
            turn_time(1.8, pan),
            &[YawRight],
            pan,
            true,
            "the brush outline wraps the ridge just the same",
        ),
        Phase {
            cue: Cue::Hills,
            ..eased(0.5, &[], 0.0, false, "hills of three heights raised ahead")
        },
        eased(
            turn_time(0.25, 0.5),
            &[PitchUp],
            0.5,
            false,
            "looking a little further ahead",
        ),
        eased(
            turn_time(1.8, pan),
            &[YawLeft],
            pan,
            false,
            "the marker over hills and hollows, wrapped like a sheet laid on them",
        ),
        eased(
            turn_time(1.8, pan),
            &[YawRight],
            pan,
            true,
            "the brush outline over the hills",
        ),
        eased(
            3.0,
            &[Forward],
            1.0,
            false,
            "walking onto the raised ground (W)",
        ),
        eased(
            turn_time(1.2, pan),
            &[YawLeft],
            pan,
            false,
            "the marker from close up",
        ),
        eased(1.5, &[], 0.0, true, "the brush outline from close up"),
    ]
}

fn cue(app: &mut App, cue: Cue) {
    match cue {
        Cue::None => {}
        Cue::Flood => {
            for k in 0..30 {
                let a = k as f64 * 0.52;
                app.world_mut()
                    .resource_scope(|world, mut fluid: Mut<Fluid>| {
                        let sim = world.resource::<Simulation>();
                        sim.inject(
                            &mut fluid,
                            sim.drum.from_water([
                                a.cos() * 7.0,
                                (k % 3) as f64 * 3.0 - 3.0,
                                a.sin() * 7.0,
                            ]),
                            1500,
                        )
                    });
            }
        }
        Cue::Enlarge => {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            Dial::Diameter.set(&mut settings, 30.0);
            Dial::Width.set(&mut settings, 16.0);
        }
        Cue::Equalize => app.world_mut().resource_mut::<Settings>().equalize_thrust(),
        Cue::Hills => {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            let site = sim.drum.site;
            for (k, (arc, y, height)) in [(-6.0, 1.0, 0.6), (-9.0, -1.5, 1.2), (-12.0, 2.5, 2.0)]
                .into_iter()
                .enumerate()
            {
                let phi = site.phi + arc / sim.drum.ring.radius.0 as f64;
                for _ in 0..20 {
                    sim.drum
                        .landscape
                        .sculpt(phi, site.y + y, 1.5 + k as f64 * 0.5, height / 20.0);
                }
            }
        }
    }
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
    let script = match std::env::args().nth(1).as_deref() {
        None => script(),
        Some("marker") => {
            app.world_mut().resource_mut::<Aim>().engaged = true;
            marker_script()
        }
        Some(other) => panic!("unknown script {other:?}: the only one is `marker`"),
    };
    let mut frame = 0u32;
    for phase in script {
        cue(&mut app, phase.cue);
        let frames = (phase.seconds * FPS as f32).round() as u32;
        for _ in 0..frames {
            {
                let player = *app.world().resource::<Player>();
                let target = app.world().resource::<Aim>().target;
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                sim.avatar_input = player.input(phase.pilot);
                if phase.sculpting
                    && let Some(target) = target
                {
                    let amount = BRUSH_RATE.0 * frame_time.0;
                    sim.sculpt(target.point.to_array(), BRUSH_SIZE.0 as f64, amount as f64);
                }
                app.world_mut().resource_mut::<Aim>().brush = phase.sculpting.then_some(BRUSH_SIZE);
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
    faders: [Fader; 12],
    gains: [f32; 12],
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
            faders: [Fader::default(); 12],
            gains: [0.0; 12],
        }
    }

    fn frame(&mut self, levels: [f64; 12]) {
        let dt = Seconds(1.0 / FPS as f32);
        let gains = Thruster::ALL.map(|t| {
            let wanted = if t.turns() {
                0.0
            } else {
                audio::gain(levels[t as usize])
            };
            self.faders[t as usize].follow(wanted, dt)
        });
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
