//! A first-person video of the avatar on the ring: each thruster firing on its own for the widget
//! to show, then walking, a hop, flying in bursts, turning round and walking against the spin,
//! wading through water and out of it, and the ring made bigger with the thrusters equalized to
//! it, rendered headless frame by frame into `target/record/` as PNGs beside the thrusters'
//! voices as a WAV and an SRT with the avatar's readouts, for ffmpeg to stitch (see `just record`).
//! The body turns to look, so looking round is a matter of the turning thrusters. Run with
//! `marker` as its argument, it records the crosshair's marker wrapping the ground instead, and
//! with `water`, the water: poured in, waded through, seen from under and from above, set
//! flowing by a change of spin, and poured from the crosshair.
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use bevy::prelude::*;
use game::core::audio::{self, Fader, Placement, Voice};
use game::core::avatar::{self, Gyros, TURN_RATE, Thruster};
use game::core::fluid::Fluid;
use game::core::math::{cross, norm, quat_from_basis, quat_rotate};
use game::core::units::Litres;
use game::core::units::Seconds;
use game::systems::aim::Aim;
use game::systems::controls::{BRUSH_RATE, BRUSH_SIZE, INJECT_DEPTH};
use game::systems::player::{PilotInput, Player};
use game::systems::scene::Sky;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;
use game::systems::testing;

const FPS: u32 = 30;
/// The video's size, and the factor the frames are drawn larger by before they are scaled down
/// to it: one sample a pixel leaves the water's ripples and its edge against the shore finer
/// than the pixels that have to carry them, which crawls from frame to frame and eats the
/// bitrate that the rest of the picture wants (see `just record`).
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const SUPERSAMPLE: u32 = 2;

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
    /// Spin the ring up by half again.
    SpinUp,
    /// Raise the ground under the avatar into a bank, hollow a basin out of the ground ahead,
    /// and pour water in all round the ring, enough to fill the basin over head height.
    Basin,
    /// Make the ring big, raise a headland across it, and flood the whole ring round the
    /// headland deep enough for the water to show its own colour rather than its bed's.
    Sea,
    /// Put the avatar back in its body over the deep water, at rest, for the sea to take it.
    Dive,
    /// Put the avatar, as a ghost, at a viewpoint of the survey, by night or by day.
    Look(View),
}

/// A viewpoint of the survey: where the eye is and what it looks at, each a place about the
/// pool in metres round the ring, along the axis and over the ground (under it, and out
/// through the glass, when negative), and whether the sun should stand over the pool or
/// behind the ring first.
#[derive(Clone, Copy, PartialEq)]
struct View {
    eye: [f64; 3],
    at: [f64; 3],
    daylight: bool,
    /// Whether the places round the ring are measured from the water's edge rather than from
    /// the pool: a shot of a shore wants to stand where the shore turns out to be.
    ashore: bool,
}

/// Where the pool was laid in: its wheel angle and its place along the axis, which the site
/// leaves behind as the avatar moves about.
#[derive(Resource, Clone, Copy)]
struct Pool {
    phi: f64,
    y: f64,
    /// How far round the ring from `phi` the water's edge lies, once a sea stands in the ring.
    shore: f64,
}

/// A stretch of the script: how long it lasts and the thrusters held, at what level.
struct Phase {
    seconds: f32,
    pilot: PilotInput,
    caption: &'static str,
    cue: Cue,
    /// Whether the sculpting brush is held on the crosshair throughout.
    sculpting: bool,
    /// Whether water is poured at the crosshair throughout.
    pouring: bool,
    /// A viewpoint the eye is held at for the whole stretch: a ghost let go of falls out
    /// through the ring within a second, taking the view with it.
    hold: Option<View>,
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
        pouring: false,
        hold: None,
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
            pouring: false,
            hold: None,
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

/// The water: poured in all round the ring and left to settle, looked into and across, waded
/// into until the eye is under it, looked at from below, flown out of and looked down on,
/// set flowing by spinning the ring up, and poured from the crosshair.
fn water_script() -> Vec<Phase> {
    use Thruster::*;
    let phase = |seconds, held: &[Thruster], level: f32, caption| {
        let mut pilot = PilotInput::default();
        for thruster in held {
            pilot.levels[*thruster as usize] = level;
        }
        Phase {
            seconds,
            pilot,
            caption,
            cue: Cue::None,
            sculpting: false,
            pouring: false,
            hold: None,
        }
    };
    let glance = 0.5;
    vec![
        Phase {
            cue: Cue::Basin,
            ..phase(
                3.5,
                &[Down],
                1.0,
                "held down (Shift) on the bed of a pool dammed between two ridges: everything seen through the water, dimmed and coloured by its depth",
            )
        },
        phase(
            turn_time(1.1, glance),
            &[PitchUp, Down],
            glance,
            "looking up at the surface from below (mouse up)",
        ),
        phase(
            3.0,
            &[Down],
            1.0,
            "the surface from below: the world above within the critical angle, the bed mirrored beyond it",
        ),
        phase(3.0, &[Up], 1.0, "thrusting up out of the water (Space)"),
        phase(1.5, &[], 0.0, "rising"),
        phase(
            turn_time(1.5, glance),
            &[PitchDown],
            glance,
            "looking down at the pool from above (mouse down)",
        ),
        phase(
            4.0,
            &[],
            0.0,
            "the ring mirrored in the water where it lies flat, the sun glinting where it ripples, the bed bent by them",
        ),
        phase(3.0, &[], 0.0, "falling back in"),
        Phase {
            cue: Cue::SpinUp,
            ..phase(
                7.0,
                &[],
                0.0,
                "the ring spun up by half (F1): the water is left behind, and flows and churns",
            )
        },
        Phase {
            pouring: true,
            ..phase(
                4.0,
                &[],
                0.0,
                "pouring water from the crosshair (LMB): foam where it churns, ripples riding the flow",
            )
        },
        phase(4.0, &[], 0.0, "the pour settling"),
        {
            let view = View {
                eye: [-2.0, -10.0, 2.0],
                at: [-2.0, 0.0, 1.0],
                daylight: true,
                ashore: false,
            };
            Phase {
                cue: Cue::Look(view),
                hold: Some(view),
                ..phase(
                    4.0,
                    &[],
                    0.0,
                    "from outside the ring's end: the water seen in through the glass disc",
                )
            }
        },
        {
            let view = View {
                eye: [-2.0, 25.0, -25.0],
                at: [-2.0, 0.0, 0.0],
                daylight: true,
                ashore: false,
            };
            Phase {
                cue: Cue::Look(view),
                hold: Some(view),
                ..phase(
                    4.0,
                    &[],
                    0.0,
                    "the whole ring from 35 m out, the water lying in it",
                )
            }
        },
    ]
}

/// The water's colour: a pool between two ridges, shallow enough that the sun reaches its bed
/// and comes back off the sand, and then a sea laid all the way round a bigger ring, deep enough
/// that away from its shore only the water's own colour comes back. Each is looked at from the
/// shore, from under the water, from over it, from outside the glass and from off the ring.
fn sea_script() -> Vec<Phase> {
    use Thruster::*;
    let phase = |seconds, held: &[Thruster], level: f32, caption| {
        let mut pilot = PilotInput::default();
        for thruster in held {
            pilot.levels[*thruster as usize] = level;
        }
        Phase {
            seconds,
            pilot,
            caption,
            cue: Cue::None,
            sculpting: false,
            pouring: false,
            hold: None,
        }
    };
    // every held shot waits for its own sun: the ring turns once in about six seconds, so a
    // place on it sees three of daylight and three of night, and a shot that does not wait for
    // the turn of the day runs into the dark halfway through
    let held = |ashore, seconds, eye, at, daylight, caption| {
        let view = View {
            eye,
            at,
            daylight,
            ashore,
        };
        Phase {
            cue: Cue::Look(view),
            hold: Some(view),
            ..phase(seconds, &[], 0.0, caption)
        }
    };
    let mut script = vec![Phase {
        cue: Cue::Basin,
        ..phase(
            0.5,
            &[],
            0.0,
            "a pool dammed between two ridges, a metre deep over its bed",
        )
    }];
    // the pool: shallow water over pale sand, which is what a lagoon is
    script.extend([
        held(
            false,
            DAY,
            [8.0, 0.0, 1.7],
            [-6.0, 0.0, 1.0],
            true,
            "from the near ridge: the sand the water laid down under itself, and over it the water, greener the further the sun's light has to cross to come back",
        ),
        held(
            false,
            DAY,
            [-2.0, 0.0, 0.6],
            [-8.0, 0.0, 0.6],
            true,
            "on the bed, under the water: nothing yet stands between the eye and the sand",
        ),
        held(
            false,
            DAY,
            [-2.0, 0.0, 0.6],
            [-4.0, 0.5, 3.0],
            true,
            "up at the surface from below: the world above gathered into the circle within the critical angle, the bed mirrored beyond it",
        ),
        held(
            false,
            DAY,
            [-2.0, 0.0, 8.5],
            [-6.0, 1.0, 0.0],
            true,
            "from up by the axis: the shallows pale at the shore and darkening toward the middle",
        ),
        held(
            false,
            DAY,
            [-2.0, 0.0, -3.5],
            [-2.0, 0.0, 1.5],
            true,
            "from outside, in through the glass floor: the pool from underneath, lit through its own surface",
        ),
        held(
            false,
            DAY,
            [8.0, 0.0, 1.7],
            [-6.0, 0.0, 1.0],
            false,
            "the same pool by night, with only the light bounced round the ring left to come back out of it",
        ),
    ]);
    // the sea: the same water, deep enough to keep its own colour
    script.push(Phase {
        cue: Cue::Sea,
        ..phase(
            4.0,
            &[],
            0.0,
            "the ring made bigger and flooded all the way round: a sea four metres deep, a headland standing out of it",
        )
    });
    script.extend([
        held(
            false,
            DAY,
            [0.0, 0.0, 1.7],
            [15.0, 6.0, -1.0],
            true,
            "from the headland: the shallows over the terrace round its foot, then blue where the bed drops away and the sand no longer answers",
        ),
        held(
            true,
            DAY,
            [-2.0, 0.0, 1.6],
            [6.0, 7.0, -1.0],
            true,
            "at the water's edge: grass to the line the water reaches, then the pale bed the water laid down under itself, and past it the deep",
        ),
        held(
            true,
            DAY,
            [-0.5, 0.0, 0.3],
            [3.0, 6.0, -0.3],
            true,
            "along the water's edge from a hand's height: the sheet giving out over the bank rather than ending at a rim the grid cut for it",
        ),
    ]);
    // down through the surface and back out of it, carried by the thrusters rather than cut to
    script.extend([
        Phase {
            cue: Cue::Dive,
            ..phase(2.0, &[], 0.0, "let go over the deep, falling toward the sea")
        },
        phase(
            3.0,
            &[Down],
            1.0,
            "down through the surface (Shift): the water closes over the eye",
        ),
        phase(
            3.0,
            &[],
            0.0,
            "under the sea: no bed answers here, so what comes back is the water's own colour, the blue that is left of the sun after four metres of it",
        ),
        phase(
            turn_time(1.2, 0.5),
            &[PitchUp],
            0.5,
            "looking up (mouse up)",
        ),
        phase(3.0, &[], 0.0, "the surface from below, the ring beyond it"),
        phase(3.0, &[Up], 1.0, "up and out again (Space)"),
    ]);
    script.extend([
        held(
            false,
            DAY,
            [0.0, 0.0, 14.0],
            [16.0, 3.0, 0.0],
            true,
            "from up by the axis: the sea running away round the ring, pale over the shelves and blue over the deep",
        ),
        held(
            false,
            DAY,
            [20.0, 0.0, -5.0],
            [20.0, 0.0, 2.0],
            true,
            "from outside, in through the glass floor: the sea from underneath",
        ),
        held(
            false,
            DAY,
            [0.0, -16.0, 4.0],
            [0.0, 0.0, 3.0],
            true,
            "from outside the ring's end, in through the glass disc and the length of the sea",
        ),
        held(
            false,
            DAY,
            [0.0, 34.0, -34.0],
            [0.0, 0.0, 0.0],
            true,
            "the whole ring from off its axis: a band of water closed on itself, lit from within",
        ),
        held(
            false,
            DAY,
            [0.0, 0.0, 1.7],
            [26.0, 0.0, 4.0],
            false,
            "the sea by night: the sun behind the ring, and the water showing only what the ring bounces round to it",
        ),
    ]);
    script
}

/// One frame from each of many viewpoints about the pool, by night and then by day: under the
/// water, at its surface, over it, out through the glass, in through the glass and the water
/// from outside, and from far off.
fn survey_script() -> Vec<Phase> {
    let views: [([f64; 3], [f64; 3], &'static str); 12] = [
        (
            [-2.0, 0.0, 0.6],
            [-8.0, 0.0, 0.6],
            "under water on the bed, along the pool",
        ),
        (
            [-2.0, 0.0, 0.6],
            [-4.0, 0.5, 3.0],
            "under water, up at the surface",
        ),
        (
            [-2.0, -2.0, 0.6],
            [-2.0, -8.0, 1.0],
            "under water, at a cap through the water",
        ),
        (
            [0.0, 0.0, 2.8],
            [-7.0, 0.0, 1.5],
            "at the surface, across the pool",
        ),
        (
            [8.0, 0.0, 1.7],
            [-6.0, 0.0, 1.0],
            "from the near ridge, over the pool",
        ),
        (
            [-2.0, 0.0, 8.5],
            [-6.0, 1.0, 0.0],
            "from up by the axis, down at the pool",
        ),
        (
            [0.0, 0.0, 3.0],
            [0.0, -20.0, 3.0],
            "from inside, out through a cap into space",
        ),
        (
            [0.0, 0.0, 3.0],
            [25.0, 0.0, 1.7],
            "from inside, along the ring",
        ),
        (
            [-2.0, 0.0, -3.5],
            [-2.0, 0.0, 1.5],
            "from outside the glass, in through the wall at the pool",
        ),
        (
            [-2.0, -9.0, 1.0],
            [-2.0, 0.0, 1.0],
            "from outside a cap, in through the glass and the water",
        ),
        (
            [0.0, 40.0, -40.0],
            [0.0, 0.0, 0.0],
            "the ring from 60 m out",
        ),
        (
            [0.0, 200.0, -200.0],
            [0.0, 0.0, 0.0],
            "the ring from 300 m out",
        ),
    ];
    let mut script = vec![Phase {
        seconds: 1.0 / FPS as f32,
        pilot: PilotInput::default(),
        caption: "the pool laid in",
        cue: Cue::Basin,
        sculpting: false,
        pouring: false,
        hold: None,
    }];
    for daylight in [false, true] {
        for (eye, at, caption) in views {
            script.push(Phase {
                seconds: 1.0 / FPS as f32,
                pilot: PilotInput::default(),
                caption,
                cue: Cue::Look(View {
                    eye,
                    at,
                    daylight,
                    ashore: false,
                }),
                sculpting: false,
                pouring: false,
                hold: None,
            });
        }
    }
    script
}

/// Hold the eye at a viewpoint about the pool, as a ghost.
fn hold(app: &mut App, view: View) {
    let pool = *app.world().resource::<Pool>();
    let shore = if view.ashore { pool.shore } else { 0.0 };
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.avatar_mut().solid = false;
    let eye = spot(&sim, pool, [view.eye[0] + shore, view.eye[1], view.eye[2]]);
    let at = spot(&sim, pool, [view.at[0] + shore, view.at[1], view.at[2]]);
    stand(&mut sim, eye, at);
}

fn cue(app: &mut App, cue: Cue) {
    match cue {
        Cue::None => {}
        Cue::Look(view) => {
            app.world_mut().resource_mut::<Settings>().collisions = false;
            // start the shot half its own length before the sun stands highest over the pool,
            // or lowest behind the ring, so that it is as bright at both its ends as it is at
            // its start: the ring turns once in about six seconds, so a place on it sees three
            // of daylight, and a shot that starts anywhere in that turn ends in the dark. The
            // sun keeps its place among the stars while the wheel turns under it, so how far
            // over the pool it ever gets is set by how far along the axis it stands.
            let over = |app: &App| -app.world().resource::<Sky>().sun.x as f64;
            let highest = {
                let sun = app.world().resource::<Sky>().sun;
                (1.0 - (sun.y * sun.y) as f64).sqrt()
            };
            let spin = app.world().resource::<Simulation>().drum.spin.0 as f64;
            let half = (spin * DAY as f64 / 2.0).min(std::f64::consts::FRAC_PI_2);
            let wanted = highest * half.cos() * if view.daylight { 1.0 } else { -1.0 };
            let mut before = over(app);
            let mut waited = 0.0;
            loop {
                hold(app, view);
                let now = over(app);
                let climbing = now > before;
                before = now;
                let ready = if view.daylight {
                    climbing && now >= wanted
                } else {
                    !climbing && now <= wanted
                };
                if ready {
                    break;
                }
                testing::watch(app, Seconds(1.0 / FPS as f32));
                waited += 1.0 / FPS as f32;
                assert!(
                    waited < 60.0,
                    "the sun never came round: it stands at {:?} with the ring spinning at {} rad/s",
                    app.world().resource::<Sky>().sun,
                    app.world().resource::<Simulation>().drum.spin.0
                );
            }
            hold(app, view);
        }
        Cue::Flood => {
            for k in 0..30 {
                app.world_mut()
                    .resource_scope(|world, mut fluid: Mut<Fluid>| {
                        let sim = world.resource::<Simulation>();
                        let radius = sim.drum.ring.radius.0 as f64;
                        let turn = (-7.0 + (k % 3) as f64 * 1.5 - 1.5) / radius;
                        let on = sim.drum.wall_point(turn, (k % 5) as f64 - 2.0);
                        let (_, out) = sim.drum.depth_and_outward(on);
                        let height = 2.5 + (k / 5) as f64 * 0.5;
                        let centre = [
                            on[0] - out[0] * height,
                            on[1] - out[1] * height,
                            on[2] - out[2] * height,
                        ];
                        sim.inject(&mut fluid, centre, 1500)
                    });
            }
        }
        Cue::Enlarge => {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            Dial::Diameter.set(&mut settings, 30.0);
            Dial::Width.set(&mut settings, 16.0);
        }
        Cue::Equalize => app.world_mut().resource_mut::<Settings>().equalize_thrust(),
        Cue::Basin => {
            let pool = {
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                let site = sim.drum.site;
                let radius = sim.drum.ring.radius.0 as f64;
                // two broad ridges across the ring, a valley between them for the pool
                for crest in [RIDGE_NEAR, RIDGE_FAR] {
                    let phi = site.phi + crest / radius;
                    for y in [-5.5, -2.75, 0.0, 2.75, 5.5] {
                        for _ in 0..20 {
                            sim.drum
                                .landscape
                                .sculpt(phi, site.y + y, 8.0, RIDGE_HEIGHT / 20.0);
                        }
                    }
                }
                Pool {
                    phi: site.phi,
                    y: site.y,
                    shore: 0.0,
                }
            };
            app.world_mut().insert_resource(pool);
            // the water is laid in low and in small heaps, since water dropped from a height in
            // a small ring lands with the spin it lacked and sloshes about
            for k in 0..15 {
                app.world_mut()
                    .resource_scope(|world, mut fluid: Mut<Fluid>| {
                        let sim = world.resource::<Simulation>();
                        let arc = -1.5 - (k % 3) as f64 * 2.5;
                        let y = (k / 3) as f64 * 1.5 - 3.0;
                        let centre = spot(sim, pool, [arc, y, HEAP_CLEARANCE]);
                        sim.inject(&mut fluid, centre, POOL_PARTICLES / 15)
                    });
                // each heap settles before the next lands beside it
                testing::run(app, Seconds(HEAP_INTERVAL));
            }
            testing::run(app, Seconds(20.0));
            // then the avatar takes its place on the pool's bed where it lies deepest, looking
            // along the pool
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            let eye = spot(&sim, pool, [POOL_DEEP, 0.0, avatar::EYE_HEIGHT.0 as f64]);
            stand(&mut sim, eye, [eye[0] + 0.3, eye[1], eye[2] + 6.0]);
        }
        Cue::Sea => {
            {
                let mut settings = app.world_mut().resource_mut::<Settings>();
                Dial::Diameter.set(&mut settings, SEA_DIAMETER);
                Dial::Width.set(&mut settings, SEA_WIDTH);
                settings.equalize_thrust();
            }
            testing::run(app, Seconds(1.0));
            let pool = {
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                let site = sim.drum.site;
                // a headland: a dome standing clear of the water, on a terrace laid under it
                // wide enough to hold a stretch of shallows round its foot. The ring is thirty
                // metres across and the sea in it four deep, so the terrace can only be so
                // wide before it is the ring: a shore here is steep, as a shore on a small
                // island is.
                for _ in 0..20 {
                    sim.drum.landscape.sculpt(
                        site.phi,
                        site.y,
                        HEADLAND_SPREAD,
                        HEADLAND_HEIGHT / 20.0,
                    );
                    sim.drum.landscape.sculpt(
                        site.phi,
                        site.y,
                        TERRACE_SPREAD,
                        TERRACE_HEIGHT / 20.0,
                    );
                }
                Pool {
                    phi: site.phi,
                    y: site.y,
                    shore: 0.0,
                }
            };
            app.world_mut().insert_resource(pool);
            // enough water to stand SEA_DEPTH deep over the whole floor, laid in heaps all
            // round: how many particles that is depends on how coarse the water has grown by
            // then, so each heap is asked for by the volume it carries rather than by a count
            let target = {
                let ring = app.world().resource::<Simulation>().drum.ring;
                let floor =
                    std::f64::consts::TAU * ring.radius.0 as f64 * 2.0 * ring.half_width.0 as f64;
                Litres((floor * SEA_DEPTH * 1000.0) as f32)
            };
            for k in 0..SEA_HEAPS {
                app.world_mut()
                    .resource_scope(|world, mut fluid: Mut<Fluid>| {
                        let sim = world.resource::<Simulation>();
                        let turn = k as f64 / SEA_HEAPS as f64 * std::f64::consts::TAU;
                        let y = ((k % 5) as f64 - 2.0) * sim.drum.ring.half_width.0 as f64 * 0.4;
                        let on = sim.drum.wall_point(turn, y);
                        let (_, out) = sim.drum.depth_and_outward(on);
                        let centre = [
                            on[0] - out[0] * HEAP_CLEARANCE,
                            on[1] - out[1] * HEAP_CLEARANCE,
                            on[2] - out[2] * HEAP_CLEARANCE,
                        ];
                        let left = target.0 - fluid.litres().0;
                        let heap = left / (SEA_HEAPS - k) as f32;
                        let count = heap / fluid.resolution().litres_per_particle().0;
                        sim.inject(&mut fluid, centre, count.max(0.0).round() as u32)
                    });
                testing::run(app, Seconds(0.4));
            }
            testing::run(app, Seconds(60.0));
            // where the water's edge lies on the headland's flank: the first place going out
            // from its top where the ground no longer stands above the level the sea found
            let shore = {
                let litres = app.world().resource::<Fluid>().litres().0 as f64;
                let sim = app.world().resource::<Simulation>();
                let level = sim.drum.landscape.flooded(litres / 1000.0).level.0 as f64;
                let radius = sim.drum.ring.radius.0 as f64;
                let mut arc = 0.0;
                while arc < TERRACE_SPREAD * 2.0
                    && sim.drum.landscape.sample(pool.phi + arc / radius, pool.y).0 > level
                {
                    arc += SHORE_STEP;
                }
                arc
            };
            app.world_mut().insert_resource(Pool { shore, ..pool });
            app.world_mut().resource_mut::<Settings>().collisions = true;
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.avatar_mut().solid = true;
            let eye = spot(&sim, pool, [0.0, 0.0, avatar::EYE_HEIGHT.0 as f64]);
            let along = spot(&sim, pool, [30.0, 0.0, 3.0]);
            stand(&mut sim, eye, along);
        }
        Cue::Dive => {
            let pool = *app.world().resource::<Pool>();
            app.world_mut().resource_mut::<Settings>().collisions = true;
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.avatar_mut().solid = true;
            let eye = spot(&sim, pool, [DIVE_ARC, 0.0, DIVE_HEIGHT]);
            let ahead = spot(&sim, pool, [DIVE_ARC + 8.0, 0.0, DIVE_HEIGHT - 1.5]);
            stand(&mut sim, eye, ahead);
        }
        Cue::SpinUp => {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            let spin = settings.spin.0 * 1.5;
            Dial::Spin.set(&mut settings, spin);
        }
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

/// The water demo's pool: the flat floor between two ridges across the ring, each a row of
/// broad mounds this tall, holding this many particles laid in heaps this far clear of the
/// ground; the avatar starts on its bed at the near ridge's foot.
const RIDGE_NEAR: f64 = 8.0;
const RIDGE_FAR: f64 = -12.0;
const RIDGE_HEIGHT: f64 = 1.4;
const POOL_PARTICLES: u32 = 7500;
const HEAP_CLEARANCE: f64 = 1.8;
const HEAP_INTERVAL: f32 = 1.0;
/// Where the pool lies deepest: the middle of the valley between the ridges.
const POOL_DEEP: f64 = -2.0;

/// The sea: a ring wide enough for the water to run deep, a headland raised across it to stand
/// on and to shelve away into the shallows, and this many heaps of water laid all round it.
const SEA_DIAMETER: f32 = 30.0;
const SEA_WIDTH: f32 = 16.0;
const HEADLAND_HEIGHT: f64 = 8.0;
const HEADLAND_SPREAD: f64 = 11.0;
/// The terrace the headland stands on, which holds the shallows round its foot.
const TERRACE_HEIGHT: f64 = 3.2;
const TERRACE_SPREAD: f64 = 17.0;
const SEA_DEPTH: f64 = 5.0;
const SEA_HEAPS: usize = 64;
/// Where the avatar is let go over the sea, clear of the headland's shelf, and how far over the
/// ground it starts: high enough to be over the water rather than in it.
const DIVE_ARC: f64 = 24.0;
const DIVE_HEIGHT: f64 = 6.0;
/// How finely the headland's flank is walked to find where the water's edge falls on it.
const SHORE_STEP: f64 = 0.1;
/// How long a held shot lasts. A place on the ring sees about three seconds of daylight in
/// every turn, and a shot held longer than that ends in the dark whatever it was of.
const DAY: f32 = 2.5;

/// A place about the pool, in the frame: `arc` metres round the ring from it, `y` along the
/// axis and `height` over the ground there, which is under the ground, and out through the
/// glass, when negative.
fn spot(sim: &Simulation, pool: Pool, [arc, y, height]: [f64; 3]) -> [f64; 3] {
    let drum = &sim.drum;
    let radius = drum.ring.radius.0 as f64;
    let phi = pool.phi + arc / radius;
    let axial = pool.y + y;
    let ground = drum.landscape.sample(phi, axial).0;
    let on = drum.wall_point(phi - drum.site.phi, axial - drum.site.y);
    let (_, out) = drum.depth_and_outward(on);
    let lift = ground + height;
    [
        on[0] - out[0] * lift,
        on[1] - out[1] * lift,
        on[2] - out[2] * lift,
    ]
}

/// Put the avatar at rest with its eye at `eye`, facing `target`, standing up on the ring.
fn stand(sim: &mut Simulation, eye: [f64; 3], target: [f64; 3]) {
    let (_, outward) = sim.drum.depth_and_outward(eye);
    let mut back = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
    let len = norm(&back);
    back = back.map(|c| c / len);
    let mut right = cross(&outward.map(|c| -c), &back);
    if norm(&right) < 1e-6 {
        // looking straight up or down, any way round is upright: face along the axis
        right = cross(&[0.0, 1.0, 0.0], &back);
    }
    let len = norm(&right);
    right = right.map(|c| c / len);
    let up = cross(&back, &right);
    let q = quat_from_basis(&right, &up, &back);
    let head = quat_rotate(&q, &avatar::eye_offset());
    sim.avatar_mut()
        .place([eye[0] - head[0], eye[1] - head[1], eye[2] - head[2]], q);
    sim.gyros = Gyros::holding(sim.avatar());
}

fn main() {
    let out = Path::new("target/record");
    let _ = fs::remove_dir_all(out);
    fs::create_dir_all(out).expect("create target/record");
    let mut app = testing::headless();
    let image = testing::render_to_image(&mut app, WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE);
    testing::watch(&mut app, Seconds(0.5));

    let frame_time = Seconds(1.0 / FPS as f32);
    let mut srt = String::new();
    let mut soundtrack = Soundtrack::new(out.join("thrusters.wav"));
    let script = match std::env::args().nth(1).as_deref() {
        None => script(),
        Some("marker") => {
            app.world_mut().resource_mut::<Aim>().engaged = true;
            marker_script()
        }
        Some("water") => water_script(),
        Some("survey") => survey_script(),
        Some("sea") => sea_script(),
        Some(other) => {
            panic!("unknown script {other:?}: the others are `marker`, `water`, `survey` and `sea`")
        }
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
            if phase.pouring
                && let Some(target) = app.world().resource::<Aim>().target
            {
                let at = target.point + target.normal * INJECT_DEPTH.0 as f64;
                app.world_mut()
                    .resource_scope(|world, mut fluid: Mut<Fluid>| {
                        let settings = world.resource::<Settings>();
                        let count = settings.flow.0 * frame_time.0
                            / fluid.resolution().litres_per_particle().0;
                        let sim = world.resource::<Simulation>();
                        sim.inject(&mut fluid, at.to_array(), count.round() as u32)
                    });
            }
            if let Some(view) = phase.hold {
                hold(&mut app, view);
            }
            testing::watch(&mut app, frame_time);
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
                WIDTH * SUPERSAMPLE,
                HEIGHT * SUPERSAMPLE,
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
