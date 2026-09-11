//! A sweep over everything the ring can be made into, looking for what no scene should ever do.
//!
//! Nothing here is a picture that was aimed at. Every case is drawn at random from the whole
//! range the dials allow — rings from a room across to a hundred kilometres, widths from a
//! corridor to longer than the ring is round, spins from none to one no body would survive, air
//! from vacuum to a hundred atmospheres of fog, seas from dry to brim full — and looked at from
//! wherever the eye can stand: on the ground, adrift in the middle, at the axis, pressed against
//! the glass, out in space, and inside the ground itself. What is asserted of each is only what
//! holds of every scene alike, and every frame is written to `target/tmp/hunt/` to be looked at.
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::fluid::Fluid;
use game::core::math::{Vec3d, cross, norm, quat_from_basis, quat_rotate};
use game::core::units::{
    EARTH_GRAVITY, KilogramsPerCubicMetre, Metres, Pascals, Radians, RadiansPerSecond, Seconds,
};
use game::systems::air::{Air, Suspension};
use game::systems::drum::GROUND_DEPTH;
use game::systems::drum::Ring;
use game::systems::player::{PilotInput, Player};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing::{self, Headless};
use rand::prelude::*;
use rand::rngs::SmallRng;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
/// How many worlds the sweep visits. Every one of them is built, settled and drawn three times.
const CASES: usize = 400;
/// The seed the sweep is drawn from, so that a case that fails can be looked at again.
const SEED: u64 = 0x5011_D1CE;
/// How far off its aim the head is turned for the last frame of a case, in degrees. However big
/// or small the ring is, a turn this small moves the picture by a fraction of a pixel.
const HAIR: f64 = 0.01;
/// How long a walker is given to come down from a hop and rub off what it landed with, in
/// seconds: a ring a hundred kilometres across turns slowly, and a hop in it is a long flight.
const SETTLING: f32 = 120.0;
/// How much of a frame may land either way between two draws of the very same scene: a pixel or
/// two of a large ring, and nothing more.
const STEADY: f64 = 1e-4;
/// How much of a frame may stand apart from its own neighbours, and how much of it may leap
/// under a hair's turn of the head. Rings far longer than they are round put their whole length
/// into a few pixels, and a frame of almost nothing but sky is metered up until its stars are
/// bright, so neither is ever nothing; what these catch is a picture coming apart.
const SPECKLED: f64 = 0.06;
const LEAPING: f64 = 0.3;
/// How much of a frame may be as bright as a frame can be in every channel. A frame of almost
/// nothing but sky is metered up until its stars are bright, and the sun is the sun; a third of
/// a picture gone white is something else.
const BLOWN: f64 = 0.35;

/// What a case is: a world, and somewhere to stand in it.
#[derive(Clone, Copy, Debug)]
struct Case {
    ring: Ring,
    spin: RadiansPerSecond,
    /// The air's pressure at the rim, in atmospheres, and what it carries.
    atmospheres: f64,
    carrying: Suspension,
    /// How much water the ring holds, as a share of what would drown the whole of it.
    sea: f64,
    /// How rough the ground is, as a share of the ring's radius.
    hills: f64,
    /// Where the eye stands, measured on the ring itself rather than from the site, which moves
    /// under it: how far round the wheel, how far along the axis, how far in from the wall.
    eye: Vec3d,
    /// Where it looks, in the frame the ring itself gives the place it stands: out from the
    /// axis, along the axis, and round the ring. None of the three moves when the site does.
    look: Vec3d,
    /// How much of a hair's breadth the head is turned for the last frame.
    nudge: f64,
    /// How far round the wheel has turned, which is where the sun stands.
    angle: Radians,
}

/// What was found in a case: what it looked like, and whether the world behind it holds together.
#[derive(Clone, Copy, Debug, Default)]
struct Reading {
    /// The share of the frame that is as bright as a frame can be in every channel.
    blown: f64,
    /// The share of it that stands apart from every one of its neighbours.
    speckle: f64,
    /// How many distinct colours the frame holds, as a share of its pixels: a frame that failed
    /// to draw is one flat colour.
    variety: f64,
    /// The share of the frame that changed wholesale when the head was turned by a hair.
    jitter: f64,
    /// The same with nothing whatever changed, which is what `jitter` is worth nothing below.
    /// A ring kilometres across is drawn from single-precision numbers as big as it is, so a
    /// handful of pixels may land either way; a picture that moves is something else.
    still: f64,
    /// Whether every number the world is made of is still a number.
    finite: bool,
}

#[test]
#[ignore = "wants a GPU"]
fn no_world_the_dials_allow_is_drawn_wrongly() {
    let mut hunt = Hunt::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut worst: Vec<(f64, usize, Case, Reading)> = Vec::new();
    for (n, case) in cases(CASES).into_iter().enumerate() {
        let reading = hunt.visit(case);
        if !reading.finite {
            wrong.push(format!(
                "{n}: the world stopped being made of numbers. {case:?}"
            ));
        }
        if reading.variety <= 0.0 {
            wrong.push(format!("{n}: nothing was drawn at all. {case:?}"));
        }
        if reading.still > STEADY {
            wrong.push(format!(
                "{n}: {:.3} of the frame changed with nothing changed. {case:?}",
                reading.still
            ));
        }
        if reading.blown > BLOWN {
            wrong.push(format!(
                "{n}: {:.3} of the frame is as bright as a frame can be. {case:?}",
                reading.blown
            ));
        }
        if reading.speckle > SPECKLED {
            wrong.push(format!(
                "{n}: {:.3} of the frame stands apart from its neighbours. {case:?}",
                reading.speckle
            ));
        }
        if reading.jitter > LEAPING {
            wrong.push(format!(
                "{n}: {:.3} of the frame leapt under a hair's turn. {case:?}",
                reading.jitter
            ));
        }
        worst.push((reading.jitter.max(reading.speckle * 20.0), n, case, reading));
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("a finite score"));
    for (_, n, case, reading) in worst.iter().take(20) {
        println!("{n:3} {reading:?} {case:?}");
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// What a viewer actually does: stand on the ground of a ring and walk it, turn, hop and wade,
/// in rings of every size the dials allow. Nothing about the picture is asserted here, only that
/// the world the picture is of stays a world — the ground holds the walker up, the drum keeps it
/// in, and nothing it does turns any of its numbers into something that is not a number.
#[test]
#[ignore = "wants a GPU"]
fn walking_any_ring_leaves_it_standing() {
    use game::core::avatar::Thruster::*;
    let mut app = testing::headless();
    let mut wrong: Vec<String> = Vec::new();
    for diameter in [8.0f32, 21.0, 200.0, 2_000.0, 20_000.0, 100_000.0] {
        let ring = Ring {
            radius: Metres(diameter / 2.0),
            half_width: Metres((diameter / 4.0).clamp(1.0, 500.0)),
        };
        app.world_mut().resource_mut::<Fluid>().clear();
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.reset();
            sim.resize(ring);
            sim.drum.spin = standing_spin(ring);
            sim.drum.target_spin = standing_spin(ring);
            sim.drum.angle = Radians(0.0);
            sim.drum.landscape.flatten(Metres(0.0));
            // stand the walker upright where the last ring left it, and put the site under it,
            // so that the same walk is walked in the same numbers however it got here
            let up = [-1.0, 0.0, 0.0];
            let back = [0.0, 0.0, 1.0];
            let q = quat_from_basis(&cross(&up, &back), &up, &back);
            let stand = [
                -(GROUND_DEPTH.0 as f64 + avatar::standing_height()),
                0.0,
                0.0,
            ];
            sim.avatar_mut().place(stand, q);
            sim.avatar_mut().v = [0.0; 3];
            sim.avatar_mut().w = [0.0; 3];
            sim.avatar_mut().ground = None;
            sim.avatar_mut().solid = true;
            sim.gyros = Gyros::holding(sim.avatar());
            let here = sim.avatar().p;
            sim.resite(here);
            let brush = (diameter as f64 / 40.0).max(2.0);
            for k in 0..4 {
                let at = sim
                    .drum
                    .wall_point(brush * (k as f64 - 1.5) / ring.radius.0 as f64, 0.0);
                sim.sculpt(
                    at,
                    brush,
                    if k % 2 == 0 {
                        brush / 8.0
                    } else {
                        -brush / 8.0
                    },
                );
            }
        }
        let centre = {
            let sim = app.world().resource::<Simulation>();
            sim.drum.wall_point(0.0, 0.0)
        };
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let sim = world.resource::<Simulation>();
                sim.inject(&mut fluid, centre, 4_000);
            });
        testing::run(&mut app, Seconds(1.0));
        let walk: [(&[game::core::avatar::Thruster], f32); 7] = [
            (&[Forward], 3.0),
            (&[YawLeft], 1.0),
            (&[Forward, Up], 1.5),
            (&[Forward], 3.0),
            (&[Left], 2.0),
            (&[YawRight, Back], 2.0),
            (&[], 2.0),
        ];
        for (held, seconds) in walk {
            let pilot = PilotInput::firing(held);
            for _ in 0..(seconds / 0.1).round() as usize {
                {
                    let player = *app.world().resource::<Player>();
                    let mut sim = app.world_mut().resource_mut::<Simulation>();
                    sim.avatar_input = player.input(pilot);
                }
                testing::run(&mut app, Seconds(0.1));
                let sim = app.world().resource::<Simulation>();
                let p = sim.avatar().p;
                let reach = sim.shapes()[sim.avatar().shape].reach();
                if !p.iter().all(|c| c.is_finite()) || !sim.avatar().v.iter().all(|c| c.is_finite())
                {
                    wrong.push(format!(
                        "{diameter} m: the walker stopped being a number at {p:?}"
                    ));
                    break;
                }
                let inside = sim.drum.height_above_glass(p);
                if inside < -reach {
                    wrong.push(format!(
                        "{diameter} m: the walker went {inside:.3} m through the glass"
                    ));
                    break;
                }
                let axial = sim.drum.axial(p).abs();
                if axial > sim.drum.ring.half_width.0 as f64 + reach {
                    wrong.push(format!(
                        "{diameter} m: the walker went {axial:.3} m out past the cap"
                    ));
                    break;
                }
            }
        }
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.avatar_input = Default::default();
        }
        // a hop in a ring kilometres across takes its time coming down, and what it lands with
        // takes longer still to rub off, so the walker is given as long as it needs
        let mut waited = 0.0;
        while waited < SETTLING {
            testing::run(&mut app, Seconds(0.5));
            waited += 0.5;
            let footing = app.world().resource::<Simulation>().footing();
            if !footing.airborne && footing.ground_speed < 0.5 {
                break;
            }
        }
        let footing = app.world().resource::<Simulation>().footing();
        if footing.airborne || !(0.3..3.0).contains(&footing.weight) {
            wrong.push(format!(
                "{diameter} m: after {waited} s the walker is {footing:?} rather than standing at one g"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The drum is a sealed vessel: nothing that is in it gets out of it, however hard it is thrown
/// at the hull. Which is worth its own test because a sphere is judged to be in the drum or out
/// of it by where its centre stands, and the hull pushes whatever is outside it further out — so
/// a body whose centre crosses the inner face between one settling of the contacts and the next
/// meets the outside of the hull and is thrown clear. That makes the hull a trapdoor rather than
/// a wall, and the way out of it is for a body still touching the hull to count as inside.
#[test]
#[ignore = "wants a GPU"]
fn nothing_thrown_at_the_hull_gets_out_of_it() {
    let mut app = testing::headless();
    let mut wrong: Vec<String> = Vec::new();
    let fastest = app.world().resource::<Simulation>().body_params.max_speed.0 as f64;
    for diameter in [8.0f32, 21.0, 200.0, 2_000.0, 100_000.0] {
        let ring = Ring {
            radius: Metres(diameter / 2.0),
            half_width: Metres((diameter / 4.0).clamp(1.0, 500.0)),
        };
        // at a cap, into the floor, and at the corner where the two meet
        for way in [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.7, 0.7, 0.0],
        ] {
            {
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                sim.reset();
                sim.resize(ring);
                sim.drum.spin = standing_spin(ring);
                sim.drum.target_spin = standing_spin(ring);
                sim.drum.angle = Radians(0.0);
                sim.drum.landscape.flatten(Metres(0.0));
                let up = [-1.0, 0.0, 0.0];
                let back = [0.0, 0.0, 1.0];
                let q = quat_from_basis(&cross(&up, &back), &up, &back);
                let stand = [
                    -(GROUND_DEPTH.0 as f64 + avatar::standing_height()),
                    0.0,
                    0.0,
                ];
                sim.avatar_mut().place(stand, q);
                // as fast as the ring itself could ever get a body going, and no faster than
                // the solver lets anything go: a drum a few metres long has no room to build up
                // a speed it would take a hundred to reach
                let room = sim.drum.ring.half_width.0 as f64 + sim.drum.ring.radius.0 as f64;
                let thrown = fastest.min((2.0 * EARTH_GRAVITY.0 * room).sqrt()) * 0.99;
                sim.avatar_mut().v = way.map(|c| c * thrown);
                sim.avatar_mut().w = [0.0; 3];
                sim.avatar_mut().ground = None;
                sim.avatar_mut().solid = true;
                sim.gyros = Gyros::holding(sim.avatar());
                let here = sim.avatar().p;
                sim.resite(here);
            }
            let reach = {
                let sim = app.world().resource::<Simulation>();
                sim.shapes()[sim.avatar().shape].reach()
            };
            for _ in 0..200 {
                testing::run(&mut app, Seconds(0.05));
                let sim = app.world().resource::<Simulation>();
                let p = sim.avatar().p;
                let out = -sim.drum.height_above_glass(p);
                let past = sim.drum.axial(p).abs() - sim.drum.ring.half_width.0 as f64;
                if out > reach || past > reach {
                    wrong.push(format!(
                        "{diameter} m: a body thrown {way:?} ended {out:.2} m through the wall and {past:.2} m past the cap"
                    ));
                    break;
                }
            }
        }
    }
    // and a body already half through the hull belongs to the room it came from: it is put back
    // in rather than met by the outside and thrown clear
    for diameter in [21.0f32, 200.0, 2_000.0] {
        let ring = Ring {
            radius: Metres(diameter / 2.0),
            half_width: Metres((diameter / 4.0).clamp(1.0, 500.0)),
        };
        let reach = {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.reset();
            sim.resize(ring);
            sim.drum.spin = standing_spin(ring);
            sim.drum.target_spin = standing_spin(ring);
            sim.drum.angle = Radians(0.0);
            sim.drum.landscape.flatten(Metres(0.0));
            sim.shapes()[sim.avatar().shape].reach()
        };
        // far enough past the cap that the body's own bulk has crossed it, and not so far that
        // the bulk is clear of the hull altogether
        for past in [0.05, 0.2, 0.45] {
            {
                let mut sim = app.world_mut().resource_mut::<Simulation>();
                let up = [-1.0, 0.0, 0.0];
                let back = [0.0, 0.0, 1.0];
                let q = quat_from_basis(&cross(&up, &back), &up, &back);
                let half = sim.drum.ring.half_width.0 as f64;
                let stand = [
                    -(GROUND_DEPTH.0 as f64 + avatar::standing_height()),
                    half + past - sim.drum.site.y,
                    0.0,
                ];
                sim.avatar_mut().place(stand, q);
                sim.avatar_mut().v = [0.0, 1.0, 0.0];
                sim.avatar_mut().w = [0.0; 3];
                sim.avatar_mut().ground = None;
                sim.avatar_mut().solid = true;
                sim.gyros = Gyros::holding(sim.avatar());
            }
            testing::run(&mut app, Seconds(2.0));
            let sim = app.world().resource::<Simulation>();
            let over = sim.drum.axial(sim.avatar().p).abs() - sim.drum.ring.half_width.0 as f64;
            if over > reach {
                wrong.push(format!(
                    "{diameter} m: a body {past:.2} m through the cap was pushed out to {over:.2} m past it rather than back in"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The worlds the sweep visits.
fn cases(count: usize) -> Vec<Case> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let mut cases = Vec::with_capacity(count);
    for _ in 0..count {
        let diameter = log_uniform(&mut rng, 6.0, 100_000.0);
        let width = log_uniform(&mut rng, 2.0, 20_000.0);
        let ring = Ring {
            radius: Metres(diameter as f32 / 2.0),
            half_width: Metres(width as f32 / 2.0),
        };
        let standing = standing_spin(ring).0 as f64;
        let spin = match rng.random_range(0..6) {
            0 => 0.0,
            1 => standing * 0.05,
            2 => standing,
            3 => standing * rng.random_range(1.0..4.0),
            4 => log_uniform(&mut rng, 0.001, 999.0),
            _ => standing * rng.random_range(0.2..1.0),
        }
        .min(999.0);
        let eye = standpoint(&mut rng, ring);
        cases.push(Case {
            ring,
            spin: RadiansPerSecond(spin as f32),
            atmospheres: match rng.random_range(0..4) {
                0 => 0.0,
                1 => 1.0,
                _ => log_uniform(&mut rng, 0.01, 200.0),
            },
            carrying: match rng.random_range(0..4) {
                0 => Suspension::CLEAR,
                1 => Suspension::DUST,
                2 => loaded(Suspension::FOG, log_uniform(&mut rng, 1e-6, 1e-3)),
                _ => loaded(Suspension::SMOKE, log_uniform(&mut rng, 1e-9, 1e-5)),
            },
            sea: match rng.random_range(0..3) {
                0 => 0.0,
                _ => rng.random_range(0.0..1.0),
            },
            hills: match rng.random_range(0..3) {
                0 => 0.0,
                _ => rng.random_range(0.0..0.02),
            },
            eye,
            look: gaze(&mut rng, ring, eye),
            nudge: rng.random_range(0.2..1.0),
            angle: Radians(rng.random_range(0.0..std::f64::consts::TAU)),
        });
    }
    cases
}

/// Somewhere the eye can stand, as a place on the ring: how far round it, how far along its
/// axis, and how far in from its wall. Height is picked from the places a viewer actually ends
/// up in rather than evenly, since the interesting ones — the ground, the glass, the axis and
/// space outside — are a vanishing share of the volume.
fn standpoint(rng: &mut SmallRng, ring: Ring) -> Vec3d {
    let radius = ring.radius.0 as f64;
    let half_width = ring.half_width.0 as f64;
    let height = match rng.random_range(0..10) {
        // stood on the ground, which is where a viewer spends nearly all of its time
        0..=2 => 1.7,
        // buried in the ground
        3 => -0.5,
        // pressed against the glass, and a hand's breadth off it
        4 => 0.02,
        5 => 0.3,
        // adrift between the wall and the far side
        6 | 7 => radius * rng.random_range(0.2..1.8),
        // at the axis itself, where the ring's own frame is singular
        8 => radius,
        // out in space
        _ => -radius * rng.random_range(0.2..3.0),
    };
    [
        radius * rng.random_range(-3.0..3.0),
        half_width * rng.random_range(-1.6..1.6),
        height,
    ]
}

/// Where an eye standing at a place looks. An eye within the ring sees the ring whichever way it
/// turns, so it turns anywhere; one out in space sees a speck of ring in a sky of nothing, and a
/// frame of nothing tells us nothing, so it is pointed back at the ring, which from out there
/// lies the way the ring's own frame calls outward.
fn gaze(rng: &mut SmallRng, ring: Ring, eye: Vec3d) -> Vec3d {
    let anywhere = direction(rng);
    let height = eye[2];
    if height > 0.0 && height < ring.radius.0 as f64 * 2.0 {
        return anywhere;
    }
    let spread = rng.random_range(0.0..0.6);
    let d = [
        1.0 + anywhere[0] * spread,
        anywhere[1] * spread,
        anywhere[2] * spread,
    ];
    let len = norm(&d).max(1e-9);
    d.map(|c| c / len)
}

fn round_unit(v: &mut Vec3d, len: f64) -> Vec3d {
    v.map(|c| c / len)
}

fn direction(rng: &mut SmallRng) -> Vec3d {
    loop {
        let d = [
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ];
        let len = norm(&d);
        if len > 1e-3 {
            return d.map(|c| c / len);
        }
    }
}

fn log_uniform(rng: &mut SmallRng, low: f64, high: f64) -> f64 {
    rng.random_range(low.ln()..high.ln()).exp()
}

fn loaded(carrying: Suspension, loading: f64) -> Suspension {
    Suspension {
        loading: KilogramsPerCubicMetre(loading),
        ..carrying
    }
}

/// The one app every case is played in: a process holds one GPU device, so the worlds are made
/// and unmade in place rather than one app to a case.
struct Hunt {
    app: Headless,
    image: Handle<Image>,
    dir: PathBuf,
    seen: usize,
}

impl Hunt {
    fn new() -> Hunt {
        let mut app = testing::headless();
        app.world_mut().resource_mut::<Settings>().collisions = false;
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hunt");
        fs::create_dir_all(&dir).expect("create the folder");
        Hunt {
            app,
            image,
            dir,
            seen: 0,
        }
    }

    fn visit(&mut self, case: Case) -> Reading {
        self.seen += 1;
        self.build(case);
        // the eye takes a moment to adapt, and what it was shown last was another world
        for _ in 0..45 {
            self.aim(case, 0.0);
            testing::watch(&mut self.app, Seconds(1.0 / 30.0));
        }
        self.aim(case, 0.0);
        testing::draw_into(&mut self.app, &self.image);
        testing::frame(&mut self.app, Seconds(0.0));
        let first = testing::capture(&mut self.app, &self.image);
        self.aim(case, 0.0);
        testing::frame(&mut self.app, Seconds(0.0));
        let again = testing::capture(&mut self.app, &self.image);
        self.aim(case, case.nudge);
        testing::frame(&mut self.app, Seconds(0.0));
        let nudged = testing::capture(&mut self.app, &self.image);
        let mut reading = read(&again, self.finite());
        reading.still = leapt(&first, &again);
        reading.jitter = leapt(&again, &nudged);
        for (tag, frame) in [("0", &first), ("a", &again), ("b", &nudged)] {
            image::save_buffer(
                self.dir.join(format!("{:03}{tag}.png", self.seen)),
                frame,
                WIDTH,
                HEIGHT,
                image::ColorType::Rgba8,
            )
            .expect("write frame");
        }
        reading
    }

    fn build(&mut self, case: Case) {
        self.app.insert_resource(Air {
            pressure: Pascals(101_325.0 * case.atmospheres),
            carries: case.carrying,
            ..Air::default()
        });
        self.app.world_mut().resource_mut::<Fluid>().clear();
        {
            let mut sim = self.app.world_mut().resource_mut::<Simulation>();
            sim.reset();
            sim.resize(case.ring);
            sim.drum.spin = case.spin;
            sim.drum.target_spin = case.spin;
            sim.drum.angle = case.angle;
            sim.drum.landscape.flatten(Metres(0.0));
            sim.avatar_mut().solid = false;
        }
        if case.hills > 0.0 {
            let radius = case.ring.radius.0 as f64;
            let mut rng = SmallRng::seed_from_u64(case.angle.0.to_bits());
            for _ in 0..6 {
                let phi = rng.random_range(-0.4..0.4);
                let y = case.ring.half_width.0 as f64 * rng.random_range(-0.9..0.9);
                let brush = radius * rng.random_range(0.005..0.05);
                let raise = radius * case.hills * rng.random_range(-1.0..1.0);
                let mut sim = self.app.world_mut().resource_mut::<Simulation>();
                let at = sim.drum.wall_point(phi, y);
                sim.sculpt(at, brush, raise);
            }
        }
        if case.sea > 0.0 {
            let centre = {
                let sim = self.app.world().resource::<Simulation>();
                sim.drum.wall_point(0.0, 0.0)
            };
            let count = (case.sea * 20_000.0) as u32;
            self.app
                .world_mut()
                .resource_scope(|world, mut fluid: Mut<Fluid>| {
                    let sim = world.resource::<Simulation>();
                    sim.inject(&mut fluid, centre, count);
                });
        }
        testing::run(&mut self.app, Seconds(0.2));
    }

    /// Stand the eye where the case asks, as a ghost, with the head turned `nudge` of a hair off
    /// its aim. Where it stands is a place on the ring carried into whatever frame the site
    /// happens to be in now, so that the site moving under it leaves it where it was.
    fn aim(&mut self, case: Case, nudge: f64) {
        let mut sim = self.app.world_mut().resource_mut::<Simulation>();
        sim.avatar_mut().solid = false;
        let radius = sim.drum.ring.radius.0 as f64;
        let (arc, axial, height) = (case.eye[0], case.eye[1], case.eye[2]);
        let on = sim
            .drum
            .wall_point(arc / radius - sim.drum.site.phi, axial - sim.drum.site.y);
        let (_, out) = sim.drum.depth_and_outward(on);
        let at = [
            on[0] - out[0] * height,
            on[1] - out[1] * height,
            on[2] - out[2] * height,
        ];
        // the way the eye looks is given in the ring's own frame at the place it stands, so
        // that the site moving under it turns neither the eye nor what the eye is looking at
        let along = [0.0, 1.0, 0.0];
        let mut round = cross(&out, &along);
        let len = norm(&round).max(1e-9);
        let round = round_unit(&mut round, len);
        let aside = direction(&mut SmallRng::seed_from_u64(case.angle.0.to_bits()));
        let turned = nudge * HAIR.to_radians();
        let mix = [
            case.look[0] + aside[0] * turned,
            case.look[1] + aside[1] * turned,
            case.look[2] + aside[2] * turned,
        ];
        let look = [
            out[0] * mix[0] + along[0] * mix[1] + round[0] * mix[2],
            out[1] * mix[0] + along[1] * mix[1] + round[1] * mix[2],
            out[2] * mix[0] + along[2] * mix[1] + round[2] * mix[2],
        ];
        let len = norm(&look).max(1e-9);
        let back = look.map(|c| -c / len);
        let mut right = cross(&out.map(|c| -c), &back);
        if norm(&right) < 1e-3 {
            right = cross(&along, &back);
        }
        let len = norm(&right);
        let right = right.map(|c| c / len);
        let up = cross(&back, &right);
        let turn = quat_from_basis(&right, &up, &back);
        let head = quat_rotate(&turn, &avatar::eye_offset());
        let p = [at[0] - head[0], at[1] - head[1], at[2] - head[2]];
        sim.avatar_mut().place(p, turn);
        sim.avatar_mut().v = [0.0; 3];
        sim.avatar_mut().w = [0.0; 3];
        sim.gyros = Gyros::holding(sim.avatar());
    }

    /// Whether every number the world is made of is still a number.
    fn finite(&self) -> bool {
        let sim = self.app.world().resource::<Simulation>();
        let avatar = sim.avatar();
        let body = avatar
            .p
            .iter()
            .chain(avatar.v.iter())
            .chain(avatar.w.iter())
            .all(|c| c.is_finite())
            && avatar.q.iter().all(|c| c.is_finite());
        let ground = (0..16).all(|k| {
            let phi = k as f64 * std::f64::consts::TAU / 16.0;
            sim.drum.landscape.sample(phi, 0.0).0.is_finite()
        });
        body && ground
    }
}

/// The share of two frames that changed wholesale between them, channel for channel.
fn leapt(before: &[u8], after: &[u8]) -> f64 {
    let mut leapt = 0usize;
    for (a, b) in before.chunks_exact(4).zip(after.chunks_exact(4)) {
        if (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 60) {
            leapt += 1;
        }
    }
    leapt as f64 / (before.len() / 4) as f64
}

/// What a frame says about itself.
fn read(pixels: &[u8], finite: bool) -> Reading {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let at = |x: usize, y: usize, c: usize| pixels[4 * (y * width + x) + c] as i32;
    let neighbours = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
    let mut blown = 0usize;
    let mut speckles = 0usize;
    let mut seen = HashSet::new();
    for y in 0..height {
        for x in 0..width {
            let here = [at(x, y, 0), at(x, y, 1), at(x, y, 2)];
            if here.iter().all(|&c| c >= 254) {
                blown += 1;
            }
            seen.insert((here[0] / 4, here[1] / 4, here[2] / 4));
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                continue;
            }
            let around = |dx: i32, dy: i32, c: usize| {
                at((x as i32 + dx) as usize, (y as i32 + dy) as usize, c)
            };
            let apart = (0..3)
                .map(|c| {
                    neighbours
                        .iter()
                        .map(|(dx, dy)| (here[c] - around(*dx, *dy, c)).abs())
                        .min()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            let lit = neighbours
                .iter()
                .map(|(dx, dy)| (0..3).map(|c| around(*dx, *dy, c)).max().unwrap_or(0))
                .max()
                .unwrap_or(0);
            if apart > 24 && lit > 40 {
                speckles += 1;
            }
        }
    }
    let count = (width * height) as f64;
    Reading {
        blown: blown as f64 / count,
        speckle: speckles as f64 / count,
        variety: (seen.len() as f64 - 1.0).max(0.0) / count,
        jitter: 0.0,
        still: 0.0,
        finite,
    }
}
