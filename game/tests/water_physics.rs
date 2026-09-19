//! Water as water is: what is asserted here is what is known of the real thing, measured, and
//! held against what the solver does with the same water in the same place. The ring's spin is
//! the gravity, so everything is measured in the ring's own turning frame, where a particle's
//! energy is what it has of motion less what the spin has given it for standing far from the
//! axis.
//!
//! Every measurement is also a reading on a scoreboard: how far it is from the real thing, as a
//! share of a scale of its own, written down beside the least it has ever been, which is kept
//! in `water_physics.json`. A reading may never be further off than that again, so that work on
//! the water can only ever bring it nearer to water; `just reference` prints the board.
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::fluid::{Fluid, Particle};
use game::core::math::norm;
use game::core::units::{Metres, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::Ring;
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;
use serde::{Deserialize, Serialize};

/// A ring wide enough round that its spin weighs nearly the same from the ground to the top of
/// any water stood on it here, and short enough along its axis for the water to reach from
/// one end to the other.
const RING: Ring = Ring {
    radius: Metres(30.0),
    half_width: Metres(8.0),
};

/// One thing measured of the water: what it came to, what it comes to in real water, and how
/// far off that is as a share of the reading's own scale.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct Reading {
    measured: f64,
    expected: f64,
    error: f64,
}

const BASELINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/water_physics.json");
/// Readings repeat from run to run to within this share of themselves, and this much.
const REPEATS_WITHIN: (f64, f64) = (0.05, 0.002);

fn readings() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("water_physics")
}

fn baseline() -> BTreeMap<String, Reading> {
    let kept = fs::read_to_string(BASELINE).expect("read the baseline");
    serde_json::from_str(&kept).expect("the baseline is a map of readings")
}

/// Hold a measurement against the real thing, write it down for the scoreboard, and fail if
/// it is further from the real thing than it has been before.
fn score(name: &str, measured: f64, expected: f64, scale: f64) -> f64 {
    let error = (measured - expected).abs() / scale;
    let reading = Reading {
        measured,
        expected,
        error,
    };
    fs::create_dir_all(readings()).expect("make the readings folder");
    let written = serde_json::to_string(&reading).expect("a reading is plain numbers");
    fs::write(readings().join(format!("{name}.json")), written).expect("write the reading");
    println!("{name}: {measured:.4}, in real water {expected:.4}, off by {error:.4}");
    let best = baseline()
        .get(name)
        .map_or(f64::INFINITY, |kept| kept.error);
    assert!(
        error <= best * (1.0 + REPEATS_WITHIN.0) + REPEATS_WITHIN.1,
        "{name} is off by {error:.4}, and has been off by as little as {best:.4}"
    );
    error
}

/// The board: every reading of the last run beside the least it has been off by. With
/// `WATER_PHYSICS_ACCEPT` set, the readings become the baseline, which they may only do when
/// none of them is worse than it.
#[test]
#[ignore = "reads what the measurements wrote"]
fn the_scoreboard() {
    let kept = baseline();
    let mut board = BTreeMap::new();
    for entry in fs::read_dir(readings()).expect("the measurements have been run") {
        let path = entry.expect("a reading").path();
        let name = path
            .file_stem()
            .expect("a name")
            .to_string_lossy()
            .into_owned();
        let written = fs::read_to_string(&path).expect("read the reading");
        let reading: Reading = serde_json::from_str(&written).expect("a reading");
        board.insert(name, reading);
    }
    println!(
        "{:<44}{:>12}{:>12}{:>10}{:>10}",
        "reading", "measured", "real", "off by", "best"
    );
    for (name, reading) in &board {
        let best = kept.get(name).map_or(f64::NAN, |kept| kept.error);
        println!(
            "{name:<44}{:>12.4}{:>12.4}{:>10.4}{best:>10.4}",
            reading.measured, reading.expected, reading.error
        );
    }
    if std::env::var_os("WATER_PHYSICS_ACCEPT").is_some() {
        for (name, reading) in &board {
            let best = kept.get(name).map_or(f64::INFINITY, |kept| kept.error);
            assert!(
                reading.error <= best * (1.0 + REPEATS_WITHIN.0) + REPEATS_WITHIN.1,
                "{name} is worse than the baseline, which only ever gets better"
            );
        }
        let mut accepted = kept;
        accepted.extend(board);
        let written = serde_json::to_string_pretty(&accepted).expect("plain numbers");
        fs::write(BASELINE, written + "\n").expect("write the baseline");
    }
}

fn ring_of_water(ring: Ring) -> testing::Headless {
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.collisions = false;
    }
    let mut sim = Simulation::new(ring);
    sim.drum.spin = standing_spin(ring);
    sim.drum.target_spin = standing_spin(ring);
    sim.drum.landscape.flatten(Metres(0.0));
    app.insert_resource(sim);
    testing::run(&mut app, Seconds(0.2));
    app
}

/// Stand water on the glass all the way round the ring, at rest on a lattice at its rest
/// spacing: a band of it from `along.0` to `along.1` of the axis, `depth` deep.
fn stand_a_band(app: &mut App, along: (f64, f64), depth: f64) -> usize {
    stand_a_band_moving(app, along, depth, |_| 0.0)
}

/// The same, each particle set moving along the axis as fast as `moving` says for where
/// along the axis it stands.
fn stand_a_band_moving(
    app: &mut App,
    along: (f64, f64),
    depth: f64,
    moving: impl Fn(f64) -> f64,
) -> usize {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let spacing = fluid.resolution().lattice().0 as f64;
            let radius = sim.drum.ring.radius.0 as f64;
            let first = sim.drum.wall_point(0.0, along.0 - sim.drum.site.y);
            sim.drum.settle_water(first);
            let layers = (depth / spacing).round() as usize;
            let rows = ((along.1 - along.0) / spacing).round() as usize;
            let mut added = 0;
            for layer in 0..layers {
                let height = (layer as f64 + 0.5) * spacing;
                let round = (std::f64::consts::TAU * (radius - height) / spacing).floor() as usize;
                for row in 0..rows {
                    let axial = along.0 + (row as f64 + 0.5) * spacing;
                    for k in 0..round {
                        let turn = std::f64::consts::TAU * k as f64 / round as f64;
                        let glass = sim.drum.wall_point(turn, axial - sim.drum.site.y);
                        let (_, outward) = sim.drum.depth_and_outward(glass);
                        let at = [0, 1, 2].map(|c| glass[c] - outward[c] * height);
                        let velocity =
                            sim.drum
                                .water_frame()
                                .vector_to_water([0.0, moving(axial), 0.0]);
                        added += fluid.add(Particle {
                            position: sim.drum.to_water(at),
                            velocity,
                            foam: 0.0,
                        }) as usize;
                    }
                }
            }
            added
        })
}

/// A particle about the ring's axis: how far from it, how far along it, and how fast it moves
/// in the water's frame.
struct Parcel {
    from_axis: f64,
    along: f64,
    speed: f64,
}

fn parcels(app: &mut App) -> Vec<Parcel> {
    let particles = testing::particles(app);
    let sim = app.world().resource::<Simulation>();
    let radius = sim.drum.ring.radius.0 as f64;
    particles
        .into_iter()
        .map(|p| {
            let [x, y, z] = sim.drum.from_water(p.position);
            Parcel {
                from_axis: (x + radius).hypot(z),
                along: y + sim.drum.site.y,
                speed: norm(&p.velocity),
            }
        })
        .collect()
}

/// The energy of the water in the ring's frame, per particle, in joules a kilogram: of its
/// motion, and of where it stands in the spin.
fn energy(parcels: &[Parcel], spin: f64) -> (f64, f64) {
    let n = parcels.len() as f64;
    let moving = parcels.iter().map(|p| 0.5 * p.speed * p.speed).sum::<f64>() / n;
    let standing = parcels
        .iter()
        .map(|p| -0.5 * spin * spin * p.from_axis * p.from_axis)
        .sum::<f64>()
        / n;
    (moving, standing)
}

/// Expected: a column of water twice as tall as it is thick, stood against the glass at one
/// end of the ring and let go, collapses and runs along the floor as a tongue. This is the
/// experiment of Martin and Moyce, and of every dam that has broken: the tongue's tip soon
/// runs at a steady speed, which for a column this shape they measured at about 1.9 times the
/// square root of the weight times the column's thickness, the ground's drag on it included,
/// and which no water can run faster than twice the square root of the weight times its
/// height. A column six particles thick is coarser than the experiment's water, and its tip a
/// little slower for it. All the while the water only ever loses energy, to its own churning: it never has more than it began with.
#[test]
#[ignore = "wants a GPU"]
fn a_collapsing_column_runs_out_as_fast_as_water_does() {
    let mut app = ring_of_water(RING);
    let (thick, tall) = (1.92, 3.84);
    let low = -(RING.half_width.0 as f64);
    let count = stand_a_band(&mut app, (low, low + thick), tall);
    let spin = standing_spin(RING).0 as f64;
    let weight = spin * spin * RING.radius.0 as f64;
    let began = energy(&parcels(&mut app), spin);
    let began = began.0 + began.1;
    let mut fronts = Vec::new();
    let mut most = began;
    let step = 0.1;
    for k in 1..=30 {
        testing::run(&mut app, Seconds(step as f32));
        let water = parcels(&mut app);
        assert_eq!(water.len(), count);
        // the tip of the tongue: all but a hundredth of the water is behind it
        let mut along: Vec<f64> = water.iter().map(|p| p.along - low).collect();
        along.sort_by(|a, b| a.partial_cmp(b).expect("a place"));
        let front = along[along.len() * 99 / 100];
        let (moving, standing) = energy(&water, spin);
        most = most.max(moving + standing);
        fronts.push((k as f64 * step, front));
    }
    let reached = |far: f64| {
        let k = fronts
            .iter()
            .position(|(_, front)| *front >= far)
            .unwrap_or_else(|| panic!("the tongue never ran {far} m: {fronts:?}"));
        let ((t0, f0), (t1, f1)) = (fronts[k.max(1) - 1], fronts[k]);
        t0 + (t1 - t0) * (far - f0) / (f1 - f0).max(1e-9)
    };
    // once the tip has run three thicknesses the column has all but fallen, and the tip runs steadily
    let speed = (6.0 * thick - 3.0 * thick) / (reached(6.0 * thick) - reached(3.0 * thick));
    let measured = speed / (weight * thick).sqrt();
    score("dam break: tongue speed", measured, 1.9, 1.9);
    let fallen = weight * tall / 2.0;
    score(
        "dam break: energy gained",
        (most - began).max(0.0),
        0.0,
        fallen,
    );
}

/// Expected: water two metres deep standing still all round a narrow ring stays two metres
/// deep. Water hardly squashes: it takes kilometres of it to press what is under them a
/// hundredth closer, so the top of the water stands as high as the water put in fills, to
/// within what its particles are coarse by. And water at rest stays at rest.
#[test]
#[ignore = "wants a GPU"]
fn deep_water_stands_as_deep_as_it_fills() {
    let ring = Ring {
        half_width: Metres(1.6),
        ..RING
    };
    let mut app = ring_of_water(ring);
    let half = ring.half_width.0 as f64;
    let depth = 1.92;
    stand_a_band(&mut app, (-half, half), depth);
    let radius = RING.radius.0 as f64;
    let mean_height = |water: &[Parcel]| {
        water.iter().map(|p| radius - p.from_axis).sum::<f64>() / water.len() as f64
    };
    let stood = mean_height(&parcels(&mut app));
    testing::run(&mut app, Seconds(20.0));
    let water = parcels(&mut app);
    // water is put in as closely packed as water at rest is, and no weight of water that a
    // ring holds presses it closer: what is put in stands as high on the whole as it was put
    score(
        "standing water: squashed by its own weight",
        stood - mean_height(&water),
        0.0,
        depth / 2.0,
    );
    let mut heights: Vec<f64> = water.iter().map(|p| radius - p.from_axis).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).expect("a height"));
    // particles stand half a spacing inside the surface they make up
    let top = heights[heights.len() * 98 / 100] + 0.16;
    let speed = water.iter().map(|p| p.speed).sum::<f64>() / water.len() as f64;
    let spin = standing_spin(ring).0 as f64;
    score("standing water: depth", top, depth, depth);
    // against how fast the same water would be moving had it fallen its own depth
    score(
        "standing water: speed at rest",
        speed,
        0.0,
        (2.0 * spin * spin * radius * depth).sqrt(),
    );
}

/// Expected: a bucketful of water let go in the air falls as anything does, exactly as far as
/// free flight among the stars takes it, the reading being a frame or two late. The air hardly
/// slows something that heavy over a fall of a second: it lands within a few hundredths of
/// the way of where it would in a vacuum.
#[test]
#[ignore = "wants a GPU"]
fn a_bucketful_of_water_falls_through_the_air_as_through_a_vacuum() {
    let fallen = |air: bool| {
        let mut app = ring_of_water(RING);
        app.world_mut().resource_mut::<Settings>().air = air;
        testing::run(&mut app, Seconds(0.1));
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let glass = sim.drum.wall_point(0.0, 0.0);
                let (_, outward) = sim.drum.depth_and_outward(glass);
                let at = [0, 1, 2].map(|c| glass[c] - outward[c] * 12.0);
                sim.drum.settle_water(glass);
                fluid.add(Particle {
                    position: sim.drum.to_water(at),
                    velocity: [0.0; 3],
                    foam: 0.0,
                });
            });
        let began = app.world().resource::<Simulation>().time.0 as f64;
        testing::run(&mut app, Seconds(1.2));
        let water = parcels(&mut app);
        let flown = app.world().resource::<Simulation>().time.0 as f64 - began;
        (water[0].from_axis - (RING.radius.0 as f64 - 12.0), flown)
    };
    let ((in_air, _), (in_vacuum, flown)) = (fallen(true), fallen(false));
    // what is let go at rest in the ring flies straight among the stars, at the speed the
    // ring carried it round at, and so is this far from the axis after a time
    let spin = standing_spin(RING).0 as f64;
    let out = (RING.radius.0 as f64 - 12.0) * ((1.0 + spin * spin * flown * flown).sqrt() - 1.0);
    score("free fall: distance in a vacuum", in_vacuum, out, out);
    // a bucketful is far too heavy for the air to hold back over a second's fall: by less
    // than a hundredth of the way
    score(
        "free fall: what the air holds back",
        (in_vacuum - in_air) / in_vacuum,
        0.0,
        1.0,
    );
}

/// Expected: a body of water let go in the air falls exactly as a single bucketful beside it
/// does. Its own particles push only on each other, which moves none of its weight anywhere:
/// it neither hangs in the air nor hurries.
#[test]
#[ignore = "wants a GPU"]
fn a_body_of_water_falls_as_one_bucketful_does() {
    let mut app = ring_of_water(RING);
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let glass = sim.drum.wall_point(0.0, 0.0);
            sim.drum.settle_water(glass);
            let (_, outward) = sim.drum.depth_and_outward(glass);
            let above = |along: f64| {
                [
                    glass[0] - outward[0] * 12.0,
                    along,
                    glass[2] - outward[2] * 12.0,
                ]
            };
            fluid.add(Particle {
                position: sim.drum.to_water(above(5.0)),
                velocity: [0.0; 3],
                foam: 0.0,
            });
            assert_eq!(sim.inject(&mut fluid, above(-3.0), 300), 300);
        });
    // where each began, and where it is a while later: the two readings are as late as each other
    let read = |app: &mut App| {
        let water = parcels(app);
        let (alone, body): (Vec<&Parcel>, Vec<&Parcel>) = water.iter().partition(|p| p.along > 2.0);
        assert_eq!(alone.len(), 1);
        let body_at = body.iter().map(|p| p.from_axis).sum::<f64>() / body.len() as f64;
        (alone[0].from_axis, body_at)
    };
    testing::run(&mut app, Seconds(0.05));
    let began = read(&mut app);
    testing::run(&mut app, Seconds(1.2));
    let ended = read(&mut app);
    // what is let go at rest flies straight among the stars, and is further from the axis by a
    // share of how far from it it began
    let (one, many) = (ended.0 / began.0 - 1.0, ended.1 / began.1 - 1.0);
    score(
        "free fall: a body of water against one bucketful",
        many,
        one,
        one,
    );
}

/// Expected: a bucketful of water lying against the glass end of the ring, a third of the way
/// out from the axis, does not stay there. The glass only holds it off along the axis, so it
/// flies on across the glass as it would through the air, straight among the stars at the
/// speed the ring carried it round at, and whatever the glass and the air drag it round with
/// them only throws it outward the faster: after three seconds it is no nearer the axis than
/// free flight takes it.
#[test]
#[ignore = "wants a GPU"]
fn water_on_the_glass_end_runs_out_to_the_rim() {
    let mut app = ring_of_water(RING);
    let from_axis = RING.radius.0 as f64 / 3.0;
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let against = RING.half_width.0 as f64 - fluid.resolution().margin().0 as f64;
            let glass = sim.drum.wall_point(0.0, against - sim.drum.site.y);
            let (_, outward) = sim.drum.depth_and_outward(glass);
            let height = RING.radius.0 as f64 - from_axis;
            let at = [0, 1, 2].map(|c| glass[c] - outward[c] * height);
            sim.drum.settle_water(glass);
            fluid.add(Particle {
                position: sim.drum.to_water(at),
                velocity: [0.0; 3],
                foam: 0.0,
            });
        });
    let spin = standing_spin(RING).0 as f64;
    let mut last = from_axis;
    for second in 1..=3 {
        testing::run(&mut app, Seconds(1.0));
        let water = parcels(&mut app);
        let flown = from_axis * (1.0 + (spin * second as f64).powi(2)).sqrt();
        let reach = flown.min(RING.radius.0 as f64 - 0.16);
        score(
            &format!("water on the end glass: from the axis after {second} s"),
            water[0].from_axis,
            reach,
            reach,
        );
        assert!(
            water[0].from_axis >= last,
            "the water came back toward the axis"
        );
        last = water[0].from_axis;
    }
}

/// A bucketful of water let go this far under the axis of a ring of air or none, and how many
/// motes of spray are in flight a while after.
fn spray_shed_falling(air: bool, seconds: f32) -> usize {
    let mut app = ring_of_water(RING);
    app.world_mut().resource_mut::<Settings>().air = air;
    testing::run(&mut app, Seconds(0.1));
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let glass = sim.drum.wall_point(0.0, 0.0);
            let (_, outward) = sim.drum.depth_and_outward(glass);
            let at = [0, 1, 2].map(|c| glass[c] - outward[c] * 25.0);
            sim.drum.settle_water(glass);
            fluid.add(Particle {
                position: sim.drum.to_water(at),
                velocity: [0.0; 3],
                foam: 0.0,
            });
        });
    testing::run(&mut app, Seconds(seconds));
    testing::spray(&mut app).len()
}

/// Expected: the air tears spray off water that flies through it fast, and only the air does.
/// A bucketful let go high in the ring meets the air faster and faster as it falls, and well
/// before it lands it trails spray; the same bucketful falling through no air sheds none. Once
/// it has landed and the spray with it, none is left in the air.
#[test]
#[ignore = "wants a GPU"]
fn the_air_tears_spray_off_falling_water() {
    let (in_air, in_vacuum) = (
        spray_shed_falling(true, 3.0),
        spray_shed_falling(false, 3.0),
    );
    println!("{in_air} motes in flight in air, {in_vacuum} in a vacuum");
    assert!(in_air > 0, "water falling through the air shed no spray");
    assert_eq!(in_vacuum, 0, "water falling through no air shed spray");
    let landed = spray_shed_falling(true, 12.0);
    assert_eq!(
        landed, 0,
        "{landed} motes of spray still fly after the water has landed"
    );
}

/// Expected: water at rest throws up no spray.
#[test]
#[ignore = "wants a GPU"]
fn still_water_sheds_no_spray() {
    let mut app = ring_of_water(RING);
    stand_a_band(&mut app, (-2.0, 2.0), 1.0);
    testing::run(&mut app, Seconds(10.0));
    let flying = testing::spray(&mut app).len();
    assert_eq!(flying, 0, "{flying} motes of spray over water at rest");
}

/// Expected: water standing between the two glass ends of a narrow ring, set swaying from
/// end to end, sways as a standing wave does: once over in the time the theory of waves gives
/// for a wave twice as long as the water is from end to end, which for water of depth h and
/// weight g is 2 pi over the root of g k tanh(k h), k being pi over that length. And it keeps
/// swaying: water is so thin that a sway of seconds in metres of it loses next to nothing in
/// a turn, some three thousandths of its height.
#[test]
#[ignore = "wants a GPU"]
fn water_sways_between_the_ends_as_a_standing_wave() {
    let ring = Ring {
        half_width: Metres(3.2),
        ..RING
    };
    let mut app = ring_of_water(ring);
    let half = ring.half_width.0 as f64;
    let (long, depth) = (2.0 * half, 1.28);
    let k = std::f64::consts::PI / long;
    stand_a_band_moving(&mut app, (-half, half), depth, |along| {
        0.3 * (k * (along + half)).sin()
    });
    let spin = standing_spin(ring).0 as f64;
    let weight = spin * spin * (RING.radius.0 as f64 - depth / 2.0);
    let period = std::f64::consts::TAU / (weight * k * (k * depth).tanh()).sqrt();
    let began = app.world().resource::<Simulation>().time.0 as f64;
    let mut swayed = Vec::new();
    for _ in 0..(4.5 * period / 0.05) as usize {
        testing::run(&mut app, Seconds(0.05));
        let water = parcels(&mut app);
        let at = app.world().resource::<Simulation>().time.0 as f64 - began;
        swayed.push((
            at,
            water.iter().map(|p| p.along).sum::<f64>() / water.len() as f64,
        ));
    }
    // the water's middle is furthest one way, then the other, half a sway apart: the turns of
    // its path, each the furthest it gets before it comes back
    let turns: Vec<(f64, f64)> = swayed
        .windows(3)
        .filter(|w| (w[1].1 - w[0].1) * (w[2].1 - w[1].1) < 0.0)
        .map(|w| w[1])
        .collect();
    assert!(
        turns.len() >= 3,
        "the water turned only {} times",
        turns.len()
    );
    score(
        "swaying water: time of a sway",
        turns[2].0 - turns[0].0,
        period,
        period,
    );
    let kept = (turns[2].1 / turns[0].1).abs();
    score("swaying water: height kept over a sway", kept, 0.997, 1.0);
}
