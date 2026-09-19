//! Water and the portals: what goes in at one comes out of the other, all of it and at the
//! speed it went in with, and water standing over a pair presses through it, so that two pools
//! joined by a pair find one level as two vessels joined by a pipe do.
use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::math::Vec3d;
use game::core::units::{Metres, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{CapSide, Drum, DrumSurface, MouthColour};
use game::systems::sim::Simulation;
use game::systems::testing;

const DIAMETER: Metres = Metres(1.6);

/// A point over the ground `spinward` metres round the ring from the wheel's own zero and
/// `along` the axis from its middle, wherever the drum's frame has moved to.
fn on_the_ground(drum: &Drum, spinward: f64, along: f64, height: f64) -> Vec3d {
    let turn = spinward / drum.ring.radius.0 as f64 - drum.site.phi;
    let glass = drum.wall_point(turn, along - drum.site.y);
    let (_, outward) = drum.depth_and_outward(glass);
    let lift = drum.ground(glass) + height;
    [0, 1, 2].map(|k| glass[k] - outward[k] * lift)
}

fn put(app: &mut App, colour: MouthColour, spinward: f64, along: f64) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let at = on_the_ground(&sim.drum, spinward, along, 0.0);
    let wanted = sim
        .drum
        .mouth_at(at, DrumSurface::Wall, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    let fit = sim.drum.fit_mouth(colour, wanted, DIAMETER);
    sim.drum.put_mouth(colour, fit, DIAMETER);
    assert!(sim.drum.mouths.get(colour).is_some(), "the mouth fits");
}

fn pour(app: &mut App, spinward: f64, along: f64, height: f64, count: u32) -> u32 {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let at = on_the_ground(&sim.drum, spinward, along, height);
            sim.inject(&mut fluid, at, count)
        })
}

/// A particle as the drum's frame has it: where it is, how high over the ground, and how fast
/// it rises from it.
struct Drop {
    at: Vec3d,
    height: f64,
    rising: f64,
    speed: f64,
}

fn water(app: &mut App) -> Vec<Drop> {
    let particles = testing::particles(app);
    let drum = &app.world().resource::<Simulation>().drum;
    let frame = drum.water_frame();
    particles
        .into_iter()
        .map(|p| {
            let at = drum.from_water(p.position);
            let (_, outward) = drum.depth_and_outward(at);
            let v = frame.vector_from_water(p.velocity);
            Drop {
                at,
                height: drum.height_above_glass(at) - drum.ground(at),
                rising: -(0..3).map(|k| v[k] * outward[k]).sum::<f64>(),
                speed: (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt(),
            }
        })
        .collect()
}

fn near(drum: &Drum, p: Vec3d, spinward: f64, along: f64, within: f64) -> bool {
    let at = on_the_ground(drum, spinward, along, 0.0);
    (0..3).map(|k| (p[k] - at[k]).powi(2)).sum::<f64>() < within * within
}

const BLUE: (f64, f64) = (4.0, 3.0);
const ORANGE: (f64, f64) = (-5.0, -3.0);

/// Water dropped into a portal comes out of the other rising as fast as it is falling in, all
/// of it, and none of it inside the ground.
#[test]
#[ignore = "wants a GPU"]
fn water_dropped_into_a_portal_comes_out_of_the_other_as_fast() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    put(&mut app, MouthColour::Blue, BLUE.0, BLUE.1);
    put(&mut app, MouthColour::Orange, ORANGE.0, ORANGE.1);
    testing::run(&mut app, Seconds(1.5));
    let dropped = pour(&mut app, BLUE.0, BLUE.1, 1.0, 120);
    let mean = |drops: &[&Drop]| drops.iter().map(|d| d.rising).sum::<f64>() / drops.len() as f64;
    let mut speeds = None;
    let mut most_out = 0;
    for _ in 0..40 {
        testing::run(&mut app, Seconds(0.05));
        let drops = water(&mut app);
        assert_eq!(drops.len() as u32, dropped, "no water is lost or made");
        let drum = &app.world().resource::<Simulation>().drum;
        let lowest = drops.iter().map(|d| d.height).fold(f64::INFINITY, f64::min);
        assert!(lowest > -0.05, "water is {lowest} m over the ground");
        let (out, falling): (Vec<&Drop>, Vec<&Drop>) = drops
            .iter()
            .partition(|d| near(drum, d.at, ORANGE.0, ORANGE.1, 2.0));
        most_out = most_out.max(out.len());
        if speeds.is_none() && 10 * out.len() > drops.len() {
            speeds = Some((-mean(&falling), mean(&out)));
        }
    }
    assert!(
        3 * most_out as u32 > dropped,
        "{most_out} of {dropped} particles came out of the orange portal"
    );
    let (falling, rising) = speeds.expect("a tenth of the water came out");
    assert!(
        falling > 1.0 && (0.7 * falling..1.5 * falling).contains(&rising),
        "water falling in at {falling} m/s came out rising at {rising} m/s"
    );
}

/// Two basins side by side round the ring, each between two dams that reach from cap to cap,
/// with a portal in the floor of each: where the dams stand and where the portals are, in
/// metres round the ring from the site.
const DAMS: [f64; 3] = [1.0, 12.0, 23.0];
const FLOORS: [f64; 2] = [6.5, 17.5];

fn build_the_basins(app: &mut App) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let drum = &mut sim.drum;
    for dam in DAMS {
        for along in [-5.0, -2.5, 0.0, 2.5, 5.0] {
            drum.sculpt(on_the_ground(drum, dam, along, 0.0), 2.5, 3.0);
        }
    }
}

/// How many particles the basin between two dams holds, the level nine in ten of them lie
/// under, and how fast they move.
fn basin(app: &mut App, dams: [f64; 2]) -> (usize, f64, f64) {
    let drops = water(app);
    let drum = &app.world().resource::<Simulation>().drum;
    let held: Vec<&Drop> = drops
        .iter()
        .filter(|d| {
            let round = (drum.turn_to(d.at) + drum.site.phi).rem_euclid(std::f64::consts::TAU);
            (dams[0]..dams[1]).contains(&(round * drum.ring.radius.0 as f64))
        })
        .collect();
    let mut levels: Vec<f64> = held.iter().map(|d| d.height).collect();
    levels.sort_by(f64::total_cmp);
    let speed = held.iter().map(|d| d.speed).sum::<f64>() / held.len().max(1) as f64;
    let level = levels.get(levels.len() * 9 / 10).copied().unwrap_or(0.0);
    (held.len(), level, speed)
}

/// Water poured into one of two basins whose floors a pair of portals joins runs through the
/// pair into the other, which nothing else lets it into, as water in two vessels joined by a
/// pipe does: the levels close on each other, and the water slows as they do rather than
/// being stirred by the pair.
#[test]
#[ignore = "wants a GPU"]
fn a_pool_fills_the_pool_a_pair_joins_it_to() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    build_the_basins(&mut app);
    let diameter = Metres(3.0);
    for (colour, floor) in MouthColour::BOTH.into_iter().zip(FLOORS) {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let at = on_the_ground(&sim.drum, floor, 0.0, 0.0);
        let wanted = sim
            .drum
            .mouth_at(at, DrumSurface::Wall, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let fit = sim.drum.fit_mouth(colour, wanted, diameter);
        sim.drum.put_mouth(colour, fit, diameter);
        assert!(sim.drum.mouths.get(colour).is_some(), "the mouth fits");
    }
    testing::run(&mut app, Seconds(1.5));
    let mut poured = 0;
    for k in 0..20 {
        let along = [-3.5, 0.0, 3.5][k % 3];
        let round = FLOORS[0] + [-2.5, 2.5][k % 2];
        poured += pour(&mut app, round, along, 1.5, 300);
        testing::run(&mut app, Seconds(0.5));
    }
    let basins = [[DAMS[0], DAMS[1]], [DAMS[1], DAMS[2]]];
    let mut heads = Vec::new();
    let mut speeds = Vec::new();
    for _ in 0..6 {
        testing::run(&mut app, Seconds(10.0));
        let (a, b) = (basin(&mut app, basins[0]), basin(&mut app, basins[1]));
        assert!(
            (a.0 + b.0) as f64 > 0.98 * poured as f64,
            "the basins hold {} and {} of {poured} particles",
            a.0,
            b.0
        );
        heads.push(a.1 - b.1);
        speeds.push(a.2.max(b.2));
    }
    let (a, b) = (basin(&mut app, basins[0]), basin(&mut app, basins[1]));
    assert!(
        b.0 as f64 > 0.3 * poured as f64,
        "the basin the water was poured into holds {a:?}, the other {b:?}"
    );
    assert!(
        heads[5] < 0.6 * heads[0] && heads[5] > -0.05,
        "the one pool stood over the other by {heads:?} m"
    );
    assert!(
        speeds[5] < 0.05 && speeds[5] < speeds[0],
        "the water moved at {speeds:?} m/s"
    );
}

/// A portal in the floor of a pool a metre deep, and its pair in the cap beside the pool, its
/// lower edge a metre over the water. The water over the one in the floor is pressed into it by
/// its own weight, and comes out of the one in the cap, from where it falls back into the pool
/// it came from: both portals are the same way up in the ring's spin, so the water runs round
/// for as long as the ring turns, as a steady stream, the pool neither draining nor the stream
/// dying.
#[test]
#[ignore = "wants a GPU"]
fn a_pool_runs_round_through_a_portal_in_its_floor_and_one_over_it() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    build_the_basins(&mut app);
    let (round, depth, rise) = (FLOORS[0], 1.0, 1.0);
    let half_width = app.world().resource::<Simulation>().drum.ring.half_width.0 as f64;
    let mut poured = 0;
    for k in 0..11 {
        let along = [-3.5, 0.0, 3.5][k % 3];
        poured += pour(&mut app, round + [-2.5, 2.5][k % 2], along, 1.5, 300);
        testing::run(&mut app, Seconds(0.5));
    }
    testing::run(&mut app, Seconds(8.0));
    let basin_dams = [DAMS[0], DAMS[1]];
    let still = basin(&mut app, basin_dams);
    assert!(
        (0.7 * depth..1.4 * depth).contains(&still.1) && still.2 < 0.4,
        "the pool stands {still:?} before the portals are put"
    );

    put(&mut app, MouthColour::Blue, round, -half_width + 3.0);
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let over = still.1 + rise + DIAMETER.0 as f64 / 2.0;
        let at = on_the_ground(&sim.drum, round, -half_width, over);
        let (_, outward) = sim.drum.depth_and_outward(at);
        let wanted = sim.drum.mouth_at(
            at,
            DrumSurface::Cap(CapSide::Low),
            outward.map(|c| -c),
            [0.0, 0.0, 1.0],
        );
        let fit = sim.drum.fit_mouth(MouthColour::Orange, wanted, DIAMETER);
        sim.drum.put_mouth(MouthColour::Orange, fit, DIAMETER);
        assert!(sim.drum.mouths.get(MouthColour::Orange).is_some());
    }
    testing::run(&mut app, Seconds(6.0));

    let mut streams = Vec::new();
    for _ in 0..8 {
        testing::run(&mut app, Seconds(1.5));
        let drops = water(&mut app);
        assert_eq!(drops.len() as u32, poured, "no water is lost or made");
        let falling: Vec<&Drop> = drops.iter().filter(|d| d.height > still.1 + 0.4).collect();
        let speed = falling.iter().map(|d| d.speed).sum::<f64>() / falling.len().max(1) as f64;
        let pool = basin(&mut app, basin_dams);
        let apart = testing::surface_demand(&mut app).droplets;
        eprintln!(
            "falling {} at {speed:.2} m/s, {apart} of all the water on their own | pool {pool:?}",
            falling.len()
        );
        streams.push((falling.len() as f64, speed, pool.0));
    }
    let mean = streams.iter().map(|s| s.0).sum::<f64>() / streams.len() as f64;
    let spread = streams
        .iter()
        .map(|s| (s.0 - mean).abs())
        .fold(0.0, f64::max);
    assert!(
        mean > 40.0 && spread < 0.4 * mean,
        "the stream holds {:?} particles from moment to moment",
        streams.iter().map(|s| s.0).collect::<Vec<_>>()
    );
    assert!(
        streams.iter().all(|s| (1.5..8.0).contains(&s.1)),
        "the stream runs at {:?} m/s",
        streams.iter().map(|s| s.1).collect::<Vec<_>>()
    );
    assert!(
        streams.iter().all(|s| s.2 as f64 > 0.95 * poured as f64),
        "the pool holds {:?} of {poured} particles",
        streams.iter().map(|s| s.2).collect::<Vec<_>>()
    );
}
