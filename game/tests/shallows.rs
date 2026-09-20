//! The water that lies on the ground, held to what water does: a lake lies still, a sway has
//! the time its length and depth give a wave and keeps its height, and water poured out of the
//! air comes to lie on the ground with none of it lost.

use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::shallows::Shallows;
use game::core::units::{Litres, Metres, Seconds};
use game::systems::drum::Ring;
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

const GROUND: f32 = 2.0;

fn ring_with_ground(ring: Ring) -> testing::Headless {
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
    }
    let mut sim = Simulation::new(ring);
    sim.drum.spin = standing_spin(ring);
    sim.drum.target_spin = standing_spin(ring);
    sim.drum.landscape.flatten(Metres(GROUND));
    app.insert_resource(sim);
    testing::run(&mut app, Seconds(0.2));
    app
}

/// A lake over sculpted ground, with a bank it does not cover and an island it does not drown,
/// does not move at all: not slowly, not within a tolerance, but not.
#[test]
#[ignore = "wants a GPU"]
fn a_lake_on_sculpted_ground_lies_exactly_still() {
    let ring = Ring {
        radius: Metres(30.0),
        half_width: Metres(8.0),
    };
    let mut app = ring_with_ground(ring);
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let island = sim.drum.wall_point(0.1, 0.0);
        sim.drum.sculpt(island, 3.0, 2.5);
        let shoal = sim.drum.wall_point(-0.2, 2.0);
        sim.drum.sculpt(shoal, 4.0, 0.6);
        let pit = sim.drum.wall_point(0.4, -3.0);
        sim.drum.sculpt(pit, 3.0, -1.0);
    }
    let level = GROUND + 1.0;
    app.world_mut()
        .resource_mut::<Shallows>()
        .stand(Metres(level), [0.0; 2]);
    testing::run(&mut app, Seconds(5.0));

    let water = testing::ground_water(&mut app);
    let wet = water.iter().filter(|(cell, bed)| cell.face > *bed).count();
    let dry = water.len() - wet;
    assert!(
        wet > 1000 && dry > 50,
        "{wet} cells wet and {dry} dry: the ground was to leave some of each"
    );
    for (cell, bed) in &water {
        assert_eq!(cell.flow, [0.0; 2], "water moves over ground at {bed} m");
        assert!(
            cell.face == level || cell.face == *bed,
            "the water's face stands at {} m over ground at {bed} m, where the lake stands at {level} m",
            cell.face
        );
    }
}

/// Water tilted along the axis between the end glasses sways as a standing wave: with the time
/// a wave of that length takes in water that deep, which is longer than it would be were the
/// water shallow to the wave, and keeping its height from one sway to the next but for what the
/// bed's drag takes.
#[test]
#[ignore = "wants a GPU"]
fn water_sways_between_the_ends_as_a_standing_wave() {
    let ring = Ring {
        radius: Metres(30.0),
        half_width: Metres(3.2),
    };
    let mut app = ring_with_ground(ring);
    let (long, depth) = (2.0 * ring.half_width.0 as f64, 1.28);
    app.world_mut()
        .resource_mut::<Shallows>()
        .stand(Metres(GROUND + depth as f32), [0.0, 0.02]);
    let chart = {
        testing::run(&mut app, Seconds(0.0));
        app.world()
            .resource::<Shallows>()
            .charted()
            .expect("a chart")
    };
    let k = std::f64::consts::PI / long;
    let spin = standing_spin(ring).0 as f64;
    let weight = spin * spin * (ring.radius.0 as f64 - GROUND as f64 - depth / 2.0);
    let period = std::f64::consts::TAU / (weight * k * (k * depth).tanh()).sqrt();

    // how high the water's first way of swaying stands, which no other way of swaying adds to
    let mut swayed = Vec::new();
    let tick = 0.05;
    for n in 0..(3.2 * period / tick) as usize {
        testing::run(&mut app, Seconds(tick as f32));
        let water = testing::ground_water(&mut app);
        let rows = chart.size[1] as usize;
        let height: f64 = (0..rows)
            .map(|row| {
                let along = (row as f64 + 0.5) * chart.cell[1].0 as f64;
                let face = water[row * chart.size[0] as usize].0.face as f64;
                (face - GROUND as f64 - depth) * (k * along).cos()
            })
            .sum::<f64>()
            * 2.0
            / rows as f64;
        swayed.push(((n + 1) as f64 * tick, height));
    }
    let crossings: Vec<f64> = swayed
        .windows(2)
        .filter(|w| w[0].1 < 0.0 && w[1].1 >= 0.0)
        .map(|w| w[0].0 + tick * -w[0].1 / (w[1].1 - w[0].1))
        .collect();
    assert!(
        crossings.len() >= 3,
        "the water swayed {} times: {swayed:?}",
        crossings.len()
    );
    let measured = (crossings[2] - crossings[0]) / 2.0;
    assert!(
        (measured / period - 1.0).abs() < 0.01,
        "a sway took {measured:.3} s, where a wave {:.1} m long in water {depth} m deep takes {period:.3} s",
        2.0 * long
    );
    let lows: Vec<f64> = crossings
        .windows(2)
        .map(|c| {
            swayed
                .iter()
                .filter(|(t, _)| *t > c[0] && *t < c[1])
                .map(|(_, h)| *h)
                .fold(f64::MAX, f64::min)
        })
        .collect();
    let kept = lows[1] / lows[0];
    assert!(
        kept > 0.985 && kept < 1.0 + 1e-3,
        "the sway kept {kept:.4} of its height over a sway: {lows:?}"
    );
}

/// Water let go in the air falls as water in flight and, come down, lies on the ground as the
/// ground's water: none of it is left in flight, and all of it lies there.
#[test]
#[ignore = "wants a GPU"]
fn water_poured_out_of_the_air_comes_to_lie_on_the_ground() {
    let ring = Ring {
        radius: Metres(30.0),
        half_width: Metres(8.0),
    };
    let mut app = ring_with_ground(ring);
    let poured = app
        .world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let above = sim.drum.wall_point(0.0, 0.0);
            let above = [above[0] - GROUND as f64 - 3.0, above[1], above[2]];
            let particles = sim.inject(&mut fluid, above, 2000);
            particles as f64 * fluid.resolution().litres_per_particle().0 as f64 / 1000.0
        });
    assert!(poured > 1.0, "{poured} cubic metres were poured");
    testing::run(&mut app, Seconds(8.0));

    let flying = testing::particles(&mut app).len();
    assert_eq!(
        flying, 0,
        "particles are in flight still, where all the water was to have come down"
    );
    let lying = cubic_metres_lying(&mut app, ring);
    let particle = poured / 2000.0;
    assert!(
        (lying - poured).abs() < particle,
        "{lying:.4} cubic metres lie on the ground of the {poured:.4} poured"
    );
}

/// A heap poured where the chart of the ground closes on itself spreads both ways round the
/// ring, and all the while is as much water as was poured.
#[test]
#[ignore = "wants a GPU"]
fn a_heap_spreading_round_the_ring_is_neither_made_nor_lost() {
    let ring = Ring {
        radius: Metres(30.0),
        half_width: Metres(8.0),
    };
    let mut app = ring_with_ground(ring);
    let seam = app.world().resource::<Simulation>().drum.ground_chart().1[0];
    let mut poured = 0.0;
    for burst in 0..10 {
        let mut shallows = app.world_mut().resource_mut::<Shallows>();
        for k in 0..200 {
            let n = (burst * 200 + k) as f64;
            let spread = |turn: f64| ((n * turn) % 1.0 - 0.5) * 4.0;
            shallows.pour([seam + spread(0.618034), spread(0.754878)], Litres(30.0));
            poured += 0.03;
        }
        testing::run(&mut app, Seconds(0.1));
    }
    for _ in 0..4 {
        testing::run(&mut app, Seconds(1.0));
        let lying = cubic_metres_lying(&mut app, ring);
        assert!(
            (lying / poured - 1.0).abs() < 1e-5,
            "{lying:.4} cubic metres lie on the ground of the {poured:.4} poured"
        );
    }
    let water = testing::ground_water(&mut app);
    let wet = water.iter().filter(|(cell, bed)| cell.face > *bed).count();
    assert!(wet > 4000, "the heap spread over {wet} cells only");
}

/// Water poured on the top of a hill runs down it all round, and all the while is as much
/// water as was poured.
#[test]
#[ignore = "wants a GPU"]
fn water_running_down_a_hill_is_neither_made_nor_lost() {
    let ring = Ring {
        radius: Metres(10.5),
        half_width: Metres(6.0),
    };
    let mut app = ring_with_ground(ring);
    let at = {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let top = sim.drum.wall_point(0.1, 1.0);
        sim.drum.sculpt(top, 5.0, 2.0);
        let top = sim.drum.to_water(top);
        let radius = ring.radius.0 as f64;
        [radius * top[2].atan2(radius + top[0]), top[1]]
    };
    testing::run(&mut app, Seconds(0.2));
    for burst in 0..20 {
        let mut shallows = app.world_mut().resource_mut::<Shallows>();
        for k in 0..10 {
            let n = (burst * 10 + k) as f64;
            let spread = |turn: f64| ((n * turn) % 1.0 - 0.5) * 2.0;
            shallows.pour(
                [at[0] + spread(0.618034), at[1] + spread(0.754878)],
                Litres(50.0),
            );
        }
        testing::run(&mut app, Seconds(0.05));
    }
    for _ in 0..6 {
        testing::run(&mut app, Seconds(1.0));
        let lying = cubic_metres_lying(&mut app, ring);
        assert!(
            (lying / 10.0 - 1.0).abs() < 1e-5,
            "{lying:.4} cubic metres lie on the ground of the 10 poured"
        );
    }
}

/// The water lying on the ground, by the chart's cells: a cell holds less the higher it is
/// taken, the ring being shorter round the nearer its axis.
fn cubic_metres_lying(app: &mut testing::Headless, ring: Ring) -> f64 {
    let chart = app
        .world()
        .resource::<Shallows>()
        .charted()
        .expect("a chart");
    let radius = ring.radius.0 as f64;
    let held = |height: f64| height - height * height / (2.0 * radius);
    let area = chart.cell[0].0 as f64 * chart.cell[1].0 as f64;
    testing::ground_water(app)
        .iter()
        .map(|(cell, bed)| (held(cell.face as f64) - held(*bed as f64)).max(0.0) * area)
        .sum()
}
