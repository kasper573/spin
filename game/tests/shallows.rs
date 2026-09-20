//! The water that lies on the ground, held to what water does: a lake lies still, and a sway
//! has the time its length and depth give a wave and keeps its height.

use bevy::prelude::*;
use game::core::shallows::Shallows;
use game::core::units::{Metres, Seconds};
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
