use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::Metres;
use game::core::units::{RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::controls::BRUSH_SIZE;
use game::systems::drum::{
    DEFAULT_RING, Drum, Ground, Landscape, PATCH, Patch, Place, Ring, Round, Site,
};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

fn at_angle(land: &Landscape, phi: f64, along: f64) -> Place {
    Place {
        round: Round::at_angle(phi, land.grid()),
        along,
    }
}

#[test]
fn sculpting_raises_ground_under_the_brush_only() {
    let mut land = Landscape::new(DEFAULT_RING);
    assert!(land.is_empty());
    for _ in 0..10 {
        land.sculpt(at_angle(&land, 1.0, 0.0), 0.5, 0.05);
    }
    assert!(!land.is_empty());
    let (h, _, _) = land.sample(at_angle(&land, 1.0, 0.0));
    assert!(h > 0.3 && h <= 0.5, "height {h}");
    let (far, _, _) = land.sample(at_angle(&land, 1.0 + std::f64::consts::PI, 0.0));
    assert_eq!(far, 0.0);
    land.flatten(Metres(0.0));
    assert!(land.is_empty());
}

/// The brush is a size in metres, not a share of the ring: put to work the same way anywhere,
/// on a ring of any size, it raises the same mound, as tall in the middle and as wide.
#[test]
fn a_brush_raises_the_same_mound_on_a_ring_of_any_size() {
    let brush = BRUSH_SIZE.0 as f64;
    let offsets = [0.0, 0.3, 0.6, 0.9, 1.1];
    let mut first: Option<Vec<f64>> = None;
    for radius in [10.5f64, 100.0, 5_000.0, 1e6, 1e9] {
        let ring = Ring {
            radius: Metres(radius as f32),
            half_width: Metres((radius * 0.1).max(6.0) as f32),
        };
        let mut land = Landscape::new(ring);
        let grid = land.grid();
        // somewhere between the cells rather than on one of them
        let at = Place {
            round: Round {
                cell: Round::at_angle(1.0, grid).cell,
                across: 0.5,
            },
            along: 0.5 * grid.along,
        };
        for _ in 0..10 {
            land.sculpt(at, brush, 0.05 * brush);
        }
        let mound: Vec<f64> = offsets
            .iter()
            .flat_map(|k| {
                let off = k * brush;
                [
                    land.sample(Place {
                        round: at.round.on(off, grid),
                        along: at.along,
                    })
                    .0,
                    land.sample(Place {
                        round: at.round,
                        along: at.along + off,
                    })
                    .0,
                ]
            })
            .collect();
        assert!(
            mound[0] > 0.3 * brush,
            "on a ring of radius {radius} m a brush {brush} m wide raised the ground {} m",
            mound[0]
        );
        assert!(
            mound[8] == 0.0 && mound[9] == 0.0,
            "on a ring of radius {radius} m the brush raised the ground past its edge: {mound:?}"
        );
        let far = land
            .sample(at_angle(&land, 1.0 + std::f64::consts::PI, at.along))
            .0;
        assert_eq!(far, 0.0, "on a ring of radius {radius} m the far side rose");
        match &first {
            None => first = Some(mound),
            Some(small) => {
                for (a, b) in small.iter().zip(&mound) {
                    assert!(
                        (a - b).abs() < 0.02 * brush,
                        "on a ring of radius {radius} m the mound is {mound:?}, \
                         on the smallest {small:?}"
                    );
                }
            }
        }
    }
}

/// A body of water finds the level at which the cells of ground lower than it hold exactly its
/// volume, whether it only half fills a hollow or drowns a hill.
#[test]
fn water_finds_the_level_the_ground_holds_it_at() {
    let ring = DEFAULT_RING;
    let mut land = Landscape::flat(ring, Metres(0.5));
    for _ in 0..20 {
        land.sculpt(at_angle(&land, 1.0, 1.0), 2.0, 0.1);
        land.sculpt(at_angle(&land, 3.0, -2.0), 1.5, -0.05);
    }
    let grid = land.grid();
    let heights: Vec<f64> = (0..grid.round)
        .flat_map(|cell| (-grid.rows..=grid.rows).map(move |row| (cell, row)))
        .map(|(cell, row)| land.height(cell, row) as f64)
        .collect();
    let area = std::f64::consts::TAU * ring.radius.0 as f64 * 2.0 * ring.half_width.0 as f64;
    let per_cell = area / heights.len() as f64;
    let held = |level: f64| heights.iter().map(|h| (level - h).max(0.0)).sum::<f64>() * per_cell;
    for cubic_metres in [0.01, 0.5, 5.0, 50.0, 2000.0] {
        let (mut under, mut over) = (0.0, land.max_height() as f64 + cubic_metres / area);
        for _ in 0..100 {
            let mid = 0.5 * (under + over);
            if held(mid) < cubic_metres {
                under = mid;
            } else {
                over = mid;
            }
        }
        let level = 0.5 * (under + over);
        let wet = heights.iter().filter(|h| **h < level).count() as f64;
        let depth = held(level) / (per_cell * wet);
        let flood = land.flooded(cubic_metres);
        assert!(
            (flood.level.0 as f64 - level).abs() < 1e-5
                && (flood.covered as f64 - wet / heights.len() as f64).abs() < 1e-5
                && (flood.depth.0 as f64 - depth).abs() < 1e-5,
            "{cubic_metres} m³ stands at {level} m over {wet} cells, {depth} m deep, not {flood:?}"
        );
    }
}

#[test]
fn heights_are_clamped_on_load() {
    let mut land = Landscape::new(DEFAULT_RING);
    let mut heights = vec![0.2; (PATCH * PATCH) as usize];
    heights[..3].copy_from_slice(&[f32::NAN, -1.0, 100.0]);
    land.load(&Ground {
        base: f32::NAN,
        patches: vec![
            Patch {
                round: 0,
                along: 0,
                heights,
            },
            Patch {
                round: 1,
                along: 0,
                heights: vec![0.3; 3],
            },
            Patch {
                round: land.grid().patches_round(),
                along: 0,
                heights: vec![0.3; (PATCH * PATCH) as usize],
            },
        ],
    });
    assert_eq!(land.base(), 0.0);
    assert_eq!(land.height(0, 0), 0.0);
    assert_eq!(land.height(0, 1), 0.0);
    assert!(land.height(0, 2) <= DEFAULT_RING.max_height().0);
    assert_eq!(land.height(0, 3), 0.2);
    assert_eq!(
        land.height(PATCH, 0),
        0.0,
        "a patch of the wrong size was laid"
    );
    assert_eq!(land.patches().count(), 1, "a patch off the ring was laid");
}

#[test]
fn raised_ground_is_a_wall_where_it_is_raised_only() {
    let mut drum = Drum {
        site: Site::at(0.7, 0.0, Drum::default().ring),
        ..Drum::default()
    };
    let under = drum.wall_point(0.0, 0.0);
    for _ in 0..20 {
        drum.sculpt(under, 1.5, 0.05);
    }
    let inside = [under[0] - 0.6, under[1], under[2]];
    let hit = drum.penetrations(inside).iter().next();
    assert!(
        hit.is_some_and(|pen| pen.depth > 0.3 && pen.normal[0] < -0.9),
        "no ground under the brush: {hit:?}"
    );
    let elsewhere = drum.wall_point(1.2, 0.0);
    let (_, outward) = drum.depth_and_outward(elsewhere);
    let clear = [
        elsewhere[0] - 0.6 * outward[0],
        elsewhere[1],
        elsewhere[2] - 0.6 * outward[2],
    ];
    let hit = drum.penetrations(clear).iter().next();
    assert!(hit.is_none(), "ground away from the brush: {hit:?}");
}

#[test]
fn water_settles_on_top_of_raised_ground() {
    let mut app = testing::headless();
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(1.0);
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        sim.drum.spin = RadiansPerSecond(1.0);
        let crest = sim.drum.wall_point(0.8, 0.0);
        for _ in 0..40 {
            sim.drum.sculpt(crest, 3.0, 0.05);
        }
    }
    for k in 0..8 {
        let a = k as f64 * 0.8;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let axis = sim.drum.ring.radius.0 as f64;
                sim.inject(&mut fluid, [a.cos() * 7.5 - axis, 0.0, a.sin() * 7.5], 400)
            });
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(6.0));
    let height = app
        .world()
        .resource::<Simulation>()
        .drum
        .landscape
        .max_height() as f64;
    assert!(height > 1.5, "landscape height {height}");
    assert_water_lies_on_the_ground(&mut app, 0.06);
}

/// Water poured onto a mound on a ring kilometres across, far round it from where the wheel's
/// angles start, runs down it and lies on the ground round its foot, never in it.
#[test]
#[ignore = "wants a GPU"]
fn water_poured_on_a_mound_of_a_big_ring_lies_on_it() {
    let ring = Ring {
        radius: Metres(5000.0),
        half_width: Metres(500.0),
    };
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.equalize_thrust();
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(0.2));
    let (turn, y) = (2.5, 200.0);
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let crest = sim.drum.wall_point(turn, y);
        for _ in 0..30 {
            sim.drum.sculpt(crest, 3.0, 0.1);
        }
    }
    for k in 0..6 {
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let radius = sim.drum.ring.radius.0 as f64;
                let on = sim.drum.wall_point(turn + (k as f64 - 2.5) / radius, y);
                let (_, out) = sim.drum.depth_and_outward(on);
                let lift = sim.drum.ground(on) + 1.5;
                let above = [on[0] - out[0] * lift, on[1], on[2] - out[2] * lift];
                sim.inject(&mut fluid, above, 300)
            });
        testing::run(&mut app, Seconds(0.3));
    }
    testing::run(&mut app, Seconds(5.0));
    assert_eq!(app.world().resource::<Fluid>().len(), 1800);
    assert_water_lies_on_the_ground(&mut app, 0.06);
}

/// Ground raised under water already standing on it lifts the water with it, however much has
/// been sculpted elsewhere since the water was put down.
#[test]
#[ignore = "wants a GPU"]
fn ground_raised_under_standing_water_lifts_it() {
    let ring = Ring {
        radius: Metres(100.0),
        half_width: Metres(20.0),
    };
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.equalize_thrust();
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(0.2));
    let (turn, y) = (0.2, 4.0);
    for k in 0..4 {
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let on = sim.drum.wall_point(turn + (k as f64 - 1.5) / 100.0, y);
                let (_, out) = sim.drum.depth_and_outward(on);
                let lift = sim.drum.ground(on) + 1.0;
                let above = [on[0] - out[0] * lift, on[1], on[2] - out[2] * lift];
                sim.inject(&mut fluid, above, 300)
            });
        testing::run(&mut app, Seconds(0.3));
    }
    testing::run(&mut app, Seconds(2.0));
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        for k in 0..100 {
            let far = turn + 1.0 + (k % 50) as f64 * 0.09;
            let across = if k < 50 { -14.0 } else { 14.0 };
            let at = sim.drum.wall_point(far, across);
            sim.drum.sculpt(at, 1.0, 0.2);
        }
    }
    assert!(
        app.world()
            .resource::<Simulation>()
            .drum
            .landscape
            .patches()
            .count()
            >= 100
    );
    for _ in 0..40 {
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            let under = sim.drum.wall_point(turn, y);
            sim.drum.sculpt(under, 2.5, 0.04);
        }
        testing::run(&mut app, Seconds(1.0 / 30.0));
    }
    testing::run(&mut app, Seconds(2.0));
    let sim = app.world().resource::<Simulation>();
    let under = sim.drum.wall_point(turn, y);
    assert!(sim.drum.ground(under) > 1.5, "the mound was not raised");
    assert_water_lies_on_the_ground(&mut app, 0.06);
}

/// No particle of the water is further into the ground under it than this.
fn assert_water_lies_on_the_ground(app: &mut App, tolerance: f64) {
    let particles = testing::particles(app);
    let sim = app.world().resource::<Simulation>();
    for p in &particles {
        let at = sim.drum.from_water(p.position);
        let over = sim.drum.height_above_glass(at);
        let ground = sim.drum.ground(at);
        assert!(
            over >= ground - tolerance,
            "particle inside terrain: {over} m over the glass, ground at {ground} m"
        );
    }
}
