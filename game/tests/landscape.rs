use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::{RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{DEFAULT_RING, Drum, Landscape, Site};
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::testing;

#[test]
fn sculpting_raises_ground_under_the_brush_only() {
    let mut land = Landscape::new(DEFAULT_RING);
    assert!(land.is_empty());
    for _ in 0..10 {
        land.sculpt(1.0, 0.0, 0.5, 0.05);
    }
    assert!(!land.is_empty());
    let (h, _, _) = land.sample(1.0, 0.0);
    assert!(h > 0.3 && h <= 0.5, "height {h}");
    let (far, _, _) = land.sample(1.0 + std::f64::consts::PI, 0.0);
    assert_eq!(far, 0.0);
    land.flatten(game::core::units::Metres(0.0));
    assert!(land.is_empty());
}

#[test]
fn heights_are_clamped_on_load() {
    let mut land = Landscape::new(DEFAULT_RING);
    land.load(&[f32::NAN, -1.0, 100.0]);
    assert_eq!(land.height_at(0, 0), 0.0);
    assert_eq!(land.height_at(0, 1), 0.0);
    assert!(land.height_at(0, 2) <= DEFAULT_RING.max_height().0);
}

#[test]
fn raised_ground_is_a_wall_where_it_is_raised_only() {
    let mut drum = Drum {
        site: Site::at(0.7, 0.0, Drum::default().ring),
        ..Drum::default()
    };
    for _ in 0..20 {
        drum.landscape.sculpt(0.7, 0.0, 1.5, 0.05);
    }
    let under = drum.wall_point(0.0, 0.0);
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
        for _ in 0..40 {
            sim.drum.landscape.sculpt(0.8, 0.0, 3.0, 0.05);
        }
    }
    for k in 0..8 {
        let a = k as f64 * 0.8;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let sim = world.resource::<Simulation>();
                let at = sim.drum.from_water([a.cos() * 7.5, 0.0, a.sin() * 7.5]);
                sim.inject(&mut fluid, at, 400)
            });
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(6.0));
    let particles = testing::particles(&mut app);
    let sim = app.world().resource::<Simulation>();
    let radius = sim.drum.ring.radius.0 as f64;
    let height = sim.drum.landscape.max_height() as f64;
    assert!(height > 1.5, "landscape height {height}");
    for p in &particles {
        let [x, y, z] = p.position;
        let r = (x * x + z * z).sqrt();
        let (h, _, _) = sim.drum.landscape.sample(z.atan2(x), y);
        assert!(
            r <= radius - h + 0.06,
            "particle inside terrain: r {r}, ground at {}",
            radius - h
        );
    }
}
