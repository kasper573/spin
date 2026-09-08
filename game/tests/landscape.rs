use bevy::prelude::*;
use game::core::fluid::{Fluid, PARTICLE_SPACING};
use game::core::units::{Radians, RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{Drum, Landscape, RADIUS, wheel_angle};
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::testing;

#[test]
fn sculpting_raises_ground_under_the_brush_only() {
    let mut land = Landscape::new();
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
    let mut land = Landscape::new();
    land.load(&[f32::NAN, -1.0, 100.0]);
    assert_eq!(land.height_at(0, 0), 0.0);
    assert_eq!(land.height_at(0, 1), 0.0);
    assert!(land.height_at(0, 2) <= RADIUS - 1.0);
}

#[test]
fn terrain_pushes_particles_out_and_turns_with_the_drum() {
    let mut drum = Drum {
        angle: Radians(0.7),
        ..Drum::default()
    };
    for _ in 0..20 {
        drum.landscape
            .sculpt(wheel_angle(9.0, 0.0, 0.7), 0.0, 1.5, 0.05);
    }
    let mut p = [9.9, 0.0, 0.0];
    let contact = drum.confine(&mut p, 0.05);
    assert!(!contact.is_empty());
    let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
    assert!(r < 9.1, "particle left at radius {r}");
    let mut elsewhere = [0.0, 0.0, 9.8];
    let contact = drum.confine(&mut elsewhere, 0.05);
    assert!(contact.is_empty());
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
        let a = k as f32 * 0.8;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [a.cos() * 7.5, 0.0, a.sin() * 7.5],
                    400,
                )
            });
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(6.0));
    let particles = testing::particles(&mut app);
    let sim = app.world().resource::<Simulation>();
    let angle = sim.drum.angle.0;
    let height = sim.drum.landscape.max_height() as f64;
    assert!(height > 1.5, "landscape height {height}");
    for p in &particles {
        let [x, y, z] = p.position.map(|v| v as f64);
        let r = (x * x + z * z).sqrt();
        let (h, _, _) = sim.drum.landscape.sample(wheel_angle(x, z, angle), y);
        assert!(
            r <= RADIUS as f64 - h + PARTICLE_SPACING as f64,
            "particle inside terrain: r {r}, ground at {}",
            RADIUS as f64 - h
        );
    }
}
