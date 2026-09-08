use game::core::units::{Radians, RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{Drum, Landscape, RADIUS, wheel_angle};
use game::systems::sim::Simulation;

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
    land.reset();
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
            .sculpt(wheel_angle(3.0, 0.0, 0.7), 0.0, 0.6, 0.05);
    }
    let mut p = [3.2, 0.0, 0.0];
    let contact = drum.confine(&mut p, 0.05);
    assert!(!contact.is_empty());
    let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
    assert!(r < 2.9, "particle left at radius {r}");
    let mut elsewhere = [0.0, 0.0, 3.2];
    let contact = drum.confine(&mut elsewhere, 0.05);
    assert!(contact.is_empty());
}

#[test]
fn water_settles_on_top_of_raised_ground() {
    let mut sim = Simulation::default();
    sim.drum.target_spin = RadiansPerSecond(1.6);
    sim.drum.spin = RadiansPerSecond(1.6);
    for _ in 0..40 {
        sim.drum.landscape.sculpt(0.8, 0.0, 1.2, 0.03);
    }
    for k in 0..8 {
        let a = k as f32 * 0.8;
        sim.inject([a.cos() * 2.0, 0.0, a.sin() * 2.0], 150);
        sim.advance_exact(Seconds(0.2));
    }
    sim.advance_exact(Seconds(5.0));
    let inside = sim
        .fluid
        .particles()
        .filter(|p| {
            let (pen, _) = sim.drum.landscape.penetration(
                p.position[0] as f64,
                p.position[1] as f64,
                p.position[2] as f64,
                sim.drum.angle.0,
                0.0,
            );
            pen > 0.06
        })
        .count();
    assert_eq!(inside, 0);
}
