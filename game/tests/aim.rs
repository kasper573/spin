//! Where the crosshair rests and what it outlines.
use bevy::math::DVec3;
use game::core::units::Seconds;
use game::systems::aim::{self, AimPoint};
use game::systems::drum::GROUND_DEPTH;
use game::systems::sim::Simulation;
use game::systems::testing;

/// The outline of a brush on uneven ground lies on the ground, just above it, everywhere
/// round it: every point is the brush's radius from its centre along the ground and at the
/// ground's height there, hill or hollow, so the outline wraps the ground like a sheet laid
/// over it.
#[test]
fn the_outline_wraps_the_ground_under_it() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.1));
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let site = sim.drum.site;
        for k in 0..6 {
            let a = k as f64 * 1.1;
            let (phi, y) = (site.phi + 0.02 * a.cos(), 0.5 + 2.0 * a.sin());
            sim.drum
                .landscape
                .sculpt(phi, y, 1.5, 1.2 * (k as f64 + 1.0));
        }
    }
    let sim = app.world().resource::<Simulation>();
    let eye = DVec3::new(-3.0, 0.0, 0.0);
    let (radius, lift) = (2.0, 0.02);
    for (dy, dz) in [(0.0, 0.0), (1.0, -1.0), (-2.0, 1.5), (2.5, 2.5)] {
        let dir = (DVec3::new(-0.5, dy, dz) - eye).normalize();
        let hit = aim::cast(eye, dir, &sim.drum).expect("the ground is in front of the eye");
        let outline = aim::outline(&sim.drum, hit, radius, lift);
        assert!(outline.len() >= 48);
        let centre_turn = sim.drum.turn_to(hit.point.to_array());
        let centre_axial = sim.drum.axial(hit.point.to_array());
        let ring_radius = sim.drum.ring.radius.0 as f64;
        let mut heights_seen = Vec::new();
        for p in &outline {
            let p = p.to_array();
            let ground = sim
                .drum
                .landscape
                .sample(sim.drum.wheel_angle_of(p), sim.drum.axial(p))
                .0;
            let above = sim.drum.height_above_glass(p);
            assert!(
                (above - ground - lift).abs() < 1e-6,
                "an outline point is {above} m above the glass over ground {ground} m high"
            );
            let along = (sim.drum.turn_to(p) - centre_turn) * ring_radius;
            let across = sim.drum.axial(p) - centre_axial;
            let distance = (along * along + across * across).sqrt();
            assert!(
                (distance - radius).abs() < 1e-6,
                "an outline point is {distance} m from the centre"
            );
            heights_seen.push(ground);
        }
        let (lowest, highest) = heights_seen
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| {
                (lo.min(*h), hi.max(*h))
            });
        assert!(
            highest - lowest > 0.3 || highest <= GROUND_DEPTH.0 as f64 + 1e-6,
            "the ground under the outline is flat: {lowest}..{highest}"
        );
    }
    let flat: AimPoint =
        aim::cast(eye, DVec3::new(0.0, 1.0, 0.0), &sim.drum).expect("the cap is above the eye");
    let outline = aim::outline(&sim.drum, flat, radius, lift);
    for p in &outline {
        assert!(
            (p.y - flat.point.y + lift).abs() < 1e-9,
            "a cap outline point is off the cap"
        );
    }
}
