use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::RADIUS;
use game::systems::sim::Simulation;

fn radius(p: [f32; 3]) -> f32 {
    (p[0] * p[0] + p[2] * p[2]).sqrt()
}

#[test]
fn water_stays_inside_the_drum() {
    let mut sim = Simulation::default();
    sim.drum.target_spin = RadiansPerSecond(3.0);
    sim.drum.spin = RadiansPerSecond(3.0);
    sim.inject([2.0, 0.0, 0.0], 800);
    sim.advance_exact(Seconds(2.0));
    assert_eq!(sim.fluid.len(), 800);
    for p in sim.fluid.particles() {
        assert!(radius(p.position) <= RADIUS + 1e-4, "{:?}", p.position);
        assert!(p.position[1].abs() <= 0.6 + 1e-4, "{:?}", p.position);
        assert!(p.velocity.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn spinning_drum_throws_water_onto_the_glass() {
    let mut sim = Simulation::default();
    sim.drum.target_spin = RadiansPerSecond(2.0);
    sim.drum.spin = RadiansPerSecond(2.0);
    sim.inject([1.5, 0.0, 0.5], 600);
    sim.advance_exact(Seconds(6.0));
    let mean_radius = sim
        .fluid
        .particles()
        .map(|p| radius(p.position))
        .sum::<f32>()
        / sim.fluid.len() as f32;
    assert!(mean_radius > 3.0, "mean radius {mean_radius}");
}

#[test]
fn injection_counts_litres() {
    let mut sim = Simulation::default();
    let added = sim.inject([0.0, 0.0, 0.0], 250);
    assert_eq!(added, 250);
    assert!((sim.water().0 - 250.0).abs() < 1e-3);
    sim.fluid.clear();
    assert_eq!(sim.water().0, 0.0);
}

#[test]
fn reset_keeps_parameters_and_target_spin() {
    let mut sim = Simulation::default();
    sim.params.viscosity = 0.7;
    sim.drum.target_spin = RadiansPerSecond(1.0);
    sim.inject([0.0, 0.0, 0.0], 10);
    sim.reset();
    assert!(sim.fluid.is_empty());
    assert_eq!(sim.params.viscosity, 0.7);
    assert_eq!(sim.drum.target_spin, RadiansPerSecond(1.0));
}
