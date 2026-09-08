//! A fixed workload: a spinning drum a third full of water with a few rafts, timed per substep.
use std::time::Instant;

use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::sim::Simulation;

fn main() {
    let mut sim = Simulation::default();
    sim.drum.target_spin = RadiansPerSecond(1.6);
    sim.drum.spin = RadiansPerSecond(1.6);
    for round in 0..40 {
        let a = round as f32 * 0.5;
        sim.inject(
            [
                a.cos() * 2.6,
                ((round % 3) as f32 - 1.0) * 0.3,
                a.sin() * 2.6,
            ],
            200,
        );
        sim.advance_exact(Seconds(0.2));
    }
    for k in 0..4 {
        let a = k as f64 * 1.3;
        let n = [-a.cos(), 0.0, -a.sin()];
        sim.spawn_raft([a.cos() * 3.2, 0.0, a.sin() * 3.2], n);
    }
    sim.advance_exact(Seconds(3.0));

    let particles = sim.fluid.len();
    let steps = 180;
    let start = Instant::now();
    for _ in 0..steps {
        sim.substep();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_step_ms = elapsed / steps as f64 * 1000.0;
    let per_particle_us = elapsed / steps as f64 / particles as f64 * 1e6;
    println!("particles: {particles}, rafts: {}", sim.rafts.len());
    println!("substep: {per_step_ms:.2} ms, {per_particle_us:.3} us per particle");
}
