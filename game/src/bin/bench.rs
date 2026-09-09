//! A fixed workload: the ring world under one g a sixth full of water, then
//! half full, run headless for a few hundred frames of one substep each, issued as fast as the
//! machine allows, and timed per frame, GPU included.
use bevy::platform::time::Instant;
use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::Seconds;
use game::systems::sim::{SUBSTEP_RATE, Simulation};
use game::systems::testing;

fn main() {
    let mut app = testing::headless();
    for round in 0..40 {
        let a = round as f64 * 0.5;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let sim = world.resource::<Simulation>();
                sim.inject(
                    &mut fluid,
                    sim.drum.from_water([
                        a.cos() * 8.0,
                        ((round % 3) as f64 - 1.0) * 3.0,
                        a.sin() * 8.0,
                    ]),
                    500,
                )
            });
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(3.0));
    measure(&mut app, "a sixth full");

    for round in 0..75 {
        let a = round as f64 * 0.7 + 0.3;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let sim = world.resource::<Simulation>();
                sim.inject(
                    &mut fluid,
                    sim.drum.from_water([
                        a.cos() * 7.0,
                        ((round % 5) as f64 - 2.0) * 2.0,
                        a.sin() * 7.0,
                    ]),
                    600,
                )
            });
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(3.0));
    measure(&mut app, "half full");
}

fn measure(app: &mut App, label: &str) {
    testing::particles(app);
    let particles = app.world().resource::<Fluid>().len();
    let frames = 300;
    let start = Instant::now();
    for _ in 0..frames {
        testing::frame(app, SUBSTEP_RATE.period());
    }
    testing::settle(app);
    testing::particles(app);
    let elapsed = start.elapsed().as_secs_f64();
    let per_frame_ms = elapsed / frames as f64 * 1000.0;
    let per_particle_us = elapsed / frames as f64 / particles as f64 * 1e6;
    println!("{label}: {particles} particles");
    println!(
        "frame: {per_frame_ms:.2} ms ({:.0} fps), {per_particle_us:.3} us per particle",
        1000.0 / per_frame_ms
    );
}
