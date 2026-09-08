//! A fixed workload: a spinning drum a third full of water with a few rafts, then half full, run
//! headless for a few hundred frames of one substep each and timed per frame, GPU included.
use bevy::platform::time::Instant;
use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::sim::{SUBSTEP_RATE, Simulation};
use game::systems::testing;

fn main() {
    let mut app = testing::headless();
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        sim.drum.target_spin = RadiansPerSecond(1.6);
        sim.drum.spin = RadiansPerSecond(1.6);
    }
    for round in 0..40 {
        let a = round as f32 * 0.5;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [
                        a.cos() * 2.6,
                        ((round % 3) as f32 - 1.0) * 0.3,
                        a.sin() * 2.6,
                    ],
                    200,
                )
            });
        testing::run(&mut app, Seconds(0.2));
    }
    for k in 0..4 {
        let a = k as f64 * 1.3;
        let n = [-a.cos(), 0.0, -a.sin()];
        app.world_mut()
            .resource_mut::<Simulation>()
            .spawn_raft([a.cos() * 3.2, 0.0, a.sin() * 3.2], n);
    }
    testing::run(&mut app, Seconds(3.0));
    measure(&mut app, "a third full");

    for round in 0..75 {
        let a = round as f32 * 0.7 + 0.3;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [
                        a.cos() * 2.4,
                        ((round % 5) as f32 - 2.0) * 0.2,
                        a.sin() * 2.4,
                    ],
                    200,
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
        testing::run(app, SUBSTEP_RATE.period());
    }
    testing::particles(app);
    let elapsed = start.elapsed().as_secs_f64();
    let per_frame_ms = elapsed / frames as f64 * 1000.0;
    let per_particle_us = elapsed / frames as f64 / particles as f64 * 1e6;
    println!(
        "{label}: {particles} particles, {} rafts",
        app.world().resource::<Simulation>().rafts().len()
    );
    println!(
        "frame: {per_frame_ms:.2} ms ({:.0} fps), {per_particle_us:.3} us per particle",
        1000.0 / per_frame_ms
    );
}
