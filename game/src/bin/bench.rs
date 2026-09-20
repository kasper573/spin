//! A fixed workload: the ring world under one g with a lake poured into it, then with rain
//! falling into the lake, run headless for a few hundred frames of one substep each, issued as
//! fast as the machine allows, and timed per frame, GPU included.
use bevy::platform::time::Instant;
use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::shallows::Shallows;
use game::core::units::Seconds;
use game::systems::sim::{SUBSTEP_RATE, Simulation};
use game::systems::testing;

fn main() {
    let mut app = testing::headless();
    for round in 0..115 {
        pour(&mut app, round, 600);
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(3.0));
    measure(&mut app, "a lake", 0);
    measure(&mut app, "rain on the lake", 250);
}

/// Water let go near the axis, a different way out from it each round.
fn pour(app: &mut App, round: u32, particles: u32) {
    let a = round as f64 * 0.7 + 0.3;
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let axis = sim.drum.ring.radius.0 as f64;
            sim.inject(
                &mut fluid,
                [
                    a.cos() * 7.0 - axis,
                    ((round % 5) as f64 - 2.0) * 2.0,
                    a.sin() * 7.0,
                ],
                particles,
            )
        });
}

fn measure(app: &mut App, label: &str, raining: u32) {
    testing::particles(app);
    let frames = 300;
    let mut flying = 0;
    let start = Instant::now();
    for frame in 0..frames {
        if raining > 0 {
            pour(app, frame, raining);
        }
        testing::frame(app, SUBSTEP_RATE.period());
        flying += app.world().resource::<Fluid>().len();
    }
    testing::settle(app);
    testing::particles(app);
    let elapsed = start.elapsed().as_secs_f64();
    let per_frame_ms = elapsed / frames as f64 * 1000.0;
    let lying = app.world().resource::<Shallows>().litres().0 / 1000.0;
    println!(
        "{label}: {lying:.0} cubic metres lying, {} particles in flight",
        flying / frames as usize
    );
    println!(
        "frame: {per_frame_ms:.2} ms ({:.0} fps)",
        1000.0 / per_frame_ms
    );
}
