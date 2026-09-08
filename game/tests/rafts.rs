use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::RADIUS;
use game::systems::settings::Settings;
use game::systems::sim::{MAX_RAFTS, Simulation};
use game::systems::testing;

fn fill(app: &mut App, spin: f32) {
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(spin);
    app.world_mut().resource_mut::<Simulation>().drum.spin = RadiansPerSecond(spin);
    for k in 0..12 {
        let a = k as f32 * 0.52;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [a.cos() * 2.7, (k % 3) as f32 * 0.3 - 0.3, a.sin() * 2.7],
                    150,
                )
            });
        testing::run(app, Seconds(0.2));
    }
    testing::run(app, Seconds(3.0));
}

#[test]
fn rafts_float_and_ride_with_the_glass() {
    let mut app = testing::headless();
    fill(&mut app, 1.6);
    for k in 0..3 {
        let a = k as f64 * 2.0;
        let n = [-a.cos(), 0.0, -a.sin()];
        assert!(
            app.world_mut()
                .resource_mut::<Simulation>()
                .spawn_raft([a.cos() * 3.1, 0.0, a.sin() * 3.1], n)
        );
    }
    testing::run(&mut app, Seconds(10.0));
    let sim = app.world().resource::<Simulation>();
    for body in sim.rafts() {
        let r = (body.p[0] * body.p[0] + body.p[2] * body.p[2]).sqrt();
        assert!(r < RADIUS as f64 && r > 2.6, "raft radius {r}");
        assert!(body.p[1].abs() <= 0.6, "raft y {}", body.p[1]);
        let tangential = (body.v[0] * body.p[2] - body.v[2] * body.p[0]) / r;
        let glass = sim.drum.spin.0 as f64 * r;
        assert!(
            (tangential - glass).abs() < 1.0,
            "slip {} vs {glass}",
            tangential
        );
        let spin = (body.w[0] * body.w[0] + body.w[1] * body.w[1] + body.w[2] * body.w[2]).sqrt();
        assert!(spin < 6.0, "tumbling at {spin} rad/s");
        assert!(body.wet > 0.0, "raft never touched the water");
    }
}

#[test]
fn raft_count_is_capped() {
    let mut sim = Simulation::default();
    let mut spawned = 0;
    for k in 0..20 {
        let a = k as f64 * 0.4;
        if sim.spawn_raft(
            [a.cos() * 3.0, 0.0, a.sin() * 3.0],
            [-a.cos(), 0.0, -a.sin()],
        ) {
            spawned += 1;
        }
    }
    assert_eq!(spawned, MAX_RAFTS);
    assert_eq!(sim.rafts().len(), MAX_RAFTS);
}
