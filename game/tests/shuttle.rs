use bevy::prelude::*;
use game::core::math::norm;
use game::core::shuttle::THRUST;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::RADIUS;
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::testing;

fn thrust_toward_axis(app: &mut App) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let p = sim.shuttle().p;
    let d = norm(&p);
    sim.shuttle_input.thrust = [-p[0] / d * THRUST, -p[1] / d * THRUST, -p[2] / d * THRUST];
}

fn state(app: &App) -> &Simulation {
    app.world().resource::<Simulation>()
}

fn state_mut(app: &mut App) -> Mut<'_, Simulation> {
    app.world_mut().resource_mut::<Simulation>()
}

fn radius(sim: &Simulation) -> f64 {
    let p = sim.shuttle().p;
    (p[0] * p[0] + p[2] * p[2]).sqrt()
}

#[test]
fn ghost_shuttle_flies_into_the_drum() {
    let mut app = testing::headless();
    assert!(!state(&app).drum.encloses(state(&app).shuttle().p));
    for _ in 0..60 {
        thrust_toward_axis(&mut app);
        testing::run(&mut app, Seconds(0.1));
    }
    let sim = state(&app);
    assert!(
        sim.drum.encloses(sim.shuttle().p),
        "at {:?}",
        sim.shuttle().p
    );
}

#[test]
fn solid_shuttle_outside_stays_outside() {
    let mut app = testing::headless();
    app.world_mut().resource_mut::<Settings>().collisions = true;
    state_mut(&mut app).shuttle_mut().solid = true;
    for _ in 0..80 {
        thrust_toward_axis(&mut app);
        testing::run(&mut app, Seconds(0.1));
    }
    let sim = state(&app);
    let p = sim.shuttle().p;
    assert!(!sim.drum.encloses(p), "entered the drum at {p:?}");
    assert!(
        p[0].abs() < 10.0 && p[1].abs() < 10.0 && p[2].abs() < 10.0,
        "flew off to {p:?}"
    );
}

#[test]
fn solid_shuttle_inside_falls_to_the_floor_upright_and_stays_in() {
    let mut app = testing::headless();
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(1.6);
    {
        let mut sim = state_mut(&mut app);
        sim.drum.spin = RadiansPerSecond(1.6);
        sim.shuttle_mut()
            .place([0.5, 0.0, 0.0], [0.383, 0.0, 0.0, 0.924]);
        sim.shuttle_mut().solid = true;
    }
    app.world_mut().resource_mut::<Settings>().collisions = true;
    testing::run(&mut app, Seconds(12.0));
    let sim = state(&app);
    let body = sim.shuttle();
    let r = radius(sim);
    assert!(r > 2.7 && r < RADIUS as f64, "radius {r}");
    assert!(body.p[1].abs() < 0.6, "y {}", body.p[1]);
    let up = body.rotate(&[0.0, 1.0, 0.0]);
    let inward = [-body.p[0] / r, 0.0, -body.p[2] / r];
    let alignment = up[0] * inward[0] + up[1] * inward[1] + up[2] * inward[2];
    assert!(alignment > 0.95, "up {up:?} vs inward {inward:?}");
    let tangential = (body.v[0] * body.p[2] - body.v[2] * body.p[0]) / r;
    assert!(
        (tangential - 1.6 * r).abs() < 0.5,
        "slip {tangential} vs {}",
        1.6 * r
    );

    for _ in 0..60 {
        let mut sim = state_mut(&mut app);
        let p = sim.shuttle().p;
        sim.shuttle_input.thrust = [p[0] / r * THRUST, 0.0, p[2] / r * THRUST];
        testing::run(&mut app, Seconds(0.1));
    }
    let sim = state(&app);
    assert!(
        sim.drum.encloses(sim.shuttle().p),
        "escaped to {:?}",
        sim.shuttle().p
    );
}

#[test]
fn flight_assist_brakes_to_a_stop_in_vacuum() {
    let mut app = testing::headless();
    state_mut(&mut app).shuttle_input.thrust = [THRUST, 0.0, 0.0];
    testing::run(&mut app, Seconds(1.0));
    assert!(
        state(&app).shuttle().v[0] > 2.0,
        "v {:?}",
        state(&app).shuttle().v
    );
    state_mut(&mut app).shuttle_input.thrust = [0.0; 3];
    testing::run(&mut app, Seconds(3.0));
    assert!(
        norm(&state(&app).shuttle().v) < 0.05,
        "v {:?}",
        state(&app).shuttle().v
    );
}

#[test]
fn ghost_shuttle_in_the_drum_is_carried_round_and_flung_out() {
    let mut app = testing::headless();
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(1.0);
    {
        let mut sim = state_mut(&mut app);
        sim.drum.spin = RadiansPerSecond(1.0);
        sim.shuttle_mut()
            .place([2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    }
    testing::run(&mut app, Seconds(1.5));
    {
        let sim = state(&app);
        let body = sim.shuttle();
        let r = radius(sim);
        assert!(sim.drum.encloses(body.p), "already out at {:?}", body.p);
        let tangential = (body.v[0] * body.p[2] - body.v[2] * body.p[0]) / r;
        assert!((tangential - r).abs() < 0.5, "slip {tangential} vs {r}");
    }
    testing::run(&mut app, Seconds(6.0));
    let sim = state(&app);
    assert!(
        !sim.drum.encloses(sim.shuttle().p),
        "still inside at {:?}",
        sim.shuttle().p
    );
}
