use game::core::math::norm;
use game::core::shuttle::THRUST;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::RADIUS;
use game::systems::sim::Simulation;

fn thrust_toward_axis(sim: &mut Simulation) {
    let p = sim.shuttle().p;
    let d = norm(&p);
    sim.shuttle_input.thrust = [-p[0] / d * THRUST, -p[1] / d * THRUST, -p[2] / d * THRUST];
}

fn radius(sim: &Simulation) -> f64 {
    let p = sim.shuttle().p;
    (p[0] * p[0] + p[2] * p[2]).sqrt()
}

#[test]
fn ghost_shuttle_flies_into_the_drum() {
    let mut sim = Simulation::default();
    assert!(!sim.drum.encloses(sim.shuttle().p));
    for _ in 0..60 {
        thrust_toward_axis(&mut sim);
        sim.advance_exact(Seconds(0.1));
    }
    assert!(
        sim.drum.encloses(sim.shuttle().p),
        "at {:?}",
        sim.shuttle().p
    );
}

#[test]
fn solid_shuttle_outside_stays_outside() {
    let mut sim = Simulation::default();
    sim.shuttle_mut().solid = true;
    for _ in 0..80 {
        thrust_toward_axis(&mut sim);
        sim.advance_exact(Seconds(0.1));
    }
    let p = sim.shuttle().p;
    assert!(!sim.drum.encloses(p), "entered the drum at {p:?}");
    assert!(
        p[0].abs() < 10.0 && p[1].abs() < 10.0 && p[2].abs() < 10.0,
        "flew off to {p:?}"
    );
}

#[test]
fn solid_shuttle_inside_falls_to_the_floor_upright_and_stays_in() {
    let mut sim = Simulation::default();
    sim.drum.spin = RadiansPerSecond(1.6);
    sim.drum.target_spin = RadiansPerSecond(1.6);
    sim.shuttle_mut()
        .place([0.5, 0.0, 0.0], [0.383, 0.0, 0.0, 0.924]);
    sim.shuttle_mut().solid = true;
    sim.advance_exact(Seconds(12.0));
    let body = sim.shuttle();
    let r = radius(&sim);
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
        let p = sim.shuttle().p;
        sim.shuttle_input.thrust = [p[0] / r * THRUST, 0.0, p[2] / r * THRUST];
        sim.advance_exact(Seconds(0.1));
    }
    assert!(
        sim.drum.encloses(sim.shuttle().p),
        "escaped to {:?}",
        sim.shuttle().p
    );
}

#[test]
fn flight_assist_brakes_to_a_stop_in_vacuum() {
    let mut sim = Simulation::default();
    sim.shuttle_input.thrust = [THRUST, 0.0, 0.0];
    sim.advance_exact(Seconds(1.0));
    assert!(sim.shuttle().v[0] > 2.0, "v {:?}", sim.shuttle().v);
    sim.shuttle_input.thrust = [0.0; 3];
    sim.advance_exact(Seconds(3.0));
    assert!(norm(&sim.shuttle().v) < 0.05, "v {:?}", sim.shuttle().v);
}

#[test]
fn ghost_shuttle_in_the_drum_is_carried_round_and_flung_out() {
    let mut sim = Simulation::default();
    sim.drum.spin = RadiansPerSecond(1.0);
    sim.drum.target_spin = RadiansPerSecond(1.0);
    sim.shuttle_mut()
        .place([2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    sim.advance_exact(Seconds(1.5));
    let body = sim.shuttle();
    let r = radius(&sim);
    assert!(sim.drum.encloses(body.p), "already out at {:?}", body.p);
    let tangential = (body.v[0] * body.p[2] - body.v[2] * body.p[0]) / r;
    assert!((tangential - r).abs() < 0.5, "slip {tangential} vs {r}");
    sim.advance_exact(Seconds(6.0));
    assert!(
        !sim.drum.encloses(sim.shuttle().p),
        "still inside at {:?}",
        sim.shuttle().p
    );
}
