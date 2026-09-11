use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::{Metres, RadiansPerSecond, Seconds};
use game::systems::drum::Ring;
use game::systems::persistence::{Snapshot, apply, snapshot};
use game::systems::player::Player;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

#[test]
fn snapshot_round_trips_through_json() {
    let mut app = testing::headless();
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        sim.drum.target_spin = RadiansPerSecond(1.5);
        let at = sim.drum.wall_point(0.3, 0.1);
        sim.drum.sculpt(at, 1.5, 0.2);
    }
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let axis = sim.drum.ring.radius.0 as f64;
            sim.inject(&mut fluid, [7.0 - axis, 0.1, 0.0], 40)
        });
    testing::run(&mut app, Seconds(1.0));
    let particles = testing::particles(&mut app);
    assert_eq!(particles.len(), 40);
    let settings = Settings {
        flow: game::core::units::LitresPerSecond(900.0),
        ..Settings::default()
    };
    let world = app.world_mut();
    let mut sim = world.resource_mut::<Simulation>();
    Player.teleport(&mut sim, [1.0, 2.0, 3.0], [0.0; 3]);
    let sim = world.resource::<Simulation>();
    let fluid = world.resource::<Fluid>();

    let json = serde_json::to_string(&snapshot(&settings, sim, fluid)).unwrap();
    let restored: Snapshot = serde_json::from_str(&json).unwrap();
    let (shuttle, attitude, angle, ground, water) = (
        sim.avatar().p,
        sim.avatar().q,
        sim.drum.angle.0,
        sim.drum.landscape.ground(),
        sim.drum.water,
    );

    let mut settings2 = Settings::default();
    let mut sim2 = Simulation::default();
    let mut fluid2 = world.resource_mut::<Fluid>();
    apply(&restored, &mut settings2, &mut sim2, &mut fluid2);

    assert_eq!(settings2, settings);
    assert_eq!(fluid2.len(), 40);
    assert!(!ground.patches.is_empty());
    assert_eq!(sim2.drum.landscape.ground(), ground);
    assert_eq!(
        (sim2.drum.water.round, sim2.drum.water.y),
        (water.round, water.y)
    );
    assert!((sim2.drum.angle.0 - angle).abs() < 1e-9);
    assert_eq!(sim2.avatar().p, shuttle);
    for (a, b) in sim2.avatar().q.iter().zip(attitude) {
        assert!(
            (a - b).abs() < 1e-9,
            "attitude {:?} vs {attitude:?}",
            sim2.avatar().q
        );
    }
    assert_eq!(restored.fluid.len(), 40 * 7);
    assert_eq!(restored.fluid[0], particles[0].position[0] as f32);
}

/// A ring kilometres across keeps what was sculpted on it and where its water lies, far round
/// it from where the wheel's angles start.
#[test]
fn a_big_ring_keeps_its_ground_and_its_water_through_a_save() {
    let ring = Ring {
        radius: Metres(5000.0),
        half_width: Metres(500.0),
    };
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.equalize_thrust();
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(0.2));
    let (turn, y) = (2.0, 120.0);
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let at = sim.drum.wall_point(turn, y);
        for _ in 0..10 {
            sim.drum.sculpt(at, 2.0, 0.2);
        }
    }
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let on = sim.drum.wall_point(turn, y);
            let (_, out) = sim.drum.depth_and_outward(on);
            let lift = sim.drum.ground(on) + 1.5;
            let above = [on[0] - out[0] * lift, on[1], on[2] - out[2] * lift];
            sim.inject(&mut fluid, above, 40)
        });
    testing::run(&mut app, Seconds(0.5));
    testing::particles(&mut app);
    let world = app.world_mut();
    let sim = world.resource::<Simulation>();
    let json = serde_json::to_string(&snapshot(
        world.resource::<Settings>(),
        sim,
        world.resource::<Fluid>(),
    ))
    .unwrap();
    let restored: Snapshot = serde_json::from_str(&json).unwrap();
    let (ground, water) = (sim.drum.landscape.ground(), sim.drum.water);
    let mut settings = Settings::default();
    let mut sim = Simulation::default();
    world.resource_scope(|_, mut fluid: Mut<Fluid>| {
        apply(&restored, &mut settings, &mut sim, &mut fluid);
        assert_eq!(fluid.len(), 40);
    });
    assert!(!ground.patches.is_empty());
    assert_eq!(sim.drum.landscape.ground(), ground);
    assert_eq!(
        (sim.drum.water.round, sim.drum.water.y),
        (water.round, water.y)
    );
    app.insert_resource(settings);
    app.insert_resource(sim);
    testing::run(&mut app, Seconds(0.1));
    let particles = testing::particles(&mut app);
    let sim = app.world().resource::<Simulation>();
    for p in &particles {
        let at = sim.drum.from_water(p.position);
        let over = sim.drum.height_above_glass(at) - sim.drum.ground(at);
        assert!(
            over > 0.0 && over < 3.0,
            "a particle came back {over} m over the ground under it"
        );
    }
}

#[test]
fn settings_are_sanitized_on_load() {
    let text = r#"{"spin": 99.0, "flow": null, "viscosity": "bad", "diameter": -5.0}"#;
    let settings = serde_json::from_str::<Settings>(text)
        .unwrap_or_default()
        .sanitized();
    assert!(Dial::Spin.get(&settings) <= Dial::Spin.max());
    assert!(Dial::Diameter.get(&settings) >= Dial::Diameter.min());
}

#[test]
fn dials_step_within_their_range_in_a_few_hundred_clicks() {
    let mut settings = Settings::default();
    let mut clicks = 0;
    while settings.spin.0 < Dial::Spin.max() {
        Dial::Spin.adjust(&mut settings, 1);
        clicks += 1;
        assert!(
            clicks < 400,
            "{clicks} clicks and only at {:?}",
            settings.spin
        );
    }
    assert_eq!(settings.spin, RadiansPerSecond(999.0));
    Dial::Spin.adjust(&mut settings, -1);
    assert!(
        settings.spin.0 < 999.0 && settings.spin.0 >= 900.0,
        "one click down from the top: {:?}",
        settings.spin
    );
    let mut settings = Settings::default();
    Dial::Spin.adjust(&mut settings, 1);
    assert!(
        (settings.spin.0 - 1.05).abs() < 1e-6,
        "fine steps near one: {:?}",
        settings.spin
    );
    for _ in 0..100 {
        Dial::Flow.adjust(&mut settings, -1);
    }
    assert_eq!(settings.flow.0, 0.0);
}
