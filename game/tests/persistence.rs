use bevy::prelude::*;
use game::core::fluid::Fluid;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::persistence::{Snapshot, apply, snapshot};
use game::systems::player::Player;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;
use game::systems::testing;

#[test]
fn snapshot_round_trips_through_json() {
    let mut app = testing::headless();
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        sim.drum.target_spin = RadiansPerSecond(1.5);
        sim.drum.landscape.sculpt(0.3, 0.1, 1.5, 0.2);
    }
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            world
                .resource::<Simulation>()
                .inject(&mut fluid, [7.0, 0.1, 0.0], 40)
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
    let (shuttle, attitude, angle, heights) = (
        sim.avatar().p,
        sim.avatar().q,
        sim.drum.angle.0,
        sim.drum.landscape.heights().to_vec(),
    );

    let mut settings2 = Settings::default();
    let mut sim2 = Simulation::default();
    let mut fluid2 = world.resource_mut::<Fluid>();
    apply(&restored, &mut settings2, &mut sim2, &mut fluid2);

    assert_eq!(settings2, settings);
    assert_eq!(fluid2.len(), 40);
    assert_eq!(sim2.drum.landscape.heights(), heights);
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
    assert_eq!(restored.fluid[0], particles[0].position[0]);
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
