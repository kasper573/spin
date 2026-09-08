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
        sim.spawn_raft([3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        sim.drum.landscape.sculpt(0.3, 0.1, 0.5, 0.2);
    }
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            world
                .resource::<Simulation>()
                .inject(&mut fluid, [2.0, 0.1, 0.0], 40)
        });
    testing::run(&mut app, Seconds(1.0));
    let particles = testing::particles(&mut app);
    assert_eq!(particles.len(), 40);
    let settings = Settings {
        flow: game::core::units::LitresPerSecond(900.0),
        ..Settings::default()
    };
    let mut player = Player::default();
    let world = app.world_mut();
    let mut sim = world.resource_mut::<Simulation>();
    player.teleport(sim.shuttle_mut(), [1.0, 2.0, 3.0], [0.0; 3]);
    let sim = world.resource::<Simulation>();
    let fluid = world.resource::<Fluid>();

    let json = serde_json::to_string(&snapshot(&settings, sim, fluid, &player)).unwrap();
    let restored: Snapshot = serde_json::from_str(&json).unwrap();
    let (raft, shuttle, angle, heights) = (
        sim.rafts()[0].p,
        sim.shuttle().p,
        sim.drum.angle.0,
        sim.drum.landscape.heights().to_vec(),
    );

    let mut settings2 = Settings::default();
    let mut sim2 = Simulation::default();
    let mut fluid2 = world.resource_mut::<Fluid>();
    let mut player2 = Player::default();
    apply(
        &restored,
        &mut settings2,
        &mut sim2,
        &mut fluid2,
        &mut player2,
    );

    assert_eq!(settings2, settings);
    assert_eq!(fluid2.len(), 40);
    assert_eq!(sim2.rafts().len(), 1);
    let (a, b) = (sim2.rafts()[0].p, raft);
    assert!(
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5,
        "raft {a:?} vs {b:?}"
    );
    assert_eq!(sim2.drum.landscape.heights(), heights);
    assert!((sim2.drum.angle.0 - angle).abs() < 1e-9);
    assert_eq!(sim2.shuttle().p, shuttle);
    assert_eq!(player2, player);
    assert_eq!(restored.fluid.len(), 40 * 7);
    assert_eq!(restored.fluid[0], particles[0].position[0]);
}

#[test]
fn settings_are_sanitized_on_load() {
    let text = r#"{"spin": 99.0, "flow": null, "viscosity": "bad", "brush_size": -5.0}"#;
    let settings = serde_json::from_str::<Settings>(text)
        .unwrap_or_default()
        .sanitized();
    assert!(Dial::Spin.get(&settings) <= Dial::Spin.max());
    assert!(Dial::BrushSize.get(&settings) >= Dial::BrushSize.min());
}

#[test]
fn dials_step_within_their_range() {
    let mut settings = Settings::default();
    for _ in 0..100 {
        Dial::Spin.adjust(&mut settings, 1);
    }
    assert_eq!(settings.spin, RadiansPerSecond(3.0));
    Dial::Spin.adjust(&mut settings, -1);
    assert_eq!(settings.spin, RadiansPerSecond(2.75));
    for _ in 0..100 {
        Dial::Flow.adjust(&mut settings, -1);
    }
    assert_eq!(settings.flow.0, 0.0);
}
