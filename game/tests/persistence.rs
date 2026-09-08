use bevy::prelude::*;
use game::core::fly_camera::FlyCamera;
use game::core::units::{MetresPerSecond, RadiansPerSecond, Seconds};
use game::systems::persistence::{Snapshot, apply, pose_camera, snapshot};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;

#[test]
fn snapshot_round_trips_through_json() {
    let mut sim = Simulation::default();
    sim.drum.target_spin = RadiansPerSecond(1.5);
    sim.inject([2.0, 0.1, 0.0], 40, true);
    sim.spawn_raft([3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], true);
    sim.drum.landscape.sculpt(0.3, 0.1, 0.5, 0.2);
    sim.advance_exact(Seconds(1.0));
    let settings = Settings {
        flow: game::core::units::LitresPerSecond(900.0),
        ..Settings::default()
    };
    let camera = Transform::from_xyz(1.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
    let fly = FlyCamera {
        speed: MetresPerSecond(7.0),
    };

    let json = serde_json::to_string(&snapshot(&settings, &sim, &camera, &fly)).unwrap();
    let restored: Snapshot = serde_json::from_str(&json).unwrap();

    let mut settings2 = Settings::default();
    let mut sim2 = Simulation::default();
    let pose = apply(&restored, &mut settings2, &mut sim2);
    let mut camera2 = Transform::default();
    let mut fly2 = FlyCamera::default();
    pose_camera(&pose, &mut camera2, &mut fly2);

    assert_eq!(settings2, settings);
    assert_eq!(sim2.fluid.len(), sim.fluid.len());
    assert_eq!(sim2.rafts.len(), 1);
    assert_eq!(sim2.drum.landscape.heights(), sim.drum.landscape.heights());
    assert!((sim2.drum.angle - sim.drum.angle).abs() < 1e-9);
    assert!((camera2.translation - camera.translation).length() < 1e-5);
    assert_eq!(fly2.speed, MetresPerSecond(7.0));
    let a = sim.fluid.particle(7);
    let b = sim2.fluid.particle(7);
    assert_eq!(a.position, b.position);
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
