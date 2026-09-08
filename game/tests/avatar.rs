use bevy::prelude::*;
use game::core::avatar::{self, EYE_HEIGHT, JUMP_HEIGHT, WALK_SPEED};
use game::core::math::norm;
use game::core::units::{EARTH_GRAVITY, RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{FLOOR_RADIUS, RADIUS};
use game::systems::player::{PilotAxes, Player};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

fn state(app: &App) -> &Simulation {
    app.world().resource::<Simulation>()
}

fn state_mut(app: &mut App) -> Mut<'_, Simulation> {
    app.world_mut().resource_mut::<Simulation>()
}

fn radius(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[2] * p[2]).sqrt()
}

/// Speed relative to the ground under the avatar's feet.
fn ground_slip(sim: &Simulation) -> f64 {
    let body = sim.avatar();
    let wall = sim.drum.wall_velocity(body.p);
    norm(&[
        body.v[0] - wall[0],
        body.v[1] - wall[1],
        body.v[2] - wall[2],
    ])
}

fn weight_in_g(sim: &Simulation) -> f64 {
    let body = sim.avatar();
    body.ground
        .map(|g| g.support.0 * body.inv_m / EARTH_GRAVITY.0)
        .unwrap_or(0.0)
}

fn hold(app: &mut App, axes: PilotAxes, seconds: f32) {
    let steps = (seconds / 0.1).round() as usize;
    for _ in 0..steps {
        {
            let player = *app.world().resource::<Player>();
            let mut sim = state_mut(app);
            sim.avatar_input = player.input(sim.avatar(), axes);
        }
        testing::run(app, Seconds(0.1));
    }
    state_mut(app).avatar_input = default();
}

fn ghost(app: &mut App) {
    app.world_mut().resource_mut::<Settings>().collisions = false;
    state_mut(app).avatar_mut().solid = false;
}

#[test]
fn standing_on_the_ring_weighs_one_g_at_eye_height() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(3.0));
    let sim = state(&app);
    assert_eq!(sim.drum.spin, standing_spin());
    let g = weight_in_g(sim);
    assert!((g - 1.0).abs() < 0.03, "weight {g} g");
    assert!(ground_slip(sim) < 0.05, "slip {}", ground_slip(sim));
    let eye = avatar::eye(sim.avatar());
    let height = FLOOR_RADIUS as f64 - radius(eye);
    assert!(
        (height - EYE_HEIGHT.0 as f64).abs() < 0.05,
        "eye {height} m above the ground"
    );
    let up = sim.avatar().rotate(&[0.0, 1.0, 0.0]);
    let inward = [-eye[0] / radius(eye), 0.0, -eye[2] / radius(eye)];
    let alignment = up[0] * inward[0] + up[2] * inward[2];
    assert!(alignment > 0.99, "up {up:?} vs inward {inward:?}");
}

#[test]
fn walking_reaches_walking_speed_and_stops_again() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    let forward = PilotAxes {
        motion: Vec3::new(0.0, 0.0, -1.0),
        ..default()
    };
    hold(&mut app, forward, 2.0);
    let slip = ground_slip(state(&app));
    assert!(
        (slip - WALK_SPEED.0 as f64).abs() < 0.15,
        "walking at {slip} m/s"
    );
    let heavier = weight_in_g(state(&app));
    assert!(heavier > 1.2, "weight {heavier} g while walking spinward");
    testing::run(&mut app, Seconds(2.0));
    let slip = ground_slip(state(&app));
    assert!(slip < 0.05, "still moving at {slip} m/s");
}

#[test]
fn a_jump_rises_like_on_earth_and_lands_moving_with_the_ground() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    let start = state(&app).avatar().p;
    let jump = PilotAxes {
        jump: true,
        ..default()
    };
    hold(&mut app, jump, 0.1);
    let mut airborne = 0.0;
    let mut highest = 0.0f64;
    for _ in 0..40 {
        testing::run(&mut app, Seconds(0.05));
        let sim = state(&app);
        if sim.avatar().ground.is_some() {
            break;
        }
        airborne += 0.05;
        highest = highest.max(radius(start) - radius(sim.avatar().p));
    }
    // Straight flight in the inertial frame until the curving floor comes back up: a little
    // shorter than 2v/g because the ring is small enough for the speed of the jump itself to
    // add to the centrifugal gravity.
    let r = radius(start);
    let (v, floor) = (avatar::jump_speed(), (r * standing_spin().0 as f64).powi(2));
    let expected = 2.0 * r * v / (v * v + floor);
    assert!(
        (airborne + 0.1 - expected).abs() < 0.12,
        "airborne {} s, expected {expected}",
        airborne + 0.1
    );
    assert!(
        (highest - JUMP_HEIGHT.0 as f64).abs() < 0.1,
        "rose {highest} m"
    );
    testing::run(&mut app, Seconds(1.0));
    let sim = state(&app);
    assert!(sim.avatar().ground.is_some(), "never landed");
    assert!(ground_slip(sim) < 0.1, "slip {}", ground_slip(sim));
    let travelled = norm(&[
        sim.avatar().p[0] - start[0],
        0.0,
        sim.avatar().p[2] - start[2],
    ]);
    assert!(travelled > 5.0, "the ground carried it only {travelled} m");
}

#[test]
fn flight_assist_brakes_a_ghost_to_a_stop_in_vacuum() {
    let mut app = testing::headless();
    ghost(&mut app);
    {
        let mut player = Player::default();
        let mut sim = state_mut(&mut app);
        player.teleport(sim.avatar_mut(), [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]);
        sim.avatar_input.motion = [1.0, 0.0, 0.0];
    }
    testing::run(&mut app, Seconds(1.0));
    assert!(
        state(&app).avatar().v[0] > 2.0,
        "v {:?}",
        state(&app).avatar().v
    );
    state_mut(&mut app).avatar_input.motion = [0.0; 3];
    testing::run(&mut app, Seconds(3.0));
    assert!(
        norm(&state(&app).avatar().v) < 0.05,
        "v {:?}",
        state(&app).avatar().v
    );
}

#[test]
fn a_ghost_in_the_drum_is_carried_round_and_flung_out() {
    let mut app = testing::headless();
    ghost(&mut app);
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(1.0);
    {
        let mut sim = state_mut(&mut app);
        sim.drum.spin = RadiansPerSecond(1.0);
        sim.avatar_mut()
            .place([6.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    }
    testing::run(&mut app, Seconds(2.0));
    {
        let sim = state(&app);
        let body = sim.avatar();
        let r = radius(body.p);
        assert!(sim.drum.encloses(body.p), "already out at {:?}", body.p);
        let tangential = (body.v[0] * body.p[2] - body.v[2] * body.p[0]) / r;
        assert!(tangential > 0.75 * r, "carried at {tangential} vs {r}");
    }
    testing::run(&mut app, Seconds(20.0));
    let sim = state(&app);
    assert!(
        !sim.drum.encloses(sim.avatar().p),
        "still inside at {:?}",
        sim.avatar().p
    );
}

#[test]
fn a_solid_avatar_outside_stays_outside() {
    let mut app = testing::headless();
    {
        let mut player = Player::default();
        let mut sim = state_mut(&mut app);
        player.teleport(sim.avatar_mut(), [0.0, 12.0, 0.0], [0.0, 0.0, 0.0]);
    }
    testing::run(&mut app, Seconds(3.0));
    let sim = state(&app);
    let p = sim.avatar().p;
    assert!(!sim.drum.encloses(p), "entered the drum at {p:?}");
    assert!(
        radius(p) < RADIUS as f64 + 5.0 && p[1].abs() < 20.0,
        "flew off to {p:?}"
    );
}
