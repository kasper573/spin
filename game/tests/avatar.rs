use bevy::prelude::*;
use game::core::audio::{self, Placement, Voice};
use game::core::avatar::{self, EYE_HEIGHT, SPOOL_TIME, Thruster, WALK_SPEED, equalized_thrust};
use game::core::fluid::Fluid;
use game::core::math::norm;
use game::core::units::{EARTH_GRAVITY, Metres, RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::DEFAULT_RING;
use game::systems::player::{PilotInput, Player};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{Simulation, standing_gravity, standing_spin};
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

fn hold(app: &mut App, pilot: PilotInput, seconds: f32) {
    let steps = (seconds / 0.1).round() as usize;
    for _ in 0..steps {
        {
            let player = *app.world().resource::<Player>();
            let mut sim = state_mut(app);
            sim.avatar_input = player.input(sim.avatar(), pilot);
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
    assert_eq!(sim.drum.spin, standing_spin(DEFAULT_RING));
    let g = weight_in_g(sim);
    assert!((g - 1.0).abs() < 0.03, "weight {g} g");
    assert!(ground_slip(sim) < 0.05, "slip {}", ground_slip(sim));
    let eye = avatar::eye(sim.avatar());
    let height = DEFAULT_RING.floor_radius().0 as f64 - radius(eye);
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
fn forward_thrust_on_the_ground_walks_and_stops_again() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    hold(&mut app, PilotInput::firing(&[Thruster::Forward]), 2.0);
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
fn thrusters_spool_up_and_down() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    let spool = SPOOL_TIME.0;
    hold(
        &mut app,
        PilotInput::firing(&[Thruster::Forward]),
        spool / 3.0,
    );
    let level = state(&app).thrusters.level(Thruster::Forward);
    assert!(
        (level - 1.0 / 3.0).abs() < 0.05,
        "a third of the way up: {level}"
    );
    hold(&mut app, PilotInput::firing(&[Thruster::Forward]), spool);
    let levels = state(&app).thrusters.levels();
    assert_eq!(levels, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    testing::run(&mut app, Seconds(spool / 2.0));
    let level = state(&app).thrusters.level(Thruster::Forward);
    assert!((level - 0.5).abs() < 0.05, "half way down: {level}");
    testing::run(&mut app, Seconds(spool));
    assert_eq!(state(&app).thrusters.levels(), [0.0; 8]);
}

#[test]
fn every_thruster_at_once_reads_full_and_cancels_out() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    hold(&mut app, PilotInput::firing(&Thruster::ALL), 2.0);
    let sim = state(&app);
    assert_eq!(sim.thrusters.levels(), [1.0; 8]);
    assert_eq!(sim.thrusters.net(), [0.0; 3]);
    assert_eq!(sim.thrusters.roll(), 0.0);
    assert!(sim.avatar().ground.is_some(), "left the ground");
    let g = weight_in_g(sim);
    assert!((g - 1.0).abs() < 0.05, "weight {g} g");
    assert!(
        ground_slip(sim) < 0.05,
        "moving at {} m/s",
        ground_slip(sim)
    );
}

#[test]
fn up_thrust_lifts_off_and_the_ground_comes_back_to_meet_it() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    let start = state(&app).avatar().p;
    hold(&mut app, PilotInput::firing(&[Thruster::Up]), 0.8);
    let sim = state(&app);
    assert!(sim.avatar().ground.is_none(), "still on the ground");
    // full thrust beats the standing gravity by the margin it was equalized with, less what
    // spooling up cost
    let rising = radius(start) - radius(sim.avatar().p);
    let thrust = equalized_thrust(EARTH_GRAVITY).0;
    let climb = (thrust - EARTH_GRAVITY.0) * (0.8 - SPOOL_TIME.0 as f64 / 2.0);
    assert!(rising > 0.3, "rose only {rising} m");
    let inward = -(sim.avatar().v[0] * sim.avatar().p[0] + sim.avatar().v[2] * sim.avatar().p[2])
        / radius(sim.avatar().p);
    assert!(
        inward > climb * 0.6 && inward < climb * 1.1,
        "climbing at {inward} m/s, thrust nets {climb} m/s"
    );
    let mut airborne = 0.8;
    let mut highest = rising;
    for _ in 0..80 {
        testing::run(&mut app, Seconds(0.05));
        let sim = state(&app);
        if sim.avatar().ground.is_some() {
            break;
        }
        airborne += 0.05;
        highest = highest.max(radius(start) - radius(sim.avatar().p));
    }
    assert!(highest > 1.0, "rose {highest} m");
    assert!(airborne < 4.0, "never landed");
    testing::run(&mut app, Seconds(1.0));
    let sim = state(&app);
    assert!(sim.avatar().ground.is_some(), "never settled");
    assert!(ground_slip(sim) < 0.15, "slip {}", ground_slip(sim));
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
    }
    hold(&mut app, PilotInput::firing(&[Thruster::Right]), 1.0);
    assert!(
        state(&app).avatar().v[0] > 2.0,
        "v {:?}",
        state(&app).avatar().v
    );
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
        radius(p) < DEFAULT_RING.radius.0 as f64 + 5.0 && p[1].abs() < 20.0,
        "flew off to {p:?}"
    );
}

#[test]
fn a_thruster_sounds_from_a_quarter_to_full_as_its_level_rises() {
    assert_eq!(audio::gain(0.0), 0.0);
    assert!((audio::gain(0.001) - 0.25).abs() < 0.01);
    assert!((audio::gain(0.5) - 0.625).abs() < 1e-6);
    assert_eq!(audio::gain(1.0), 1.0);
    let mut voice = Voice::new(0);
    let samples: Vec<f32> = (0..audio::SAMPLE_RATE.get())
        .map(|_| voice.sample())
        .collect();
    let loudness = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(loudness > 0.05 && loudness < 0.6, "voice at {loudness}");
    assert!(samples.iter().all(|s| s.abs() <= 1.0));
}

#[test]
fn every_thruster_is_heard_from_where_it_sits() {
    let ears = |t: Thruster| Placement::around(t.mount()).ears();
    let [left, right] = ears(Thruster::Left);
    assert!(
        right > left * 1.5,
        "the left thruster sits on the right: {left} {right}"
    );
    let [left, right] = ears(Thruster::Right);
    assert!(
        left > right * 1.5,
        "the right thruster sits on the left: {left} {right}"
    );
    let [left, right] = ears(Thruster::RollLeft);
    assert!(
        right > left * 1.5,
        "the roll left thruster is at the right shoulder"
    );
    for centred in [
        Thruster::Forward,
        Thruster::Back,
        Thruster::Up,
        Thruster::Down,
    ] {
        let [left, right] = ears(centred);
        assert!(
            (left - right).abs() < 1e-6,
            "{centred:?} is off centre: {left} {right}"
        );
    }
    let behind = ears(Thruster::Forward)[0];
    let in_front = ears(Thruster::Back)[0];
    assert!(
        behind < in_front,
        "the forward thruster fires behind the head"
    );
    for thruster in Thruster::ALL {
        let [left, right] = ears(thruster);
        let (near, far) = (left.max(right), left.min(right));
        assert!(
            far > near * 0.3,
            "{thruster:?} is heard in one ear only: {left} {right}"
        );
    }
}

/// Half fill the drum with water, moving with the glass.
fn flood(app: &mut App) {
    for k in 0..30 {
        let a = k as f32 * 0.52;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world.resource::<Simulation>().inject(
                    &mut fluid,
                    [a.cos() * 7.0, (k % 3) as f32 * 3.0 - 3.0, a.sin() * 7.0],
                    1500,
                )
            });
        testing::run(app, Seconds(0.2));
    }
    testing::run(app, Seconds(6.0));
}

#[test]
fn the_thrusters_push_through_water_and_out_of_it() {
    let mut app = testing::headless();
    flood(&mut app);
    let sim = state(&app);
    assert!(
        sim.avatar().wet > 0.5,
        "standing dry: wet {}",
        sim.avatar().wet
    );
    let start = radius(sim.avatar().p);
    // a hop is a chord through the ring, so the rise is measured at its highest
    let mut risen: f64 = 0.0;
    for _ in 0..12 {
        hold(&mut app, PilotInput::firing(&[Thruster::Up]), 0.1);
        risen = risen.max(start - radius(state(&app).avatar().p));
    }
    assert!(risen > 1.5, "up thrust through water rose only {risen} m");
    testing::run(&mut app, Seconds(4.0));
    let sim = state(&app);
    assert!(
        sim.avatar().wet > 0.3 && sim.avatar().wet < 0.95,
        "does not float: wet {}",
        sim.avatar().wet
    );
    hold(&mut app, PilotInput::firing(&[Thruster::Forward]), 3.0);
    let speed = ground_slip(state(&app));
    assert!(
        speed > 2.0 && speed < 6.0,
        "forward thrust through water moves at {speed} m/s"
    );
}

#[test]
fn a_bigger_ring_weighs_more_until_the_thrusters_are_equalized() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, 30.0);
        Dial::Width.set(&mut settings, 16.0);
    }
    testing::run(&mut app, Seconds(6.0));
    let settings = app.world().resource::<Settings>().clone();
    let sim = state(&app);
    assert_eq!(sim.drum.ring.radius, Metres(15.0));
    assert_eq!(sim.drum.ring.half_width, Metres(8.0));
    assert_eq!(sim.drum.spin, standing_spin(DEFAULT_RING));
    let expected = standing_gravity(sim.drum.spin, sim.drum.ring).0 / EARTH_GRAVITY.0;
    let g = weight_in_g(sim);
    assert!(
        (g - expected).abs() < 0.05,
        "weight {g} g, expected {expected}"
    );
    assert!(expected > 1.3, "the bigger ring pulls {expected} g");
    assert!(ground_slip(sim) < 0.1, "slip {}", ground_slip(sim));
    let eye = avatar::eye(sim.avatar());
    let height = sim.drum.ring.floor_radius().0 as f64 - radius(eye);
    assert!(
        (height - EYE_HEIGHT.0 as f64).abs() < 0.05,
        "eye {height} m above the new ground"
    );
    let same_power = sim.thrusters.power;
    assert_eq!(same_power, settings.thrust);
    let hop = |app: &mut App| {
        let start = radius(state(app).avatar().p);
        hold(app, PilotInput::firing(&[Thruster::Up]), 1.0);
        let risen = start - radius(state(app).avatar().p);
        testing::run(app, Seconds(5.0));
        assert!(state(app).avatar().ground.is_some(), "never came down");
        risen
    };
    let heavy = hop(&mut app);
    app.world_mut().resource_mut::<Settings>().equalize_thrust();
    testing::run(&mut app, Seconds(0.1));
    let sim = state(&app);
    let equalized = equalized_thrust(standing_gravity(sim.drum.spin, sim.drum.ring));
    assert_eq!(sim.thrusters.power, equalized);
    assert!(
        equalized.0 > same_power.0 * 1.3,
        "equalized to {} from {}",
        equalized.0,
        same_power.0
    );
    let light = hop(&mut app);
    assert!(
        light > heavy * 2.0 && light > 1.0,
        "a hop rose {heavy} m before equalizing and {light} m after"
    );
}
