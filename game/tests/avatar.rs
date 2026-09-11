use bevy::prelude::*;
use game::core::audio::{self, Placement, Voice};
use game::core::avatar::{
    self, EYE_HEIGHT, Gyros, SPOOL_TIME, Thruster, WALK_SPEED, equalized_thrust,
};
use game::core::fluid::Fluid;
use game::core::math::{
    Quatd, Vec3d, add_scaled, cross, dot, norm, quat_about_y, quat_conjugate, quat_from_basis,
    quat_mul, quat_rotate,
};
use game::core::units::{EARTH_GRAVITY, Metres, RadiansPerSecond, Seconds};
use game::core::vessel::Vessel;
use game::systems::drum::{DEFAULT_RING, GROUND_DEPTH};
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

/// A point given about the drum's axis: across it, toward the site and spinward, and along it
/// from the middle.
fn about_axis(sim: &Simulation, [x, y, z]: Vec3d) -> Vec3d {
    [x - sim.drum.ring.radius.0 as f64, y - sim.drum.site.y, z]
}

/// How high the avatar's centre is above the glass.
fn altitude(sim: &Simulation) -> f64 {
    sim.drum.height_above_glass(sim.avatar().p)
}

/// Speed over the ground, which stands still in the drum's frame.
fn ground_slip(sim: &Simulation) -> f64 {
    norm(&sim.avatar().v)
}

fn weight_in_g(sim: &Simulation) -> f64 {
    let body = sim.avatar();
    body.ground
        .map(|g| g.support.0 * body.inv_m / EARTH_GRAVITY.0)
        .unwrap_or(0.0)
}

fn hold(app: &mut App, pilot: PilotInput, seconds: f32) {
    hold_watching(app, pilot, seconds);
}

/// Hold the keys down and return the avatar's ground slip every tenth of a second.
fn hold_watching(app: &mut App, pilot: PilotInput, seconds: f32) -> Vec<f64> {
    let steps = (seconds / 0.1).round() as usize;
    let mut slips = Vec::with_capacity(steps);
    for _ in 0..steps {
        {
            let player = *app.world().resource::<Player>();
            let mut sim = state_mut(app);
            sim.avatar_input = player.input(pilot);
        }
        testing::run(app, Seconds(0.1));
        slips.push(ground_slip(state(app)));
    }
    state_mut(app).avatar_input = default();
    slips
}

/// Stand the hull upright on the local vertical where it is, facing the way it faces and
/// keeping its motion, with the gyros holding that: off the ground they hold the attitude the
/// drum's air carried it off with, so a hop or a spell adrift in water moving against the drum
/// leaves it leaning.
fn level(app: &mut App) {
    let mut sim = state_mut(app);
    let body = sim.avatar();
    let (p, v, w) = (body.p, body.v, body.w);
    let up = sim.drum.depth_and_outward(p).1.map(|x| -x);
    let ahead = body.rotate(&[0.0, 0.0, -1.0]);
    let mut forward = ahead;
    add_scaled(&mut forward, &up, -dot(&ahead, &up));
    let len = norm(&forward);
    let forward = forward.map(|c| c / len);
    let right = cross(&forward, &up);
    let back = forward.map(|c| -c);
    sim.avatar_mut()
        .place(p, quat_from_basis(&right, &up, &back));
    sim.avatar_mut().v = v;
    sim.avatar_mut().w = w;
    sim.gyros = Gyros::holding(sim.avatar());
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
    let height = sim.drum.height_above_glass(eye) - GROUND_DEPTH.0 as f64;
    assert!(
        (height - EYE_HEIGHT.0 as f64).abs() < 0.05,
        "eye {height} m above the ground"
    );
    let up = sim.avatar().rotate(&[0.0, 1.0, 0.0]);
    let inward = sim.drum.depth_and_outward(eye).1.map(|x| -x);
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
    let mut forward_only = [0.0; 12];
    forward_only[Thruster::Forward as usize] = 1.0;
    assert_eq!(levels, forward_only);
    testing::run(&mut app, Seconds(spool / 2.0));
    let level = state(&app).thrusters.level(Thruster::Forward);
    assert!((level - 0.5).abs() < 0.05, "half way down: {level}");
    testing::run(&mut app, Seconds(spool));
    assert_eq!(state(&app).thrusters.levels(), [0.0; 12]);
}

#[test]
fn every_thruster_at_once_reads_full_and_cancels_out() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(1.0));
    hold(&mut app, PilotInput::firing(&Thruster::ALL), 2.0);
    let sim = state(&app);
    assert_eq!(sim.thrusters.levels(), [1.0; 12]);
    assert_eq!(sim.thrusters.net(), [0.0; 3]);
    assert_eq!(sim.thrusters.roll(), 0.0);
    assert_eq!(sim.thrusters.pitch(), 0.0);
    assert_eq!(sim.thrusters.yaw(), 0.0);
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
    let floor = altitude(state(&app));
    hold(&mut app, PilotInput::firing(&[Thruster::Up]), 0.8);
    let sim = state(&app);
    assert!(sim.avatar().ground.is_none(), "still on the ground");
    // full thrust beats the standing gravity by the margin it was equalized with, less what
    // spooling up cost
    let rising = altitude(sim) - floor;
    let thrust = equalized_thrust(EARTH_GRAVITY).0;
    let climb = (thrust - EARTH_GRAVITY.0) * (0.8 - SPOOL_TIME.0 as f64 / 2.0);
    assert!(rising > 0.3, "rose only {rising} m");
    let up = sim.drum.depth_and_outward(sim.avatar().p).1.map(|x| -x);
    let inward = dot(&sim.avatar().v, &up);
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
        highest = highest.max(altitude(sim) - floor);
    }
    assert!(highest > 1.0, "rose {highest} m");
    assert!(airborne < 4.0, "never landed");
    testing::run(&mut app, Seconds(1.0));
    let sim = state(&app);
    assert!(sim.avatar().ground.is_some(), "never settled");
    assert!(ground_slip(sim) < 0.15, "slip {}", ground_slip(sim));
    // the ground stands still under a hop; only the Coriolis turn moves the landing a little
    let travelled = norm(&[
        0.0,
        sim.avatar().p[1] - start[1],
        sim.avatar().p[2] - start[2],
    ]);
    assert!(
        travelled < 2.0,
        "landed {travelled} m from where it took off"
    );
}

#[test]
fn flight_assist_brakes_a_ghost_to_a_stop_in_vacuum() {
    let mut app = testing::headless();
    ghost(&mut app);
    {
        let mut player = Player;
        let mut sim = state_mut(&mut app);
        let axis = -(DEFAULT_RING.radius.0 as f64);
        player.teleport(&mut sim, [axis, 20.0, 0.0], [axis, 0.0, 0.0]);
    }
    hold(&mut app, PilotInput::firing(&[Thruster::Right]), 1.0);
    assert!(
        state(&app).avatar().v[0] > 2.0,
        "v {:?}",
        state(&app).avatar().v
    );
    testing::run(&mut app, Seconds(3.0));
    // at rest among the stars, which turn past the drum
    let sim = state(&app);
    let stars = sim.drum.star_velocity(sim.avatar().p);
    let v = sim.avatar().v;
    let adrift = norm(&[v[0] - stars[0], v[1] - stars[1], v[2] - stars[2]]);
    assert!(adrift < 0.05, "v {v:?} against the stars' {stars:?}");
}

#[test]
fn a_ghost_in_the_drum_is_carried_round_and_flung_out() {
    let mut app = testing::headless();
    ghost(&mut app);
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(1.0);
    {
        let mut sim = state_mut(&mut app);
        sim.drum.spin = RadiansPerSecond(1.0);
        let inside = about_axis(&sim, [6.0, 0.0, 0.0]);
        sim.avatar_mut().place(inside, [0.0, 0.0, 0.0, 1.0]);
        sim.avatar_mut().v = sim.drum.star_velocity(inside);
    }
    testing::run(&mut app, Seconds(2.0));
    {
        let sim = state(&app);
        let body = sim.avatar();
        assert!(sim.drum.encloses(body.p), "already out at {:?}", body.p);
        // carried round with the air: within a quarter of the drum's own speed there, and
        // drifting outward
        let round = DEFAULT_RING.radius.0 as f64 + body.p[0];
        assert!(
            body.v[2].abs() < 0.25 * round && body.v[1].abs() < 0.5 && body.v[0] > 0.0,
            "moving through the air at {:?}",
            body.v
        );
    }
    testing::run(&mut app, Seconds(20.0));
    // flung out to the glass, where the air ends and the assist holds it among the stars
    let sim = state(&app);
    assert!(altitude(sim) < 0.1, "still inside at {:?}", sim.avatar().p);
}

/// Where the avatar is and which way it faces among the stars: its place and attitude about
/// the axis turned back by how far the drum has turned.
fn among_the_stars(sim: &Simulation) -> (Vec3d, Quatd, Vec3d) {
    let (p, q) = about_the_axis(sim);
    let back = quat_about_y(sim.drum.angle.0);
    let spin = sim.drum.angular_velocity();
    let body = sim.avatar();
    let w = quat_rotate(
        &back,
        &[
            body.w[0] + spin[0],
            body.w[1] + spin[1],
            body.w[2] + spin[2],
        ],
    );
    (quat_rotate(&back, &p), quat_mul(&back, &q), w)
}

/// A ghost at rest in space outside the ring is untouched by anything the ring does: spun up
/// hard, spun down and resized under it, it neither moves, turns nor starts turning among the
/// stars.
#[test]
fn a_ghost_outside_is_untouched_by_the_ring_however_it_spins() {
    let mut app = testing::headless();
    ghost(&mut app);
    {
        let mut sim = state_mut(&mut app);
        let outside = about_axis(&sim, [40.0, 3.0, 0.0]);
        let axis = about_axis(&sim, [0.0, 3.0, 0.0]);
        Player.teleport(&mut sim, outside, axis);
    }
    testing::run(&mut app, Seconds(6.0));
    let start = among_the_stars(state(&app));
    let check = |app: &mut App, what: &str| {
        let now = among_the_stars(state(app));
        let moved = norm(&[
            now.0[0] - start.0[0],
            now.0[1] - start.0[1],
            now.0[2] - start.0[2],
        ]);
        let turned = angle_between(&start.1, &now.1);
        let turning = norm(&now.2);
        assert!(
            moved < 0.5 && turned < 0.01 && turning < 0.01,
            "{what}: the ghost moved {moved} m, turned {turned} rad and turns at {turning} rad/s among the stars"
        );
    };
    for spin in [2.0, 4.0, 8.0, 12.0, 0.0] {
        app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(spin);
        testing::run(&mut app, Seconds(6.0));
        check(&mut app, &format!("at {spin} rad/s"));
    }
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, 60.0);
        Dial::Width.set(&mut settings, 30.0);
    }
    testing::run(&mut app, Seconds(3.0));
    check(&mut app, "resized");
}

#[test]
fn a_solid_avatar_outside_stays_outside() {
    let mut app = testing::headless();
    {
        let mut player = Player;
        let mut sim = state_mut(&mut app);
        let axis = -(DEFAULT_RING.radius.0 as f64);
        player.teleport(&mut sim, [axis, 12.0, 0.0], [axis, 0.0, 0.0]);
    }
    testing::run(&mut app, Seconds(3.0));
    let sim = state(&app);
    let p = sim.avatar().p;
    assert!(!sim.drum.encloses(p), "entered the drum at {p:?}");
    assert!(
        altitude(sim) > -5.0 && sim.drum.axial(p).abs() < 20.0,
        "flew off to {p:?}"
    );
}

/// A solid avatar standing by a cap and rolling toward it leans its head against the glass
/// and no further: the whole hull, head included, collides, so the view never leaves the ring.
#[test]
fn a_solid_avatar_cannot_lean_its_head_through_the_glass() {
    let mut touched = false;
    for (thruster, side) in [(Thruster::RollLeft, -1.0), (Thruster::RollRight, 1.0)] {
        let mut app = testing::headless();
        let half_width = state(&app).drum.ring.half_width.0 as f64;
        {
            let mut sim = state_mut(&mut app);
            let body = sim.avatar();
            let (mut p, q) = (body.p, body.q);
            p[1] = side * (half_width - avatar::RADIUS - 0.2);
            sim.avatar_mut().place(p, q);
            sim.gyros = Gyros::holding(sim.avatar());
        }
        testing::run(&mut app, Seconds(0.5));
        let mut least = f64::INFINITY;
        for _ in 0..30 {
            hold(&mut app, PilotInput::firing(&[thruster]), 0.1);
            let sim = state(&app);
            let eye = avatar::eye(sim.avatar());
            let clearance = half_width - side * sim.drum.axial(eye);
            assert!(
                sim.drum.encloses(eye) && clearance > 0.2,
                "rolling {thruster:?} put the eye {clearance} m inside the cap at {eye:?}"
            );
            least = least.min(clearance);
        }
        touched |= least < 0.4;
    }
    assert!(touched, "neither roll brought the head to the glass");
}

#[test]
fn a_thruster_sounds_from_a_quarter_to_full_as_its_level_rises() {
    let full = audio::gain(1.0);
    assert_eq!(audio::gain(0.0), 0.0);
    assert!((audio::gain(0.001) / full - 0.25).abs() < 0.01);
    assert!((audio::gain(0.5) / full - 0.625).abs() < 1e-6);
    assert!(full > 0.4 && full <= 0.6, "full loudness {full}");
    let mut fader = audio::Fader::default();
    let mut rising = Vec::new();
    for _ in 0..30 {
        rising.push(fader.follow(full, Seconds(1.0 / 60.0)));
    }
    assert!(rising[0] < full * 0.2, "clicks on: {}", rising[0]);
    assert!(
        rising.windows(2).all(|w| w[1] >= w[0]),
        "eases in: {rising:?}"
    );
    assert!(
        rising[29] > full * 0.9,
        "half a second in, still at {}",
        rising[29]
    );
    let mut falling = Vec::new();
    for _ in 0..120 {
        falling.push(fader.follow(0.0, Seconds(1.0 / 60.0)));
    }
    assert!(falling[0] > full * 0.8, "cuts off: {}", falling[0]);
    assert!(
        falling[29] > full * 0.2,
        "half a second out, already at {}",
        falling[29]
    );
    assert!(fader.silent(), "never settles silent: {}", falling[119]);
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

/// Half fill the drum with water, at rest in it.
fn flood(app: &mut App) {
    for k in 0..30 {
        let a = k as f64 * 0.52;
        app.world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let at = about_axis(
                    &sim,
                    [a.cos() * 7.0, (k % 3) as f64 * 3.0 - 3.0, a.sin() * 7.0],
                );
                sim.inject(&mut fluid, at, 1500)
            });
        testing::run(app, Seconds(0.2));
    }
    testing::run(app, Seconds(6.0));
}

/// An avatar left to itself under still water comes to rest and stays there. Water pushes on a
/// body it covers from every side at once, and a body that answers each push with another is a
/// body that shakes: the eye is at its head, so a shake there is a shake of everything seen.
#[test]
#[ignore = "wants a GPU"]
fn an_avatar_under_still_water_comes_to_rest() {
    let mut app = testing::headless();
    flood(&mut app);
    testing::run(&mut app, Seconds(6.0));
    let mut flips = 0;
    let mut wets = Vec::new();
    let mut before = state(&app).submerged();
    for _ in 0..60 {
        testing::run(&mut app, Seconds(1.0 / 60.0));
        let now = state(&app).submerged();
        wets.push(state(&app).avatar().wet);
        if now != before {
            flips += 1;
        }
        before = now;
    }
    let least = wets.iter().cloned().fold(f64::MAX, f64::min);
    let most = wets.iter().cloned().fold(0.0, f64::max);
    assert!(
        flips < 2,
        "the eye crosses the water {flips} times in a second while the avatar floats: wet runs {least:.3} to {most:.3}"
    );
}

#[test]
#[ignore = "wants a GPU"]
fn the_thrusters_push_through_water_and_out_of_it() {
    let mut app = testing::headless();
    flood(&mut app);
    let sim = state(&app);
    assert!(
        sim.avatar().wet > 0.3,
        "standing dry: wet {}",
        sim.avatar().wet
    );
    level(&mut app);
    let start = altitude(state(&app));
    // a hop is a chord through the ring, so the rise is measured at its highest
    let mut risen: f64 = 0.0;
    for _ in 0..12 {
        hold(&mut app, PilotInput::firing(&[Thruster::Up]), 0.1);
        risen = risen.max(altitude(state(&app)) - start);
    }
    assert!(risen > 1.5, "up thrust through water rose only {risen} m");
    testing::run(&mut app, Seconds(4.0));
    let sim = state(&app);
    assert!(
        sim.avatar().wet > 0.3 && sim.avatar().wet < 0.95,
        "does not float: wet {}",
        sim.avatar().wet
    );
    // the avatar porpoises through the water, so its speed is judged over the last two seconds
    let slips = hold_watching(&mut app, PilotInput::firing(&[Thruster::Forward]), 3.0);
    let settled = &slips[slips.len() - 20..];
    let speed = settled.iter().sum::<f64>() / settled.len() as f64;
    assert!(
        speed > 0.5 && speed < 6.0,
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
    // the frame follows the glass, so the avatar keeps its footing on the new floor
    let height = altitude(sim) - GROUND_DEPTH.0 as f64;
    assert!(
        sim.avatar().ground.is_some() && height > 0.0 && height < 2.0 * avatar::RADIUS,
        "centre {height} m above the new ground"
    );
    let same_power = sim.thrusters.power;
    assert_eq!(same_power, settings.thrust);
    let hop = |app: &mut App| {
        level(app);
        let start = altitude(state(app));
        hold(app, PilotInput::firing(&[Thruster::Up]), 1.0);
        let risen = altitude(state(app)) - start;
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

/// Where the avatar is and which way it faces about the drum's axis, which a ghost outside
/// the ring keeps whatever happens to the ring.
fn about_the_axis(sim: &Simulation) -> (Vec3d, Quatd) {
    let site = sim.drum.site;
    let radius = sim.drum.ring.radius.0 as f64;
    let body = sim.avatar();
    let turned = quat_about_y(-site.phi);
    let p = quat_rotate(&turned, &body.p);
    let (sin, cos) = site.phi.sin_cos();
    (
        [radius * cos + p[0], site.y + p[1], radius * sin + p[2]],
        quat_mul(&turned, &body.q),
    )
}

fn angle_between(a: &Quatd, b: &Quatd) -> f64 {
    let d = quat_mul(&quat_conjugate(a), b);
    2.0 * norm(&[d[0], d[1], d[2]]).atan2(d[3].abs())
}

/// The ring is resized under a ghost outside it, a metre at a time as the dial is held, and
/// the ghost moves and turns in every frame of it as smoothly as in the frame before: the
/// wall moves, not the ghost.
#[test]
fn resizing_the_ring_leaves_a_ghost_outside_where_it_is() {
    let mut app = testing::headless();
    ghost(&mut app);
    {
        let mut sim = state_mut(&mut app);
        let mut player = Player;
        let outside = about_axis(&sim, [30.0, 5.0, 0.0]);
        let axis = about_axis(&sim, [0.0, 5.0, 0.0]);
        player.teleport(&mut sim, outside, axis);
    }
    testing::run(&mut app, Seconds(0.5));
    let frame = Seconds(1.0 / 60.0);
    let drift = |from: &(Vec3d, Quatd), to: &(Vec3d, Quatd)| {
        let d = [
            to.0[0] - from.0[0],
            to.0[1] - from.0[1],
            to.0[2] - from.0[2],
        ];
        (norm(&d), angle_between(&from.1, &to.1))
    };
    let a = about_the_axis(state(&app));
    testing::run(&mut app, frame);
    let mut last = about_the_axis(state(&app));
    let (mut usual, mut turned) = drift(&a, &last);
    for diameter in 22..=60 {
        {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            Dial::Diameter.set(&mut settings, diameter as f32);
        }
        testing::run(&mut app, frame);
        let now = about_the_axis(state(&app));
        let (moved, jumped) = drift(&last, &now);
        assert!(
            (moved - usual).abs() < 0.05 && (jumped - turned).abs() < 0.01,
            "resizing to {diameter} m moved the ghost {moved} m and turned it {jumped} rad in a frame, against {usual} m and {turned} rad the frame before"
        );
        (last, usual, turned) = (now, moved, jumped);
    }
    assert_eq!(state(&app).drum.ring.radius, Metres(30.0));
}
