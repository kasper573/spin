//! What is seen of the water from everywhere a viewer can be: on the ground, under the water,
//! at its surface, up by the axis, outside the glass floor, outside the glass ends and far
//! off, by day and by night. Each sight is drawn twice, with the water and with it hidden,
//! and the water must change the picture wherever it lies in view, with its own colours, and
//! nowhere else. Every frame is written to `target/sights/` to be looked at as well.
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::fluid::Fluid;
use game::core::math::{cross, norm, quat_from_basis, quat_rotate};
use game::core::units::{Metres, Seconds};
use game::systems::drum::{DEFAULT_RING, Ring};
use game::systems::scene::Sky;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing::{self, Headless};
use game::systems::water::WaterMesh;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// A pixel counts as changed when a channel moves by this much, out of 255.
const CHANGED: i32 = 20;

/// A place about the ring, in the frame: `arc` metres round the ring from wheel angle zero,
/// `y` along the axis and `height` over the ground there, out through the glass when negative.
fn spot(sim: &Simulation, [arc, y, height]: [f64; 3]) -> [f64; 3] {
    let drum = &sim.drum;
    let radius = drum.ring.radius.0 as f64;
    let phi = arc / radius;
    let ground = drum.landscape.sample(phi, y).0;
    let on = drum.wall_point(phi - drum.site.phi, y - drum.site.y);
    let (_, out) = drum.depth_and_outward(on);
    let lift = ground + height;
    [
        on[0] - out[0] * lift,
        on[1] - out[1] * lift,
        on[2] - out[2] * lift,
    ]
}

/// Hold the viewer as a ghost with its eye at `eye` looking at `at`, upright on the ring.
fn look(app: &mut App, eye: [f64; 3], at: [f64; 3]) {
    app.world_mut().resource_mut::<Settings>().collisions = false;
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.avatar_mut().solid = false;
    let eye = spot(&sim, eye);
    let target = spot(&sim, at);
    let (_, outward) = sim.drum.depth_and_outward(eye);
    let mut back = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
    let len = norm(&back);
    back = back.map(|c| c / len);
    let mut right = cross(&outward.map(|c| -c), &back);
    if norm(&right) < 1e-6 {
        right = cross(&[0.0, 1.0, 0.0], &back);
    }
    let len = norm(&right);
    right = right.map(|c| c / len);
    let up = cross(&back, &right);
    let q = quat_from_basis(&right, &up, &back);
    let head = quat_rotate(&q, &avatar::eye_offset());
    sim.avatar_mut()
        .place([eye[0] - head[0], eye[1] - head[1], eye[2] - head[2]], q);
    sim.gyros = Gyros::holding(sim.avatar());
}

/// Wait, held at the viewpoint, for the sun to stand over it or to shine from behind the ring.
fn wait_for_sun(app: &mut App, eye: [f64; 3], at: [f64; 3], daylight: bool) {
    let mut waited = 0.0;
    loop {
        look(app, eye, at);
        let sun = app.world().resource::<Sky>().sun.x;
        if (daylight && sun < -0.55) || (!daylight && sun > 0.3) {
            return;
        }
        testing::run(app, Seconds(1.0 / 30.0));
        waited += 1.0 / 30.0;
        assert!(waited < 30.0, "the sun never came round");
    }
}

fn inject(app: &mut App, at: [f64; 3], count: u32) {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let sim = world.resource::<Simulation>();
            sim.inject(&mut fluid, spot(sim, at), count)
        });
}

/// Heaps of water laid all round the ring until it stands half full, and left to settle.
fn fill_half(app: &mut App) {
    for k in 0..44 {
        let arc = k as f64 * 1.5;
        let y = ((k % 4) as f64 - 1.5) * 2.5;
        inject(app, [arc, y, 1.8], 1500);
        testing::run(app, Seconds(0.5));
    }
    testing::run(app, Seconds(40.0));
}

/// Two ridges across the ring with a pool laid in the valley between them, and left to settle.
fn pool(app: &mut App) {
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let radius = sim.drum.ring.radius.0 as f64;
        for crest in [8.0, -12.0] {
            for y in [-5.5, -2.75, 0.0, 2.75, 5.5] {
                for _ in 0..20 {
                    sim.drum
                        .landscape
                        .sculpt(crest / radius, y, 8.0, 1.4 / 20.0);
                }
            }
        }
    }
    for k in 0..15 {
        let arc = -1.5 - (k % 3) as f64 * 2.5;
        let y = (k / 3) as f64 * 1.5 - 3.0;
        inject(app, [arc, y, 1.8], 500);
        testing::run(app, Seconds(1.0));
    }
    testing::run(app, Seconds(20.0));
}

/// The ring spun up by half, so the water is left behind and churns as it catches up.
fn churn(app: &mut App) {
    fill_half(app);
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        let spin = settings.spin.0 * 1.5;
        Dial::Spin.set(&mut settings, spin);
    }
    testing::run(app, Seconds(7.0));
}

/// A heap of water let go high over the pool, caught as it breaks on the surface.
fn spray(app: &mut App) {
    inject(app, [-2.0, 0.0, 4.5], 300);
    testing::run(app, Seconds(1.25));
}

struct Sight {
    app: Headless,
    image: Handle<Image>,
    dir: PathBuf,
}

impl Sight {
    fn new(scene: &str, build: fn(&mut App)) -> Sight {
        Sight::sized(scene, DEFAULT_RING, build)
    }

    /// A sight of a ring of a given size, with the scene built in it.
    fn sized(scene: &str, ring: Ring, build: fn(&mut App)) -> Sight {
        let mut app = testing::headless();
        if ring.radius != DEFAULT_RING.radius || ring.half_width != DEFAULT_RING.half_width {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            settings.diameter = Metres(ring.radius.0 * 2.0);
            settings.width = Metres(ring.half_width.0 * 2.0);
            settings.spin = standing_spin(ring);
            settings.equalize_thrust();
            app.insert_resource(Simulation::new(ring));
        }
        testing::run(&mut app, Seconds(0.5));
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        build(&mut app);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("sights")
            .join(scene);
        fs::create_dir_all(&dir).expect("create the sights folder");
        Sight { app, image, dir }
    }

    fn shown(&mut self, shown: bool) {
        let mut query = self
            .app
            .world_mut()
            .query_filtered::<&mut Visibility, With<WaterMesh>>();
        for mut visibility in query.iter_mut(self.app.world_mut()) {
            *visibility = if shown {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
    }

    fn capture(&mut self, name: &str) -> Vec<u8> {
        testing::frame(&mut self.app, Seconds(0.0));
        let pixels = testing::capture(&mut self.app, &self.image);
        image::save_buffer(
            self.dir.join(format!("{name}.png")),
            &pixels,
            WIDTH,
            HEIGHT,
            image::ColorType::Rgba8,
        )
        .expect("write frame");
        pixels
    }

    /// The view from `eye` toward `at`, with the water and without, once the sun stands as
    /// asked; the frame without the water is drawn from the very same moment.
    fn view(&mut self, name: &str, eye: [f64; 3], at: [f64; 3], daylight: bool) -> View {
        wait_for_sun(&mut self.app, eye, at, daylight);
        testing::run(&mut self.app, Seconds(1.0 / 30.0));
        self.held(name, eye, at, daylight)
    }

    /// The view from `eye` toward `at` as the water stands now, with it and without.
    fn held(&mut self, name: &str, eye: [f64; 3], at: [f64; 3], daylight: bool) -> View {
        look(&mut self.app, eye, at);
        let tag = if daylight { "day" } else { "night" };
        let with = self.capture(&format!("{name}_{tag}"));
        self.shown(false);
        let without = self.capture(&format!("{name}_{tag}_hidden"));
        self.shown(true);
        View::compare(name, tag, &with, &without)
    }
}

/// How a sight differs with and without the water.
#[derive(Debug)]
struct View {
    name: String,
    /// The share of the frame the water changed.
    changed: f64,
    /// The mean colour of the water where it changed the frame, out of 255.
    water: [f64; 3],
    /// The frame itself, with the water in it.
    pixels: Vec<u8>,
}

impl View {
    fn compare(name: &str, tag: &str, with: &[u8], without: &[u8]) -> View {
        let mut changed = 0usize;
        let mut sum = [0.0; 3];
        for (a, b) in with.chunks_exact(4).zip(without.chunks_exact(4)) {
            let moved = (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > CHANGED);
            if moved {
                changed += 1;
                for c in 0..3 {
                    sum[c] += a[c] as f64;
                }
            }
        }
        let n = changed.max(1) as f64;
        View {
            name: format!("{name} by {tag}"),
            changed: changed as f64 / (WIDTH * HEIGHT) as f64,
            water: sum.map(|s| s / n),
            pixels: with.to_vec(),
        }
    }

    fn luminance(&self) -> f64 {
        0.2126 * self.water[0] + 0.7152 * self.water[1] + 0.0722 * self.water[2]
    }

    /// The water must fill at least this share of the frame.
    fn seen(&self, at_least: f64) -> &View {
        assert!(
            self.changed >= at_least,
            "{}: the water changes only {:.2}% of the frame, at least {:.2}% was expected",
            self.name,
            self.changed * 100.0,
            at_least * 100.0
        );
        self
    }

    /// The water must stand differently here than in another sight of the same place.
    fn differs_from(&self, other: &View, at_least: f64) -> &View {
        let mut moved = 0usize;
        for (a, b) in self
            .pixels
            .chunks_exact(4)
            .zip(other.pixels.chunks_exact(4))
        {
            if (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > CHANGED) {
                moved += 1;
            }
        }
        let share = moved as f64 / (WIDTH * HEIGHT) as f64;
        assert!(
            share >= at_least,
            "{}: only {:.2}% of the frame stands apart from {}, at least {:.2}% was expected",
            self.name,
            share * 100.0,
            other.name,
            at_least * 100.0
        );
        self
    }

    /// Nothing of the water may show.
    fn unseen(&self) -> &View {
        assert!(
            self.changed < 0.005,
            "{}: the water changes {:.2}% of the frame, where none of it should show",
            self.name,
            self.changed * 100.0
        );
        self
    }

    /// What the water adds must be water-coloured: bluer than it is red.
    fn blue(&self) -> &View {
        assert!(
            self.water[2] > self.water[0] * 1.3,
            "{}: the water shows as {:?}, not blue over red",
            self.name,
            self.water
        );
        self
    }
}

/// By night the water shows only the light bounced round the ring, green, never the sun's
/// white glints and foam, and the eye's adaptation to the dark cannot make it brighter than
/// by day.
fn unlit_by_night(day: &View, night: &View) {
    assert!(
        night.luminance() < day.luminance() && night.water[0] < 0.6 * night.water[1],
        "{}: the water shows the sun by night ({:?}) against by day ({:?})",
        night.name,
        night.water,
        day.water
    );
}

/// The ring half full of water, seen from inside and out.
#[test]
fn half_a_ring_of_water_is_seen_from_everywhere() {
    let mut sight = Sight::new("half", fill_half);
    // the water stands three metres over the ground all the way round
    let under = ([0.0, 0.0, 0.8], [-8.0, 0.0, 0.8]);
    let waterline = ([0.0, 0.0, 2.9], [-8.0, 0.0, 2.6]);
    let over = ([0.0, 0.0, 4.5], [-8.0, 0.0, 3.0]);
    let axis = ([0.0, 0.0, 9.0], [-4.0, 1.0, 0.0]);
    let out_the_end = ([0.0, 0.0, 4.0], [0.0, -20.0, 4.0]);
    let through_floor = ([0.0, 0.0, -3.5], [0.0, 0.0, 1.5]);
    // under the water, facing the glass disc that closes the ring's end
    let at_the_end = ([0.0, -5.0, 0.8], [0.0, -20.0, 0.8]);
    let end_on_axis = ([0.0, -10.0, 2.0], [0.0, 0.0, 2.0]);
    let end_off_axis = ([0.0, -10.0, 9.0], [0.0, 0.0, 5.0]);
    let from_35m = ([0.0, 25.0, -25.0], [0.0, 0.0, 0.0]);
    let from_300m = ([0.0, 200.0, -200.0], [0.0, 0.0, 0.0]);

    sight.view("under", under.0, under.1, true).seen(0.3).blue();
    sight
        .view("waterline", waterline.0, waterline.1, true)
        .seen(0.2);
    let over_day = sight.view("over", over.0, over.1, true);
    over_day.seen(0.15);
    sight.view("axis", axis.0, axis.1, true).seen(0.15);
    sight
        .view("out_the_end", out_the_end.0, out_the_end.1, true)
        .seen(0.05);
    sight
        .view("at_the_end", at_the_end.0, at_the_end.1, true)
        .seen(0.3)
        .blue();
    sight
        .view("through_floor", through_floor.0, through_floor.1, true)
        .unseen();
    let end_day = sight.view("end_on_axis", end_on_axis.0, end_on_axis.1, true);
    end_day.seen(0.1).blue();
    sight
        .view("end_off_axis", end_off_axis.0, end_off_axis.1, true)
        .seen(0.05);
    sight
        .view("from_35m", from_35m.0, from_35m.1, true)
        .seen(0.003);
    sight
        .view("from_300m", from_300m.0, from_300m.1, true)
        .seen(0.00005);

    let over_night = sight.view("over", over.0, over.1, false);
    unlit_by_night(&over_day, &over_night);
    let end_night = sight.view("end_on_axis", end_on_axis.0, end_on_axis.1, false);
    unlit_by_night(&end_day, &end_night);
    sight.view("under", under.0, under.1, false).seen(0.3);
    sight.view("axis", axis.0, axis.1, false).seen(0.1);
    sight
        .view("through_floor", through_floor.0, through_floor.1, false)
        .unseen();
    sight
        .view("from_35m", from_35m.0, from_35m.1, false)
        .seen(0.001);
}

/// A pool between two ridges, seen from its bed, its shore, over it, and from outside.
#[test]
fn a_pool_is_seen_from_everywhere() {
    let mut sight = Sight::new("pool", pool);
    let under = ([-2.0, 0.0, 0.6], [-8.0, 0.0, 0.6]);
    let shore = ([3.0, 0.0, 1.7], [-6.0, 0.0, 1.0]);
    let ridge = ([8.0, 0.0, 1.7], [-6.0, 0.0, 1.0]);
    let axis = ([-2.0, 0.0, 8.5], [-6.0, 1.0, 0.0]);
    let end_through = ([-2.0, -10.0, 1.0], [-2.0, 0.0, 1.0]);
    let through_floor = ([-2.0, 0.0, -3.5], [-2.0, 0.0, 1.5]);
    let from_35m = ([-2.0, 25.0, -25.0], [-2.0, 0.0, 0.0]);

    sight.view("under", under.0, under.1, true).seen(0.3);
    let shore_day = sight.view("shore", shore.0, shore.1, true);
    shore_day.seen(0.1);
    sight.view("ridge", ridge.0, ridge.1, true).seen(0.05);
    sight.view("axis", axis.0, axis.1, true).seen(0.05);
    sight
        .view("end_through", end_through.0, end_through.1, true)
        .seen(0.02);
    sight
        .view("through_floor", through_floor.0, through_floor.1, true)
        .unseen();
    sight
        .view("from_35m", from_35m.0, from_35m.1, true)
        .seen(0.001);

    let shore_night = sight.view("shore", shore.0, shore.1, false);
    unlit_by_night(&shore_day, &shore_night);
    sight.view("under", under.0, under.1, false).seen(0.3);
    sight
        .view("end_through", end_through.0, end_through.1, false)
        .seen(0.01);
}

/// Water in flight: a heap splashing into the pool, its droplets seen from the shore and from
/// outside the glass end. The sun is brought round before the heap is let go, since the splash
/// is over in a moment.
#[test]
fn a_splash_is_seen_in_flight() {
    let shore = ([3.5, 2.5, 2.4], [-2.0, 0.0, 2.2]);
    let end = ([-2.0, -10.0, 3.0], [-2.0, 0.0, 2.5]);
    let mut sight = Sight::new("spray", pool);
    for (name, (eye, at), least) in [("splash", shore, 0.05), ("splash_from_the_end", end, 0.02)] {
        wait_for_sun(&mut sight.app, eye, at, true);
        let still = sight.held(&format!("{name}_still"), eye, at, true);
        spray(&mut sight.app);
        let drops = testing::surface_demand(&mut sight.app).droplets;
        assert!(drops > 0, "{name}: the splash threw no droplets at all");
        let flying = sight.held(name, eye, at, true);
        flying.seen(least);
        flying.differs_from(&still, 0.01);
    }
}

/// A pool on a ring two hundred metres across, where the same water covers a much smaller
/// part of the world: it must still be seen from its shore, from under it and from outside.
#[test]
fn a_pool_on_a_large_ring_is_seen_from_everywhere() {
    let ring = Ring {
        radius: Metres(100.0),
        half_width: Metres(20.0),
    };
    let mut sight = Sight::sized("large", ring, pool);
    let under = ([-2.0, 0.0, 0.6], [-8.0, 0.0, 0.6]);
    let shore = ([3.0, 0.0, 1.7], [-6.0, 0.0, 1.0]);
    let over = ([-2.0, 0.0, 6.0], [-6.0, 0.0, 0.0]);
    let end_through = ([-2.0, -30.0, 1.5], [-2.0, 0.0, 1.0]);
    let through_floor = ([-2.0, 0.0, -3.5], [-2.0, 0.0, 1.5]);

    sight.view("under", under.0, under.1, true).seen(0.3);
    sight.view("shore", shore.0, shore.1, true).seen(0.05);
    sight.view("over", over.0, over.1, true).seen(0.02);
    sight
        .view("end_through", end_through.0, end_through.1, true)
        .seen(0.002);
    sight
        .view("through_floor", through_floor.0, through_floor.1, true)
        .unseen();
}

/// Churning water, thrown about by the ring spinning up: it must still read as water from
/// just over it and from under it, not as a pattern painted on the ground.
#[test]
fn churning_water_is_seen_from_everywhere() {
    let mut sight = Sight::new("churn", churn);
    let over = ([0.0, 0.0, 3.4], [-14.0, 0.0, 2.9]);
    let under = ([0.0, 0.0, 0.8], [-8.0, 0.0, 0.8]);
    let end_on_axis = ([0.0, -10.0, 2.0], [0.0, 0.0, 2.0]);
    // deep under the water, looking down at the bed through all of it
    let down = ([0.0, 0.0, 2.4], [-1.5, 0.0, 0.0]);

    sight.view("over", over.0, over.1, true).seen(0.3);
    sight.view("down", down.0, down.1, true).seen(0.3);
    sight.view("under", under.0, under.1, true).seen(0.3);
    sight
        .view("end_on_axis", end_on_axis.0, end_on_axis.1, true)
        .seen(0.1);
}
