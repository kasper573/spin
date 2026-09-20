//! What is seen of the water from everywhere a viewer can be: on the ground, under the water,
//! at its surface, up by the axis, outside the glass floor, outside the glass ends and far
//! off, by day and by night. Each sight is drawn twice, with the water and with it hidden,
//! and the water must change the picture wherever it lies in view, with its own colours, and
//! nowhere else. Every frame is written to `target/sights/` to be looked at as well.
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::fluid::{Fluid, Particle};
use game::core::math::{cross, norm, quat_from_basis, quat_rotate};
use game::core::shallows::Shallows;
use game::core::units::{Metres, Radians, Seconds};
use game::systems::drum::{DEFAULT_RING, GROUND_DEPTH, Place, Ring, Round};
use game::systems::scene::SUN_DIRECTION;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing::{self, Headless};
use game::systems::water::WaterMesh;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// The eye adapts to the light of a sight before it is drawn, and light is the same however
/// big the picture is, so it adapts on a small one: drawing is dear where there is no GPU.
const ADAPTING_WIDTH: u32 = WIDTH / 8;
const ADAPTING_HEIGHT: u32 = HEIGHT / 8;
/// A pixel counts as changed when a channel moves by this much, out of 255.
const CHANGED: i32 = 20;
/// By how much a colour channel must differ for the water to have dimmed a pixel: what still
/// water does to a dim ground seen straight down through it, which is lose it the little light
/// the way down and back up takes.
const DIMMED: i32 = 8;
/// How long the eye is given to adapt to the light of a sight before it is drawn, and in how
/// many steps: the eye adapts several stops a second, so a second of it is plenty.
const ADAPTING: f32 = 1.0;
const ADAPTING_STEPS: usize = 15;

/// A place about the ring, in the frame: `arc` metres round the ring from wheel angle zero,
/// `y` along the axis and `height` over the ground there, out through the glass when negative.
fn spot(sim: &Simulation, [arc, y, height]: [f64; 3]) -> [f64; 3] {
    let drum = &sim.drum;
    let at = place(sim, arc, y);
    let ground = drum.landscape.sample(at).0;
    let turn = drum.site.round.arc_to(at.round, drum.landscape.grid()) / drum.ring.radius.0 as f64;
    let on = drum.wall_point(turn, y - drum.site.y);
    let (_, out) = drum.depth_and_outward(on);
    let lift = ground + height;
    [
        on[0] - out[0] * lift,
        on[1] - out[1] * lift,
        on[2] - out[2] * lift,
    ]
}

/// The point of the ground `arc` metres round the ring from wheel angle zero and `y` along the
/// axis from the middle.
fn place(sim: &Simulation, arc: f64, y: f64) -> Place {
    Place {
        round: Round::default().on(arc, sim.drum.landscape.grid()),
        along: y,
    }
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

/// Turn the wheel so the sun stands over the site, or behind the ring from it. The sun keeps
/// its place among the stars and the wheel turns under it, so a sight sets the turn it wants
/// rather than running the ring round to it.
fn face_the_sun(app: &mut App, daylight: bool) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    // the sun stands over the site where its reach along the site's own up is greatest, and
    // the site's up is -x in the drum's frame
    let over = (SUN_DIRECTION.z as f64).atan2(SUN_DIRECTION.x as f64) + std::f64::consts::PI;
    let turn = if daylight {
        over
    } else {
        over + std::f64::consts::PI
    };
    sim.drum.angle = Radians(sim.drum.site.phi - turn);
}

/// Hold the eye at the viewpoint, under the sun it asked for, while the eye adapts to the
/// light there: a ghost let go of falls out through the ring, so it is stood up again at every
/// step of the while, and the wheel is turned back under the sun with it.
fn settle_the_eye(app: &mut App, eye: [f64; 3], at: [f64; 3], daylight: bool) {
    for _ in 0..ADAPTING_STEPS {
        look(app, eye, at);
        face_the_sun(app, daylight);
        testing::watch(app, Seconds(ADAPTING / ADAPTING_STEPS as f32));
    }
    look(app, eye, at);
    face_the_sun(app, daylight);
}

fn inject(app: &mut App, at: [f64; 3], count: u32) {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let centre = spot(&sim, at);
            sim.inject(&mut fluid, centre, count)
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

/// A third of the wheel under water, held there by a ridge across the ring at either end of it,
/// and left to settle.
fn fill_a_third(app: &mut App) {
    ridges(app, [-2.0, 24.0]);
    for k in 0..15 {
        let arc = 4.0 + (k % 5) as f64 * 3.5;
        let y = (k / 5) as f64 * 3.0 - 3.0;
        inject(app, [arc, y, 1.8], 500);
        testing::run(app, Seconds(1.0));
    }
    testing::run(app, Seconds(20.0));
}

/// Ridges raised across the ring from end to end, with their crests this far round it.
fn ridges(app: &mut App, crests: [f64; 2]) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let across = (sim.drum.ring.half_width.0 as f64 / 2.75).floor() as i32;
    for crest in crests {
        for y in (-across..=across).map(|k| k as f64 * 2.75) {
            let at = place(&sim, crest, y);
            for _ in 0..20 {
                sim.drum.landscape.sculpt(at, 8.0, 1.4 / 20.0);
            }
        }
    }
}

/// Two ridges across the ring with a pool laid in the valley between them, and left to settle.
fn pool(app: &mut App) {
    pool_of(app, 500);
}

/// The same between the ends of a ring several times as wide, with as much more water as
/// stands as deep in it once it has run out level from end to end.
fn wide_pool(app: &mut App) {
    pool_of(app, 1800);
}

fn pool_of(app: &mut App, heap: u32) {
    ridges(app, [8.0, -12.0]);
    for k in 0..15 {
        let arc = -1.5 - (k % 3) as f64 * 2.5;
        let y = (k / 3) as f64 * 1.5 - 3.0;
        inject(app, [arc, y, 1.8], heap);
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
    /// The picture the eye adapts on, which is never looked at.
    glimpse: Handle<Image>,
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
        let glimpse = testing::render_to_image(&mut app, ADAPTING_WIDTH, ADAPTING_HEIGHT);
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        build(&mut app);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("sights")
            .join(scene);
        fs::create_dir_all(&dir).expect("create the sights folder");
        Sight {
            app,
            image,
            glimpse,
            dir,
        }
    }

    /// Take all the water out of the ring.
    fn empty(&mut self) {
        self.app.world_mut().resource_mut::<Fluid>().clear();
        testing::run(&mut self.app, Seconds(0.1));
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
        // what is drawn by what an earlier pass of the same frame drew shows only in the frame
        // after the eye is opened again
        testing::frame(&mut self.app, Seconds(0.0));
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
        testing::draw_into(&mut self.app, &self.glimpse);
        settle_the_eye(&mut self.app, eye, at, daylight);
        testing::draw_into(&mut self.app, &self.image);
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
    /// The share of the frame the water at least dimmed.
    dimmed: f64,
    /// The mean colour of the water where it changed the frame, out of 255.
    water: [f64; 3],
    /// The frame itself, with the water in it, and with it hidden.
    pixels: Vec<u8>,
    hidden: Vec<u8>,
}

impl View {
    fn compare(name: &str, tag: &str, with: &[u8], without: &[u8]) -> View {
        let mut changed = 0usize;
        let mut dimmed = 0usize;
        let mut sum = [0.0; 3];
        for (a, b) in with.chunks_exact(4).zip(without.chunks_exact(4)) {
            let differs_by = |by: i32| (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > by);
            dimmed += usize::from(differs_by(DIMMED));
            let moved = differs_by(CHANGED);
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
            dimmed: dimmed as f64 / (WIDTH * HEIGHT) as f64,
            water: sum.map(|s| s / n),
            pixels: with.to_vec(),
            hidden: without.to_vec(),
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

    /// The water must at least dim this share of the frame: all that calm water does to a
    /// ground seen straight down through it by night, however still it lies.
    fn seen_dimly(&self, at_least: f64) -> &View {
        assert!(
            self.dimmed >= at_least,
            "{}: the water dims only {:.2}% of the frame, at least {:.2}% was expected",
            self.name,
            self.dimmed * 100.0,
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

/// How green a pixel is, as green's share of its light, or nothing where it is too dark to say.
fn greenness(pixel: &[u8]) -> Option<f64> {
    let light: f64 = pixel[..3].iter().map(|&c| c as f64).sum();
    (light > 60.0).then(|| pixel[1] as f64 / light)
}

/// The share of the frame where the ground shows as the water's bed with no water drawn over
/// it: where it has lost the green it has in the same sight of the ring dry, though the water
/// changes nothing at all there, however clear it may be. The water is kept a little clear of
/// the glass, which leaves a sliver of wet ground bare along it, so only ground bare for
/// `SLIVER` pixels every way round counts.
fn bare_bed(view: &View, dry: &View) -> f64 {
    const SLIVER: usize = 4;
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let bare: Vec<bool> = view
        .pixels
        .chunks_exact(4)
        .zip(view.hidden.chunks_exact(4))
        .zip(dry.hidden.chunks_exact(4))
        .map(|((with, hidden), dry)| {
            let watered = (0..3).any(|c| (with[c] as i32 - hidden[c] as i32).abs() > 2);
            let bedded = matches!(
                (greenness(hidden), greenness(dry)),
                (Some(wet), Some(dry)) if dry - wet > 0.08
            );
            bedded && !watered
        })
        .collect();
    let wide = (SLIVER..height - SLIVER)
        .flat_map(|y| (SLIVER..width - SLIVER).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            (y - SLIVER..=y + SLIVER)
                .all(|v| (x - SLIVER..=x + SLIVER).all(|u| bare[v * width + u]))
        })
        .count();
    wide as f64 / (width * height) as f64
}

/// The share of the frame that shows the ground's dirt: red over green over blue.
fn dirt(view: &View) -> f64 {
    let dirt = view
        .hidden
        .chunks_exact(4)
        .filter(|p| {
            p[0] > 60 && p[0] as f64 > 1.15 * p[1] as f64 && p[1] as f64 > 1.3 * p[2] as f64
        })
        .count();
    dirt as f64 / (WIDTH * HEIGHT) as f64
}

/// Where a viewer outside the near end of the ring stands, `aside` metres round the ring from
/// where the water was laid, looking in through the glass at the ground on the far side.
fn from_outside(aside: f64) -> ([f64; 3], [f64; 3]) {
    ([aside - 9.0, 17.0, 8.0], [aside + 8.0, -6.0, 2.0])
}

/// The ground must show as the water's bed under the water and nowhere else, all the way round
/// the ring, for a viewer standing still outside it.
#[test]
#[ignore = "wants a GPU"]
fn the_bed_lies_under_the_water_for_a_viewer_standing_outside() {
    let mut sight = Sight::new("bed_still", fill_a_third);
    let (eye, at) = from_outside(0.0);
    let views: Vec<View> = (0..10)
        .map(|second| sight.view(&format!("second_{second}"), eye, at, true))
        .collect();
    sight.empty();
    let dry = sight.view("dry", eye, at, true);
    for view in &views {
        view.seen(0.1);
        let bare = bare_bed(view, &dry);
        assert!(
            bare < 0.002,
            "{}: {:.2}% of the frame is bed with no water over it",
            view.name,
            bare * 100.0
        );
    }
}

/// The same for a viewer moving from side to side, whose site on the ring moves with them.
#[test]
#[ignore = "wants a GPU"]
fn the_bed_lies_under_the_water_for_a_viewer_moving_outside() {
    let mut sight = Sight::new("bed_moving", fill_a_third);
    let asides: Vec<f64> = (0..12).map(|k| (k as f64 * 0.7).sin() * 9.0).collect();
    let views: Vec<View> = asides
        .iter()
        .enumerate()
        .map(|(k, &aside)| {
            let (eye, at) = from_outside(aside);
            sight.view(&format!("aside_{k}"), eye, at, true)
        })
        .collect();
    sight.empty();
    for (k, (view, &aside)) in views.iter().zip(&asides).enumerate() {
        let (eye, at) = from_outside(aside);
        let dry = sight.view(&format!("dry_{k}"), eye, at, true);
        view.seen(0.05);
        let bare = bare_bed(view, &dry);
        assert!(
            bare < 0.002,
            "{}: {:.2}% of the frame is bed with no water over it",
            view.name,
            bare * 100.0
        );
    }
}

/// Flat ground shows its grass right up to the glass at either end, from wherever over it the
/// viewer stands: its dirt is only ever seen from the side or from below.
#[test]
#[ignore = "wants a GPU"]
fn flat_ground_shows_no_dirt_from_over_it() {
    let mut sight = Sight::new("edges", |_| {});
    for (k, aside) in [0.0, 3.0, 9.0, -6.0].into_iter().enumerate() {
        for (end, toward) in [("far", 1.0), ("near", -1.0)] {
            let eye = [aside, -3.0 * toward, 1.7];
            let at = [aside + 6.0, 6.0 * toward, 0.0];
            let view = sight.view(&format!("{end}_{k}"), eye, at, true);
            let dirt = dirt(&view);
            assert!(
                dirt < 0.0005,
                "{}: {:.2}% of the frame is dirt",
                view.name,
                dirt * 100.0
            );
        }
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
#[ignore = "wants a GPU"]
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
    sight.view("axis", axis.0, axis.1, false).seen_dimly(0.4);
    sight
        .view("through_floor", through_floor.0, through_floor.1, false)
        .unseen();
    sight
        .view("from_35m", from_35m.0, from_35m.1, false)
        .seen(0.001);
}

/// Water lying on the ground a metre and a half deep all the way round the ring, as the ground's
/// own water keeps it, seen from under it, from over it and from the axis.
#[test]
#[ignore = "wants a GPU"]
fn water_lying_on_the_ground_is_seen_from_everywhere() {
    let mut sight = Sight::new("lying", |app: &mut App| {
        app.world_mut()
            .resource_mut::<Shallows>()
            .stand(Metres(GROUND_DEPTH.0 + 1.5), [0.0; 2]);
        testing::run(app, Seconds(1.0));
    });
    let under = ([0.0, 0.0, 0.8], [-8.0, 0.0, 0.8]);
    let over = ([0.0, 0.0, 3.0], [-8.0, 0.0, 1.5]);
    let axis = ([0.0, 0.0, 9.0], [-4.0, 1.0, 0.0]);
    sight.view("under", under.0, under.1, true).seen(0.3).blue();
    sight.view("over", over.0, over.1, true).seen(0.15);
    sight.view("axis", axis.0, axis.1, true).seen(0.15);
}

/// A pool between two ridges, seen from its bed, its shore, over it, and from outside.
#[test]
#[ignore = "wants a GPU"]
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

/// Water in flight: a heap splashing into the pool, the water it throws up and the spray
/// that comes off it seen from the shore and from outside the glass end. The sun is brought round before the heap is let go, since the splash
/// is over in a moment.
#[test]
#[ignore = "wants a GPU"]
fn a_splash_is_seen_in_flight() {
    let shore = ([3.5, 2.5, 2.4], [-2.0, 0.0, 2.2]);
    let end = ([-2.0, -10.0, 3.0], [-2.0, 0.0, 2.5]);
    let mut sight = Sight::new("spray", pool);
    for (name, (eye, at), least) in [("splash", shore, 0.05), ("splash_from_the_end", end, 0.02)] {
        testing::draw_into(&mut sight.app, &sight.glimpse.clone());
        settle_the_eye(&mut sight.app, eye, at, true);
        testing::draw_into(&mut sight.app, &sight.image.clone());
        let still = sight.held(&format!("{name}_still"), eye, at, true);
        spray(&mut sight.app);
        let drops = testing::surface_demand(&mut sight.app).droplets;
        assert!(drops > 0, "{name}: the splash threw no droplets at all");
        let motes = testing::spray(&mut sight.app).len();
        assert!(motes > 0, "{name}: the splash threw no spray at all");
        let flying = sight.held(name, eye, at, true);
        flying.seen(least);
        flying.differs_from(&still, 0.01);
    }
}

/// The share of the water in view that flashes: whose brightness jumps by more than a tenth of
/// the way from black to white from one frame to the next, a thirtieth of a second on, and
/// jumps back in the one after. Sunlight sweeping over the water as the ring turns and ripples
/// running across it change a pixel one way and leave it changed, which is no flash; and a
/// star crossing behind the water flashes with no water there too, which is no flash of the
/// water's.
fn flashing(sight: &mut Sight, eye: [f64; 3], at: [f64; 3]) -> f64 {
    testing::draw_into(&mut sight.app, &sight.glimpse.clone());
    settle_the_eye(&mut sight.app, eye, at, true);
    testing::draw_into(&mut sight.app, &sight.image.clone());
    let frames: Vec<[Vec<u8>; 2]> = (0..7)
        .map(|k| {
            testing::frame(&mut sight.app, Seconds(1.0 / 30.0));
            look(&mut sight.app, eye, at);
            let wet = sight.capture(&format!("frame_{k}"));
            sight.shown(false);
            let dry = sight.capture(&format!("dry_{k}"));
            sight.shown(true);
            [wet, dry]
        })
        .collect();
    let light = |frame: &[u8], pixel: usize| {
        frame[4 * pixel..4 * pixel + 3]
            .iter()
            .map(|&c| c as i32)
            .sum::<i32>()
    };
    let flashes = |run: &[[Vec<u8>; 2]], shown: usize, pixel: usize| {
        let there = light(&run[1][shown], pixel) - light(&run[0][shown], pixel);
        let back = light(&run[2][shown], pixel) - light(&run[1][shown], pixel);
        there.abs() > 75 && back.abs() > 75 && there.signum() != back.signum()
    };
    let water: Vec<usize> = (0..(WIDTH * HEIGHT) as usize)
        .filter(|&pixel| (light(&frames[3][0], pixel) - light(&frames[3][1], pixel)).abs() > 24)
        .collect();
    assert!(
        water.len() > (WIDTH * HEIGHT) as usize / 20,
        "hardly any water is in view"
    );
    // the water bends what is seen through it, so a star behind it flashes a little way from
    // where it does with the water gone
    const BLOCK: usize = 16;
    let across = (WIDTH as usize).div_ceil(BLOCK);
    let block_of = |pixel: usize| {
        (
            pixel / WIDTH as usize / BLOCK,
            pixel % WIDTH as usize / BLOCK,
        )
    };
    frames
        .windows(3)
        .map(|run| {
            let starry: HashSet<usize> = (0..(WIDTH * HEIGHT) as usize)
                .filter(|&pixel| flashes(run, 1, pixel))
                .map(|pixel| block_of(pixel).0 * across + block_of(pixel).1)
                .collect();
            let by_a_star = |pixel: usize| {
                let (row, column) = block_of(pixel);
                (row.saturating_sub(1)..=row + 1).any(|r| {
                    (column.saturating_sub(1)..=column + 1)
                        .any(|c| starry.contains(&(r * across + c)))
                })
            };
            let flashed = water
                .iter()
                .filter(|&&pixel| flashes(run, 0, pixel) && !by_a_star(pixel))
                .count();
            flashed as f64 / water.len() as f64
        })
        .fold(0.0, f64::max)
}

/// Expected: a pool left to settle lies still to the eye of someone standing on its shore:
/// the bed shows steadily through it, and nothing on it flashes.
#[test]
#[ignore = "wants a GPU"]
fn a_settled_pool_does_not_flicker() {
    let mut sight = Sight::new("flicker", pool);
    testing::run(&mut sight.app, Seconds(20.0));
    let worst = flashing(&mut sight, [3.0, 2.0, 2.6], [0.0, 0.5, 0.0]);
    println!("at worst {worst:.5} of the water flashed");
    assert!(worst < 0.0001, "{worst:.5} of the water flashed");
}

/// Expected: the water lying round a ring a third full, seen from outside through the glass
/// end, lies as still to the eye as a pool does from its shore.
#[test]
#[ignore = "wants a GPU"]
fn water_seen_from_outside_does_not_flicker() {
    let mut sight = Sight::new("flicker_outside", fill_a_third);
    testing::run(&mut sight.app, Seconds(20.0));
    let (eye, at) = from_outside(0.0);
    let worst = flashing(&mut sight, eye, at);
    println!("at worst {worst:.5} of the water flashed");
    assert!(worst < 0.0001, "{worst:.5} of the water flashed");
}

/// Parcels of water let go on their own in the air, a row of them, from clear at one end to
/// all froth at the other.
fn parcels_in_the_air(app: &mut App) {
    parcels_let_go(app, 2.5);
}

/// A row of parcels of water let go on their own, this high above the ground.
fn parcels_let_go(app: &mut App, height: f64) {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let sim = world.resource::<Simulation>();
            for k in 0..5 {
                let at = spot(sim, [-3.0 + 1.5 * k as f64, 0.0, height]);
                fluid.add(Particle {
                    position: sim.drum.to_water(at),
                    velocity: [0.0; 3],
                    foam: k as f32 / 4.0,
                });
            }
        });
    testing::run(app, Seconds(0.1));
}

/// Expected: a bucketful of water let go on its own in the air is the same water as any
/// other, drawn as the surface would be there had its grid been fine enough: a globe of clear
/// water that shows what is behind it upside down, as a ball of water does, and the whiter the
/// more of it is froth. Every parcel of the row is on its own, and every one is seen.
#[test]
#[ignore = "wants a GPU"]
fn a_parcel_on_its_own_is_a_globe_of_water() {
    let (eye, at) = ([0.0, -5.0, 2.2], [0.0, 0.0, 2.3]);
    let mut sight = Sight::new("parcels", |_| {});
    testing::draw_into(&mut sight.app, &sight.glimpse.clone());
    settle_the_eye(&mut sight.app, eye, at, true);
    testing::draw_into(&mut sight.app, &sight.image.clone());
    parcels_in_the_air(&mut sight.app);
    let drops = testing::surface_demand(&mut sight.app).droplets;
    assert_eq!(drops, 5, "the parcels let go are not all on their own");
    let seen = sight.held("row", eye, at, true);
    seen.seen(0.004);
}

/// Expected: water that has come down on dry ground on its own is no cloud of drops any
/// more, and nothing white: it lies on the ground as a clear puddle, which shows as the ground
/// seen through a little water and brightens next to none of it.
#[test]
#[ignore = "wants a GPU"]
fn water_come_down_on_dry_ground_lies_in_puddles() {
    let (eye, at) = ([0.0, -3.5, 1.6], [0.0, 0.0, 0.0]);
    let mut sight = Sight::new("puddles", |_| {});
    testing::draw_into(&mut sight.app, &sight.glimpse.clone());
    settle_the_eye(&mut sight.app, eye, at, true);
    testing::draw_into(&mut sight.app, &sight.image.clone());
    parcels_let_go(&mut sight.app, 0.3);
    testing::run(&mut sight.app, Seconds(2.0));
    let drops = testing::surface_demand(&mut sight.app).droplets;
    assert_eq!(drops, 5, "the parcels come down are not all on their own");
    let seen = sight.held("row", eye, at, true);
    seen.seen(0.0003);
    let light = |p: &[u8]| p[0] as i32 + p[1] as i32 + p[2] as i32;
    let (changed, whitened) = seen
        .pixels
        .chunks_exact(4)
        .zip(seen.hidden.chunks_exact(4))
        .filter(|(with, without)| (light(with) - light(without)).abs() > 24)
        .fold((0, 0), |(changed, whitened), (with, without)| {
            let whiter = light(with) - light(without) > 150;
            (changed + 1, whitened + whiter as usize)
        });
    let share = whitened as f64 / changed as f64;
    println!("{share:.3} of the puddles is much brighter than the ground they lie on");
    assert!(share < 0.05, "{share:.3} of the water come down is white");
}

/// A pool on a ring two hundred metres across, where the same water covers a much smaller
/// part of the world: it must still be seen from its shore, from under it and from outside.
#[test]
#[ignore = "wants a GPU"]
fn a_pool_on_a_large_ring_is_seen_from_everywhere() {
    let ring = Ring {
        radius: Metres(100.0),
        half_width: Metres(20.0),
    };
    let mut sight = Sight::sized("large", ring, wide_pool);
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
#[ignore = "wants a GPU"]
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
