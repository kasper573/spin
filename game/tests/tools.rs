//! The tools: one is out at a time or none, its number key brings it out and puts it away,
//! only the tool that is out answers to the mouse, and it is a thing in the world: seen
//! before the eye, lighting what is near when it fires, casting a shadow with the body behind
//! it, and mirrored with that body in the water and the glass. The sights are written to
//! `target/tmp/tools/` to be looked at as well.
use std::fs;
use std::path::PathBuf;

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use game::core::avatar::Gyros;
use game::core::fluid::Fluid;
use game::core::math::quat_mul;
use game::core::units::{Radians, Seconds};
use game::systems::aim::Aim;
use game::systems::figure::Mirrored;
use game::systems::scene::SUN_DIRECTION;
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::testing::{self, Headless};
use game::systems::tools::Toolbelt;
use game::systems::tools::land_tool::LandTool;
use game::systems::tools::water_tool::WaterTool;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAME: Seconds = Seconds(1.0 / 60.0);
/// A pixel counts as changed when a channel moves by this much, out of 255; and as touched
/// when it moves at all, between two frames of the very same moment.
const CHANGED: i32 = 24;
const TOUCHED: i32 = 1;

fn wielded(app: &App) -> Option<usize> {
    app.world().resource::<Toolbelt>().wielded()
}

fn hold(app: &mut App, button: MouseButton, seconds: f32) {
    testing::button(app, button, true);
    testing::run(app, Seconds(seconds));
    testing::button(app, button, false);
    testing::run(app, FRAME);
}

/// Tip the avatar's view down by this much, and hold it there.
fn look_down(app: &mut App, angle: f64) {
    turn(app, [(-angle / 2.0).sin(), 0.0, 0.0, (-angle / 2.0).cos()]);
}

fn turn(app: &mut App, by: [f64; 4]) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let (p, q) = (sim.avatar().p, quat_mul(&sim.avatar().q, &by));
    sim.avatar_mut().place(p, q);
    sim.gyros = Gyros::holding(sim.avatar());
}

/// Turn the wheel so the sun stands over the site, or behind the ring from it.
fn face_the_sun(app: &mut App, daylight: bool) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let over = (SUN_DIRECTION.z as f64).atan2(SUN_DIRECTION.x as f64) + std::f64::consts::PI;
    let turn = if daylight {
        over
    } else {
        over + std::f64::consts::PI
    };
    sim.drum.angle = Radians(sim.drum.site.phi - turn);
}

/// A ring that stands still under the sun, so that a sight keeps the light it was given.
fn still(app: &mut App, daylight: bool) {
    face_the_sun(app, daylight);
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.drum.spin.0 = 0.0;
    sim.drum.target_spin.0 = 0.0;
    sim.avatar_mut().solid = false;
    app.world_mut().resource_mut::<Settings>().spin.0 = 0.0;
    app.world_mut().resource_mut::<Settings>().collisions = false;
}

struct Sights {
    app: Headless,
    image: Handle<Image>,
    dir: PathBuf,
}

impl Sights {
    fn new(daylight: bool) -> Sights {
        let mut app = testing::headless();
        testing::run(&mut app, Seconds(0.5));
        still(&mut app, daylight);
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tools");
        fs::create_dir_all(&dir).expect("create the tools folder");
        Sights { app, image, dir }
    }

    /// A ring left spinning under the sun, with the water tool out: water lies down in it, and
    /// the wheel has to be turned back under the sun for as long as the eye adapts.
    fn spinning() -> Sights {
        let mut app = testing::headless();
        testing::run(&mut app, Seconds(0.5));
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tools");
        fs::create_dir_all(&dir).expect("create the tools folder");
        Sights { app, image, dir }
    }

    /// Let the eye adapt and whatever is moving settle, then draw the view.
    fn draw(&mut self, name: &str) -> Vec<u8> {
        self.hold_still(1.0);
        self.glance(name)
    }

    /// Draw the view as it stands this very moment.
    fn glance(&mut self, name: &str) -> Vec<u8> {
        testing::frame(&mut self.app, Seconds(0.0));
        let pixels = testing::capture(&mut self.app, &self.image);
        image::save_buffer(
            self.dir.join(format!("{name}.png")),
            &pixels,
            WIDTH,
            HEIGHT,
            image::ColorType::Rgba8,
        )
        .expect("write the sight");
        pixels
    }

    /// A ghost at rest drifts, so it is put back where it was at every frame.
    fn hold_still(&mut self, seconds: f32) {
        let (p, q) = {
            let sim = self.app.world().resource::<Simulation>();
            (sim.avatar().p, sim.avatar().q)
        };
        for _ in 0..(seconds * 30.0) as usize {
            let mut sim = self.app.world_mut().resource_mut::<Simulation>();
            sim.avatar_mut().place(p, q);
            sim.avatar_mut().v = [0.0; 3];
            sim.avatar_mut().w = [0.0; 3];
            sim.gyros = Gyros::holding(sim.avatar());
            testing::watch(&mut self.app, Seconds(1.0 / 30.0));
        }
    }
}

/// How much land lies about a point of the ground, in cubic metres: the ground's height
/// summed over a square of it wide enough to hold whatever a tool does there in a second or
/// two. The ground is measured round the ring as the ring lays it out, along the glass.
fn land_about(app: &App, point: DVec3) -> f64 {
    let sim = app.world().resource::<Simulation>();
    // the frame lies on the glass under the site, the ring's axis a radius over it along -x
    let radius = sim.drum.ring.radius.0 as f64;
    let axis = DVec3::new(-radius, point.y, 0.0);
    let (reach, step) = (5.0, 0.125);
    let across = (reach / step) as i32;
    let mut land = 0.0;
    for i in -across..=across {
        let round = DQuat::from_rotation_y(i as f64 * step / radius);
        for j in -across..=across {
            let at = axis + round * (point - axis) + DVec3::Y * (j as f64 * step);
            land += sim.drum.ground(at.to_array()) * step * step;
        }
    }
    land
}

/// How many pixels of a part of the picture differ between two frames, the part given by
/// its edges as shares of the picture.
fn changed(a: &[u8], b: &[u8], part: [f32; 4]) -> usize {
    moved(a, b, part, CHANGED)
}

fn moved(a: &[u8], b: &[u8], [left, top, right, bottom]: [f32; 4], by: i32) -> usize {
    let mut count = 0;
    for y in (top * HEIGHT as f32) as u32..(bottom * HEIGHT as f32) as u32 {
        for x in (left * WIDTH as f32) as u32..(right * WIDTH as f32) as u32 {
            let i = ((y * WIDTH + x) * 4) as usize;
            let differs = (0..3).any(|c| (a[i + c] as i32 - b[i + c] as i32).abs() >= by);
            count += usize::from(differs);
        }
    }
    count
}

/// How much more of one colour channel than of the other two a part of a frame has gained
/// over another frame, summed over its pixels.
fn gained(
    before: &[u8],
    after: &[u8],
    channel: usize,
    [left, top, right, bottom]: [f32; 4],
) -> i64 {
    let mut sum = 0;
    for y in (top * HEIGHT as f32) as u32..(bottom * HEIGHT as f32) as u32 {
        for x in (left * WIDTH as f32) as u32..(right * WIDTH as f32) as u32 {
            let i = ((y * WIDTH + x) * 4) as usize;
            let gain = |c: usize| after[i + c] as i64 - before[i + c] as i64;
            let others = (0..3).filter(|c| *c != channel).map(gain).sum::<i64>() / 2;
            sum += gain(channel) - others;
        }
    }
    sum
}

const WHOLE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const LOWER_RIGHT: [f32; 4] = [0.5, 0.5, 1.0, 1.0];
const UPPER_LEFT: [f32; 4] = [0.0, 0.0, 0.5, 0.45];

/// A tool's number key brings it out, putting away whichever was out, and the key of the
/// tool that is out puts it away, leaving none.
#[test]
fn a_tools_key_brings_it_out_and_puts_it_away() {
    let mut app = testing::headless();
    testing::run(&mut app, FRAME);
    assert_eq!(
        wielded(&app),
        None,
        "the harness starts with every tool away"
    );
    for (key, out) in [
        (KeyCode::Digit1, Some(0)),
        (KeyCode::Digit2, Some(1)),
        (KeyCode::Digit1, Some(0)),
        (KeyCode::Digit1, None),
        (KeyCode::Digit2, Some(1)),
        (KeyCode::Digit2, None),
    ] {
        testing::tap(&mut app, key);
        testing::run(&mut app, FRAME);
        assert_eq!(wielded(&app), out, "after {key:?}");
    }
}

/// Only the tool that is out answers to the mouse, and each to its own buttons: the left
/// button pours with the water tool and raises the ground with the land tool, the right one
/// lowers the ground with the land tool and does nothing with the water tool, and with no
/// tool out, or the viewer away from the controls, neither does anything.
#[test]
#[ignore = "wants a GPU"]
fn only_the_tool_that_is_out_answers_to_the_mouse() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    look_down(&mut app, 0.3);
    testing::run(&mut app, FRAME);
    let water = |app: &App| app.world().resource::<Fluid>().len();
    let high = |app: &App| {
        app.world()
            .resource::<Simulation>()
            .drum
            .landscape
            .max_height()
    };
    let flat = high(&app);

    for button in [MouseButton::Left, MouseButton::Right] {
        hold(&mut app, button, 0.5);
    }
    assert_eq!(water(&app), 0, "water poured with no tool out");
    assert_eq!(high(&app), flat, "the ground was worked with no tool out");

    testing::tap(&mut app, KeyCode::Digit1);
    hold(&mut app, MouseButton::Right, 0.5);
    assert_eq!(water(&app), 0, "the water tool poured on the right button");
    assert_eq!(high(&app), flat, "the water tool worked the ground");
    app.world_mut().resource_mut::<Aim>().engaged = false;
    hold(&mut app, MouseButton::Left, 0.5);
    assert_eq!(
        water(&app),
        0,
        "water poured with the viewer away from the controls"
    );
    app.world_mut().resource_mut::<Aim>().engaged = true;
    testing::button(&mut app, MouseButton::Left, true);
    testing::run(&mut app, Seconds(0.25));
    assert!(
        app.world().resource::<WaterTool>().pouring,
        "the water tool does not say it is pouring"
    );
    testing::run(&mut app, Seconds(0.25));
    testing::button(&mut app, MouseButton::Left, false);
    testing::run(&mut app, FRAME);
    let poured = water(&app);
    let flow = app.world().resource::<Settings>().flow.0;
    let asked = flow * 0.5
        / app
            .world()
            .resource::<Fluid>()
            .resolution()
            .litres_per_particle()
            .0;
    assert!(
        (poured as f32 - asked).abs() < 0.1 * asked,
        "half a second of the left button poured {poured} particles, not the {asked} the flow asks for"
    );
    assert_eq!(high(&app), flat, "the water tool worked the ground");
    assert!(
        !app.world().resource::<WaterTool>().pouring,
        "the water tool still says it is pouring"
    );

    testing::tap(&mut app, KeyCode::Digit2);
    let level = {
        let under = app.world().resource::<Aim>().target.expect("aimed").point;
        land_about(&app, under)
    };
    testing::button(&mut app, MouseButton::Left, true);
    testing::run(&mut app, Seconds(0.5));
    let tool = app.world().resource::<LandTool>();
    assert!(
        tool.raising && !tool.lowering,
        "the left barrel is not the one firing"
    );
    testing::run(&mut app, Seconds(0.5));
    testing::button(&mut app, MouseButton::Left, false);
    testing::run(&mut app, FRAME);
    assert_eq!(water(&app), poured, "the land tool poured water");
    // what the dial asks for is land by volume, wherever under the crosshair it is put
    let asked = app.world().resource::<Settings>().build.0 as f64 / 1000.0;
    let under = app.world().resource::<Aim>().target.expect("aimed").point;
    let raised = land_about(&app, under) - level;
    assert!(
        (raised - asked).abs() < 0.1 * asked,
        "a second of the left button put down {raised} m3 of land, with {asked} m3 a second on its dial"
    );
    // the ground stops giving way at the glass, so the right button may take up less
    let before = land_about(&app, under);
    hold(&mut app, MouseButton::Right, 1.0);
    let lowered = before - land_about(&app, under);
    assert!(
        lowered > 0.4 * asked && lowered < 1.1 * asked,
        "a second of the right button took up {lowered} m3 of land, with {asked} m3 a second on its dial"
    );
    let stands = |app: &App| {
        let sim = app.world().resource::<Simulation>();
        sim.drum.ground(under.to_array())
    };

    testing::tap(&mut app, KeyCode::Digit2);
    let before = stands(&app);
    hold(&mut app, MouseButton::Left, 0.5);
    assert_eq!(
        stands(&app),
        before,
        "the land tool worked the ground after it was put away"
    );
    let tool = app.world().resource::<LandTool>();
    assert!(
        !tool.raising && !tool.lowering,
        "a barrel fires with the tool away"
    );
}

/// The wheel turns the dial of the tool that is out, a step a click, and no other: the water
/// tool's flow, the land tool's, which widens its brush with it, and with no tool out nothing.
#[test]
fn the_wheel_turns_the_dial_of_the_tool_that_is_out() {
    let mut app = testing::headless();
    testing::run(&mut app, FRAME);
    let dials = |app: &App| {
        let settings = app.world().resource::<Settings>();
        (settings.flow.0, settings.build.0)
    };
    let (flow, build) = dials(&app);
    testing::wheel(&mut app, 1);
    testing::run(&mut app, FRAME);
    assert_eq!(
        dials(&app),
        (flow, build),
        "the wheel turned a dial with no tool out"
    );

    testing::tap(&mut app, KeyCode::Digit1);
    testing::run(&mut app, FRAME);
    testing::wheel(&mut app, 1);
    testing::run(&mut app, FRAME);
    let (more, same) = dials(&app);
    assert!(
        more > flow,
        "the wheel turned up did not raise the flow: {more}"
    );
    assert_eq!(
        same, build,
        "the water tool's wheel turned the land tool's dial"
    );
    testing::wheel(&mut app, -1);
    testing::run(&mut app, FRAME);
    assert_eq!(
        dials(&app),
        (flow, build),
        "a click down did not undo a click up"
    );

    testing::tap(&mut app, KeyCode::Digit2);
    testing::run(&mut app, FRAME);
    testing::wheel(&mut app, -1);
    testing::run(&mut app, FRAME);
    let (same, less) = dials(&app);
    assert!(
        less < build,
        "the wheel turned down did not lower the build-up: {less}"
    );
    assert_eq!(
        same, flow,
        "the land tool's wheel turned the water tool's dial"
    );

    let brush = |app: &App| app.world().resource::<Aim>().brush.expect("a brush is out");
    let narrow = brush(&app);
    testing::wheel(&mut app, 4);
    testing::run(&mut app, FRAME);
    assert!(
        brush(&app) > narrow,
        "more land a second did not widen the brush: {:?} from {narrow:?}",
        brush(&app)
    );
}

/// A tool that is out is seen before the eye, low and to the right where a hand would hold
/// it, and nowhere else; each tool looks its own way; and put away again it is gone.
#[test]
#[ignore = "wants a GPU"]
fn the_tool_that_is_out_is_seen_before_the_eye() {
    let mut sights = Sights::new(true);
    let bare = sights.draw("day-no-tool");
    testing::tap(&mut sights.app, KeyCode::Digit1);
    let water = sights.draw("day-water-tool");
    testing::tap(&mut sights.app, KeyCode::Digit2);
    let land = sights.draw("day-land-tool");
    testing::tap(&mut sights.app, KeyCode::Digit2);
    let away = sights.draw("day-put-away");

    let region = (WIDTH * HEIGHT / 4) as usize;
    for (name, tool) in [("water", &water), ("land", &land)] {
        let seen = changed(&bare, tool, LOWER_RIGHT);
        assert!(
            seen > region / 12,
            "the {name} tool changes only {seen} pixels of the lower right of the view"
        );
        let elsewhere = changed(&bare, tool, UPPER_LEFT);
        assert!(
            elsewhere < region / 200,
            "the {name} tool changes {elsewhere} pixels of the upper left of the view"
        );
    }
    let apart = changed(&water, &land, LOWER_RIGHT);
    assert!(
        apart > region / 25,
        "the two tools differ in only {apart} pixels"
    );
    let left = changed(&bare, &away, WHOLE);
    assert!(
        left < region / 200,
        "a tool put away still changes {left} pixels of the view"
    );
}

/// A barrel that fires lights its ring, and the ring lights what is near it: by night the
/// view gains blue while the water tool pours and while the land tool's left barrel fires,
/// and red while its right barrel does, and loses it again when the button is let go.
#[test]
#[ignore = "wants a GPU"]
fn a_firing_barrel_lights_its_ring_and_what_is_near() {
    let mut sights = Sights::new(false);
    look_down(&mut sights.app, 0.35);
    // firing works the ring, which is not what is looked at here
    sights.app.world_mut().resource_mut::<Settings>().flow.0 = 0.0;
    sights.app.world_mut().resource_mut::<Settings>().build.0 = 0.0;
    for (key, name, shots) in [
        (KeyCode::Digit1, "water", vec![(MouseButton::Left, 2)]),
        (
            KeyCode::Digit2,
            "land",
            vec![(MouseButton::Left, 2), (MouseButton::Right, 0)],
        ),
    ] {
        testing::tap(&mut sights.app, key);
        let dark = sights.draw(&format!("night-{name}-tool"));
        for (button, channel) in shots {
            testing::button(&mut sights.app, button, true);
            let lit = sights.draw(&format!("night-{name}-tool-firing-{button:?}"));
            testing::button(&mut sights.app, button, false);
            let glow = gained(&dark, &lit, channel, WHOLE);
            assert!(
                glow > (WIDTH * HEIGHT) as i64,
                "the {name} tool firing on {button:?} gains the view only {glow} of its colour"
            );
            let after = sights.draw(&format!("night-{name}-tool-after-{button:?}"));
            let left = changed(&dark, &after, WHOLE);
            assert!(
                left < (WIDTH * HEIGHT / 400) as usize,
                "{left} pixels stay lit after the {name} tool stops firing on {button:?}"
            );
        }
    }
}

/// The viewer sees its own figure in the glass: taking away what the mirrors are told of it
/// changes what they show, there where the figure stands in them.
#[test]
#[ignore = "wants a GPU"]
fn the_viewer_sees_its_own_figure_in_the_glass() {
    let mut sights = Sights::new(true);
    testing::tap(&mut sights.app, KeyCode::Digit1);
    // up to a cap of the ring, face on
    {
        let mut sim = sights.app.world_mut().resource_mut::<Simulation>();
        let half_width = sim.drum.ring.half_width.0 as f64;
        let (mut p, q) = (sim.avatar().p, sim.avatar().q);
        p[1] += half_width - 2.0;
        sim.avatar_mut().place(p, q);
    }
    // the site follows the avatar along the axis, and where the avatar is held is told from it
    testing::run(&mut sights.app, FRAME);
    let quarter = std::f64::consts::FRAC_PI_4;
    turn(
        &mut sights.app,
        [0.0, (-quarter).sin(), 0.0, (-quarter).cos()],
    );
    let mirrored = sights.draw("glass-figure");
    let limbs: Vec<Entity> = sights
        .app
        .world_mut()
        .query_filtered::<Entity, With<Mirrored>>()
        .iter(sights.app.world())
        .collect();
    for entity in limbs {
        sights
            .app
            .world_mut()
            .entity_mut(entity)
            .remove::<Mirrored>();
    }
    let unmirrored = sights.glance("glass-no-figure");
    let middle = [0.3, 0.2, 0.7, 0.95];
    let seen = changed(&mirrored, &unmirrored, middle);
    assert!(
        seen > 2000,
        "the figure shows in only {seen} pixels of the glass it stands in front of"
    );
}

/// And in still water it stands in, looking down into it. Water mirrors little of what
/// stands straight over it, a fiftieth of its light, over the sunlit ground that shows through
/// it: the figure is there, and faint.
#[test]
#[ignore = "wants a GPU"]
fn the_viewer_sees_its_own_figure_in_the_water() {
    let mut sights = Sights::spinning();
    testing::tap(&mut sights.app, KeyCode::Digit1);
    sights
        .app
        .world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let mut sim = world.resource_mut::<Simulation>();
            let mut p = sim.avatar().p;
            p[2] -= 3.0;
            sim.inject(&mut fluid, p, 9000)
        });
    testing::run(&mut sights.app, Seconds(12.0));
    look_down(&mut sights.app, 1.35);
    for _ in 0..30 {
        face_the_sun(&mut sights.app, true);
        testing::watch(&mut sights.app, Seconds(1.0 / 30.0));
    }
    face_the_sun(&mut sights.app, true);
    let mirrored = sights.glance("water-figure");
    let limbs: Vec<Entity> = sights
        .app
        .world_mut()
        .query_filtered::<Entity, With<Mirrored>>()
        .iter(sights.app.world())
        .collect();
    for entity in limbs {
        sights
            .app
            .world_mut()
            .entity_mut(entity)
            .remove::<Mirrored>();
    }
    let unmirrored = sights.glance("water-no-figure");
    // the figure's image lies under the figure, which is down the middle of a view tipped
    // this far down, and nowhere else
    let under = [0.25, 0.35, 0.75, 1.0];
    let seen = moved(&mirrored, &unmirrored, under, TOUCHED);
    let elsewhere = moved(&mirrored, &unmirrored, [0.0, 0.0, 1.0, 0.25], TOUCHED);
    assert!(
        elsewhere * 100 < seen,
        "taking the figure away touched {elsewhere} pixels of the water far from it"
    );
    assert!(
        seen > 10000,
        "the figure shows in only {seen} pixels of the water it stands over"
    );
}
