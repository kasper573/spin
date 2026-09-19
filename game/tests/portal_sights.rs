//! What is seen of the portals: alone and filled in, opening, open, and from behind. Each sight
//! is drawn with the portals and without them, and what they add to the picture is what is
//! judged. Every frame is written to `target/tmp/portal_sights/` to be looked at as well.
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::fluid::Fluid;
use game::core::math::{
    Quatd, Vec3d, cross, dot, norm, quat_conjugate, quat_from_basis, quat_rotate,
};
use game::core::units::{KilogramsPerCubicMetre, Metres, Radians, Seconds};
use game::systems::aim;
use game::systems::air::{Air, Suspension};
use game::systems::body::AvatarBody;
use game::systems::drum::{
    CapSide, DEFAULT_RING, Drum, DrumSurface, MouthColour, Mouths, OPENING, Place, Ring, Round,
};
use game::systems::player::PlayerCamera;
use game::systems::scene::{SUN_DIRECTION, Vantages};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing::{self, Headless};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const ADAPTING_WIDTH: u32 = WIDTH / 8;
const ADAPTING_HEIGHT: u32 = HEIGHT / 8;
/// A pixel counts as changed when a channel moves by this much, out of 255.
const CHANGED: i32 = 20;
const ADAPTING: f32 = 1.0;
const ADAPTING_STEPS: usize = 15;
const DIAMETER: Metres = Metres(2.0);

/// A place in the ring, fixed to the wheel: `height` over the ground at a wheel angle and a
/// place along the axis. Where that is in the drum's frame changes as the frame follows the
/// viewer round.
#[derive(Clone, Copy)]
struct Spot {
    round: f64,
    along: f64,
    height: f64,
}

impl Spot {
    fn of(drum: &Drum, p: Vec3d) -> Spot {
        Spot {
            round: drum.site.phi + drum.turn_to(p),
            along: drum.axial(p),
            height: drum.height_above_glass(p) - drum.ground(p),
        }
    }

    fn at(self, drum: &Drum) -> Vec3d {
        let glass = drum.wall_point(self.round - drum.site.phi, self.along - drum.site.y);
        let (_, outward) = drum.depth_and_outward(glass);
        let lift = drum.ground(glass) + self.height;
        [0, 1, 2].map(|k| glass[k] - outward[k] * lift)
    }
}

/// How an eye at `eye` is turned that looks at `at`, upright on the ring.
fn facing(drum: &Drum, eye: Vec3d, at: Vec3d) -> Quatd {
    let (_, outward) = drum.depth_and_outward(eye);
    let back = [0, 1, 2].map(|k| eye[k] - at[k]);
    let back = back.map(|c| c / norm(&back));
    let mut right = cross(&outward.map(|c| -c), &back);
    if norm(&right) < 1e-6 {
        right = cross(&[0.0, 1.0, 0.0], &back);
    }
    let right = right.map(|c| c / norm(&right));
    let up = cross(&back, &right);
    quat_from_basis(&right, &up, &back)
}

/// Hold the viewer as a ghost with its eye at `eye`, turned so.
fn hold(app: &mut App, eye: Vec3d, q: Quatd) {
    app.world_mut().resource_mut::<Settings>().collisions = false;
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.avatar_mut().solid = false;
    let head = quat_rotate(&q, &avatar::eye_offset());
    sim.avatar_mut()
        .place([0, 1, 2].map(|k| eye[k] - head[k]), q);
    sim.gyros = Gyros::holding(sim.avatar());
}

fn look(app: &mut App, eye: Spot, at: Spot) {
    let drum = &app.world().resource::<Simulation>().drum;
    let (eye, at) = (eye.at(drum), at.at(drum));
    let q = facing(drum, eye, at);
    hold(app, eye, q);
}

/// Turn the drum so that the sun stands over a wheel angle, or under it.
fn face_the_sun(app: &mut App, lit: f64, daylight: bool) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let over = (SUN_DIRECTION.z as f64).atan2(SUN_DIRECTION.x as f64) + std::f64::consts::PI;
    let turn = if daylight {
        over
    } else {
        over + std::f64::consts::PI
    };
    sim.drum.angle = Radians(lit - turn);
}

/// Put a mouth of a colour on the ground where it is wanted, its top the way `up` points.
fn put(app: &mut App, colour: MouthColour, at: Spot, up: Vec3d) {
    put_on(app, colour, DrumSurface::Wall, at, up);
}

fn put_on(app: &mut App, colour: MouthColour, surface: DrumSurface, at: Spot, up: Vec3d) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    let at = at.at(&sim.drum);
    let wanted = sim.drum.mouth_at(at, surface, up, [0.0, 0.0, 1.0]);
    let fit = sim.drum.fit_mouth(colour, wanted, DIAMETER);
    sim.drum.put_mouth(colour, fit, DIAMETER);
    assert!(sim.drum.mouths.get(colour).is_some(), "the mouth fits");
}

struct Sight {
    /// The wheel angle the sun is held over, or under by night.
    lit: f64,
    app: Headless,
    image: Handle<Image>,
    glimpse: Handle<Image>,
    dir: PathBuf,
}

/// What the portals add to a sight: the share of the frame they change, and the frame with
/// them and without.
struct View {
    name: String,
    changed: Vec<bool>,
    pixels: Vec<u8>,
}

impl Sight {
    fn new(scene: &str) -> Sight {
        Sight::on(scene, DEFAULT_RING)
    }

    fn on(scene: &str, ring: Ring) -> Sight {
        let mut app = testing::headless();
        {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            settings.diameter = Metres(ring.radius.0 * 2.0);
            settings.width = Metres(ring.half_width.0 * 2.0);
            settings.spin = standing_spin(ring);
            settings.equalize_thrust();
        }
        app.insert_resource(Simulation::new(ring));
        testing::run(&mut app, Seconds(0.5));
        let glimpse = testing::render_to_image(&mut app, ADAPTING_WIDTH, ADAPTING_HEIGHT);
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("portal_sights")
            .join(scene);
        fs::create_dir_all(&dir).expect("create the sights folder");
        Sight {
            lit: 0.0,
            app,
            image,
            glimpse,
            dir,
        }
    }

    fn drum(&self) -> &Drum {
        &self.app.world().resource::<Simulation>().drum
    }

    /// Let the eye adapt to the light at a viewpoint, held there as a ghost.
    fn settle(&mut self, eye: Spot, at: Spot, daylight: bool) {
        testing::draw_into(&mut self.app, &self.glimpse);
        for _ in 0..ADAPTING_STEPS {
            look(&mut self.app, eye, at);
            face_the_sun(&mut self.app, self.lit, daylight);
            testing::watch(&mut self.app, Seconds(ADAPTING / ADAPTING_STEPS as f32));
        }
        look(&mut self.app, eye, at);
        face_the_sun(&mut self.app, self.lit, daylight);
        testing::draw_into(&mut self.app, &self.image);
        // what is seen through a portal is drawn as large as the view was the frame before
        for _ in 0..2 {
            testing::frame(&mut self.app, Seconds(0.0));
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

    fn without_mouths(&mut self) -> Mouths {
        let mut sim = self.app.world_mut().resource_mut::<Simulation>();
        std::mem::take(&mut sim.drum.mouths)
    }

    /// The view as it stands, with the portals and with them taken away.
    fn held(&mut self, name: &str) -> View {
        let with = self.capture(name);
        let mouths = self.without_mouths();
        let without = self.capture(&format!("{name}_without"));
        let mut sim = self.app.world_mut().resource_mut::<Simulation>();
        sim.drum.mouths = mouths;
        let changed = with
            .chunks_exact(4)
            .zip(without.chunks_exact(4))
            .map(|(a, b)| (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > CHANGED))
            .collect();
        View {
            name: name.to_owned(),
            changed,
            pixels: with,
        }
    }

    fn view(&mut self, name: &str, eye: Spot, at: Spot, daylight: bool) -> View {
        self.settle(eye, at, daylight);
        self.held(name)
    }
}

impl View {
    fn share(&self) -> f64 {
        self.changed.iter().filter(|c| **c).count() as f64 / self.changed.len() as f64
    }

    /// The mean colour of the changed pixels of the rows between two shares of the way down
    /// the changed part of the frame.
    fn colour(&self, from: f64, to: f64) -> [f64; 3] {
        let rows: Vec<usize> = (0..HEIGHT as usize)
            .filter(|y| self.changed[y * WIDTH as usize..(y + 1) * WIDTH as usize].contains(&true))
            .collect();
        let (top, bottom) = (rows[0] as f64, *rows.last().expect("rows") as f64 + 1.0);
        let (mut sum, mut n) = ([0.0; 3], 0.0);
        for (k, pixel) in self.pixels.chunks_exact(4).enumerate() {
            let down = ((k / WIDTH as usize) as f64 - top) / (bottom - top);
            if self.changed[k] && (from..to).contains(&down) {
                for c in 0..3 {
                    sum[c] += pixel[c] as f64;
                }
                n += 1.0;
            }
        }
        sum.map(|s| s / f64::max(n, 1.0))
    }
}

fn luminance(colour: [f64; 3]) -> f64 {
    0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
}

/// A mouth on the ground ahead of a viewer standing on it, its top away from the viewer, and
/// where the viewer's eye is and looks.
fn on_the_ground_ahead(sight: &mut Sight, colour: MouthColour) -> (Spot, Spot) {
    let at = Spot {
        round: 0.35,
        along: 0.0,
        height: 0.0,
    };
    put(&mut sight.app, colour, at, [0.0, 0.0, 1.0]);
    let eye = Spot {
        round: 0.0,
        along: 0.0,
        height: 1.7,
    };
    (eye, at)
}

/// Expected: a disc of the mouth's colour lies on the ground ahead, foreshortened to an
/// ellipse, ringed by flames of the same colour; nothing of the ground shows through it; its
/// far side, which is its top, is clearly brighter than its near side. The same for either
/// colour, by day and by night, when the fire also lights the ground round it.
#[test]
#[ignore = "wants a GPU"]
fn a_lone_portal_is_a_filled_disc_of_its_colour_brighter_at_its_top() {
    for (colour, name) in [(MouthColour::Blue, "blue"), (MouthColour::Orange, "orange")] {
        let mut sight = Sight::new("lone");
        let (eye, at) = on_the_ground_ahead(&mut sight, colour);
        testing::run(&mut sight.app, Seconds(2.0 * OPENING.0));
        assert_eq!(sight.drum().mouths.fill(), 1.0);
        for daylight in [true, false] {
            let tag = if daylight { "day" } else { "night" };
            let view = sight.view(&format!("{name}_{tag}"), eye, at, daylight);
            assert!(
                view.share() > 0.02,
                "{}: the portal changes {:.2}% of the frame",
                view.name,
                view.share() * 100.0
            );
            let [r, _, b] = view.colour(0.2, 0.8);
            match colour {
                MouthColour::Blue => assert!(
                    b > 1.5 * r,
                    "{}: a blue portal shows {r} red, {b} blue",
                    view.name
                ),
                MouthColour::Orange => assert!(
                    r > 1.5 * b,
                    "{}: an orange portal shows {r} red, {b} blue",
                    view.name
                ),
            }
            let (top, bottom) = (
                luminance(view.colour(0.15, 0.4)),
                luminance(view.colour(0.6, 0.85)),
            );
            assert!(
                top > 1.15 * bottom,
                "{}: the top of the portal is {top}, the bottom {bottom}",
                view.name
            );
        }
    }
}

/// The player's camera sees only what is past a plane through `point` of the drum's frame,
/// on the side `inward` points to; or whatever is before it, as ever.
fn see_only_past(app: &mut App, plane: Option<(Vec3d, Vec3d)>) {
    let (eye, q) = app.world().resource::<Simulation>().eye();
    let mut cameras = app
        .world_mut()
        .query_filtered::<&mut Projection, With<PlayerCamera>>();
    for mut projection in cameras.iter_mut(app.world_mut()) {
        let Projection::Perspective(lens) = &mut *projection else {
            continue;
        };
        lens.near_clip_plane = match plane {
            Some((point, inward)) => {
                let normal = quat_rotate(&quat_conjugate(&q), &inward);
                let offset = -dot(&inward, &[0, 1, 2].map(|k| point[k] - eye[k]));
                Vec4::new(
                    normal[0] as f32,
                    normal[1] as f32,
                    normal[2] as f32,
                    offset as f32,
                )
            }
            None => Vec4::new(0.0, 0.0, -1.0, -lens.near),
        };
    }
}

/// How far apart two frames are over the disc of `radius` pixels about the middle of the
/// frame: the mean difference of a channel, out of 255.
fn apart(a: &[u8], b: &[u8], radius: f64) -> f64 {
    let (mut sum, mut n) = (0.0, 0.0);
    for (k, (a, b)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let x = (k % WIDTH as usize) as f64 - WIDTH as f64 / 2.0;
        let y = (k / WIDTH as usize) as f64 - HEIGHT as f64 / 2.0;
        if x.hypot(y) < radius {
            for c in 0..3 {
                sum += (a[c] as f64 - b[c] as f64).abs();
                n += 1.0;
            }
        }
    }
    sum / n
}

/// Looking down into an open portal on the ground, the viewer sees out of the other one: the
/// ring's far side from below, just as an eye sees it that is held where the pair takes the
/// viewer's, under the far portal, with all that is short of that portal's ground cut away.
/// The two pictures are the same over the middle of the mouth, where nothing of the flames
/// reaches; and neither is what the viewer sees with no portals there, which is the ground.
fn seen_through_as_from_where_the_pair_takes_the_eye(mut sight: Sight, orange: Spot) {
    let adapting = sight
        .app
        .world_mut()
        .query_filtered::<Entity, With<PlayerCamera>>()
        .single(sight.app.world())
        .expect("the player's camera");
    sight
        .app
        .world_mut()
        .entity_mut(adapting)
        .remove::<bevy::post_process::auto_exposure::AutoExposure>();
    let metres = 1.0 / sight.drum().ring.radius.0 as f64;
    let blue = Spot {
        round: 2.0 * metres,
        along: 1.0,
        height: 0.0,
    };
    put(&mut sight.app, MouthColour::Blue, blue, [0.0, 0.0, 1.0]);
    put(&mut sight.app, MouthColour::Orange, orange, [0.0, 0.0, 1.0]);
    testing::run(&mut sight.app, Seconds(2.0 * OPENING.0));
    assert_eq!(sight.drum().mouths.fill(), 0.0, "the pair has opened");

    let eye = Spot {
        along: 2.6,
        height: 1.4,
        ..blue
    };
    sight.lit = orange.round + std::f64::consts::PI;
    sight.settle(eye, blue, true);
    let seen = sight.capture("through");
    let mouths = sight.without_mouths();
    let ground = sight.capture("no_portals");
    sight
        .app
        .world_mut()
        .resource_mut::<Simulation>()
        .drum
        .mouths = mouths;

    // the frame follows the viewer to the far mouth, so what is known of the pair there is
    // asked for once it has
    let (from, attitude) = sight.app.world().resource::<Simulation>().eye();
    let through = sight
        .drum()
        .mouth_sight(MouthColour::Blue, from)
        .expect("a pair is seen through");
    hold(
        &mut sight.app,
        through.point(from),
        through.attitude(attitude),
    );
    testing::frame(&mut sight.app, Seconds(0.0));
    let (carried, _) = sight.app.world().resource::<Simulation>().eye();
    let far = sight
        .drum()
        .mouth_sight(MouthColour::Blue, carried)
        .expect("the pair still stands");
    sight.without_mouths();
    see_only_past(&mut sight.app, Some((far.threshold, far.inward)));
    let direct = sight.capture("direct");
    see_only_past(&mut sight.app, None);

    let middle = 90.0;
    let (same, other) = (apart(&seen, &direct, middle), apart(&seen, &ground, middle));
    assert!(
        same < 4.0,
        "what is seen through the portal is {same} from what the carried eye sees"
    );
    assert!(
        other > 5.0 * same.max(1.0),
        "what is seen through the portal is only {other} from the bare ground"
    );
}

#[test]
#[ignore = "wants a GPU"]
fn through_an_open_portal_the_far_side_is_seen_as_from_where_the_pair_takes_the_eye() {
    let orange = Spot {
        round: 2.2,
        along: -1.5,
        height: 0.0,
    };
    seen_through_as_from_where_the_pair_takes_the_eye(Sight::new("through"), orange);
}

/// The same with the far portal kilometres round a ring kilometres across from the near one:
/// what is seen through is drawn about the far portal, as exactly as what is round the viewer.
#[test]
#[ignore = "wants a GPU"]
fn a_portal_kilometres_away_is_seen_through_as_exactly() {
    let ring = Ring {
        radius: Metres(5000.0),
        half_width: Metres(500.0),
    };
    let orange = Spot {
        round: 2.2,
        along: -300.0,
        height: 0.0,
    };
    seen_through_as_from_where_the_pair_takes_the_eye(Sight::on("through_far", ring), orange);
}

/// Expected: the viewer stands between a pair of portals on the ground, looking down into
/// the one ahead, and sees out of the one behind, up past its own back: its own body stands
/// in what the portal shows, dark against the far side of the ring. With the body taken
/// away, what the portal shows changes where the body was, and hardly anything else does.
#[test]
#[ignore = "wants a GPU"]
fn the_viewers_own_body_is_seen_through_a_portal() {
    let mut sight = Sight::new("body");
    let (eye, blue) = on_the_ground_ahead(&mut sight, MouthColour::Blue);
    let orange = Spot {
        round: -0.3,
        ..blue
    };
    put(&mut sight.app, MouthColour::Orange, orange, [0.0, 0.0, 1.0]);
    testing::run(&mut sight.app, Seconds(2.0 * OPENING.0));
    let portals = sight.view("with_the_body", eye, blue, true);
    let bodies: Vec<Entity> = sight
        .app
        .world_mut()
        .query_filtered::<Entity, With<AvatarBody>>()
        .iter(sight.app.world())
        .collect();
    for body in bodies {
        sight
            .app
            .world_mut()
            .entity_mut(body)
            .insert(Visibility::Hidden);
    }
    let bare = sight.capture("without_the_body");
    let differs: Vec<bool> = portals
        .pixels
        .chunks_exact(4)
        .zip(bare.chunks_exact(4))
        .map(|(a, b)| (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > CHANGED))
        .collect();
    let within = differs
        .iter()
        .zip(&portals.changed)
        .filter(|(differs, portal)| **differs && **portal)
        .count() as f64
        / differs.len() as f64;
    assert!(
        within > 0.003,
        "the body takes up {:.2}% of the frame within the portal",
        within * 100.0
    );
}

fn is_blue(p: &[u8]) -> bool {
    p[2] > 25 && p[2] as f64 > 1.5 * p[0] as f64 && p[2] > p[1]
}

/// How much of a frame is plainly of the blue portal's colour: bluer than it is red by far.
fn blue_share(pixels: &[u8]) -> f64 {
    let blue = pixels.chunks_exact(4).filter(|p| is_blue(p)).count();
    blue as f64 / (pixels.len() / 4) as f64
}

/// The same of the disc of `radius` pixels about the middle of the frame.
fn blue_share_of_the_middle(pixels: &[u8], radius: f64) -> f64 {
    let (mut blue, mut n) = (0.0, 0.0);
    for (k, p) in pixels.chunks_exact(4).enumerate() {
        let x = (k % WIDTH as usize) as f64 - WIDTH as f64 / 2.0;
        let y = (k / WIDTH as usize) as f64 - HEIGHT as f64 / 2.0;
        if x.hypot(y) < radius {
            n += 1.0;
            if is_blue(p) {
                blue += 1.0;
            }
        }
    }
    blue / n
}

/// Expected: a portal let into a cap is open to the room only. From outside the drum, behind
/// the cap, it is a disc filled with its colour, as it is while it is alone, and nothing is
/// seen through it, though its pair stands and it is open from the front.
#[test]
#[ignore = "wants a GPU"]
fn from_behind_an_open_portal_is_filled_in() {
    let mut sight = Sight::new("behind");
    let half_width = sight.drum().ring.half_width.0 as f64;
    let on_the_cap = Spot {
        round: 0.0,
        along: -half_width,
        height: 3.0,
    };
    {
        let mut sim = sight.app.world_mut().resource_mut::<Simulation>();
        let at = on_the_cap.at(&sim.drum);
        let surface = DrumSurface::Cap(CapSide::Low);
        let wanted = sim
            .drum
            .mouth_at(at, surface, [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let fit = sim.drum.fit_mouth(MouthColour::Blue, wanted, DIAMETER);
        sim.drum.put_mouth(MouthColour::Blue, fit, DIAMETER);
        assert!(sim.drum.mouths.get(MouthColour::Blue).is_some());
    }
    let orange = Spot {
        round: 1.0,
        along: 0.0,
        height: 0.0,
    };
    put(&mut sight.app, MouthColour::Orange, orange, [0.0, 0.0, 1.0]);
    testing::run(&mut sight.app, Seconds(2.0 * OPENING.0));
    assert_eq!(sight.drum().mouths.fill(), 0.0, "the pair has opened");

    let inside = Spot {
        along: -half_width + 4.0,
        ..on_the_cap
    };
    let outside = Spot {
        along: -half_width - 4.0,
        ..on_the_cap
    };
    let front = sight.view("from_the_room", inside, on_the_cap, true);
    let behind = sight.view("from_behind", outside, on_the_cap, true);
    let middle = 70.0;
    let (open, filled) = (
        blue_share_of_the_middle(&front.pixels, middle),
        blue_share_of_the_middle(&behind.pixels, middle),
    );
    assert!(
        behind.share() > 0.02,
        "from behind the portal changes {:.2}% of the frame",
        behind.share() * 100.0
    );
    assert!(
        filled > 0.8 && open < 0.05,
        "{:.2}% of the middle of the mouth is blue from behind, {:.2}% from the room",
        filled * 100.0,
        open * 100.0
    );
}

/// Expected: when a portal's pair is put, both open from the middle out: the part of the
/// mouth that is filled with its colour shrinks steadily to nothing, leaving the ring of fire.
#[test]
#[ignore = "wants a GPU"]
fn a_pair_opens_steadily_from_the_middle_out() {
    let mut sight = Sight::new("opening");
    let (eye, blue) = on_the_ground_ahead(&mut sight, MouthColour::Blue);
    testing::run(&mut sight.app, Seconds(1.0));
    sight.settle(eye, blue, true);
    let orange = Spot { round: 2.0, ..blue };
    put(&mut sight.app, MouthColour::Orange, orange, [0.0, 0.0, 1.0]);
    let steps = 6;
    let shares: Vec<f64> = (0..=steps)
        .map(|step| {
            if step > 0 {
                testing::run(&mut sight.app, Seconds(1.2 * OPENING.0 / steps as f32));
                look(&mut sight.app, eye, blue);
            }
            blue_share(&sight.capture(&format!("step_{step}")))
        })
        .collect();
    assert!(
        shares.windows(2).all(|pair| pair[1] <= pair[0] + 0.002),
        "the blue of the mouth went {shares:?}"
    );
    assert!(
        shares[0] > 3.0 * shares[steps],
        "the blue of the mouth went {shares:?}"
    );
}

/// The mean brightness of the middle of a frame, out to a share of its height.
fn brightness_of_the_middle(pixels: &[u8], radius: f64) -> f64 {
    let (cx, cy) = (WIDTH as f64 / 2.0, HEIGHT as f64 / 2.0);
    let (mut sum, mut n) = (0.0, 0.0);
    for (k, p) in pixels.chunks_exact(4).enumerate() {
        let (x, y) = ((k % WIDTH as usize) as f64, (k / WIDTH as usize) as f64);
        if (x - cx).hypot(y - cy) < radius * HEIGHT as f64 {
            sum += luminance([p[0] as f64, p[1] as f64, p[2] as f64]);
            n += 1.0;
        }
    }
    sum / n
}

/// Expected: a mouth let into the glass at the ring's end where the sun falls on it, and its
/// pair let into the ground on the ring's dark side. The sunlight that goes in at the one
/// comes out of the other as a beam, leaning as the sun does from the first, and lands on the
/// ring where a ray sent the same way lands: that place is lit by it while the pair is open,
/// over and above whatever light it had, in a patch as wide as the mouth, with the ground
/// beside the patch no brighter than before; and where the ring's air is a fog, the air the
/// beam crosses shows it as a shaft, faintly, since the beam is only as thick as the mouth;
/// and a tool held in the beam, with the light coming from behind the eye, is lit by it as
/// the ground is. Move the first mouth into the shade and all of that is
/// gone.
#[test]
#[ignore = "wants a GPU"]
fn sunlight_goes_through_an_open_pair() {
    let mut sight = Sight::new("sunlight");
    let half_width = sight.drum().ring.half_width.0 as f64;
    let orange = Spot {
        round: std::f64::consts::PI,
        along: 0.0,
        height: 0.0,
    };
    let in_the_sun = Spot {
        round: 0.0,
        along: -half_width,
        height: 5.0,
    };
    let in_the_shade = Spot {
        round: std::f64::consts::PI,
        along: -half_width,
        height: 3.0,
    };
    face_the_sun(&mut sight.app, 0.0, true);
    testing::run(&mut sight.app, Seconds(0.1));
    let to_sun = |app: &App| {
        let vantage = app.world().resource::<Vantages>().0[0].expect("the viewer's vantage");
        vantage
            .frame
            .vector_back(vantage.sky.sun.to_array().map(f64::from))
    };
    // the first mouth's top is put square to the way the sun leans, and the second's along the
    // ring's axis, so that the beam leans round the ring as it leaves the ground
    let leans = to_sun(&sight.app);
    let top = cross(&[0.0, 1.0, 0.0], &[leans[0], 0.0, leans[2]]);
    put(&mut sight.app, MouthColour::Orange, orange, [0.0, 1.0, 0.0]);
    let cap = DrumSurface::Cap(CapSide::Low);
    put_on(&mut sight.app, MouthColour::Blue, cap, in_the_sun, top);
    testing::run(&mut sight.app, Seconds(OPENING.0 + 0.5));
    face_the_sun(&mut sight.app, 0.0, true);
    testing::frame(&mut sight.app, Seconds(0.0));

    // the way the light goes, as a ray sent down it from before the first mouth
    let (landing, beam, leaves) = {
        let world = sight.app.world();
        let drum = &world.resource::<Simulation>().drum;
        let to_sun = to_sun(&sight.app);
        let blue = in_the_sun.at(drum);
        let from = [0, 1, 2].map(|k| blue[k] + to_sun[k] * 2.0);
        let met = aim::cast(from.into(), -bevy::math::DVec3::from(to_sun), drum)
            .expect("the light lands somewhere");
        assert_eq!(
            met.mouth,
            Some(MouthColour::Blue),
            "the ray goes in at the mouth"
        );
        let lands = met.point.to_array();
        let leaves = orange.at(drum);
        let along = [0, 1, 2].map(|k| lands[k] - leaves[k]);
        let along = along.map(|c| c / norm(&along));
        let side = cross(&met.normal.to_array(), &along);
        let side = side.map(|c| c / norm(&side));
        ((lands, met.normal.to_array(), side), along, leaves)
    };
    let (lands, normal, side) = landing;
    let spot = |p: Vec3d, drum: &Drum| Spot::of(drum, p);
    let (eye, at, beside, shaft_eye, shaft_at, in_the_beam, down_the_beam) = {
        let drum = sight.drum();
        let off = |p: Vec3d, a: f64, v: Vec3d, b: f64, w: Vec3d| {
            [0, 1, 2].map(|k| p[k] + a * v[k] + b * w[k])
        };
        let middle = off(leaves, 3.0, beam, 0.0, side);
        (
            spot(off(lands, 1.6, normal, 4.0, side), drum),
            spot(lands, drum),
            spot(off(lands, 0.0, normal, 3.0, side), drum),
            spot(off(middle, 0.0, beam, 4.0, side), drum),
            spot(middle, drum),
            spot(off(middle, 0.0, beam, 0.3, side), drum),
            spot(lands, drum),
        )
    };

    sight.settle(eye, at, true);
    let lit_patch = sight.capture("patch");
    sight.settle(eye, beside, true);
    let lit_beside = sight.capture("beside");
    let fog = Air {
        carries: Suspension {
            loading: KilogramsPerCubicMetre(3.0e-4),
            ..Suspension::FOG
        },
        ..Air::default()
    };
    sight.app.insert_resource(fog);
    sight.settle(shaft_eye, shaft_at, true);
    let lit_shaft = sight.capture("shaft");
    sight.app.insert_resource(Air::default());

    testing::tap(&mut sight.app, KeyCode::Digit1);
    sight.settle(in_the_beam, down_the_beam, true);
    let lit_tool = sight.capture("tool");

    put_on(&mut sight.app, MouthColour::Blue, cap, in_the_shade, top);
    sight.settle(in_the_beam, down_the_beam, true);
    let dark_tool = sight.capture("tool_shaded");
    testing::tap(&mut sight.app, KeyCode::Digit1);
    sight.settle(eye, at, true);
    let dark_patch = sight.capture("patch_shaded");
    sight.settle(eye, beside, true);
    let dark_beside = sight.capture("beside_shaded");
    sight.app.insert_resource(fog);
    sight.settle(shaft_eye, shaft_at, true);
    let dark_shaft = sight.capture("shaft_shaded");

    let middle = |pixels: &[u8]| brightness_of_the_middle(pixels, 0.04);
    let (patch, shaded) = (middle(&lit_patch), middle(&dark_patch));
    assert!(
        patch > shaded + 40.0,
        "where the beam lands is lit: {patch:.1} against {shaded:.1} without it"
    );
    let (beside, beside_shaded) = (middle(&lit_beside), middle(&dark_beside));
    assert!(
        (beside - beside_shaded).abs() < 0.15 * beside_shaded + 3.0,
        "the ground beside the patch is as it was: {beside:.1} against {beside_shaded:.1}"
    );
    // the tool is held before the eye, low and to the right
    let tool = |pixels: &[u8]| luminance(colour_within(pixels, 0.62..0.8, 0.72..0.9));
    let (held, held_shaded) = (tool(&lit_tool), tool(&dark_tool));
    assert!(
        held > 1.1 * held_shaded,
        "a tool held in the beam is lit by it: {held:.1} against {held_shaded:.1}"
    );
    let (shaft, no_shaft) = (middle(&lit_shaft), middle(&dark_shaft));
    assert!(
        shaft > no_shaft + 0.8,
        "the air shows the beam: {shaft:.1} against {no_shaft:.1} without it"
    );
}

/// The mean colour of a part of a frame, between shares of the way across it and down it.
fn colour_within(
    pixels: &[u8],
    across: std::ops::Range<f64>,
    down: std::ops::Range<f64>,
) -> [f64; 3] {
    let (mut sum, mut n) = ([0.0; 3], 0.0);
    for (k, p) in pixels.chunks_exact(4).enumerate() {
        let x = (k % WIDTH as usize) as f64 / WIDTH as f64;
        let y = (k / WIDTH as usize) as f64 / HEIGHT as f64;
        if across.contains(&x) && down.contains(&y) {
            for c in 0..3 {
                sum[c] += p[c] as f64;
            }
            n += 1.0;
        }
    }
    sum.map(|s| s / n)
}

/// Expected: by night, an orange mouth let into the glass at the ring's end, low over the
/// ground, and a blue one in the ground across the ring, too far off for its fire to light
/// anything here. While the orange mouth is alone, the ground before it is lit by the orange
/// fire alone. Once it has, the blue fire shines out of the orange mouth too, from
/// behind it, onto the same ground. A lamp that bright, that far off, adds about a twentieth
/// to the light that is bounced round the ring by night, so that ground shows some hundredths
/// more blue than it did, and gains less than half as much red.
#[test]
#[ignore = "wants a GPU"]
fn a_mouths_fire_shines_out_of_the_other_mouth() {
    let mut sight = Sight::new("fire_through");
    let half_width = sight.drum().ring.half_width.0 as f64;
    let orange = Spot {
        round: 0.0,
        along: -half_width,
        height: 1.3,
    };
    let blue = Spot {
        round: std::f64::consts::PI,
        along: 0.0,
        height: 0.0,
    };
    let lit = Spot {
        round: 0.0,
        along: -half_width + 1.6,
        height: 0.0,
    };
    let eye = Spot {
        round: 0.25,
        along: -half_width + 3.0,
        height: 1.7,
    };
    // the ground between the mouth and the crosshair, clear of the crosshair's own marker
    let before_the_mouth = |pixels: &[u8]| colour_within(pixels, 0.3..0.45, 0.42..0.52);
    let cap = DrumSurface::Cap(CapSide::Low);
    put_on(
        &mut sight.app,
        MouthColour::Orange,
        cap,
        orange,
        [1.0, 0.0, 0.0],
    );
    testing::run(&mut sight.app, Seconds(1.0));
    sight.settle(eye, lit, false);
    let shut = before_the_mouth(&sight.capture("shut"));
    // nothing is drawn while the pair opens, so the eye stays as it had adapted, and the two
    // frames are exposed alike
    put(&mut sight.app, MouthColour::Blue, blue, [0.0, 1.0, 0.0]);
    testing::run(&mut sight.app, Seconds(2.0 * OPENING.0));
    assert_eq!(sight.drum().mouths.fill(), 0.0, "the pair has opened");
    look(&mut sight.app, eye, lit);
    face_the_sun(&mut sight.app, sight.lit, false);
    let open = before_the_mouth(&sight.capture("open"));
    let gained = [0, 1, 2].map(|c| open[c] / shut[c] - 1.0);
    assert!(
        (0.03..0.15).contains(&gained[2]) && gained[2] > 2.0 * gained[0],
        "the ground before the orange mouth gains blue light: {shut:?} shut, {open:?} open"
    );
}

/// Expected: water poured into a mouth in the ground falls out of its pair, let into the glass
/// at the ring's end, as one smooth stream: round and unbroken from the mouth down, with no
/// flat faces or corners on it, as clear as any water, and nothing standing beside it in the
/// air that is not water.
#[test]
#[ignore = "wants a GPU"]
fn water_falls_out_of_a_portal_as_a_smooth_stream() {
    let mut sight = Sight::new("stream");
    let half_width = sight.drum().ring.half_width.0 as f64;
    let blue = Spot {
        round: 0.0,
        along: 0.0,
        height: 0.0,
    };
    let orange = Spot {
        round: 0.0,
        along: -half_width,
        height: 3.5,
    };
    let radius = sight.drum().ring.radius.0 as f64;
    let eye = Spot {
        round: 4.0 / radius,
        along: 2.0 - half_width,
        height: 2.5,
    };
    let at = Spot {
        round: 0.0,
        along: 1.0 - half_width,
        height: 2.2,
    };
    put(&mut sight.app, MouthColour::Blue, blue, [0.0, 1.0, 0.0]);
    put_on(
        &mut sight.app,
        MouthColour::Orange,
        DrumSurface::Cap(CapSide::Low),
        orange,
        [0.0, 0.0, 1.0],
    );
    sight.settle(eye, at, true);
    for k in 0..12 {
        sight
            .app
            .world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let over = Spot {
                    height: 1.5,
                    ..blue
                }
                .at(&sim.drum);
                sim.inject(&mut fluid, over, 150);
            });
        look(&mut sight.app, eye, at);
        testing::watch(&mut sight.app, Seconds(0.25));
        if k % 3 == 2 {
            look(&mut sight.app, eye, at);
            sight.capture(&format!("stream_{k}"));
        }
    }
}

/// Expected: a pool with a mouth let into its floor, and its pair let into the glass at the
/// ring's end over it. Through the mouth in the glass the pool is seen from its floor: water,
/// lit from above and coloured by its depth, and never the black of space with its stars, which
/// is what is there with the water left out.
#[test]
#[ignore = "wants a GPU"]
fn a_pool_is_seen_from_its_floor_through_a_portal() {
    let mut sight = Sight::new("pool_floor");
    let half_width = sight.drum().ring.half_width.0 as f64;
    let radius = sight.drum().ring.radius.0 as f64;
    {
        let mut sim = sight.app.world_mut().resource_mut::<Simulation>();
        let across = (half_width / 2.75).floor() as i32;
        for crest in [6.0, -6.0] {
            for y in (-across..=across).map(|k| k as f64 * 2.75) {
                let at = Place {
                    round: Round::default().on(crest, sim.drum.landscape.grid()),
                    along: y,
                };
                for _ in 0..20 {
                    sim.drum.landscape.sculpt(at, 8.0, 1.4 / 20.0);
                }
            }
        }
    }
    let over = |arc: f64, along: f64, height: f64| Spot {
        round: arc / radius,
        along,
        height,
    };
    for k in 0..12 {
        sight
            .app
            .world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                let mut sim = world.resource_mut::<Simulation>();
                let at = over((k % 3) as f64 - 1.0, (k / 3) as f64 * 2.0 - 3.0, 1.5).at(&sim.drum);
                sim.inject(&mut fluid, at, 500);
            });
        testing::run(&mut sight.app, Seconds(1.0));
    }
    testing::run(&mut sight.app, Seconds(10.0));
    let orange = over(0.0, -half_width, 3.0);
    put(
        &mut sight.app,
        MouthColour::Blue,
        over(0.0, 0.0, 0.0),
        [0.0, 1.0, 0.0],
    );
    put_on(
        &mut sight.app,
        MouthColour::Orange,
        DrumSurface::Cap(CapSide::Low),
        orange,
        [0.0, 0.0, 1.0],
    );
    testing::run(&mut sight.app, Seconds(OPENING.0 + 0.5));
    let eye = over(1.0, 3.0 - half_width, 3.2);
    sight.settle(eye, orange, true);
    look(&mut sight.app, eye, orange);
    sight.capture("through");
    let mut hidden = sight
        .app
        .world_mut()
        .query_filtered::<&mut Visibility, With<game::systems::water::WaterMesh>>();
    for mut shown in hidden.iter_mut(sight.app.world_mut()) {
        *shown = Visibility::Hidden;
    }
    look(&mut sight.app, eye, orange);
    sight.capture("through_dry");
}
