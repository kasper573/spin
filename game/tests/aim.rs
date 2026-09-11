//! Where the crosshair rests and what it outlines.
use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::fluid::{Fluid, Resolution};
use game::core::math::{mat3mul, quat_mul};
use game::core::units::{Metres, Seconds};
use game::systems::aim::{self, Aim, AimPoint};
use game::systems::controls::{BRUSH_RATE, BRUSH_SIZE, INJECT_DEPTH};
use game::systems::drum::Ring;
use game::systems::drum::{Drum, GROUND_DEPTH};
use game::systems::player::PlayerCamera;
use game::systems::scene::Viewpoint;
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::sim::standing_spin;
use game::systems::testing;

/// The outline of a brush on uneven ground lies on the ground, just above it, everywhere
/// round it: every point is the brush's radius from its centre along the ground and at the
/// ground's height there, hill or hollow, so the outline wraps the ground like a sheet laid
/// over it.
#[test]
fn the_outline_wraps_the_ground_under_it() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.1));
    {
        let mut sim = app.world_mut().resource_mut::<Simulation>();
        let site = sim.drum.site;
        for k in 0..6 {
            let a = k as f64 * 1.1;
            let at = sim
                .drum
                .wall_point(0.02 * a.cos(), 0.5 + 2.0 * a.sin() - site.y);
            sim.drum.sculpt(at, 1.5, 1.2 * (k as f64 + 1.0));
        }
    }
    let sim = app.world().resource::<Simulation>();
    let eye = DVec3::new(-3.0, 0.0, 0.0);
    let (radius, lift) = (2.0, 0.02);
    for (dy, dz) in [(0.0, 0.0), (1.0, -1.0), (-2.0, 1.5), (2.5, 2.5)] {
        let dir = (DVec3::new(-0.5, dy, dz) - eye).normalize();
        let hit = aim::cast(eye, dir, &sim.drum).expect("the ground is in front of the eye");
        assert!(
            (hit.normal.length() - 1.0).abs() < 1e-6 && hit.normal.dot(dir) < 0.0,
            "the ground hit faces {:?}, looked at along {dir:?}",
            hit.normal
        );
        let outline = aim::outline(&sim.drum, hit, radius, lift);
        assert!(outline.len() >= 48);
        let centre_turn = sim.drum.turn_to(hit.point.to_array());
        let centre_axial = sim.drum.axial(hit.point.to_array());
        let ring_radius = sim.drum.ring.radius.0 as f64;
        let mut heights_seen = Vec::new();
        for p in &outline {
            let p = p.to_array();
            let ground = sim.drum.ground(p);
            let above = sim.drum.height_above_glass(p);
            assert!(
                (above - ground - lift).abs() < 1e-6,
                "an outline point is {above} m above the glass over ground {ground} m high"
            );
            let along = (sim.drum.turn_to(p) - centre_turn) * ring_radius;
            let across = sim.drum.axial(p) - centre_axial;
            let distance = (along * along + across * across).sqrt();
            assert!(
                (distance - radius).abs() < 1e-6,
                "an outline point is {distance} m from the centre"
            );
            heights_seen.push(ground);
        }
        let (lowest, highest) = heights_seen
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| {
                (lo.min(*h), hi.max(*h))
            });
        assert!(
            highest - lowest > 0.3 || highest <= GROUND_DEPTH.0 as f64 + 1e-6,
            "the ground under the outline is flat: {lowest}..{highest}"
        );
    }
    let flat: AimPoint =
        aim::cast(eye, DVec3::new(0.0, 1.0, 0.0), &sim.drum).expect("the cap is above the eye");
    let outline = aim::outline(&sim.drum, flat, radius, lift);
    for p in &outline {
        assert!(
            (p.y - flat.point.y + lift).abs() < 1e-9,
            "a cap outline point is off the cap"
        );
    }
}

/// The crosshair sits at the middle of the screen, so what it points at must be drawn there:
/// on a ring of any size, the aim point and the marker outlined about it land under the
/// crosshair, not beside it.
#[test]
#[ignore = "wants a GPU"]
fn what_the_crosshair_points_at_is_drawn_under_it() {
    for radius in [10.5, 100.0, 5000.0] {
        let ring = Ring {
            radius: Metres(radius),
            half_width: Metres((radius * 0.1).max(6.0)),
        };
        let mut app = testing::headless();
        {
            let mut settings = app.world_mut().resource_mut::<Settings>();
            settings.diameter = Metres(ring.radius.0 * 2.0);
            settings.width = Metres(ring.half_width.0 * 2.0);
            settings.spin = standing_spin(ring);
            settings.equalize_thrust();
        }
        app.insert_resource(Simulation::new(ring));
        testing::run(&mut app, Seconds(1.0));
        let image = testing::render_to_image(&mut app, 1280, 720);
        testing::draw_into(&mut app, &image);
        testing::frame(&mut app, Seconds(0.0));
        let (projection, transform) = app
            .world_mut()
            .query_filtered::<(&Projection, &GlobalTransform), With<PlayerCamera>>()
            .single(app.world())
            .expect("the scene has one camera");
        let clip_from_world = projection.get_clip_from_view() * transform.to_matrix().inverse();
        let viewpoint = *app.world().resource::<Viewpoint>();
        let sim = app.world().resource::<Simulation>();
        let eye = DVec3::from_array(avatar::eye(sim.avatar()));
        // half the screen is one in the clip's own units, so this is in half-screens
        let on_screen = |p: [f64; 3]| -> Option<DVec2> {
            let clip = clip_from_world * viewpoint.local(p).extend(1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let ndc = clip.xy() / clip.w;
            Some(DVec2::new(ndc.x as f64, ndc.y as f64) * 720.0 / 2.0)
        };
        let ahead = DVec3::from_array(mat3mul(&sim.avatar().m, &[0.0, 0.0, -1.0]));
        let up = DVec3::from_array(mat3mul(&sim.avatar().m, &[0.0, 1.0, 0.0]));
        for down in [0.0, 0.01, 0.05, 0.2, 0.6, 1.2] {
            let dir = (ahead - up * down).normalize();
            let Some(hit) = aim::cast(eye, dir, &sim.drum) else {
                continue;
            };
            let (radius, lift) = (BRUSH_SIZE.0 as f64, 0.02);
            let Some(crosshair) = on_screen((eye + dir * eye.distance(hit.point)).to_array())
            else {
                continue;
            };
            let at = on_screen(hit.point.to_array()).expect("the aim point is in front");
            let off = (at - crosshair).length();
            assert!(
                off < 2.0,
                "on a ring of radius {radius_m} m, looking {down} down, the crosshair points {off} pixels from where it is drawn",
                radius_m = ring.radius.0
            );
            let outline = aim::outline(&sim.drum, hit, radius, lift);
            let seen: Vec<DVec2> = outline
                .iter()
                .filter_map(|p| on_screen(p.to_array()))
                .collect();
            let middle = seen.iter().fold(DVec2::ZERO, |a, b| a + *b) / seen.len() as f64;
            let spread = seen
                .iter()
                .map(|p| (*p - middle).length())
                .fold(0.0, f64::max);
            assert!(
                (middle - at).length() < spread.max(2.0),
                "on a ring of radius {radius_m} m, looking {down} down, the marker is drawn {} pixels from the point it outlines, and is only {spread} pixels across",
                (middle - at).length(),
                radius_m = ring.radius.0
            );
        }
    }
}

/// Whatever size the ring is, the mouse works what the crosshair rests on the same way: the
/// brush, a few metres across on every ring, raises the ground under the crosshair as fast,
/// and the hose puts the same water down where it points, as fine and as many particles a
/// second.
#[test]
fn the_crosshair_works_the_ground_alike_on_a_ring_of_any_size() {
    let frame = 1.0 / 60.0;
    let mut mounds = Vec::new();
    for radius in [10.5f32, 5_000.0, 1e7] {
        let ring = Ring {
            radius: Metres(radius),
            half_width: Metres((radius * 0.1).max(6.0)),
        };
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
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            let down = [(-0.15f64).sin(), 0.0, 0.0, (-0.15f64).cos()];
            let (p, q) = (sim.avatar().p, quat_mul(&sim.avatar().q, &down));
            sim.avatar_mut().place(p, q);
            sim.gyros = Gyros::holding(sim.avatar());
        }
        testing::run(&mut app, Seconds(frame));
        let first = app
            .world()
            .resource::<Aim>()
            .target
            .expect("the crosshair rests on the ground");
        let sim = app.world().resource::<Simulation>();
        let distance = first
            .point
            .distance(DVec3::from_array(avatar::eye(sim.avatar())));
        assert!(
            distance < 10.0,
            "on a ring of radius {radius} m the crosshair rests {distance} m off"
        );
        let before = sim.drum.landscape.max_height();
        for _ in 0..60 {
            let target = app.world().resource::<Aim>().target.expect("aimed");
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.sculpt(
                target.point.to_array(),
                BRUSH_SIZE.0 as f64,
                (BRUSH_RATE.0 * frame) as f64,
            );
            testing::run(&mut app, Seconds(frame));
        }
        let sim = app.world().resource::<Simulation>();
        let mound = (sim.drum.landscape.max_height() - before) as f64;
        mounds.push(mound);
        let under = first.point.to_array();
        assert!(
            sim.drum.ground(under) > GROUND_DEPTH.0 as f64 + 0.3,
            "on a ring of radius {radius} m the ground under the crosshair stands {} m",
            sim.drum.ground(under)
        );
        let target = app.world().resource::<Aim>().target.expect("aimed");
        let at = target.point + target.normal * INJECT_DEPTH.0 as f64;
        let flow = app.world().resource::<Settings>().flow.0;
        let count = (flow * 0.5 / Resolution::FINEST.litres_per_particle().0).round() as u32;
        let added = app
            .world_mut()
            .resource_scope(|world, mut fluid: Mut<Fluid>| {
                world
                    .resource_mut::<Simulation>()
                    .inject(&mut fluid, at.to_array(), count)
            });
        testing::run(&mut app, Seconds(frame));
        assert_eq!(added, count, "on a ring of radius {radius} m");
        assert_eq!(
            app.world().resource::<Fluid>().resolution(),
            Resolution::FINEST,
            "on a ring of radius {radius} m"
        );
        let particles = testing::particles(&mut app);
        let sim = app.world().resource::<Simulation>();
        let near = particles
            .iter()
            .filter(|p| DVec3::from_array(sim.drum.from_water(p.position)).distance(at) < 4.0)
            .count();
        assert!(
            near * 10 >= particles.len() * 9,
            "on a ring of radius {radius} m only {near} of {} particles are where the hose points",
            particles.len()
        );
    }
    let smallest = mounds[0];
    for (radius, mound) in [10.5, 5_000.0, 1e7].iter().zip(&mounds) {
        assert!(
            smallest > 0.3 && (mound - smallest).abs() < 0.1 * smallest,
            "a second of the brush raised {mound} m on a ring of radius {radius} m, {smallest} m on the smallest"
        );
    }
}

/// The crosshair finds the ground wherever it rests, however far off that is, on a ring of any
/// size: on a mound in front of the eye, or on the bare ground far round the ring past it, and
/// always on the surface, facing the eye.
#[test]
fn the_crosshair_finds_the_ground_near_and_far_on_a_ring_of_any_size() {
    for radius in [10.5f32, 5_000.0, 1e7, 1e16] {
        let ring = Ring {
            radius: Metres(radius),
            half_width: Metres((radius * 0.1).max(6.0)),
        };
        let mut drum = Drum::new(ring);
        let r = radius as f64;
        let mound = drum.wall_point(4.0 / r, 0.5);
        for _ in 0..10 {
            drum.sculpt(mound, 1.5, 0.3);
        }
        let eye = DVec3::new(-(GROUND_DEPTH.0 as f64 + 1.6), 0.0, 0.0);
        for (dir, what) in [
            (DVec3::new(0.25, 0.1, 1.0).normalize(), "the mound"),
            (DVec3::new(0.0, 0.0, 1.0), "the far ground"),
            (
                DVec3::new(0.35, -1.0, 0.1).normalize(),
                "the ground along the axis",
            ),
        ] {
            let hit = aim::cast(eye, dir, &drum).unwrap_or_else(|| {
                panic!("on a ring of radius {r} m nothing is hit looking at {what}")
            });
            let p = hit.point.to_array();
            let over = drum.height_above_glass(p) - drum.ground(p);
            assert!(
                over.abs() < 1e-3 && hit.normal.dot(dir) < 0.0,
                "on a ring of radius {r} m looking at {what} the crosshair rests {over} m off the \
                 ground at {:?}, facing {:?}",
                hit.point,
                hit.normal
            );
            if what == "the mound" {
                assert!(
                    drum.ground(p) > GROUND_DEPTH.0 as f64 + 0.3,
                    "on a ring of radius {r} m the crosshair missed the mound"
                );
            }
        }
    }
}
