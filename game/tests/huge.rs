//! What a ring kilometres across looks like. Everything about the drawing is worked out in
//! single precision from numbers as big as the ring, and the air it holds is deep enough at
//! that size to stand between the eye and everything in it, so a ring ten kilometres across is
//! where both of those show. Every frame is written to `target/tmp/huge/` to be looked at too.
use std::fs;
use std::path::PathBuf;

use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::math::{cross, norm, quat_from_basis, quat_rotate};
use game::core::units::{Metres, Radians, Seconds};
use game::systems::air::Air;
use game::systems::drum::{Place, Ring, Round};
use game::systems::scene::SUN_DIRECTION;
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing::{self, Headless};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn spot(sim: &Simulation, [arc, y, height]: [f64; 3]) -> [f64; 3] {
    let drum = &sim.drum;
    let grid = drum.landscape.grid();
    let at = Place {
        round: Round::default().on(arc, grid),
        along: y,
    };
    let ground = drum.landscape.sample(at).0;
    let turn = drum.site.round.arc_to(at.round, grid) / drum.ring.radius.0 as f64;
    let on = drum.wall_point(turn, y - drum.site.y);
    let (_, out) = drum.depth_and_outward(on);
    let lift = ground + height;
    [
        on[0] - out[0] * lift,
        on[1] - out[1] * lift,
        on[2] - out[2] * lift,
    ]
}

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

struct Shot {
    app: Headless,
    image: Handle<Image>,
    glimpse: Handle<Image>,
    dir: PathBuf,
}

impl Shot {
    fn new(ring: Ring) -> Shot {
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
        let glimpse = testing::render_to_image(&mut app, WIDTH / 8, HEIGHT / 8);
        let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("huge");
        fs::create_dir_all(&dir).expect("create the folder");
        Shot {
            app,
            image,
            glimpse,
            dir,
        }
    }

    fn view(&mut self, name: &str, eye: [f64; 3], at: [f64; 3], daylight: bool) -> Vec<u8> {
        testing::draw_into(&mut self.app, &self.glimpse);
        for _ in 0..15 {
            look(&mut self.app, eye, at);
            face_the_sun(&mut self.app, daylight);
            testing::watch(&mut self.app, Seconds(1.0 / 15.0));
        }
        look(&mut self.app, eye, at);
        face_the_sun(&mut self.app, daylight);
        testing::draw_into(&mut self.app, &self.image);
        look(&mut self.app, eye, at);
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
}

/// The share of a frame that stands wildly apart from what is around it: a pixel that differs
/// from every one of its neighbours by more than an eye can miss, somewhere the frame is lit
/// at all, so that a star alone against the dark is not counted as one. A picture of anything
/// made of surfaces is smooth almost everywhere, whatever size those surfaces are, so this
/// catches a decision made in single precision about a surface kilometres off coming out
/// differently from one pixel to the next.
fn speckle_share(pixels: &[u8]) -> f64 {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let at = |x: usize, y: usize, c: usize| pixels[4 * (y * width + x) + c] as i32;
    let mut speckles = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let apart = (0..3)
                .map(|c| {
                    let here = at(x, y, c);
                    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .map(|(dx, dy)| {
                            (here
                                - at(
                                    x.wrapping_add(*dx as usize),
                                    y.wrapping_add(*dy as usize),
                                    c,
                                ))
                            .abs()
                        })
                        .min()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            let lit = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                .iter()
                .map(|(dx, dy)| {
                    (0..3)
                        .map(|c| {
                            at(
                                x.wrapping_add(*dx as usize),
                                y.wrapping_add(*dy as usize),
                                c,
                            )
                        })
                        .max()
                        .unwrap_or(0)
                })
                .min()
                .unwrap_or(0);
            if apart > 24 && lit > 40 {
                speckles += 1;
            }
        }
    }
    speckles as f64 / (width * height) as f64
}

/// What the air did to a frame, as a picture in its own right: what it took out of each pixel.
fn what_the_air_did(with: &[u8], without: &[u8]) -> Vec<u8> {
    with.chunks_exact(4)
        .zip(without.chunks_exact(4))
        .flat_map(|(a, b)| {
            [
                (a[0] as i32 - b[0] as i32).unsigned_abs().min(255) as u8,
                (a[1] as i32 - b[1] as i32).unsigned_abs().min(255) as u8,
                (a[2] as i32 - b[2] as i32).unsigned_abs().min(255) as u8,
                255,
            ]
        })
        .collect()
}

/// The share of a frame blown to white in every colour at once.
fn blown_share(pixels: &[u8]) -> f64 {
    let blown = pixels
        .chunks_exact(4)
        .filter(|p| p[0] >= 250 && p[1] >= 250 && p[2] >= 250)
        .count();
    blown as f64 / (pixels.len() / 4) as f64
}

/// A ring ten kilometres across is drawn as smoothly as a small one. Everything about the
/// rendering is worked out in single precision from numbers as big as the ring, so a ring this
/// size is where a lost digit shows: as pixels that stand apart from their neighbours where a
/// decision went one way for one of them and the other way for the next, or as pieces of the
/// picture blown to white where a division by nothing got into the shading. Neither may happen
/// anywhere the ring is looked at from.
#[test]
#[ignore = "wants a GPU"]
fn a_ring_kilometres_across_is_drawn_without_holes() {
    let ring = Ring {
        radius: Metres(5000.0),
        half_width: Metres(500.0),
    };
    let mut shot = Shot::new(ring);
    testing::run(&mut shot.app, Seconds(2.0));
    let views: [(&str, [f64; 3], [f64; 3], bool); 6] = [
        ("along", [0.0, 0.0, 1.7], [4000.0, 0.0, 1.7], true),
        ("across", [0.0, 0.0, 1.7], [0.0, 0.0, 3000.0], true),
        ("cap", [0.0, 0.0, 1.7], [0.0, -500.0, 1.7], true),
        ("down", [0.0, 0.0, 40.0], [0.0, 0.0, 0.0], true),
        ("far_out", [0.0, 0.0, -8000.0], [0.0, 0.0, 0.0], true),
        ("night", [0.0, 0.0, 1.7], [4000.0, 0.0, 1.7], false),
    ];
    for (name, eye, at, daylight) in views {
        let pixels = shot.view(name, eye, at, daylight);
        let speckles = speckle_share(&pixels);
        let blown = blown_share(&pixels);
        assert!(
            speckles < 0.002,
            "looking {name}, {:.3}% of the frame stands apart from everything around it",
            speckles * 100.0
        );
        assert!(
            blown < 0.001,
            "looking {name}, {:.3}% of the frame is blown to white",
            blown * 100.0
        );
    }
}

/// Kilometres of air stand between the eye and the far side of a ring this size, and that much
/// of it is seen: it takes a good part of the blue out of everything across the ring and puts
/// its own light in front of it. How much of it a ray crosses is worked out from where the ray
/// sets out against the ring's own wall, which on a ring kilometres across is a difference of
/// numbers as big as the ring, so this is also where losing that difference would show — as a
/// ring whose air does nothing at all.
#[test]
#[ignore = "wants a GPU"]
fn the_air_of_a_ring_kilometres_across_is_seen_across_it() {
    let ring = Ring {
        radius: Metres(5000.0),
        half_width: Metres(500.0),
    };
    let views: [(&str, [f64; 3], [f64; 3]); 3] = [
        ("across", [0.0, 0.0, 1.7], [0.0, 0.0, 3000.0]),
        ("along", [0.0, 0.0, 1.7], [4000.0, 0.0, 1.7]),
        ("cap", [0.0, 0.0, 1.7], [0.0, -500.0, 1.7]),
    ];
    let mut shot = Shot::new(ring);
    testing::run(&mut shot.app, Seconds(2.0));
    for (name, eye, at) in views {
        // the same frame twice over, with the ring's air and with the ring emptied of it, so
        // that nothing but the air tells them apart
        let with = shot.view(name, eye, at, true);
        shot.app.insert_resource(Air {
            pressure: game::core::units::Pascals(0.0),
            ..Air::default()
        });
        let without = shot.view(&format!("vacuum_{name}"), eye, at, true);
        shot.app.insert_resource(Air::default());
        let did = what_the_air_did(&with, &without);
        let took = did.chunks_exact(4).map(|p| p[2] as f64).sum::<f64>() / (WIDTH * HEIGHT) as f64;
        assert!(
            took > 2.0,
            "looking {name}, the air took an average of {took:.2} out of 255 of the blue, which is nothing at all"
        );
    }
}
