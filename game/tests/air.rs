//! The air the ring holds, as a medium and as what it makes the ring look like.
//!
//! Nothing here asks the renderer for a picture that was aimed at. Each of these is a law the
//! air obeys, checked either on the air itself or on a frame drawn through air made thick
//! enough, or spun hard enough, for the law to show. A habitat's air over thirty metres does
//! almost nothing to a picture; that is what the same laws say it should do, and the ones that
//! take a planet's worth of air to see are made to show by giving the ring a planet's worth.
use bevy::prelude::*;
use game::core::avatar::{self, Gyros};
use game::core::math::{cross, norm, quat_from_basis, quat_rotate};
use game::core::units::{Kelvin, Metres, Pascals, Radians, RadiansPerSecond, Seconds};
use game::systems::air::{Air, CHANNELS, Suspension};
use game::systems::settings::{Dial, Settings};
use game::systems::sim::Simulation;
use game::systems::testing;

const RED: usize = 0;
const BLUE: usize = 2;

/// Scattering off the molecules goes as the inverse fourth power of the wavelength. The eye's
/// blue is answered nearly three times as readily as its red, which is why a sky is blue, and
/// the little that is over comes from the molecules answering blue light a shade more strongly
/// than red and from their not being spheres.
#[test]
fn the_molecules_take_blue_hardest() {
    let air = Air::default();
    let measured = air.rayleigh(CHANNELS[BLUE]) / air.rayleigh(CHANNELS[RED]);
    let power = (CHANNELS[RED].0 / CHANNELS[BLUE].0).powi(4);
    assert!(
        measured > power && measured < power * 1.1,
        "blue is scattered {measured:.3} times as hard as red, against {power:.3} for the fourth power alone"
    );
}

/// What a metre of air scatters follows how many molecules stand in it, so it follows the
/// pressure and runs against the temperature; how much it slows light follows the same count.
#[test]
fn the_air_scatters_by_how_much_of_it_there_is() {
    let air = Air::default();
    let mut pressed = air;
    pressed.pressure = Pascals(air.pressure.0 * 2.0);
    let mut warmed = air;
    warmed.temperature = Kelvin(air.temperature.0 * 2.0);
    let green = CHANNELS[1];
    let twice = pressed.rayleigh(green) / air.rayleigh(green);
    let half = warmed.rayleigh(green) / air.rayleigh(green);
    assert!(
        (twice - 2.0).abs() < 1e-6,
        "twice the pressure scattered {twice:.4} times as much"
    );
    assert!(
        (half - 0.5).abs() < 1e-6,
        "twice as warm scattered {half:.4} times as much"
    );
    let slowing = pressed.slowing(green) / air.slowing(green);
    assert!(
        (slowing - 2.0).abs() < 1e-6,
        "twice the pressure slowed light {slowing:.4} times as much"
    );
}

/// A grain much narrower than the wavelength answers light as a molecule does: it takes blue
/// four times over and throws it as readily backward as forward. A grain much wider takes
/// every colour alike and throws nearly all of it forward, which is why fog, cloud and spray
/// are white and why a beam shows in dusty air.
#[test]
fn grains_take_every_colour_alike_once_they_are_wide_enough() {
    let air = Air::default();
    let with = |radius: f64| {
        let mut a = air;
        a.carries = Suspension {
            radius: Metres(radius as f32),
            ..Suspension::DUST
        };
        a
    };
    let fine = with(5e-9);
    let ratio = fine.mie(CHANNELS[BLUE]) / fine.mie(CHANNELS[RED]);
    let power = (CHANNELS[RED].0 / CHANNELS[BLUE].0).powi(4);
    assert!(
        (ratio - power).abs() < 0.05 * power,
        "grains a hundredth of a wavelength wide took blue {ratio:.3} times as hard as red, against {power:.3} for a molecule"
    );
    assert!(
        fine.carries.lobe(CHANNELS[1]) < 0.05,
        "they threw it forward by {:.3}",
        fine.carries.lobe(CHANNELS[1])
    );
    let wide = with(1e-5);
    let ratio = wide.mie(CHANNELS[BLUE]) / wide.mie(CHANNELS[RED]);
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "grains twenty wavelengths wide took blue {ratio:.3} times as hard as red"
    );
    assert!(
        wide.carries.lobe(CHANNELS[1]) > 0.7,
        "they threw it forward by only {:.3}",
        wide.carries.lobe(CHANNELS[1])
    );
    // and a wide grain stops about twice its own shadow's worth, which is what a sphere does
    let taken = wide.carries.scattering(CHANNELS[1]).taken;
    assert!(
        (taken - 2.0).abs() < 0.3,
        "a wide grain stopped {taken:.3} times its own area's worth"
    );
}

/// Grains that swallow light take more out of a ray than they turn aside, so smoke darkens
/// what is behind it where dust only whitens it.
#[test]
fn grains_that_swallow_light_take_more_than_they_turn() {
    let air = Air {
        carries: Suspension::SMOKE,
        ..Air::default()
    };
    let green = CHANNELS[1];
    let turned = air.mie(green);
    let taken = air.mie_extinction(green);
    assert!(
        turned < 0.6 * taken,
        "soot turned {turned:.3e} of the {taken:.3e} it took"
    );
}

/// A ring holds its air by spinning, so the air thins toward the axis exactly as a planet's
/// thins with height, and by the same law. A ring that does not turn holds its air nowhere in
/// particular; one that turns hard holds it in a shell against the rim.
#[test]
fn the_air_thins_toward_the_axis_as_the_ring_spins() {
    let air = Air::default();
    let at_axis =
        |spin: f32, radius: f64| (air.thinning(RadiansPerSecond(spin)) * -(radius * radius)).exp();
    assert_eq!(
        at_axis(0.0, 15.0),
        1.0,
        "air thinned in a ring that does not turn"
    );
    let gentle = at_axis(1.0, 15.0);
    let hard = at_axis(40.0, 15.0);
    assert!(
        gentle > 0.99 && gentle < 1.0,
        "a ring turning once in six seconds thinned its air to {gentle:.4} at the axis"
    );
    assert!(
        hard < 0.2,
        "a ring turning forty times a second thinned its air only to {hard:.4} at the axis"
    );
    // and warmer air stands higher, so it thins more slowly
    let mut warm = air;
    warm.temperature = Kelvin(air.temperature.0 * 2.0);
    let ratio = warm.thinning(RadiansPerSecond(1.0)) / air.thinning(RadiansPerSecond(1.0));
    assert!(
        (ratio - 0.5).abs() < 1e-9,
        "twice as warm thinned {ratio:.4} as fast"
    );
}

/// Air slows blue more than red. That is what spreads a ray crossing a density gradient into
/// its colours, and it is the whole of why a setting sun can flash green.
#[test]
fn the_air_slows_blue_more_than_red() {
    let air = Air::default();
    let red = air.slowing(CHANNELS[RED]);
    let blue = air.slowing(CHANNELS[BLUE]);
    assert!(
        blue > red,
        "blue was slowed {blue:.6e} against red's {red:.6e}"
    );
    let spread = (blue - red) / red;
    assert!(
        spread > 0.005 && spread < 0.03,
        "blue was slowed {:.3}% more than red",
        spread * 100.0
    );
}

/// A frame drawn through the air, with the eye looking across the ring from its rim.
fn seen_through(air: Air, spin: f32) -> (Vec<u8>, u32, u32) {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 180;
    let mut app = testing::headless();
    app.insert_resource(air);
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Spin.set(&mut settings, spin);
    }
    app.world_mut().resource_mut::<Simulation>().drum.spin = RadiansPerSecond(spin);
    testing::run(&mut app, Seconds(0.2));
    let image = testing::render_to_image(&mut app, WIDTH, HEIGHT);
    testing::watch(&mut app, Seconds(1.0));
    testing::frame(&mut app, Seconds(0.0));
    (testing::capture(&mut app, &image), WIDTH, HEIGHT)
}

/// The mean of each colour over a frame.
fn colours(pixels: &[u8]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let n = (pixels.len() / 4) as f64;
    for p in pixels.chunks_exact(4) {
        for c in 0..3 {
            sum[c] += p[c] as f64 / n;
        }
    }
    sum
}

/// Air deep enough turns the ring blue, and turns it blue because of the fourth power. What
/// makes a sky is the number of molecules along the way, and thirty metres of a habitat's air
/// holds about as many as a two-hundredth of the way straight up through a planet's; given two
/// hundred atmospheres of it the ring is looked at through as much air as a zenith, and the
/// blue of it rises over the red the way a zenith's does. Three times that and it rises again:
/// what is checked is that it follows the air along the way, not that it reaches a number.
#[test]
#[ignore = "wants a GPU"]
fn deep_enough_air_makes_a_sky() {
    let blue_over_red = |atmospheres: f64| {
        let air = Air {
            pressure: Pascals(101325.0 * atmospheres),
            ..Air::default()
        };
        let seen = colours(&seen_through(air, 1.0).0);
        seen[BLUE] / seen[RED].max(1e-9)
    };
    // an exposure is one number over the whole frame, so a ratio of colours sees through it
    let own = blue_over_red(1.0);
    let zenith = blue_over_red(200.0);
    let deeper = blue_over_red(600.0);
    assert!(
        zenith > own * 1.1,
        "a zenith's worth of air left the ring at {zenith:.3} blue over red against {own:.3} in its own air"
    );
    assert!(
        deeper > zenith * 1.1,
        "three zeniths' worth left it at {deeper:.3} against the one zenith's {zenith:.3}"
    );
}

/// The same air carrying grains the size of fog droplets whitens the ring instead of bluing
/// it, because a grain that wide takes every colour alike.
#[test]
#[ignore = "wants a GPU"]
fn a_fog_of_wide_grains_whitens_rather_than_blues() {
    let clear = Air {
        carries: Suspension::CLEAR,
        ..Air::default()
    };
    let foggy = Air {
        carries: Suspension::FOG,
        ..clear
    };
    let before = colours(&seen_through(clear, 1.0).0);
    let after = colours(&seen_through(foggy, 1.0).0);
    let was = before[BLUE] / before[RED].max(1e-9);
    let now = after[BLUE] / after[RED].max(1e-9);
    assert!(
        after.iter().sum::<f64>() > before.iter().sum::<f64>(),
        "the fog left the ring no brighter: {after:?} against {before:?}"
    );
    assert!(
        (now - was).abs() < 0.15 * was,
        "the fog turned the ring from {was:.3} blue over red to {now:.3}, rather than leaving its colour alone"
    );
}

/// Where the air has a gradient it bends what crosses it, and bends each colour of it by its
/// own amount, since it slows each by its own amount. A habitat's own air bends a ray crossing
/// it by a millionth of a degree, so the circumstance is forced: the ground is taken away so
/// that the ring is a glass shell with nothing in it but air, the air is made two hundred
/// times as deep, and the ring is spun hard enough to press that air into a shell against the
/// rim. The eye stands just inside the rim and looks along the axis, where the air it looks
/// through stands at one density the whole way and its gradient lies square across the ray.
///
/// What that does to the sky is what is looked at, and it is looked at against air that has no
/// gradient in it: the same ring, spun the same, holding air of the same density at the rim but
/// hot enough to stand even from rim to axis rather than stacked against the rim. The two hold
/// the same number of molecules along the ray, so they scatter the same and dim the same, and
/// the only thing that tells them apart is the gradient. Against that, the stacked air carries
/// the stars off their places, and carries each colour of them its own distance, so a star that
/// was white splits into coloured ones.
#[test]
#[ignore = "wants a GPU"]
fn a_gradient_carries_the_sky_off_its_place_and_splits_its_colours() {
    let deep = Air {
        pressure: Pascals(101325.0 * 200.0),
        ..Air::default()
    };
    let even = out_through_the_axis(deep, 0.0);
    let stacked = out_through_the_axis(deep, HARD_SPIN);

    let moved = moved_share(&even, &stacked);
    assert!(
        moved > 0.05,
        "the gradient carried only {:.2}% of the sky off where even air puts it",
        moved * 100.0
    );

    // a gradient bends what crosses it and makes nothing: the stacked air holds less along the
    // ray than the even air does, having pressed most of it out of the way, so whatever it
    // carries off its place it cannot leave the sky brighter than the even air leaves it
    let (dim, bent) = (brightness(&even), brightness(&stacked));
    assert!(
        bent <= dim,
        "the gradient left the sky at {bent:.2} against {dim:.2} without it, having made light rather than bent it"
    );

    let (white, split) = (split_share(&even), split_share(&stacked));
    assert!(
        split > white * 1.5,
        "the gradient set the sky's colours {split:.4} apart, against {white:.4} without it"
    );
}

/// The spin that presses a habitat's air into a shell a fraction of the ring thick.
const HARD_SPIN: f32 = 300.0;
const FRAME_WIDTH: u32 = 960;
const FRAME_HEIGHT: u32 = 540;

/// How bright a frame stands on the whole.
fn brightness(pixels: &[u8]) -> f64 {
    let sum: f64 = pixels
        .chunks_exact(4)
        .map(|p| (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0)
        .sum();
    sum / (pixels.len() / 4) as f64
}

/// The share of the frame that stands somewhere else in one than in the other.
fn moved_share(before: &[u8], after: &[u8]) -> f64 {
    let mut moved = 0usize;
    for (a, b) in before.chunks_exact(4).zip(after.chunks_exact(4)) {
        if (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 24) {
            moved += 1;
        }
    }
    moved as f64 / (before.len() / 4) as f64
}

/// How far the sharp detail of a frame disagrees between its red and its blue. A star is a
/// spike against the sky round it, and a white star is the same spike in every colour; one
/// whose colours have been carried to different places is a spike in one colour where there is
/// none in the other. A wash of colour over the whole sky, however blue, is not sharp detail
/// and counts for nothing here.
fn split_share(pixels: &[u8]) -> f64 {
    let (width, height) = (FRAME_WIDTH as usize, FRAME_HEIGHT as usize);
    let at = |x: usize, y: usize, c: usize| pixels[4 * (y * width + x) + c] as f64;
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sharp = [0.0f64; 3];
            for (c, s) in sharp.iter_mut().enumerate() {
                let mut around = 0.0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        around += at(x + dx - 1, y + dy - 1, c);
                    }
                }
                *s = (at(x, y, c) - around / 9.0).max(0.0);
            }
            let (red, blue) = (sharp[RED], sharp[BLUE]);
            if red + blue < 12.0 {
                continue;
            }
            sum += (red - blue).abs() / (red + blue);
            count += 1.0;
        }
    }
    sum / count.max(1.0)
}

/// The sky seen from just inside the rim, looking along the ring's axis and out through a cap,
/// with the ground taken away so that nothing but air stands between the eye and the stars.
fn out_through_the_axis(air: Air, spin: f32) -> Vec<u8> {
    let (width, height) = (FRAME_WIDTH, FRAME_HEIGHT);
    let mut app = testing::headless();
    app.insert_resource(air);
    app.world_mut().resource_mut::<Settings>().collisions = false;
    app.world_mut()
        .resource_mut::<Simulation>()
        .drum
        .landscape
        .flatten(Metres(0.0));
    testing::run(&mut app, Seconds(0.1));
    let image = testing::render_to_image(&mut app, width, height);
    // the eye is stood up and the exposure settled with the ring at rest, alike for every air
    for _ in 0..10 {
        along_the_axis(&mut app);
        testing::watch(&mut app, Seconds(0.1));
    }
    along_the_axis(&mut app);
    // the spin is then set without letting the wheel turn under it and without letting the eye
    // adapt again, so the one thing that tells the frames apart is how the air stands in them
    app.world_mut().resource_mut::<Simulation>().drum.spin = RadiansPerSecond(spin);
    testing::frame(&mut app, Seconds(0.0));
    testing::capture(&mut app, &image)
}

/// Stand the viewer as a ghost a hand's breadth inside the rim, facing along the axis, with
/// the wheel held where it started so that spinning it changes the air it holds and not where
/// the stars stand.
fn along_the_axis(app: &mut App) {
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.drum.angle = Radians(sim.drum.site.phi);
    sim.avatar_mut().solid = false;
    let on = sim.drum.wall_point(0.0, -sim.drum.site.y);
    let (_, out) = sim.drum.depth_and_outward(on);
    let lift = 0.02;
    let eye = [
        on[0] - out[0] * lift,
        on[1] - out[1] * lift,
        on[2] - out[2] * lift,
    ];
    let back = [0.0, -1.0, 0.0];
    let right = cross(&out.map(|c| -c), &back);
    let len = norm(&right);
    let right = right.map(|c| c / len);
    let up = cross(&back, &right);
    let turn = quat_from_basis(&right, &up, &back);
    let head = quat_rotate(&turn, &avatar::eye_offset());
    sim.avatar_mut()
        .place([eye[0] - head[0], eye[1] - head[1], eye[2] - head[2]], turn);
    sim.gyros = Gyros::holding(sim.avatar());
}
