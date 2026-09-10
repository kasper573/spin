//! What the viewer sees of the ring from afar.
use game::core::units::{Metres, Seconds};
use game::systems::drum::Ring;
use game::systems::player::Player;
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

/// A ghost far outside a large ring, looking back at it from off its axis, sees the ring: it
/// is never hidden behind anything, however far away it is, until it is smaller than a pixel.
/// It may be showing its night side, lit faintly through the glass by the light bounced round
/// inside it, and its rim thins below a pixel, so the ring only has to tint its pixels with
/// its own colours, green ground, brown dirt and blue water, against the near-white stars.
#[test]
fn the_ring_is_seen_from_far_away() {
    let ring = Ring {
        radius: Metres(245.0),
        half_width: Metres(6.0),
    };
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.collisions = false;
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(0.1));
    let (width, height) = (160u32, 90u32);
    let fov = 60f64.to_radians();
    for distance in [1500.0, 20000.0] {
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            sim.avatar_mut().solid = false;
            let slant = distance / 2f64.sqrt();
            let eye = sim.drum.from_water([slant, slant, 0.0]);
            let axis = sim.drum.from_water([0.0, 0.0, 0.0]);
            Player.teleport(&mut sim, eye, axis);
        }
        let image = testing::render_to_image(&mut app, width, height);
        testing::run(&mut app, Seconds(0.1));
        let bytes = testing::capture(&mut app, &image);
        let ring_pixels = bytes
            .chunks_exact(4)
            .filter(|p| {
                let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                let brightest = r.max(g).max(b);
                brightest > 3 && brightest - r.min(g).min(b) > brightest / 3
            })
            .count();
        let across =
            2.0 * ring.radius.0 as f64 / distance * (width as f64 / 2.0) / (fov / 2.0).tan();
        assert!(
            ring_pixels as f64 > across,
            "from {distance} m only {ring_pixels} pixels show the ring, which is {across} pixels across"
        );
    }
}
