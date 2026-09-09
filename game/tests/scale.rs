//! The ring can be any size: the water, the avatar, the rendering and the saves all hold at
//! sizes from metres to light years, and the water flows the same at every one of them.
use bevy::prelude::*;
use game::core::fluid::{Fluid, Resolution, SPACINGS_FROM_AXIS};
use game::core::math::norm;
use game::core::units::{EARTH_GRAVITY, Metres, RadiansPerSecond, Seconds};
use game::systems::drum::Ring;
use game::systems::persistence::{Snapshot, apply, snapshot};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

fn ring(radius: f64) -> Ring {
    Ring {
        radius: Metres(radius as f32),
        half_width: Metres((radius * 0.2).max(6.0) as f32),
    }
}

/// The initial ring world at this size, settled for a couple of seconds.
fn world(ring: Ring) -> testing::Headless {
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
        settings.equalize_thrust();
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(2.0));
    app
}

fn weight_in_g(sim: &Simulation) -> f64 {
    let body = sim.avatar();
    body.ground
        .map(|g| g.support.0 * body.inv_m / EARTH_GRAVITY.0)
        .unwrap_or(0.0)
}

/// Inject water around a point of the water's frame, about the drum's centre.
fn inject(app: &mut App, centre: [f64; 3], count: u32) -> u32 {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let sim = world.resource::<Simulation>();
            sim.inject(&mut fluid, sim.drum.from_water(centre), count)
        })
}

/// How high the avatar's centre stands above the glass.
fn altitude(sim: &Simulation) -> f64 {
    sim.drum.height_above_glass(sim.avatar().p)
}

/// Run the water for this many of its own steps.
fn run_water_steps(app: &mut App, steps: u32) {
    let step = app.world().resource::<Fluid>().step();
    testing::run(app, Seconds(step.0 * steps as f32));
}

#[test]
fn the_avatar_stands_under_one_g_on_any_size_of_ring() {
    for radius in [1e3, 1e6, 1e9, 1e12, 1e15] {
        let app = world(ring(radius));
        let sim = app.world().resource::<Simulation>();
        let g = weight_in_g(sim);
        assert!(
            sim.avatar().ground.is_some() && (g - 1.0).abs() < 0.05,
            "on a ring of radius {radius} m the avatar weighs {g} g"
        );
        let p = sim.avatar().p;
        assert!(
            (altitude(sim) - 0.8).abs() < 0.3 && p.iter().all(|x| x.is_finite()),
            "on a ring of radius {radius} m the avatar is at {p:?}"
        );
    }
}

/// The water's resolution follows the ring, so a ring of any size is the same number of
/// spacings across, and water dropped in a huge ring falls to its floor and grows a surface.
#[test]
fn water_in_a_huge_ring_settles_on_its_floor_and_is_meshed() {
    for radius in [1e5, 1e8] {
        let mut app = world(ring(radius));
        let spacing = app.world().resource::<Fluid>().resolution().spacing.0 as f64;
        assert!(
            (spacing - radius / SPACINGS_FROM_AXIS as f64).abs() < 1e-3 * spacing,
            "a ring of radius {radius} m got water spaced {spacing} m"
        );
        let above = radius - 3.0 * spacing;
        let added = inject(&mut app, [above, 0.0, 0.0], 500);
        assert_eq!(added, 500);
        run_water_steps(&mut app, 300);
        let particles = testing::particles(&mut app);
        assert_eq!(particles.len(), 500);
        let settled = particles
            .iter()
            .filter(|p| {
                let r = (p.position[0].powi(2) + p.position[2].powi(2)).sqrt();
                p.position.iter().all(|x| x.is_finite()) && r > radius - 2.5 * spacing
            })
            .count();
        assert!(
            settled >= 425,
            "{settled} of 500 particles settled on the floor of a ring of radius {radius} m"
        );
        let demand = testing::surface_demand(&mut app);
        assert!(demand.vertices > 100, "no surface: {demand:?}");
    }
}

/// The water is the same simulation at every scale: a ring a million times bigger, holding
/// its water at the same number of spacings, sees that water go the same way in the same
/// number of its own steps, which take a thousand times longer.
#[test]
fn water_flows_the_same_at_any_scale() {
    let mut settled = Vec::new();
    for scale in [1.0, 1e6] {
        let radius = (Resolution::FINEST.spacing.0 * SPACINGS_FROM_AXIS) as f64 * scale;
        let ring = Ring {
            radius: Metres(radius as f32),
            half_width: Metres((radius * 0.3) as f32),
        };
        let mut app = world(ring);
        {
            let mut sim = app.world_mut().resource_mut::<Simulation>();
            let spin = RadiansPerSecond((EARTH_GRAVITY.0 / radius).sqrt() as f32);
            sim.drum.spin = spin;
            sim.drum.target_spin = spin;
            sim.drum.landscape.flatten(Metres(0.0));
            sim.avatar_mut().solid = false;
            let mut settings = app.world_mut().resource_mut::<Settings>();
            settings.spin = spin;
            settings.collisions = false;
        }
        testing::run(&mut app, Seconds(0.1));
        let spacing = app.world().resource::<Fluid>().resolution().spacing.0 as f64;
        assert!(
            (spacing / Resolution::FINEST.spacing.0 as f64 - scale).abs() < 1e-3 * scale,
            "spacing {spacing} at scale {scale}"
        );
        inject(&mut app, [radius * 0.9, 0.0, 0.0], 400);
        run_water_steps(&mut app, 90);
        let particles = testing::particles(&mut app);
        assert_eq!(particles.len(), 400);
        let time = app.world().resource::<Fluid>().resolution().time();
        let mean = |f: &dyn Fn(&game::core::fluid::Particle) -> f64| {
            particles.iter().map(f).sum::<f64>() / particles.len() as f64
        };
        settled.push((
            mean(&|p| (p.position[0].powi(2) + p.position[2].powi(2)).sqrt() / radius),
            mean(&|p| p.position[1].abs() / radius),
            mean(&|p| norm(&p.velocity) * time / spacing),
        ));
    }
    let (small, big) = (settled[0], settled[1]);
    assert!(
        (small.0 - big.0).abs() < 0.01 && (small.1 - big.1).abs() < 0.01,
        "the water lies differently: {small:?} against {big:?}"
    );
    assert!(
        (small.2 - big.2).abs() < 0.2 * small.2.max(0.05),
        "the water moves differently: {small:?} against {big:?}"
    );
}

#[test]
fn a_huge_ring_survives_a_save() {
    let mut app = world(ring(1e12));
    inject(&mut app, [1e12 - 2e9, 0.0, 0.0], 50);
    run_water_steps(&mut app, 2);
    testing::particles(&mut app);
    let world = app.world_mut();
    let json = serde_json::to_string(&snapshot(
        world.resource::<Settings>(),
        world.resource::<Simulation>(),
        world.resource::<Fluid>(),
    ))
    .unwrap();
    let restored: Snapshot = serde_json::from_str(&json).unwrap();
    let mut settings = Settings::default();
    let mut sim = Simulation::default();
    let mut fluid = world.resource_mut::<Fluid>();
    apply(&restored, &mut settings, &mut sim, &mut fluid);
    assert_eq!(sim.drum.ring.radius, Metres(1e12));
    assert_eq!(fluid.len(), 50);
    let p = sim.avatar().p;
    assert!(
        (altitude(&sim) - 0.8).abs() < 0.3,
        "the avatar came back at {p:?}"
    );
}

/// A frame of a ring of any size, up to one a light year across, draws the ring: the ground
/// fills the bottom of the view in front of the standing avatar.
#[test]
fn a_ring_of_any_size_is_drawn() {
    for radius in [10.5, 1e7, 1e16] {
        let mut app = world(ring(radius));
        let (width, height) = (160, 90);
        let image = testing::render_to_image(&mut app, width, height);
        testing::run(&mut app, Seconds(0.1));
        let bytes = testing::capture(&mut app, &image);
        let bottom = bytes[(width * height * 3) as usize..].chunks_exact(4);
        let total = bottom.len();
        let ground = bottom
            .filter(|p| p[1] > 30 && p[1] > p[0] && p[1] > p[2])
            .count();
        assert!(
            ground * 20 > total * 19,
            "on a ring of radius {radius} m only {ground} of {total} pixels at the bottom show the ground"
        );
    }
}
