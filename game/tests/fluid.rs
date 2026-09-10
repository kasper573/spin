use bevy::prelude::*;
use game::core::fluid::{
    Fluid, MAX_BLOCKS, MAX_INDICES, MAX_VERTICES, Particle, Resolution, grid_reach,
};
use game::core::math::norm;
use game::core::units::Metres;
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::DEFAULT_RING;
use game::systems::settings::{Dial, Settings};
use game::systems::sim::{SUBSTEP_RATE, Simulation};
use game::systems::testing;

/// Inject water around a point of the water's frame, about the drum's centre.
fn inject(app: &mut App, centre: [f64; 3], count: u32) -> u32 {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            let sim = world.resource::<Simulation>();
            sim.inject(&mut fluid, sim.drum.from_water(centre), count)
        })
}

fn set_spin(app: &mut App, spin: f32) {
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(spin);
    app.world_mut().resource_mut::<Simulation>().drum.spin = RadiansPerSecond(spin);
}

/// Where the water is: its centroid and the root mean square distance from it, in spacings,
/// and its mean speed in metres per second.
fn spread(app: &mut App) -> ([f64; 3], f64, f64) {
    let particles = testing::particles(app);
    let spacing = app.world().resource::<Fluid>().resolution().spacing.0 as f64;
    let n = particles.len() as f64;
    let mut centre = [0.0; 3];
    for p in &particles {
        for (c, x) in centre.iter_mut().zip(p.position) {
            *c += x / n;
        }
    }
    let rms = (particles
        .iter()
        .map(|p| {
            (0..3)
                .map(|a| (p.position[a] - centre[a]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n)
        .sqrt()
        / spacing;
    let speed = particles.iter().map(|p| norm(&p.velocity)).sum::<f64>() / n;
    (centre, rms, speed)
}

fn still(app: &mut App) {
    set_spin(app, 0.0);
    app.world_mut()
        .resource_mut::<Simulation>()
        .drum
        .target_spin = RadiansPerSecond(0.0);
}

/// Water put down in the air of a ring that is not spinning has nothing to fall toward: it
/// stays where it was put, at rest, as a blob at its rest spacing.
#[test]
fn water_placed_in_the_air_of_a_still_ring_stays_where_it_is_put() {
    let mut app = testing::headless();
    still(&mut app);
    testing::run(&mut app, Seconds(0.5));
    let at = [5.0, 0.0, 0.0];
    assert_eq!(inject(&mut app, at, 300), 300);
    testing::run(&mut app, Seconds(3.0));
    let (centre, rms, speed) = spread(&mut app);
    let moved = norm(&[centre[0] - at[0], centre[1] - at[1], centre[2] - at[2]]);
    assert!(
        moved < 0.3 && rms < 5.0 && speed < 0.3,
        "the water moved {moved} m, spread {rms} spacings and moves at {speed} m/s"
    );
}

/// Water put down onto water takes the free room around it rather than bursting out of it.
#[test]
fn water_placed_onto_water_settles_around_it() {
    let mut app = testing::headless();
    still(&mut app);
    testing::run(&mut app, Seconds(0.5));
    let at = [5.0, 0.0, 0.0];
    for _ in 0..10 {
        assert_eq!(inject(&mut app, at, 40), 40);
        testing::run(&mut app, Seconds(0.1));
    }
    testing::run(&mut app, Seconds(2.0));
    let (centre, rms, speed) = spread(&mut app);
    let moved = norm(&[centre[0] - at[0], centre[1] - at[1], centre[2] - at[2]]);
    assert_eq!(app.world().resource::<Fluid>().len(), 400);
    assert!(
        moved < 0.3 && rms < 6.0 && speed < 0.3,
        "the water moved {moved} m, spread {rms} spacings and moves at {speed} m/s"
    );
}

/// Spinning the ring up under water hanging still in it brings the water down onto the
/// floor: the ring's gravity is nothing but its spin.
#[test]
#[ignore = "wants a GPU"]
fn spinning_up_a_still_ring_brings_placed_water_down_to_the_floor() {
    let mut app = testing::headless();
    still(&mut app);
    testing::run(&mut app, Seconds(0.5));
    assert_eq!(inject(&mut app, [5.0, 0.0, 0.0], 300), 300);
    testing::run(&mut app, Seconds(1.0));
    {
        let spin = game::systems::sim::standing_spin(DEFAULT_RING);
        app.world_mut().resource_mut::<Settings>().spin = spin;
        app.world_mut()
            .resource_mut::<Simulation>()
            .drum
            .target_spin = spin;
    }
    testing::run(&mut app, Seconds(30.0));
    let particles = testing::particles(&mut app);
    let spacing = app.world().resource::<Fluid>().resolution().spacing.0 as f64;
    let floor = DEFAULT_RING.floor_radius().0 as f64;
    let down = particles
        .iter()
        .filter(|p| (p.position[0].powi(2) + p.position[2].powi(2)).sqrt() > floor - 2.5 * spacing)
        .count();
    assert!(
        down * 10 >= particles.len() * 9,
        "only {down} of {} particles came down to the floor",
        particles.len()
    );
}

#[test]
fn injection_counts_litres() {
    let mut app = testing::headless();
    let added = inject(&mut app, [6.0, 0.0, 0.0], 500);
    assert_eq!(added, 500);
    let fluid = app.world().resource::<Fluid>();
    assert_eq!(fluid.len(), 500);
    let litres = fluid.litres().0;
    let each = Resolution::FINEST.litres_per_particle().0;
    assert!((litres - 500.0 * each).abs() < 1e-3);
}

#[test]
fn water_stays_inside_the_drum() {
    let mut app = testing::headless();
    set_spin(&mut app, 0.8);
    for k in 0..6 {
        let a = k as f64;
        inject(&mut app, [a.cos() * 7.5, 0.0, a.sin() * 7.5], 300);
        testing::run(&mut app, Seconds(0.3));
    }
    testing::run(&mut app, Seconds(4.0));
    let particles = testing::particles(&mut app);
    assert_eq!(particles.len(), 1800);
    for p in &particles {
        let [x, y, z] = p.position;
        assert!(x.is_finite() && y.is_finite() && z.is_finite(), "{p:?}");
        let r = (x * x + z * z).sqrt();
        assert!(
            r <= DEFAULT_RING.radius.0 as f64 + 1e-3,
            "particle at radius {r}"
        );
        assert!(
            y.abs() <= DEFAULT_RING.half_width.0 as f64 + 1e-3,
            "particle at y {y}"
        );
    }
}

#[test]
#[ignore = "wants a GPU"]
fn water_follows_the_ring_when_it_is_made_smaller() {
    let mut app = testing::headless();
    set_spin(&mut app, 1.0);
    for k in 0..6 {
        let a = k as f64;
        inject(
            &mut app,
            [a.cos() * 9.0, (k % 3) as f64 * 4.0 - 4.0, a.sin() * 9.0],
            300,
        );
        testing::run(&mut app, Seconds(0.3));
    }
    testing::run(&mut app, Seconds(2.0));
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, 14.0);
        Dial::Width.set(&mut settings, 6.0);
    }
    testing::run(&mut app, Seconds(4.0));
    let sim = app.world().resource::<Simulation>();
    assert_eq!(sim.drum.ring.radius, Metres(7.0));
    assert_eq!(sim.drum.ring.half_width, Metres(3.0));
    let particles = testing::particles(&mut app);
    assert_eq!(particles.len(), 1800);
    for p in &particles {
        let [x, y, z] = p.position;
        let r = (x * x + z * z).sqrt();
        assert!(r <= 7.0 + 1e-3, "particle at radius {r}");
        assert!(y.abs() <= 3.0 + 1e-3, "particle at y {y}");
    }
}

/// Water dropped at the axis of the spinning drum ends up on the glass, riding round with it:
/// at rest in the drum's own frame.
#[test]
#[ignore = "wants a GPU"]
fn spinning_drum_throws_water_onto_the_glass() {
    let mut app = testing::headless();
    set_spin(&mut app, 1.0);
    inject(&mut app, [0.0, 0.0, 0.0], 800);
    testing::run(&mut app, Seconds(12.0));
    let particles = testing::particles(&mut app);
    let near_glass = particles
        .iter()
        .filter(|p| {
            (p.position[0].powi(2) + p.position[2].powi(2)).sqrt()
                > DEFAULT_RING.floor_radius().0 as f64 - 1.5
        })
        .count();
    assert!(
        near_glass as f32 > particles.len() as f32 * 0.8,
        "{near_glass} of {} near the glass",
        particles.len()
    );
    let riding = particles.iter().filter(|p| norm(&p.velocity) < 1.5).count();
    assert!(
        riding as f32 > particles.len() as f32 * 0.7,
        "{riding} of {} ride with the glass",
        particles.len()
    );
}

#[test]
fn reset_keeps_parameters_and_target_spin() {
    let mut app = testing::headless();
    inject(&mut app, [6.0, 0.0, 0.0], 100);
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.viscosity = 0.4;
        settings.spin = RadiansPerSecond(1.5);
    }
    testing::run(&mut app, Seconds(1.0));
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.reset();
    assert_eq!(sim.params.viscosity, 0.4);
    assert_eq!(sim.drum.target_spin, RadiansPerSecond(1.5));
    assert_eq!(sim.time, Seconds(0.0));
    let mut fluid = app.world_mut().resource_mut::<Fluid>();
    fluid.clear();
    assert!(fluid.is_empty());
    assert!(fluid.add(Particle::default()));
    assert_eq!(fluid.len(), 1);
}

#[test]
fn water_grows_a_surface() {
    let mut app = testing::headless();
    assert_eq!(testing::surface_triangles(&mut app), 0);
    inject(&mut app, [6.0, 0.0, 0.0], 400);
    testing::run(&mut app, Seconds(0.5));
    let triangles = testing::surface_triangles(&mut app);
    assert!(triangles > 200, "{triangles} triangles");
    app.world_mut().resource_mut::<Fluid>().clear();
    testing::run(&mut app, Seconds(0.1));
    assert_eq!(testing::surface_triangles(&mut app), 0);
}

/// A particle flying on its own is a drop, not a surface: the grid cannot draw a blob that
/// small, so it is listed to be drawn as a sphere instead, and it takes no part in the surface
/// of the water it lands in until it joins it.
#[test]
fn a_particle_on_its_own_is_a_droplet_not_a_surface() {
    let mut app = testing::headless();
    inject(&mut app, [6.0, 0.0, 0.0], 400);
    testing::run(&mut app, Seconds(0.5));
    let body = testing::surface_demand(&mut app);
    assert_eq!(body.droplets, 0, "{body:?}");
    inject(&mut app, [-6.0, 0.0, 0.0], 1);
    testing::run(&mut app, Seconds(0.1));
    let with_drop = testing::surface_demand(&mut app);
    assert_eq!(with_drop.droplets, 1, "{with_drop:?}");
    assert!(
        with_drop.vertices.abs_diff(body.vertices) < body.vertices / 10,
        "the drop changed the surface: {body:?} then {with_drop:?}"
    );
}

/// Water at rest in the turning drum is drawn holding still in it: its surface is extracted in
/// the drum's own frame, so between two steps a vertex moves only as far as the water does,
/// not by the grid sliding under it.
#[test]
#[ignore = "wants a GPU"]
fn settled_water_holds_still_in_the_drums_frame() {
    let mut app = testing::headless();
    for k in 0..10 {
        let a = k as f64 * 0.7;
        inject(
            &mut app,
            [a.cos() * 8.0, (k % 3) as f64 * 2.0 - 2.0, a.sin() * 8.0],
            1000,
        );
        testing::run(&mut app, Seconds(0.2));
    }
    testing::run(&mut app, Seconds(10.0));
    let before = testing::surface_vertices(&mut app);
    testing::run(&mut app, SUBSTEP_RATE.period());
    let after = testing::surface_vertices(&mut app);
    assert!(before.len() > 10_000, "{} vertices", before.len());
    let mut moved: Vec<f32> = after
        .iter()
        .step_by(16)
        .map(|p| {
            before
                .iter()
                .map(|q| {
                    let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f32::MAX, f32::min)
                .sqrt()
        })
        .collect();
    moved.sort_by(|x, y| x.total_cmp(y));
    let typical = moved[moved.len() * 9 / 10];
    assert!(
        typical < 0.005,
        "the surface moved {typical} m in the drum's frame between two steps"
    );
}

/// A ring a few hundred metres across with tens of thousands of cubic metres of water in it,
/// which is more particles than the water keeps, so it has coarsened a few times.
fn big_ring(width: f32) -> testing::Headless {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, 490.0);
        Dial::Width.set(&mut settings, width);
    }
    set_spin(&mut app, 0.2);
    testing::run(&mut app, Seconds(1.0));
    let r = app.world().resource::<Simulation>().drum.ring.radius.0 as f64 - 2.0;
    let mut left = 61_359_000.0f32;
    for k in 0..200 {
        let a = k as f64 * std::f64::consts::TAU / 100.0;
        let per = app
            .world()
            .resource::<Fluid>()
            .resolution()
            .litres_per_particle()
            .0;
        let count = (left / (200 - k) as f32 / per).max(1.0) as u32;
        left -= count as f32 * per;
        let y = ((k % 7) as f32 * width / 8.0 - width / 2.0) as f64;
        inject(&mut app, [a.cos() * r, y, a.sin() * r], count);
        testing::run(&mut app, Seconds(0.1));
    }
    testing::run(&mut app, Seconds(5.0));
    app
}

#[test]
#[ignore = "wants a GPU"]
fn a_big_ring_of_water_keeps_its_whole_surface() {
    for width in [12.0, 60.0] {
        let mut app = big_ring(width);
        let demand = testing::surface_demand(&mut app);
        assert!(
            demand.vertices <= MAX_VERTICES as u32
                && demand.indices <= MAX_INDICES as u32
                && demand.blocks <= MAX_BLOCKS as u32,
            "the surface of a ring {width} m wide does not fit its buffers: {demand:?}"
        );
        assert!(demand.indices > 0, "no surface at all");
    }
}

/// Whatever size the vessel is, the finest water it may hold keeps it within the surface grid.
#[test]
fn the_finest_water_for_a_vessel_keeps_it_within_the_surface_grid() {
    for reach in [1.0, 10.5, 500.0, 1e4, 1e6, 1e9, 1e12, 1e20, 1e30] {
        let finest = Resolution::finest_for(Metres(reach));
        assert!(
            grid_reach(finest).0 >= reach && finest.spacing.0.is_finite(),
            "a vessel reaching {reach} m gets water spaced {:?}",
            finest.spacing
        );
    }
    assert_eq!(Resolution::finest_for(Metres(10.5)), Resolution::FINEST);
}

/// Drops of a few particles scattered through a ring a kilometre across: every one gets its surface,
/// however many blocks of the grid they touch between them.
#[test]
fn spray_all_over_a_big_ring_is_meshed() {
    let mut app = testing::headless();
    testing::run(&mut app, Seconds(0.5));
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        Dial::Diameter.set(&mut settings, 999.0);
        Dial::Width.set(&mut settings, 999.0);
    }
    set_spin(&mut app, 0.0);
    testing::run(&mut app, Seconds(0.5));
    let drops = 6000;
    for k in 0..drops {
        let a = k as f64 * 2.399;
        let r = 20.0 + (k % 97) as f64 * 4.9;
        let y = (k % 89) as f64 * 11.0 - 490.0;
        inject(&mut app, [a.cos() * r, y, a.sin() * r], 10);
    }
    testing::run(&mut app, Seconds(0.1));
    let demand = testing::surface_demand(&mut app);
    assert!(
        demand.vertices >= 8 * drops && demand.vertices <= MAX_VERTICES as u32,
        "{demand:?}"
    );
    assert!(demand.blocks <= MAX_BLOCKS as u32, "{demand:?}");
}
