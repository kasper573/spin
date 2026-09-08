use bevy::prelude::*;
use game::core::fluid::{Fluid, PARTICLE_MASS, Particle, REST_DENSITY};
use game::core::units::{RadiansPerSecond, Seconds};
use game::systems::drum::{HALF_WIDTH, RADIUS};
use game::systems::settings::Settings;
use game::systems::sim::Simulation;
use game::systems::testing;

fn inject(app: &mut App, centre: [f32; 3], count: u32) -> u32 {
    app.world_mut()
        .resource_scope(|world, mut fluid: Mut<Fluid>| {
            world
                .resource::<Simulation>()
                .inject(&mut fluid, centre, count)
        })
}

fn set_spin(app: &mut App, spin: f32) {
    app.world_mut().resource_mut::<Settings>().spin = RadiansPerSecond(spin);
    app.world_mut().resource_mut::<Simulation>().drum.spin = RadiansPerSecond(spin);
}

#[test]
fn injection_counts_litres() {
    let mut app = testing::headless();
    let added = inject(&mut app, [2.0, 0.0, 0.0], 500);
    assert_eq!(added, 500);
    let fluid = app.world().resource::<Fluid>();
    assert_eq!(fluid.len(), 500);
    let litres = Simulation::water(fluid).0;
    assert!((litres - 500.0 * PARTICLE_MASS / REST_DENSITY * 1000.0).abs() < 1e-3);
}

#[test]
fn water_stays_inside_the_drum() {
    let mut app = testing::headless();
    set_spin(&mut app, 1.2);
    for k in 0..6 {
        let a = k as f32;
        inject(&mut app, [a.cos() * 2.5, 0.0, a.sin() * 2.5], 300);
        testing::run(&mut app, Seconds(0.3));
    }
    testing::run(&mut app, Seconds(4.0));
    let particles = testing::particles(&mut app);
    assert_eq!(particles.len(), 1800);
    for p in &particles {
        let [x, y, z] = p.position;
        assert!(x.is_finite() && y.is_finite() && z.is_finite(), "{p:?}");
        let r = (x * x + z * z).sqrt();
        assert!(r <= RADIUS + 1e-3, "particle at radius {r}");
        assert!(y.abs() <= HALF_WIDTH + 1e-3, "particle at y {y}");
    }
}

#[test]
fn spinning_drum_throws_water_onto_the_glass() {
    let mut app = testing::headless();
    set_spin(&mut app, 1.6);
    inject(&mut app, [0.0, 0.0, 0.0], 800);
    testing::run(&mut app, Seconds(8.0));
    let particles = testing::particles(&mut app);
    let near_glass = particles
        .iter()
        .filter(|p| (p.position[0].powi(2) + p.position[2].powi(2)).sqrt() > RADIUS - 0.6)
        .count();
    assert!(
        near_glass as f32 > particles.len() as f32 * 0.8,
        "{near_glass} of {} near the glass",
        particles.len()
    );
    let riding = particles
        .iter()
        .filter(|p| {
            let [x, _, z] = p.position;
            let r = (x * x + z * z).sqrt().max(1e-6);
            let tangential = (p.velocity[0] * z - p.velocity[2] * x) / r;
            (tangential - 1.6 * r).abs() < 1.0
        })
        .count();
    assert!(
        riding as f32 > particles.len() as f32 * 0.7,
        "{riding} of {} ride with the glass",
        particles.len()
    );
}

#[test]
fn reset_keeps_parameters_and_target_spin() {
    let mut app = testing::headless();
    inject(&mut app, [2.0, 0.0, 0.0], 100);
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.viscosity = 0.4;
        settings.spin = RadiansPerSecond(2.0);
    }
    testing::run(&mut app, Seconds(1.0));
    let mut sim = app.world_mut().resource_mut::<Simulation>();
    sim.reset();
    assert_eq!(sim.params.viscosity, 0.4);
    assert_eq!(sim.drum.target_spin, RadiansPerSecond(2.0));
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
    inject(&mut app, [2.0, 0.0, 0.0], 400);
    testing::run(&mut app, Seconds(0.5));
    let triangles = testing::surface_triangles(&mut app);
    assert!(triangles > 200, "{triangles} triangles");
    app.world_mut().resource_mut::<Fluid>().clear();
    testing::run(&mut app, Seconds(0.1));
    assert_eq!(testing::surface_triangles(&mut app), 0);
}
