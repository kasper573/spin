//! The drum's wall and the patterns fixed to it.
use game::core::units::{Metres, Seconds};
use game::systems::drum::{PANE, Ring, Site};
use game::systems::settings::Settings;
use game::systems::sim::{Simulation, standing_spin};
use game::systems::testing;

/// The site's place in the patterns, worked out afresh from its angle.
fn laid_afresh(site: Site, ring: Ring) -> Site {
    Site::at(site.phi, site.y, ring)
}

fn same_place(a: Site, b: Site, ring: Ring, tolerance: f64) -> bool {
    let wrapped = |x: f64, y: f64, period: f64| {
        let d = (x - y).rem_euclid(period);
        d.min(period - d) < tolerance
    };
    (a.phi - b.phi).abs() < 1e-12
        && (a.y - b.y).abs() < 1e-12
        && wrapped(a.arc, b.arc, std::f64::consts::TAU * ring_radius(ring))
        && wrapped(a.cap[0], b.cap[0], PANE)
        && wrapped(a.cap[1], b.cap[1], PANE)
}

fn ring_radius(ring: Ring) -> f64 {
    ring.radius.0 as f64
}

/// The patterns on the glass and the ground are fixed to the wheel: however the site is
/// walked round the ring and across it, where it lies in the patterns is exactly where its
/// angle says, so the patterns never shift when the wheel is drawn about a new site.
#[test]
fn the_patterns_stay_fixed_to_the_wheel_as_the_site_moves() {
    let ring = Ring {
        radius: Metres(109.0),
        half_width: Metres(120.0),
    };
    let mut sim = Simulation::new(ring);
    let mut walk = 0.0f64;
    for step in 0..2000 {
        walk += 0.37;
        let below = [
            -0.8,
            (walk * 0.7).sin() * 3.0,
            if step % 2 == 0 { 2.3 } else { -1.9 } + walk.cos(),
        ];
        sim.resite(below);
        let site = sim.drum.site;
        assert!(
            same_place(site, laid_afresh(site, ring), ring, 1e-6),
            "after {step} moves the site is at {site:?}"
        );
    }
}

/// A ring drawn about a site that came from a save, a resize or the start is in the same
/// patterns as one walked there.
#[test]
fn a_site_from_anywhere_is_in_the_same_patterns() {
    let ring = Ring {
        radius: Metres(109.0),
        half_width: Metres(120.0),
    };
    let mut app = testing::headless();
    {
        let mut settings = app.world_mut().resource_mut::<Settings>();
        settings.diameter = Metres(ring.radius.0 * 2.0);
        settings.width = Metres(ring.half_width.0 * 2.0);
        settings.spin = standing_spin(ring);
    }
    app.insert_resource(Simulation::new(ring));
    testing::run(&mut app, Seconds(0.5));
    let sim = app.world().resource::<Simulation>();
    let site = sim.drum.site;
    assert!(
        same_place(site, laid_afresh(site, ring), ring, 1e-9),
        "{site:?}"
    );
    let mut sim = Simulation::new(ring);
    for _ in 0..50 {
        sim.resite([-0.8, 1.5, 3.0]);
    }
    let wider = Ring {
        radius: Metres(109.0),
        half_width: Metres(150.0),
    };
    sim.resize(wider);
    let site = sim.drum.site;
    assert!(
        same_place(site, laid_afresh(site, wider), wider, 1e-9),
        "{site:?}"
    );
}
