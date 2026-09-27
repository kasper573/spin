//! How the world behaves over time, played as a player plays it, through the thrusters' keys:
//! what happens does not depend on how fast the machine draws it.
use bevy::prelude::*;
use game::core::avatar::Thruster;
use game::core::math::{Vec3d, quat_rotate};
use game::core::units::Seconds;
use game::systems::drum::{Drum, Ring, Site};
use game::systems::player::{PilotInput, Player};
use game::systems::sim::Simulation;
use game::systems::testing;

/// Frame rates a player's machine may draw at, from a slow one to a fast display's.
const RATES: [f64; 3] = [60.0, 144.0, 240.0];
/// A flight: turning on the spot, walking, thrusting up off the ground and coming down again,
/// each for a whole number of frames at every rate.
const FLIGHT: [(Option<Thruster>, f64); 4] = [
    (Some(Thruster::YawLeft), 0.5),
    (Some(Thruster::Forward), 1.5),
    (Some(Thruster::Up), 0.75),
    (None, 1.0),
];
/// How far apart the same flight may end at two rates: what a single step of the bodies at a
/// walk carries the eye, and turns it by.
const SAME_PLACE_M: f64 = 0.01;
const SAME_HEADING_RAD: f64 = 0.01;
#[test]
#[ignore = "wants a GPU"]
fn a_flight_ends_where_it_would_at_any_frame_rate() {
    let ends: Vec<Flown> = RATES.iter().map(|&fps| flown_at(fps)).collect();
    let first = &ends[0];
    for (fps, other) in RATES.iter().zip(&ends).skip(1) {
        let (eye, ahead) = other.seen_from(first);
        let apart = (0..3)
            .map(|k| (eye[k] - first.eye[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let turned = (0..3)
            .map(|k| ahead[k] * first.ahead[k])
            .sum::<f64>()
            .clamp(-1.0, 1.0)
            .acos();
        eprintln!(
            "at {fps} fps the flight ends {apart:.4} m and {turned:.4} rad from where it does at {} fps",
            RATES[0]
        );
        assert!(
            apart < SAME_PLACE_M && turned < SAME_HEADING_RAD,
            "at {fps} fps the flight ends {apart:.3} m and {turned:.3} rad from where it does at {} fps",
            RATES[0]
        );
    }
}

/// Where a flight ended: the eye and the way it looks, in the frame about the site of the wall
/// the simulation had moved to.
struct Flown {
    ring: Ring,
    site: Site,
    eye: Vec3d,
    ahead: Vec3d,
}

impl Flown {
    /// The eye and the way it looks in the frame the other flight ended in.
    fn seen_from(&self, other: &Flown) -> (Vec3d, Vec3d) {
        let mut drum = Drum::new(self.ring);
        drum.site = other.site;
        let frame = drum.frame_at(self.site);
        (frame.point_back(self.eye), frame.vector_back(self.ahead))
    }
}

/// Fly the flight with frames drawn this many times a second.
fn flown_at(fps: f64) -> Flown {
    let mut app = testing::headless();
    testing::watch(&mut app, Seconds(0.5));
    for (thruster, seconds) in FLIGHT {
        let held: Vec<Thruster> = thruster.into_iter().collect();
        for _ in 0..(seconds * fps).round() as u32 {
            let player = *app.world().resource::<Player>();
            app.world_mut().resource_mut::<Simulation>().avatar_input =
                player.input(PilotInput::firing(&held));
            testing::frame_as_played(&mut app, Seconds((1.0 / fps) as f32));
        }
    }
    let sim = app.world().resource::<Simulation>();
    let (eye, attitude) = sim.eye();
    Flown {
        ring: sim.drum.ring,
        site: sim.drum.site,
        eye,
        ahead: quat_rotate(&attitude, &[0.0, 0.0, -1.0]),
    }
}
