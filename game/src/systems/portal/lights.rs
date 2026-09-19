//! The light a portal's burning rim throws: lamps of its colour spaced round the rim, a little
//! off the surface, as bright together as the fire is.
use bevy::prelude::*;

use crate::systems::drum::{FLAME_BAND, MouthColour, MouthCoords};
use crate::systems::portal::{grade, portal_colour};
use crate::systems::scene::Viewpoint;
use crate::systems::sim::{SimSet, Simulation};

pub struct RimLightPlugin;

impl Plugin for RimLightPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, place.in_set(SimSet::Observe));
    }
}

/// How many lamps stand round a rim, how far off the surface as a share of the mouth's radius,
/// what a metre of burning rim throws, in lumens, and how many radii away it is still seen by.
const LAMPS: usize = 3;
const OFF: f64 = 0.12;
const LUMENS_PER_METRE: f32 = 60_000.0;
const REACH: f32 = 12.0;

/// One of the lamps round a mouth's rim.
#[derive(Component)]
struct RimLamp {
    colour: MouthColour,
    round: f64,
}

fn spawn(mut commands: Commands) {
    for colour in MouthColour::BOTH {
        for k in 0..LAMPS {
            commands.spawn((
                RimLamp {
                    colour,
                    round: (k as f64 / LAMPS as f64 + 0.25) * std::f64::consts::TAU,
                },
                PointLight {
                    color: portal_colour(colour),
                    intensity: 0.0,
                    ..default()
                },
                Visibility::Hidden,
            ));
        }
    }
}

fn place(
    sim: Res<Simulation>,
    viewpoint: Res<Viewpoint>,
    mut lamps: Query<(&RimLamp, &mut PointLight, &mut Transform, &mut Visibility)>,
) {
    let radius = sim.drum.mouths.radius();
    for (lamp, mut light, mut transform, mut visibility) in &mut lamps {
        let Some(mouth) = sim.drum.mouths.get(lamp.colour) else {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        };
        let (sin, cos) = lamp.round.sin_cos();
        let at = sim.drum.mouth_point(
            mouth,
            MouthCoords {
                u: cos * radius,
                v: sin * radius,
                h: OFF * radius,
            },
        );
        *transform = viewpoint.place(at);
        let rim = std::f32::consts::TAU * radius as f32;
        // the fire dies down toward the mouth's bottom, and its light with it: see `portals.wgsl`
        let fire = grade(sin).powi(2) as f32;
        light.intensity = LUMENS_PER_METRE * rim / LAMPS as f32 * fire;
        light.range = REACH * radius as f32;
        light.radius = (FLAME_BAND * radius) as f32;
        visibility.set_if_neq(Visibility::Inherited);
    }
}
