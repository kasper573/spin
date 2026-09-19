//! The ring light round the end of a barrel: dark until the barrel fires, then glowing, and
//! lighting whatever is near it, as any lamp in the ring would.
use bevy::prelude::*;

use crate::core::units::Seconds;
use crate::systems::figure::Mirrored;
use crate::systems::sim::SimSet;
use crate::systems::tools::Workbench;

/// A barrel's ring light. The tool it is on says whether the barrel is firing.
#[derive(Component)]
pub struct MuzzleLight {
    pub firing: bool,
    colour: Color,
    /// How bright it is, 0 to 1, which follows the firing over a moment.
    level: f32,
    lamp: Entity,
}

impl Workbench<'_, '_, '_> {
    /// A ring light of this colour round the end of a barrel this thick, the end being at
    /// `at` and the barrel lying along z.
    pub fn muzzle_light(&mut self, colour: Color, barrel: f32, at: Vec3) -> Entity {
        let unlit = self.finish(StandardMaterial {
            base_color: Color::BLACK.mix(&colour, 0.12),
            emissive_exposure_weight: 1.0,
            perceptual_roughness: 0.2,
            ..default()
        });
        let ring = self.part(
            Torus::new(barrel - TUBE, barrel + TUBE),
            &unlit,
            Transform::from_translation(at)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        );
        let lamp = self
            .commands
            .spawn((
                PointLight {
                    color: colour,
                    intensity: 0.0,
                    range: RANGE,
                    radius: barrel,
                    ..default()
                },
                Transform::from_translation(Vec3::NEG_Y * LEAD),
                Visibility::Hidden,
                ChildOf(ring),
            ))
            .id();
        self.commands.entity(ring).insert(MuzzleLight {
            firing: false,
            colour,
            level: 0.0,
            lamp,
        });
        ring
    }
}

pub struct MuzzlePlugin;

impl Plugin for MuzzlePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, glow.in_set(SimSet::Observe));
    }
}

/// The ring's tube is this thick, and its lamp stands this far out in front of it, clear of
/// the barrel's own end.
const TUBE: f32 = 0.006;
const LEAD: f32 = 0.03;
/// What the ring gives off at full: the luminance of its tube, which has to stand out against
/// sunlit ground, and the light it throws, in lumens, as far as this.
const LUMINANCE: f32 = 400_000.0;
const LUMENS: f32 = 1_000_000.0;
const RANGE: f32 = 30.0;
/// How long the ring takes to come up to full, and to die down.
const RISE: Seconds = Seconds(0.06);

fn glow(
    time: Res<Time>,
    mut rings: Query<(
        &mut MuzzleLight,
        &mut Mirrored,
        &MeshMaterial3d<StandardMaterial>,
    )>,
    mut lamps: Query<(&mut PointLight, &mut Visibility)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let step = time.delta_secs() / RISE.0;
    for (mut ring, mut mirrored, material) in &mut rings {
        let level = if ring.firing {
            (ring.level + step).min(1.0)
        } else {
            (ring.level - step).max(0.0)
        };
        if level == ring.level {
            continue;
        }
        ring.level = level;
        let given_off = ring.colour.to_linear() * (LUMINANCE * level);
        mirrored.finish.glow = given_off;
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.emissive = given_off;
        }
        if let Ok((mut lamp, mut visibility)) = lamps.get_mut(ring.lamp) {
            lamp.intensity = LUMENS * level;
            visibility.set_if_neq(if level > 0.0 {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            });
        }
    }
}
