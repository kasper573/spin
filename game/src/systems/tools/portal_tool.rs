//! The portal tool: the left mouse button puts the blue portal where the crosshair rests and
//! the right one the orange, each taking the place of the one of its colour that stood before.
//! The middle button takes away the portal under the crosshair, or both where there is none
//! under it. The wheel turns how wide the pair is, which the screen on its back shows, and the
//! crosshair's marker shows where a portal would go and which way up: moved off whatever is
//! in its way, or marked as refused where it cannot go at all.
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::core::math::quat_rotate;
use crate::core::units::{Metres, Seconds};
use crate::systems::aim::{Aim, AimMarker};
use crate::systems::drum::{MouthColour, MouthFit};
use crate::systems::figure::Mirrored;
use crate::systems::portal::{SolidMaterial, portal_colour};
use crate::systems::settings::{Dial, Settings};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::tools::muzzle::MuzzleLight;
use crate::systems::tools::{Tool, ToolApp, Workbench, laid_along};

#[derive(Resource, Default, PartialEq)]
pub struct PortalTool {
    /// The colour of the shot the muzzle is still lit by, and for how much longer.
    pub firing: Option<(MouthColour, Seconds)>,
}

/// The colour of the last portal the tool put, which its marker is fitted as until the other
/// button is pressed.
#[derive(Resource, Default, Clone, Copy, PartialEq)]
pub struct LastPortal(pub Option<MouthColour>);

impl Tool for PortalTool {
    const REACH: Metres = Metres(MUZZLE + HANDLE.z);

    /// Two rings, one before the other.
    fn icon(at: Vec2) -> f32 {
        let ring = |centre: Vec2| {
            let off = (at - centre) / Vec2::new(0.19, 0.3);
            (off.length() - 1.0).abs() < 0.2
        };
        f32::from(ring(Vec2::new(0.36, 0.42)) || ring(Vec2::new(0.64, 0.58)))
    }

    fn model(bench: &mut Workbench) {
        let shell = bench.finish(StandardMaterial {
            base_color: SHELL,
            perceptual_roughness: 0.3,
            reflectance: 0.6,
            ..default()
        });
        let core = bench.finish(StandardMaterial {
            base_color: CORE,
            metallic: 0.6,
            perceptual_roughness: 0.4,
            ..default()
        });
        bench.handle_at(HANDLE);

        let along = |z: f32| Transform::from_xyz(0.0, 0.0, -z);
        bench.part(
            Capsule3d::new(REAR.0, REAR.1),
            &shell,
            along(REAR.2).with_rotation(laid_along()),
        );
        bench.part(
            Capsule3d::new(FRONT.0, FRONT.1),
            &shell,
            along(FRONT.2).with_rotation(laid_along()),
        );
        bench.part(
            Extrusion::new(Annulus::new(BARREL * 0.6, REAR.0 * 0.85), SEAM.0),
            &core,
            along(SEAM.1),
        );
        bench.part(
            Cylinder::new(BARREL, MUZZLE - SEAM.1),
            &core,
            along((MUZZLE + SEAM.1) / 2.0).with_rotation(laid_along()),
        );
        bench.part(
            Cuboid::new(0.036, 0.12, 0.048),
            &core,
            Transform::from_translation(HANDLE).with_rotation(Quat::from_rotation_x(-0.3)),
        );

        for prong in 0..PRONGS {
            let round = Quat::from_rotation_z(
                std::f32::consts::FRAC_PI_2 + prong as f32 * std::f32::consts::TAU / PRONGS as f32,
            );
            let splayed = round * Quat::from_rotation_y(-SPLAY);
            let root = round * Vec3::X * PRONG_ROOT.0 - Vec3::Z * PRONG_ROOT.1;
            let tip = root + splayed * Vec3::NEG_Z * ARM;
            bench.part(
                Cuboid::new(0.012, 0.018, ARM),
                &shell,
                Transform::from_translation((root + tip) / 2.0).with_rotation(splayed),
            );
            let canted = round * Quat::from_rotation_y(CANT);
            bench.part(
                Cuboid::new(0.01, 0.012, CLAW),
                &core,
                Transform::from_translation(tip + canted * Vec3::NEG_Z * CLAW / 2.0)
                    .with_rotation(canted),
            );
        }
        let ring = bench.muzzle_light(
            portal_colour(MouthColour::Blue),
            BARREL + 0.004,
            Vec3::new(0.0, 0.0, -MUZZLE),
        );
        bench.commands.entity(ring).insert(Emitter);

        for (side, colour) in [(-1.0, MouthColour::Blue), (1.0, MouthColour::Orange)] {
            let stud = bench.finish(StandardMaterial {
                base_color: Color::BLACK.mix(&portal_colour(colour), 0.25),
                emissive_exposure_weight: 1.0,
                perceptual_roughness: 0.3,
                ..default()
            });
            let lamp = bench.part(
                Cylinder::new(0.009, 0.006),
                &stud,
                Transform::from_xyz(side * 0.022, REAR.0 * 0.93, -REAR.2)
                    .with_rotation(Quat::from_rotation_z(-side * 0.38)),
            );
            bench.commands.entity(lamp).insert(Stud(colour));
        }

        bench.part(
            Cuboid::new(BACK.x, BACK.y, 0.012),
            &core,
            Transform::from_xyz(0.0, BACK_UP, BACK_AT),
        );
        bench.dial_screen(
            Dial::Portal,
            portal_colour(MouthColour::Orange),
            SCREEN,
            Transform::from_xyz(0.0, BACK_UP, BACK_AT + 0.006 + FLUSH),
        );
    }
}

pub struct PortalToolPlugin;

impl Plugin for PortalToolPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LastPortal>()
            .add_tool::<PortalTool, _>(operate)
            .add_systems(Update, (light, glow).in_set(SimSet::Observe));
    }
}

const SHELL: Color = Color::srgb(0.92, 0.93, 0.94);
const CORE: Color = Color::srgb(0.035, 0.037, 0.042);

const HANDLE: Vec3 = Vec3::new(0.0, -0.1, 0.02);
/// The two halves of the shell, each a radius, a length and how far ahead of the handle its
/// middle is, and the dark seam between them, which is this long and this far ahead.
const REAR: (f32, f32, f32) = (0.062, 0.13, 0.04);
const FRONT: (f32, f32, f32) = (0.046, 0.1, 0.27);
const SEAM: (f32, f32) = (0.07, 0.17);
/// The barrel the shell is built round: how thick it is and how far ahead it ends.
const BARREL: f32 = 0.03;
const MUZZLE: f32 = 0.43;
/// The prongs round the muzzle: how many, where each is rooted from the barrel's middle and
/// how far ahead, how long its arm is and how far that splays out, and the claw at its end,
/// canted back in toward what the tool is aimed at.
const PRONGS: usize = 3;
const PRONG_ROOT: (f32, f32) = (0.045, 0.3);
const ARM: f32 = 0.15;
const SPLAY: f32 = 0.22;
const CLAW: f32 = 0.05;
const CANT: f32 = 0.7;
/// The plate that closes the back of the shell, how far behind the handle it is and how far
/// over the barrel's middle, which keeps all of it before the eye of whoever holds the tool,
/// and the screen set in it, lifted off it only enough to be seen apart.
const BACK: Vec2 = Vec2::new(0.092, 0.062);
const BACK_AT: f32 = 0.105;
const BACK_UP: f32 = 0.034;
const SCREEN: Vec2 = Vec2::new(0.08, 0.046);
const FLUSH: f32 = 0.0004;
/// How long the muzzle stays lit by a shot, and what the studs give off.
const FLASH: Seconds = Seconds(0.25);
const LUMINANCE: f32 = 60_000.0;

/// The ring light round the muzzle.
#[derive(Component)]
struct Emitter;

/// A stud that glows in a portal's colour while that portal stands.
#[derive(Component, Clone, Copy)]
struct Stud(MouthColour);

#[allow(clippy::too_many_arguments)]
fn operate(
    mut tool: ResMut<PortalTool>,
    mut last: ResMut<LastPortal>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut aim: ResMut<Aim>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
) {
    if scroll.delta.y != 0.0 {
        Dial::Portal.adjust(&mut settings, scroll.delta.y.signum() as i32);
    }
    tool.firing = tool
        .firing
        .map(|(colour, left)| (colour, Seconds(left.0 - time.delta_secs())))
        .filter(|(_, left)| left.0 > 0.0);
    let Some(target) = aim.target else {
        return;
    };
    if mouse.just_pressed(MouseButton::Middle) {
        match target.mouth {
            Some(colour) => sim.drum.mouths.remove(colour),
            None => sim.drum.mouths.clear(),
        }
    }
    let shot = if mouse.just_pressed(MouseButton::Left) {
        Some(MouthColour::Blue)
    } else if mouse.just_pressed(MouseButton::Right) {
        Some(MouthColour::Orange)
    } else {
        None
    };
    let colour = shot.or(last.0).unwrap_or(MouthColour::Blue);
    let (_, attitude) = sim.eye();
    let seen_from = |axis: [f64; 3]| quat_rotate(&target.turn, &quat_rotate(&attitude, &axis));
    let wanted = sim.drum.mouth_at(
        target.point.to_array(),
        target.surface,
        seen_from(DVec3::Y.to_array()),
        seen_from(DVec3::X.to_array()),
    );
    let fit = sim.drum.fit_mouth(colour, wanted, settings.portal);
    if shot.is_some() && matches!(fit, MouthFit::Fits { .. }) {
        sim.drum.put_mouth(colour, fit, settings.portal);
        tool.firing = Some((colour, FLASH));
        last.0 = Some(colour);
    }
    aim.marker = Some(AimMarker {
        mouth: fit.mouth(),
        radius: Metres(settings.portal.0 / 2.0),
        top: true,
        refused: matches!(fit, MouthFit::Refused(_)),
    });
}

fn light(tool: Res<PortalTool>, mut rings: Query<&mut MuzzleLight, With<Emitter>>) {
    for mut ring in &mut rings {
        let firing = tool.firing.is_some();
        if ring.firing != firing {
            ring.firing = firing;
        }
        if let Some((colour, _)) = tool.firing
            && ring.colour != portal_colour(colour)
        {
            ring.colour = portal_colour(colour);
        }
    }
}

fn glow(
    sim: Res<Simulation>,
    mut studs: Query<(&Stud, &mut Mirrored, &MeshMaterial3d<SolidMaterial>)>,
    mut materials: ResMut<Assets<SolidMaterial>>,
) {
    for (&Stud(colour), mut mirrored, material) in &mut studs {
        let given_off = match sim.drum.mouths.get(colour) {
            Some(_) => portal_colour(colour).to_linear() * LUMINANCE,
            None => LinearRgba::BLACK,
        };
        if mirrored.finish.glow != given_off {
            mirrored.finish.glow = given_off;
            if let Some(mut material) = materials.get_mut(&material.0) {
                material.base.emissive = given_off;
            }
        }
    }
}
