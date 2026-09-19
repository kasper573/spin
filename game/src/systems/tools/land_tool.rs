//! The land tool: a twin-barrelled gun that works the ground under the crosshair. The left
//! mouse button fires the left barrel, which puts land down, and the right button the right
//! barrel, which takes it up; an arrow painted on each barrel says which way its land goes,
//! and the ring light round its muzzle is lit while it fires. The wheel turns how much land
//! goes a second, which the screen on its back shows, and the brush widens with it, which the
//! crosshair's marker shows on the ground for as long as the tool is out.
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;

use crate::core::units::{LitresPerSecond, Metres};
use crate::systems::aim::{self, Aim, AimMarker};
use crate::systems::settings::{Dial, Settings};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::tools::muzzle::MuzzleLight;
use crate::systems::tools::{Tool, ToolApp, Workbench, lying_on, side_on};

/// The brush the tool works the ground with at a flow: its radius, the same on a ring of any
/// size. It grows as the cube root of the flow, so a second of any flow raises a mound of the
/// one shape, only bigger.
pub fn brush(flow: LitresPerSecond) -> Metres {
    Metres(UNIT_BRUSH.0 * (flow.0 / 1000.0).cbrt())
}

#[derive(Resource, Default, PartialEq)]
pub struct LandTool {
    pub raising: bool,
    pub lowering: bool,
}

impl Tool for LandTool {
    const REACH: Metres = Metres(MUZZLE + HANDLE.z);

    /// Two peaks standing on level ground, the nearer one lower.
    fn icon(at: Vec2) -> f32 {
        let peak = |top: Vec2, slope: f32| at.y > top.y + (at.x - top.x).abs() * slope;
        let ground = 0.84;
        let standing = peak(Vec2::new(0.4, 0.16), 1.9) || peak(Vec2::new(0.7, 0.44), 1.5);
        f32::from(standing && at.y < ground && (0.06..0.94).contains(&at.x))
    }

    fn model(bench: &mut Workbench) {
        let frame = bench.finish(StandardMaterial {
            base_color: FRAME,
            metallic: 0.8,
            perceptual_roughness: 0.35,
            ..default()
        });
        let plating = bench.finish(StandardMaterial {
            base_color: PLATING,
            metallic: 0.3,
            perceptual_roughness: 0.55,
            ..default()
        });
        let hazard = bench.finish(StandardMaterial {
            base_color: HAZARD,
            perceptual_roughness: 0.5,
            ..default()
        });
        let ground = bench.finish(StandardMaterial {
            base_color: ARROW_GROUND,
            perceptual_roughness: 0.6,
            ..default()
        });
        bench.handle_at(HANDLE);

        let profile = ConvexPolygon::new(PROFILE.map(Vec2::from))
            .expect("the receiver's side view bulges nowhere inward");
        bench.part(
            Extrusion::new(profile, RECEIVER_WIDTH),
            &plating,
            Transform::from_rotation(side_on()),
        );
        bench.part(
            Cuboid::new(0.04, 0.12, 0.052),
            &frame,
            Transform::from_translation(HANDLE).with_rotation(Quat::from_rotation_x(-0.3)),
        );
        bench.part(
            Cuboid::new(0.08, 0.035, 0.2),
            &frame,
            Transform::from_xyz(0.0, BORE - SHROUD.y / 2.0 - 0.0175, -(BREECH + 0.12)),
        );
        for side in [-1.0, 1.0] {
            bench.part(
                Cuboid::new(0.002, 0.014, 0.2),
                &hazard,
                Transform::from_xyz(side * RECEIVER_WIDTH / 2.0, 0.03, -0.04),
            );
        }
        for clamp in [BREECH + CLAMP / 2.0, SHROUDED - CLAMP / 2.0] {
            bench.part(
                Cuboid::new(RECEIVER_WIDTH + 0.006, SHROUD.y + 0.006, CLAMP),
                &hazard,
                Transform::from_xyz(0.0, BORE, -clamp),
            );
        }

        let middle = -(BREECH + SHROUDED) / 2.0;
        let top = BORE + SHROUD.y / 2.0;
        for barrel in [Barrel::Raise, Barrel::Lower] {
            let x = barrel.side() * APART / 2.0;
            let paint = bench.finish(StandardMaterial {
                base_color: barrel.colour(),
                perceptual_roughness: 0.6,
                ..default()
            });
            bench.part(
                Cuboid::new(SHROUD.x, SHROUD.y, SHROUDED - BREECH),
                &frame,
                Transform::from_xyz(x, BORE, middle),
            );
            bench.part(
                Extrusion::new(Annulus::new(0.014, BARREL), MUZZLE - SHROUDED),
                &frame,
                Transform::from_xyz(x, BORE, -(SHROUDED + MUZZLE) / 2.0),
            );
            // an arrow lies flat on its plate: its head a wedge, its tail a bar behind it
            let (width, long, head) = ARROW;
            bench.part(
                Cuboid::new(SHROUD.x - 0.006, PAINT, long + 0.03),
                &ground,
                Transform::from_xyz(x, top + PAINT / 2.0, middle),
            );
            let lies = Vec3::new(x, top + PAINT * 1.5, middle);
            let flat = Quat::from_rotation_y(barrel.heading())
                * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            let ahead = Quat::from_rotation_y(barrel.heading()) * Vec3::NEG_Z;
            bench.part(
                Extrusion::new(
                    Triangle2d::new(
                        Vec2::new(0.0, head / 2.0),
                        Vec2::new(-width / 2.0, -head / 2.0),
                        Vec2::new(width / 2.0, -head / 2.0),
                    ),
                    PAINT,
                ),
                &paint,
                Transform::from_translation(lies + ahead * (long - head) / 2.0).with_rotation(flat),
            );
            bench.part(
                Cuboid::new(width * 0.45, PAINT, long - head),
                &paint,
                Transform::from_translation(lies - ahead * head / 2.0),
            );
            let ring =
                bench.muzzle_light(barrel.colour(), BARREL + 0.004, Vec3::new(x, BORE, -MUZZLE));
            bench.commands.entity(ring).insert(barrel);
        }

        bench.dial_screen(
            Dial::Build,
            HAZARD,
            SCREEN,
            lying_on(DECK_TOP, DECK_FOOT, FLUSH),
        );
    }
}

pub struct LandToolPlugin;

impl Plugin for LandToolPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool::<LandTool, _>(operate)
            .add_systems(Update, light.in_set(SimSet::Observe));
    }
}

const FRAME: Color = Color::srgb(0.1, 0.105, 0.115);
const PLATING: Color = Color::srgb(0.42, 0.43, 0.45);
const HAZARD: Color = Color::srgb(0.95, 0.45, 0.04);
const ARROW_GROUND: Color = Color::srgb(0.82, 0.82, 0.8);
const PUTS_DOWN: Color = Color::srgb(0.04, 0.22, 0.95);
const TAKES_UP: Color = Color::srgb(0.9, 0.05, 0.04);

/// The receiver from its side, x ahead and y up, and how wide it is. Its back slopes down from
/// the top of the deck to its foot, and the screen lies on that slope.
const DECK_TOP: Vec2 = Vec2::new(-0.04, 0.065);
const DECK_FOOT: Vec2 = Vec2::new(-0.12, -0.03);
const PROFILE: [(f32, f32); 7] = [
    (-0.09, -0.045),
    (0.1, -0.045),
    (BREECH, -0.015),
    (BREECH, 0.05),
    (0.14, DECK_TOP.y),
    (DECK_TOP.x, DECK_TOP.y),
    (DECK_FOOT.x, DECK_FOOT.y),
];
const RECEIVER_WIDTH: f32 = 0.13;
const HANDLE: Vec3 = Vec3::new(0.0, -0.09, 0.04);
/// The barrels: how high their bores lie and how far apart, how wide and tall the shroud
/// each runs in is and how thick the barrel that comes out of it, where they leave the
/// receiver, where the shrouds end and where the barrels do, and how broad the clamps that
/// hold the two together are.
const BORE: f32 = 0.02;
const APART: f32 = 0.066;
const SHROUD: Vec2 = Vec2::new(0.058, 0.05);
const BARREL: f32 = 0.022;
const BREECH: f32 = 0.16;
const SHROUDED: f32 = 0.56;
const MUZZLE: f32 = 0.62;
const CLAMP: f32 = 0.016;
/// An arrow: how wide its head is, how long it is from tip to tail, how much of that is head,
/// and how thick its paint lies.
const ARROW: (f32, f32, f32) = (0.046, 0.32, 0.11);
const PAINT: f32 = 0.002;
const SCREEN: Vec2 = Vec2::new(0.1, 0.0575);
/// The brush at a cubic metre a second.
const UNIT_BRUSH: Metres = Metres(1.26);
/// The screen is set in the slope of the back, lifted off it only enough to be seen apart.
const FLUSH: f32 = 0.0004;

/// One of the tool's barrels, which its ring light carries.
#[derive(Component, Clone, Copy)]
enum Barrel {
    Raise,
    Lower,
}

impl Barrel {
    /// Which side of the tool the barrel lies on, as the viewer holds it: the left button's
    /// barrel on the left.
    fn side(self) -> f32 {
        match self {
            Barrel::Raise => -1.0,
            Barrel::Lower => 1.0,
        }
    }

    fn colour(self) -> Color {
        match self {
            Barrel::Raise => PUTS_DOWN,
            Barrel::Lower => TAKES_UP,
        }
    }

    /// The turn about the tool's upright that points the barrel's arrow the way its land
    /// goes: out of the muzzle, or back in.
    fn heading(self) -> f32 {
        match self {
            Barrel::Raise => 0.0,
            Barrel::Lower => std::f32::consts::PI,
        }
    }
}

fn operate(
    mut tool: ResMut<LandTool>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut aim: ResMut<Aim>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
) {
    if scroll.delta.y != 0.0 {
        Dial::Build.adjust(&mut settings, scroll.delta.y.signum() as i32);
    }
    let brush = brush(settings.build);
    aim.marker = aim
        .target
        .filter(|_| brush.0 > 0.0)
        .map(|target| AimMarker {
            mouth: aim::disc_at(&sim.drum, target),
            radius: brush,
            top: false,
            refused: false,
        });
    tool.raising = aim.target.is_some() && mouse.pressed(MouseButton::Left);
    tool.lowering = aim.target.is_some() && mouse.pressed(MouseButton::Right);
    let way = f32::from(tool.raising) - f32::from(tool.lowering);
    let cubic_metres = settings.build.0 / 1000.0 * time.delta_secs() * way;
    if let Some(target) = aim.target
        && cubic_metres != 0.0
    {
        // the brush raises its middle by the amount, falling off to its rim: a third of the
        // cylinder that would stand on it
        let amount = 3.0 * cubic_metres / (std::f32::consts::PI * brush.0 * brush.0);
        sim.sculpt(target.point.to_array(), brush.0 as f64, amount as f64);
    }
}

fn light(tool: Res<LandTool>, mut rings: Query<(&Barrel, &mut MuzzleLight)>) {
    for (barrel, mut ring) in &mut rings {
        let firing = match barrel {
            Barrel::Raise => tool.raising,
            Barrel::Lower => tool.lowering,
        };
        if ring.firing != firing {
            ring.firing = firing;
        }
    }
}
