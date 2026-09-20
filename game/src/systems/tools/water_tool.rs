//! The water tool: a hose in the shape of a gun. The left mouse button pours water where the
//! crosshair rests, and the wheel turns the flow, which the screen on its back shows. Nothing
//! is seen to leave the barrel: the water comes straight down on the floor under where it is
//! aimed, over as much of it as the flow falling that fast is wide, and the ring light round the
//! muzzle is what says it is pouring. It comes down at rest over the floor: a ring this small
//! turns about as fast as any hose throws water, and water thrown against the turning that
//! fast weighs nothing.
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;

use crate::core::fluid::Fluid;
use crate::core::sheet::Sheet;
use crate::core::units::{Litres, Metres, MetresPerSecond};
use crate::systems::aim::Aim;
use crate::systems::drum::{SheetPour, SheetWindow};
use crate::systems::settings::{Dial, Settings};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::tools::muzzle::MuzzleLight;
use crate::systems::tools::{Tool, ToolApp, Workbench, laid_along, lying_on, side_on};

/// How fast the water comes down, which with the flow says how wide it falls: as fast as
/// what has fallen half a metre under the Earth's gravity.
const FALL_SPEED: MetresPerSecond = MetresPerSecond(3.0);

#[derive(Resource, Default, PartialEq)]
pub struct WaterTool {
    pub pouring: bool,
    /// Water asked for that nothing has taken yet.
    carry: Litres,
}

impl Tool for WaterTool {
    const REACH: Metres = Metres(MUZZLE + HANDLE.z);

    /// A drop: a round bulb, and the taper that runs up from where it touches the bulb's
    /// sides to a point.
    fn icon(at: Vec2) -> f32 {
        let (tip, bulb, radius) = (Vec2::new(0.5, 0.1), Vec2::new(0.5, 0.62), 0.27);
        let reach = tip.distance(bulb);
        let spread = (radius / reach).asin().tan();
        let touching = (reach * reach - radius * radius).sqrt();
        let down = at - tip;
        let tapered = down.y > 0.0 && down.x.abs() < down.y * spread && down.length() < touching;
        f32::from(tapered || at.distance(bulb) < radius)
    }

    fn model(bench: &mut Workbench) {
        let shell = bench.finish(StandardMaterial {
            base_color: SHELL,
            perceptual_roughness: 0.4,
            ..default()
        });
        let frame = bench.finish(StandardMaterial {
            base_color: FRAME,
            metallic: 0.8,
            perceptual_roughness: 0.35,
            ..default()
        });
        let trim = bench.finish(StandardMaterial {
            base_color: TRIM,
            metallic: 0.1,
            perceptual_roughness: 0.35,
            ..default()
        });
        let tank = bench.finish(StandardMaterial {
            base_color: TANK_BLUE,
            perceptual_roughness: 0.12,
            reflectance: 0.7,
            ..default()
        });
        bench.handle_at(HANDLE);

        let profile = ConvexPolygon::new(PROFILE.map(Vec2::from))
            .expect("the shell's side view bulges nowhere inward");
        bench.part(
            Extrusion::new(profile, SHELL_WIDTH),
            &shell,
            Transform::from_rotation(side_on()),
        );
        bench.part(
            Cuboid::new(0.038, 0.12, 0.05),
            &frame,
            Transform::from_translation(HANDLE).with_rotation(Quat::from_rotation_x(-0.3)),
        );
        for side in [-1.0, 1.0] {
            bench.part(
                Cuboid::new(0.002, 0.012, 0.3),
                &trim,
                Transform::from_xyz(side * SHELL_WIDTH / 2.0, 0.045, -0.1),
            );
        }
        bench.part(
            Cuboid::new(0.014, 0.004, 0.26),
            &trim,
            Transform::from_xyz(0.0, DECK_TOP.y + 0.002, -0.11),
        );

        bench.part(
            Extrusion::new(Annulus::new(0.018, BARREL), MUZZLE - BREECH),
            &frame,
            Transform::from_xyz(0.0, BORE, -(BREECH + MUZZLE) / 2.0),
        );
        for fin in 0..FINS {
            bench.part(
                Extrusion::new(Annulus::new(BARREL, BARREL + 0.008), 0.012),
                &trim,
                Transform::from_xyz(0.0, BORE, -(BREECH + 0.05 + fin as f32 * 0.045)),
            );
        }
        let tank_middle = -(TANK.0 + TANK.1) / 2.0;
        bench.part(
            Capsule3d::new(TANK_RADIUS, TANK.1 - TANK.0),
            &tank,
            Transform::from_xyz(0.0, TANK_HEIGHT, tank_middle).with_rotation(laid_along()),
        );
        for strap in [TANK.0 + 0.02, TANK.1 - 0.02] {
            bench.part(
                Extrusion::new(Annulus::new(TANK_RADIUS, TANK_RADIUS + 0.004), 0.014),
                &frame,
                Transform::from_xyz(0.0, TANK_HEIGHT, -strap),
            );
            bench.part(
                Cuboid::new(0.016, BORE - TANK_HEIGHT, 0.014),
                &frame,
                Transform::from_xyz(0.0, (BORE + TANK_HEIGHT) / 2.0, -strap),
            );
        }
        let ring = bench.muzzle_light(GLOW, BARREL + 0.004, Vec3::new(0.0, BORE, -MUZZLE));
        bench.commands.entity(ring).insert(Nozzle);

        bench.dial_screen(
            Dial::Flow,
            GLOW,
            SCREEN,
            lying_on(DECK_TOP, DECK_FOOT, FLUSH),
        );
    }
}

pub struct WaterToolPlugin;

impl Plugin for WaterToolPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool::<WaterTool, _>(operate)
            .add_systems(Update, light.in_set(SimSet::Observe));
    }
}

const SHELL: Color = Color::srgb(0.88, 0.9, 0.92);
const FRAME: Color = Color::srgb(0.1, 0.105, 0.115);
const TRIM: Color = Color::srgb(0.05, 0.35, 0.95);
const TANK_BLUE: Color = Color::srgb(0.03, 0.2, 0.55);
const GLOW: Color = Color::srgb(0.1, 0.45, 1.0);

/// The shell from its side, x ahead and y up, and how wide it is. Its back slopes down from
/// the top of the deck to its foot, and the screen lies on that slope.
const DECK_TOP: Vec2 = Vec2::new(-0.05, 0.075);
const DECK_FOOT: Vec2 = Vec2::new(-0.13, -0.02);
const PROFILE: [(f32, f32); 7] = [
    (-0.09, -0.04),
    (0.14, -0.04),
    (BREECH, 0.0),
    (BREECH, 0.055),
    (0.27, DECK_TOP.y),
    (DECK_TOP.x, DECK_TOP.y),
    (DECK_FOOT.x, DECK_FOOT.y),
];
const SHELL_WIDTH: f32 = 0.09;
const HANDLE: Vec3 = Vec3::new(0.0, -0.085, 0.035);
/// The barrel: how high its bore lies, how thick it is, where it leaves the shell and where
/// it ends, and how many cooling fins it wears.
const BORE: f32 = 0.03;
const BARREL: f32 = 0.03;
const BREECH: f32 = 0.32;
const MUZZLE: f32 = 0.6;
const FINS: usize = 5;
/// The tank slung under the barrel: how thick it is, how low it hangs, and from where to
/// where ahead of the handle it runs.
const TANK_RADIUS: f32 = 0.03;
const TANK_HEIGHT: f32 = -0.045;
const TANK: (f32, f32) = (0.2, 0.46);
const SCREEN: Vec2 = Vec2::new(0.08, 0.046);
/// The screen is set in the slope of the back, lifted off it only enough to be seen apart.
const FLUSH: f32 = 0.0004;

/// The water tool's ring light.
#[derive(Component)]
struct Nozzle;

#[allow(clippy::too_many_arguments)]
fn operate(
    mut tool: ResMut<WaterTool>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    aim: Res<Aim>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
    fluid: Res<Fluid>,
    mut sheet: ResMut<Sheet>,
    mut window: ResMut<SheetWindow>,
) {
    if scroll.delta.y != 0.0 {
        Dial::Flow.adjust(&mut settings, scroll.delta.y.signum() as i32);
    }
    let target = aim.target.filter(|_| mouse.pressed(MouseButton::Left));
    tool.pouring = target.is_some();
    let Some(target) = target else {
        tool.carry = Litres(0.0);
        return;
    };
    tool.carry.0 += settings.flow.0 * time.delta_secs();
    // a round fall of water thinning to its rim is three times as wide as an even one of the
    // same flow would be: its mean is a third of what falls in its middle
    let even = settings.flow.0 / 1000.0 / FALL_SPEED.0;
    let wide = Metres(2.0 * (3.0 * even / std::f32::consts::PI).sqrt());
    let (_, outward) = sim.drum.depth_and_outward(target.point.to_array());
    let taken = window.pour(
        &mut sim.drum,
        &mut sheet,
        fluid.is_empty(),
        SheetPour {
            at: target.point.to_array(),
            velocity: outward.map(|out| out * FALL_SPEED.0 as f64),
            wide,
            water: tool.carry,
        },
    );
    if taken {
        tool.carry = Litres(0.0);
    }
}

fn light(tool: Res<WaterTool>, mut rings: Query<&mut MuzzleLight, With<Nozzle>>) {
    for mut ring in &mut rings {
        if ring.firing != tool.pouring {
            ring.firing = tool.pouring;
        }
    }
}
