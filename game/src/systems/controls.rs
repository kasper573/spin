//! Mouse and keyboard: pointer lock, piloting the avatar, the dials (hold a key, turn the wheel),
//! the toggles, the clearing chords, and the mouse buttons that act on the crosshair.
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::core::fluid::Fluid;
use crate::core::units::{Metres, MetresPerSecond, PixelsPerSecond};
use crate::core::web;
use crate::systems::aim::Aim;
use crate::systems::drum::GROUND_DEPTH;
use crate::systems::player::{PilotInput, Player};
use crate::systems::settings::{Action, Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};

/// Water appears this far in front of the surface the crosshair rests on.
pub const INJECT_DEPTH: Metres = Metres(1.0);
/// Mouse speed (pixels per second) at which a turning thruster is asked for full.
const MOUSE_FULL_SPEED: PixelsPerSecond = PixelsPerSecond(800.0);
/// The sculpting brush: its radius, the same on a ring of any size, and how fast it raises the
/// ground.
pub const BRUSH_SIZE: Metres = Metres(2.0);
pub const BRUSH_RATE: MetresPerSecond = MetresPerSecond(1.0);

/// Destructive actions, each a digit chorded with Backspace so nothing is lost to a stray key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearAction {
    ResetAll,
    ClearWater,
    ResetLandscape,
}

impl ClearAction {
    pub const ALL: [ClearAction; 3] = [
        ClearAction::ResetAll,
        ClearAction::ClearWater,
        ClearAction::ResetLandscape,
    ];

    pub const CHORD: KeyCode = KeyCode::Backspace;

    pub fn key(self) -> KeyCode {
        match self {
            ClearAction::ResetAll => KeyCode::Digit0,
            ClearAction::ClearWater => KeyCode::Digit1,
            ClearAction::ResetLandscape => KeyCode::Digit2,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            ClearAction::ResetAll => "Backspace+0",
            ClearAction::ClearWater => "Backspace+1",
            ClearAction::ResetLandscape => "Backspace+2",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ClearAction::ResetAll => "reset everything to the initial state",
            ClearAction::ClearWater => "remove all water",
            ClearAction::ResetLandscape => "flatten landscape",
        }
    }

    pub fn apply(self, settings: &mut Settings, sim: &mut Simulation, fluid: &mut Fluid) {
        match self {
            ClearAction::ResetAll => {
                *settings = Settings::default();
                *sim = Simulation::default();
                fluid.clear();
            }
            ClearAction::ClearWater => fluid.clear(),
            ClearAction::ResetLandscape => sim.drum.landscape.flatten(GROUND_DEPTH),
        }
    }
}

#[derive(Resource, Default)]
pub struct Controls {
    /// Pointer lock held: the mouse steers and the buttons act on the crosshair.
    pub active: bool,
    /// The dial whose key is held; the mouse wheel then turns it.
    pub held_dial: Option<Dial>,
    /// Litres requested but not yet turned into whole particles.
    inject_carry: f32,
}

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Controls>().add_systems(
            Update,
            (pointer_lock, pilot, keys, mouse)
                .chain()
                .in_set(SimSet::Command),
        );
    }
}

fn pointer_lock(
    mut controls: ResMut<Controls>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cursors: Query<&mut CursorOptions>,
) {
    let was_active = controls.active;
    if !controls.active && mouse.just_pressed(MouseButton::Left) {
        controls.active = true;
    }
    if keys.just_pressed(KeyCode::Escape) {
        controls.active = false;
    }
    if controls.active && was_active && !web::pointer_locked() {
        controls.active = false;
    }
    if controls.active != was_active {
        for mut cursor in &mut cursors {
            cursor.grab_mode = if controls.active {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            cursor.visible = !controls.active;
        }
    }
}

/// Reads the thruster keys and the mouse (in the order of `Thruster::ALL`) into the avatar's
/// input; scripted input comes after. The mouse asks for pitch and yaw by how fast it moves:
/// the turning thrusters fire as long as it keeps moving, at full when it moves this fast.
pub fn pilot(
    controls: Res<Controls>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    time: Res<Time>,
    player: Res<Player>,
    mut sim: ResMut<Simulation>,
) {
    if !controls.active {
        sim.avatar_input = default();
        return;
    }
    let held = |key: KeyCode| keys.pressed(key) as u8 as f32;
    let speed = motion.delta / time.delta_secs().max(1e-3) / MOUSE_FULL_SPEED.0;
    let turning = |v: f32| v.clamp(0.0, 1.0);
    let pilot = PilotInput {
        levels: [
            held(KeyCode::KeyW),
            held(KeyCode::KeyS),
            held(KeyCode::KeyA),
            held(KeyCode::KeyD),
            held(KeyCode::Space),
            held(KeyCode::ShiftLeft).max(held(KeyCode::ShiftRight)),
            held(KeyCode::KeyQ),
            held(KeyCode::KeyE),
            turning(-speed.y),
            turning(speed.y),
            turning(-speed.x),
            turning(speed.x),
        ],
    };
    sim.avatar_input = player.input(pilot);
}

fn keys(
    mut controls: ResMut<Controls>,
    keys: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
    mut fluid: ResMut<Fluid>,
) {
    controls.held_dial = Dial::ALL.into_iter().find(|dial| keys.pressed(dial.key()));
    if let Some(dial) = controls.held_dial
        && scroll.delta.y != 0.0
    {
        dial.adjust(&mut settings, scroll.delta.y.signum() as i32);
    }
    for toggle in Toggle::ALL {
        if keys.just_pressed(toggle.key()) {
            toggle.flip(&mut settings);
        }
    }
    for action in Action::ALL {
        if keys.just_pressed(action.key()) {
            action.apply(&mut settings);
        }
    }
    if keys.pressed(ClearAction::CHORD) {
        for action in ClearAction::ALL {
            if keys.just_pressed(action.key()) {
                action.apply(&mut settings, &mut sim, &mut fluid);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mouse(
    mut controls: ResMut<Controls>,
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut aim: ResMut<Aim>,
    settings: Res<Settings>,
    mut sim: ResMut<Simulation>,
    mut fluid: ResMut<Fluid>,
) {
    aim.engaged = controls.active;
    aim.brush = None;
    if !controls.active {
        controls.inject_carry = 0.0;
        return;
    }
    let Some(target) = aim.target else {
        return;
    };
    let dt = time.delta_secs();
    if mouse.pressed(MouseButton::Left) && !mouse.just_pressed(MouseButton::Left) {
        controls.inject_carry += settings.flow.0 * dt / fluid.resolution().litres_per_particle().0;
        let count = controls.inject_carry.floor();
        controls.inject_carry -= count;
        let at = target.point + target.normal * INJECT_DEPTH.0 as f64;
        sim.inject(&mut fluid, at.to_array(), count as u32);
    } else {
        controls.inject_carry = 0.0;
    }
    if mouse.pressed(MouseButton::Right) {
        let brush = BRUSH_SIZE;
        aim.brush = Some(brush);
        let lower = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
        let amount = BRUSH_RATE.0 * dt * if lower { -1.0 } else { 1.0 };
        sim.sculpt(target.point.to_array(), brush.0 as f64, amount as f64);
    }
}
