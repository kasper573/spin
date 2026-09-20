//! Mouse and keyboard: pointer lock, piloting the avatar, the dials (hold a key, turn the
//! wheel), the toggles and the clearing chords. The tools take whatever input is left once
//! these have had theirs.
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::core::fluid::Fluid;
use crate::core::shallows::Shallows;
use crate::core::units::PixelsPerSecond;
use crate::core::web;
use crate::systems::aim::Aim;
use crate::systems::drum::GROUND_DEPTH;
use crate::systems::player::{PilotInput, Player};
use crate::systems::settings::{Action, Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::tools::ToolInput;

/// Mouse speed (pixels per second) at which a turning thruster is asked for full.
const MOUSE_FULL_SPEED: PixelsPerSecond = PixelsPerSecond(800.0);

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

    pub fn apply(
        self,
        settings: &mut Settings,
        sim: &mut Simulation,
        (fluid, shallows): (&mut Fluid, &mut Shallows),
    ) {
        match self {
            ClearAction::ResetAll => {
                *settings = Settings::default();
                *sim = Simulation::default();
                fluid.clear();
                shallows.clear();
            }
            ClearAction::ClearWater => {
                fluid.clear();
                shallows.clear();
            }
            ClearAction::ResetLandscape => sim.drum.landscape.flatten(GROUND_DEPTH),
        }
    }
}

#[derive(Resource, Default)]
pub struct Controls {
    /// Pointer lock held: the mouse steers, and the tool that is out answers to it.
    pub active: bool,
}

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Controls>().add_systems(
            Update,
            (pointer_lock, pilot, keys)
                .chain()
                .in_set(SimSet::Command)
                .before(ToolInput),
        );
    }
}

/// The click that takes the controls is used up by taking them: a tool only answers to a
/// button pressed once they are held.
fn pointer_lock(
    mut controls: ResMut<Controls>,
    mut aim: ResMut<Aim>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cursors: Query<&mut CursorOptions>,
) {
    let was_active = controls.active;
    if !controls.active && mouse.just_pressed(MouseButton::Left) {
        controls.active = true;
        mouse.reset(MouseButton::Left);
    }
    if keys.just_pressed(KeyCode::Escape) {
        controls.active = false;
    }
    if controls.active && was_active && !web::pointer_locked() {
        controls.active = false;
    }
    aim.engaged = controls.active;
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

/// A held dial takes the wheel, and a clearing chord its digit, so neither reaches the tools.
fn keys(
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut scroll: ResMut<AccumulatedMouseScroll>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
    (mut fluid, mut shallows): (ResMut<Fluid>, ResMut<Shallows>),
) {
    if let Some(dial) = Dial::held(&keys) {
        if scroll.delta.y != 0.0 {
            dial.adjust(&mut settings, scroll.delta.y.signum() as i32);
        }
        scroll.delta = Vec2::ZERO;
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
                action.apply(&mut settings, &mut sim, (&mut fluid, &mut shallows));
                keys.clear_just_pressed(action.key());
            }
        }
    }
}
