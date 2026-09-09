//! Mouse and keyboard: pointer lock, piloting the avatar, the dials (hold a key, turn the wheel),
//! the toggles, the clearing chords, and the three mouse buttons that act on the crosshair.
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::core::fluid::Fluid;
use crate::core::units::{Metres, MetresPerSecond};
use crate::core::web;
use crate::systems::aim::Aim;
use crate::systems::drum::GROUND_DEPTH;
use crate::systems::player::{PilotInput, Player};
use crate::systems::rafts::PLACEMENT_OFFSET;
use crate::systems::settings::{Action, Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};

/// Water appears this far in front of the surface the crosshair rests on.
const INJECT_DEPTH: Metres = Metres(1.0);
const MARKER_RADIUS: Metres = Metres(0.3);
/// The sculpting brush: how wide it is and how fast it raises the ground.
const BRUSH_SIZE: Metres = Metres(2.0);
const BRUSH_RATE: MetresPerSecond = MetresPerSecond(1.0);

/// Destructive actions, each a digit chorded with Backspace so nothing is lost to a stray key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearAction {
    ResetAll,
    ClearWater,
    ClearRafts,
    ResetLandscape,
}

impl ClearAction {
    pub const ALL: [ClearAction; 4] = [
        ClearAction::ResetAll,
        ClearAction::ClearWater,
        ClearAction::ClearRafts,
        ClearAction::ResetLandscape,
    ];

    pub const CHORD: KeyCode = KeyCode::Backspace;

    pub fn key(self) -> KeyCode {
        match self {
            ClearAction::ResetAll => KeyCode::Digit0,
            ClearAction::ClearWater => KeyCode::Digit1,
            ClearAction::ClearRafts => KeyCode::Digit2,
            ClearAction::ResetLandscape => KeyCode::Digit3,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            ClearAction::ResetAll => "Backspace+0",
            ClearAction::ClearWater => "Backspace+1",
            ClearAction::ClearRafts => "Backspace+2",
            ClearAction::ResetLandscape => "Backspace+3",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ClearAction::ResetAll => "reset everything to the initial state",
            ClearAction::ClearWater => "remove all water",
            ClearAction::ClearRafts => "remove all rafts",
            ClearAction::ResetLandscape => "flatten landscape",
        }
    }

    pub fn apply(
        self,
        settings: &mut Settings,
        sim: &mut Simulation,
        fluid: &mut Fluid,
        player: &mut Player,
    ) {
        match self {
            ClearAction::ResetAll => {
                *settings = Settings::default();
                *sim = Simulation::default();
                *player = Player::default();
                fluid.clear();
            }
            ClearAction::ClearWater => fluid.clear(),
            ClearAction::ClearRafts => sim.clear_rafts(),
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
    sculpting: bool,
}

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Controls>()
            .add_systems(
                Update,
                (pointer_lock, pilot, keys, mouse)
                    .chain()
                    .in_set(SimSet::Command),
            )
            .add_systems(Update, marker.in_set(SimSet::Observe));
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

/// Reads the mouse and the thruster keys (in the order of `Thruster::ALL`) into the avatar's
/// input; scripted input comes after.
pub fn pilot(
    controls: Res<Controls>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    mut player: ResMut<Player>,
    mut sim: ResMut<Simulation>,
) {
    if !controls.active {
        sim.avatar_input = default();
        return;
    }
    player.turn(motion.delta);
    let held = |key: KeyCode| keys.pressed(key) as u8 as f32;
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
        ],
    };
    sim.avatar_input = player.input(sim.avatar(), pilot);
}

fn keys(
    mut controls: ResMut<Controls>,
    keys: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
    mut fluid: ResMut<Fluid>,
    mut player: ResMut<Player>,
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
                action.apply(&mut settings, &mut sim, &mut fluid, &mut player);
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
    aim: Res<Aim>,
    settings: Res<Settings>,
    mut sim: ResMut<Simulation>,
    mut fluid: ResMut<Fluid>,
) {
    controls.sculpting = false;
    if !controls.active {
        controls.inject_carry = 0.0;
        return;
    }
    let Some(target) = aim.0 else {
        return;
    };
    let dt = time.delta_secs();
    if mouse.pressed(MouseButton::Left) && !mouse.just_pressed(MouseButton::Left) {
        controls.inject_carry += settings.flow.0 * dt / litres_per_particle();
        let count = controls.inject_carry.floor();
        controls.inject_carry -= count;
        let at = target.point + target.normal * INJECT_DEPTH.0;
        sim.inject(&mut fluid, at.to_array(), count as u32);
    } else {
        controls.inject_carry = 0.0;
    }
    if mouse.just_pressed(MouseButton::Right) {
        let at = target.point + target.normal * PLACEMENT_OFFSET.0;
        sim.spawn_raft(
            at.as_dvec3().to_array(),
            target.normal.as_dvec3().to_array(),
        );
    }
    if mouse.pressed(MouseButton::Middle) {
        controls.sculpting = true;
        let lower = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
        let amount = BRUSH_RATE.0 * dt * if lower { -1.0 } else { 1.0 };
        let phi = sim
            .drum
            .wheel_angle(target.point.x as f64, target.point.z as f64);
        sim.drum.landscape.sculpt(
            phi,
            target.point.y as f64,
            BRUSH_SIZE.0 as f64,
            amount as f64,
        );
    }
}

fn litres_per_particle() -> f32 {
    use crate::core::fluid::{PARTICLE_MASS, REST_DENSITY};
    PARTICLE_MASS / REST_DENSITY * 1000.0
}

fn marker(controls: Res<Controls>, aim: Res<Aim>, mut gizmos: Gizmos) {
    let Some(target) = aim.0 else {
        return;
    };
    let radius = if controls.sculpting {
        BRUSH_SIZE.0
    } else {
        MARKER_RADIUS.0
    };
    let rotation = Quat::from_rotation_arc(Vec3::Z, target.normal);
    let colour = if controls.active {
        Color::srgba(1.0, 1.0, 1.0, 0.9)
    } else {
        Color::srgba(1.0, 1.0, 1.0, 0.35)
    };
    gizmos.circle(
        Isometry3d::new(target.point + target.normal * 0.01, rotation),
        radius,
        colour,
    );
}
