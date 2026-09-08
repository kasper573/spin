//! Mouse and keyboard: pointer lock, spaceship flight, the dials and toggles, and the three mouse
//! buttons that act on the crosshair.
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::core::fly_camera::FlyCamera;
use crate::core::platform::ClientPlatform;
use crate::core::units::Metres;
use crate::systems::aim::Aim;
use crate::systems::rafts::PLACEMENT_OFFSET;
use crate::systems::settings::{Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};

/// Water appears this far in front of the surface the crosshair rests on.
const INJECT_DEPTH: Metres = Metres(0.4);
const MARKER_RADIUS: Metres = Metres(0.12);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    ClearWater,
    ClearRafts,
    ResetLandscape,
    ResetAll,
}

impl Action {
    pub const ALL: [Action; 4] = [
        Action::ClearWater,
        Action::ClearRafts,
        Action::ResetLandscape,
        Action::ResetAll,
    ];

    pub fn key(self) -> KeyCode {
        match self {
            Action::ClearWater => KeyCode::Backspace,
            Action::ClearRafts => KeyCode::Delete,
            Action::ResetLandscape => KeyCode::KeyL,
            Action::ResetAll => KeyCode::KeyR,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Action::ClearWater => "Backspace",
            Action::ClearRafts => "Delete",
            Action::ResetLandscape => "L",
            Action::ResetAll => "R",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Action::ClearWater => "remove all water",
            Action::ClearRafts => "remove all rafts",
            Action::ResetLandscape => "flatten landscape",
            Action::ResetAll => "reset everything",
        }
    }

    pub fn apply(self, sim: &mut Simulation) {
        match self {
            Action::ClearWater => sim.fluid.clear(),
            Action::ClearRafts => sim.rafts.clear(),
            Action::ResetLandscape => sim.drum.landscape.reset(),
            Action::ResetAll => sim.reset(),
        }
    }
}

#[derive(Resource, Default)]
pub struct Controls {
    /// Pointer lock held: the mouse steers and the buttons act on the crosshair.
    pub active: bool,
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
                (pointer_lock, fly, keys, mouse)
                    .chain()
                    .in_set(SimSet::Command),
            )
            .add_systems(Update, marker.in_set(SimSet::Observe));
    }
}

fn pointer_lock(
    mut controls: ResMut<Controls>,
    platform: Res<ClientPlatform>,
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
    if controls.active && !platform.0.pointer_locked(was_active) {
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

fn fly(
    controls: Res<Controls>,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
) {
    if !controls.active {
        return;
    }
    let dt = time.delta_secs();
    let axis =
        |neg: KeyCode, pos: KeyCode| (keys.pressed(pos) as i32 - keys.pressed(neg) as i32) as f32;
    for (mut transform, mut camera) in &mut cameras {
        FlyCamera::look(&mut transform, motion.delta);
        FlyCamera::roll(&mut transform, axis(KeyCode::KeyE, KeyCode::KeyQ), dt);
        let up =
            axis(KeyCode::ShiftLeft, KeyCode::Space) + axis(KeyCode::ShiftRight, KeyCode::Space);
        let axes = Vec3::new(
            axis(KeyCode::KeyA, KeyCode::KeyD),
            up.clamp(-1.0, 1.0),
            -axis(KeyCode::KeyS, KeyCode::KeyW),
        );
        camera.fly(&mut transform, axes, dt);
        if scroll.delta.y != 0.0 {
            camera.adjust_speed(scroll.delta.y.signum());
        }
    }
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<Settings>,
    mut sim: ResMut<Simulation>,
) {
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    for dial in Dial::ALL {
        if keys.just_pressed(dial.key()) {
            dial.adjust(&mut settings, if ctrl { -1 } else { 1 });
        }
    }
    for toggle in Toggle::ALL {
        if keys.just_pressed(toggle.key()) {
            toggle.flip(&mut settings);
        }
    }
    for action in Action::ALL {
        if keys.just_pressed(action.key()) {
            action.apply(&mut sim);
        }
    }
}

fn mouse(
    mut controls: ResMut<Controls>,
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    aim: Res<Aim>,
    settings: Res<Settings>,
    mut sim: ResMut<Simulation>,
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
        sim.inject(at.to_array(), count as u32, settings.match_wheel);
    } else {
        controls.inject_carry = 0.0;
    }
    if mouse.just_pressed(MouseButton::Right) {
        let at = target.point + target.normal * PLACEMENT_OFFSET.0;
        sim.spawn_raft(
            at.as_dvec3().to_array(),
            target.normal.as_dvec3().to_array(),
            settings.match_wheel,
        );
    }
    if mouse.pressed(MouseButton::Middle) {
        controls.sculpting = true;
        let lower = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
        let amount = settings.brush_rate.0 * dt * if lower { -1.0 } else { 1.0 };
        let phi = sim
            .drum
            .wheel_angle(target.point.x as f64, target.point.z as f64);
        sim.drum.landscape.sculpt(
            phi,
            target.point.y as f64,
            settings.brush_size.0 as f64,
            amount as f64,
        );
    }
}

fn litres_per_particle() -> f32 {
    use crate::core::fluid::{PARTICLE_MASS, REST_DENSITY};
    PARTICLE_MASS / REST_DENSITY * 1000.0
}

fn marker(controls: Res<Controls>, aim: Res<Aim>, settings: Res<Settings>, mut gizmos: Gizmos) {
    let Some(target) = aim.0 else {
        return;
    };
    let radius = if controls.sculpting {
        settings.brush_size.0
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
