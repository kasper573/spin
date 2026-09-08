//! The viewer as the avatar in the simulation: the camera rides the hull and the pilot's keys fire
//! its thrusters in the head's level frame, so that thrusting forward follows where the head is
//! turned but stays level with the hull.
use bevy::prelude::*;

use crate::core::avatar::{self, AvatarInput, Look, Thruster};
use crate::core::math::{Quatd, Vec3d};
use crate::core::rigid::Body;
use crate::systems::sim::{SimSet, Simulation};

/// The camera that rides the avatar.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerCamera;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Player {
    pub look: Look,
}

/// How hard the pilot asks each thruster to fire, 0 to 1, in the order of `Thruster::ALL`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PilotInput {
    pub levels: [f32; 8],
}

impl PilotInput {
    /// These thrusters at full, the rest idle.
    pub fn firing(thrusters: &[Thruster]) -> PilotInput {
        let mut input = PilotInput::default();
        for thruster in thrusters {
            input.levels[*thruster as usize] = 1.0;
        }
        input
    }
}

impl Player {
    /// Mouse movement in pixels turns the head.
    pub fn turn(&mut self, delta: Vec2) {
        self.look.turn(delta.x as f64, delta.y as f64);
    }

    /// The avatar's input for what the pilot holds, given the hull's attitude.
    pub fn input(&self, hull: &Body, pilot: PilotInput) -> AvatarInput {
        let level = self.level_frame(hull);
        let axis = |v: Vec3| (level * v).as_dvec3().to_array();
        AvatarInput {
            frame: [axis(Vec3::X), axis(Vec3::Y), axis(Vec3::Z)],
            levels: pilot.levels.map(|l| l as f64),
        }
    }

    /// The frame the thrusters act in: the hull's, turned to where the head looks, but not
    /// tilted with it, so looking down does not slow a walk.
    pub fn level_frame(&self, hull: &Body) -> Quat {
        quat(hull.q) * Quat::from_rotation_y(self.look.yaw as f32)
    }

    /// Where the eye is and which way it faces.
    pub fn view(&self, hull: &Body) -> Transform {
        let eye = avatar::eye(hull);
        Transform::from_xyz(eye[0] as f32, eye[1] as f32, eye[2] as f32)
            .with_rotation(self.view_rotation(hull))
    }

    /// Put the avatar upright in world space and at rest with its eye at `eye`, facing `target`.
    pub fn teleport(&mut self, hull: &mut Body, eye: Vec3d, target: Vec3d) {
        hull.place(avatar::centre_for_eye(eye), [0.0, 0.0, 0.0, 1.0]);
        self.look = Look::facing(eye, target);
    }

    fn view_rotation(&self, hull: &Body) -> Quat {
        self.level_frame(hull) * Quat::from_rotation_x(self.look.pitch as f32)
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Player>()
            .add_systems(Update, ride.in_set(SimSet::Observe));
    }
}

fn quat(q: Quatd) -> Quat {
    Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32)
}

fn ride(
    player: Res<Player>,
    sim: Res<Simulation>,
    mut cameras: Query<&mut Transform, With<PlayerCamera>>,
) {
    let view = player.view(sim.avatar());
    for mut camera in &mut cameras {
        *camera = view;
    }
}
