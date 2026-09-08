//! The viewer as a shuttle in the simulation: the camera rides the hull, the pilot's keys become
//! thrust and spin, and Enter decides whether the hull is solid or a ghost.
use bevy::prelude::*;

use crate::core::math::{Quatd, Vec3d};
use crate::core::rigid::Body;
use crate::core::shuttle::{self, Look, ROLL_RATE, ShuttleInput, THRUST};
use crate::systems::sim::{SimSet, Simulation};

/// Where the eye starts, outside the drum with the whole wheel in view.
pub const START_EYE: Vec3d = [4.9, 5.5, 7.2];

/// The camera that rides the shuttle.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerCamera;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Player {
    pub look: Look,
}

impl Default for Player {
    fn default() -> Self {
        Player {
            look: Look::facing(START_EYE, [0.0; 3]),
        }
    }
}

/// Requested motion in the view frame (x right, y up, z backward), each axis in [-1, 1], plus a
/// roll about the view axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PilotAxes {
    pub motion: Vec3,
    pub roll: f32,
}

impl Player {
    /// Mouse movement in pixels turns the head.
    pub fn turn(&mut self, delta: Vec2) {
        self.look.turn(delta.x as f64, delta.y as f64);
    }

    /// The shuttle's input for these pilot axes, given the hull's attitude.
    pub fn input(&self, hull: &Body, axes: PilotAxes) -> ShuttleInput {
        let view = self.view_rotation(hull);
        let thrust = view * (axes.motion.clamp_length_max(1.0) * THRUST as f32);
        let spin = view * Vec3::Z * (axes.roll.clamp(-1.0, 1.0) * ROLL_RATE as f32);
        ShuttleInput {
            thrust: thrust.as_dvec3().to_array(),
            spin: spin.as_dvec3().to_array(),
        }
    }

    /// Where the eye is and which way it faces.
    pub fn view(&self, hull: &Body) -> Transform {
        let eye = shuttle::eye(hull);
        Transform::from_xyz(eye[0] as f32, eye[1] as f32, eye[2] as f32)
            .with_rotation(self.view_rotation(hull))
    }

    /// Put the shuttle upright and at rest with its eye at `eye`, facing `target`.
    pub fn teleport(&mut self, hull: &mut Body, eye: Vec3d, target: Vec3d) {
        hull.place(shuttle::centre_for_eye(eye), [0.0, 0.0, 0.0, 1.0]);
        self.look = Look::facing(eye, target);
    }

    fn view_rotation(&self, hull: &Body) -> Quat {
        quat(hull.q)
            * Quat::from_rotation_y(self.look.yaw as f32)
            * Quat::from_rotation_x(self.look.pitch as f32)
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
    let view = player.view(sim.shuttle());
    for mut camera in &mut cameras {
        *camera = view;
    }
}
