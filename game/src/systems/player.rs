//! The viewer as the avatar in the simulation: the camera is fixed in the hull at the eye,
//! looking straight ahead, and the pilot's keys and mouse fire its thrusters, the mouse the
//! turning ones, so the body turns to look.
use bevy::prelude::*;

use crate::core::avatar::{self, AvatarInput, Gyros};
use crate::core::math::{Quatd, Vec3d, cross, norm, quat_from_basis};
use crate::core::rigid::Body;
use crate::systems::sim::{SimSet, Simulation};

/// The camera that rides the avatar.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerCamera;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Player;

/// How hard the pilot asks each thruster to fire, 0 to 1, in the order of `Thruster::ALL`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PilotInput {
    pub levels: [f32; 12],
}

impl PilotInput {
    /// These thrusters at full, the rest idle.
    pub fn firing(thrusters: &[avatar::Thruster]) -> PilotInput {
        let mut input = PilotInput::default();
        for thruster in thrusters {
            input.levels[*thruster as usize] = 1.0;
        }
        input
    }
}

impl Player {
    /// The avatar's input for what the pilot holds.
    pub fn input(&self, pilot: PilotInput) -> AvatarInput {
        AvatarInput {
            levels: pilot.levels.map(|l| l as f64),
        }
    }

    /// Where the eye is and which way it faces.
    pub fn view(&self, hull: &Body) -> Transform {
        let eye = avatar::eye(hull);
        Transform::from_xyz(eye[0] as f32, eye[1] as f32, eye[2] as f32).with_rotation(quat(hull.q))
    }

    /// Put the avatar at rest with its eye at `eye`, facing `target` with its head toward the
    /// world's up, and its gyros holding it there.
    pub fn teleport(&mut self, sim: &mut Simulation, eye: Vec3d, target: Vec3d) {
        let q = facing(eye, target);
        sim.avatar_mut().place(avatar::centre_for_eye(eye), q);
        sim.gyros = Gyros::holding(sim.avatar());
    }
}

/// The orientation whose forward (-z) points from `eye` to `target`, as upright as it can be.
fn facing(eye: Vec3d, target: Vec3d) -> Quatd {
    let mut back = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
    let len = norm(&back);
    if len < 1e-9 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    back = back.map(|c| c / len);
    let helper = if back[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, -1.0]
    };
    let mut right = cross(&helper, &back);
    let rl = norm(&right).max(1e-12);
    right = right.map(|c| c / rl);
    let up = cross(&back, &right);
    quat_from_basis(&right, &up, &back)
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
