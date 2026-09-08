//! Free-flying camera: every motion and rotation is in the camera's own frame, like a spacecraft.
use bevy::prelude::*;

use crate::core::units::MetresPerSecond;

const LOOK_RATE: f32 = 0.0022;
const ROLL_RATE: f32 = 1.6;
const SPEED: MetresPerSecond = MetresPerSecond(4.0);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyCamera;

impl FlyCamera {
    /// Mouse movement in pixels turns the camera about its own axes.
    pub fn look(transform: &mut Transform, delta: Vec2) {
        transform.rotate_local_y(-delta.x * LOOK_RATE);
        transform.rotate_local_x(-delta.y * LOOK_RATE);
    }

    /// `direction` in [-1, 1] rolls about the view axis.
    pub fn roll(transform: &mut Transform, direction: f32, dt: f32) {
        transform.rotate_local_z(direction * ROLL_RATE * dt);
    }

    /// `axes` is the requested motion in the camera frame (x right, y up, z backward).
    pub fn fly(transform: &mut Transform, axes: Vec3, dt: f32) {
        let step = transform.rotation * (axes.clamp_length_max(1.0) * SPEED.0 * dt);
        transform.translation += step;
    }
}
