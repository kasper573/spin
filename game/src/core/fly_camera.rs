//! Free-flying camera: every motion and rotation is in the camera's own frame, like a spacecraft.
use bevy::prelude::*;

use crate::core::units::MetresPerSecond;

const LOOK_RATE: f32 = 0.0022;
const ROLL_RATE: f32 = 1.6;
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 40.0;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct FlyCamera {
    pub speed: MetresPerSecond,
}

impl Default for FlyCamera {
    fn default() -> Self {
        FlyCamera {
            speed: MetresPerSecond(4.0),
        }
    }
}

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
    pub fn fly(&self, transform: &mut Transform, axes: Vec3, dt: f32) {
        let step = transform.rotation * (axes.clamp_length_max(1.0) * self.speed.0 * dt);
        transform.translation += step;
    }

    pub fn adjust_speed(&mut self, scroll: f32) {
        self.speed =
            MetresPerSecond((self.speed.0 * (scroll * 0.1).exp()).clamp(MIN_SPEED, MAX_SPEED));
    }

    pub fn set_speed(&mut self, speed: MetresPerSecond) {
        self.speed = MetresPerSecond(speed.0.clamp(MIN_SPEED, MAX_SPEED));
    }
}
