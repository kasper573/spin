pub mod aim;
pub mod app;
pub mod controls;
pub mod drum;
pub mod hud;
pub mod persistence;
pub mod player;
pub mod scene;
pub mod settings;
pub mod sim;
pub mod testing;
pub mod thrusters;
pub mod water;

use bevy::asset::embedded_asset;
use bevy::prelude::App;

/// Bakes every shader under `shaders/` into the binary as `embedded://game/systems/shaders/*`.
pub fn embed_shaders(app: &mut App) {
    embedded_asset!(app, "shaders/water.wgsl");
    embedded_asset!(app, "shaders/glass.wgsl");
    embedded_asset!(app, "shaders/terrain.wgsl");
    embedded_asset!(app, "shaders/stars.wgsl");
    embedded_asset!(app, "shaders/space.wgsl");
    embedded_asset!(app, "shaders/stars_prepass.wgsl");
    embedded_asset!(app, "shaders/ripples.wgsl");
    embedded_asset!(app, "shaders/optics.wgsl");
    embedded_asset!(app, "shaders/air.wgsl");
    embedded_asset!(app, "drum/shaders/drum.wgsl");
    embedded_asset!(app, "drum/shaders/columns.wgsl");
}
