pub mod aim;
pub mod air;
pub mod app;
pub mod body;
pub mod controls;
pub mod drum;
pub mod figure;
pub mod hud;
pub mod persistence;
pub mod player;
pub mod portal;
pub mod scene;
pub mod settings;
pub mod sim;
pub mod testing;
pub mod thrusters;
pub mod tools;
pub mod water;
pub mod water_column;

use bevy::asset::embedded_asset;
use bevy::prelude::App;

/// Bakes every shader under `shaders/` into the binary as `embedded://game/systems/shaders/*`.
pub fn embed_shaders(app: &mut App) {
    embedded_asset!(app, "shaders/water.wgsl");
    embedded_asset!(app, "shaders/water_column.wgsl");
    embedded_asset!(app, "shaders/spray.wgsl");
    embedded_asset!(app, "shaders/glass.wgsl");
    embedded_asset!(app, "shaders/terrain.wgsl");
    embedded_asset!(app, "shaders/stars.wgsl");
    embedded_asset!(app, "shaders/space.wgsl");
    embedded_asset!(app, "shaders/stars_prepass.wgsl");
    embedded_asset!(app, "shaders/ripples.wgsl");
    embedded_asset!(app, "shaders/optics.wgsl");
    embedded_asset!(app, "shaders/air.wgsl");
    embedded_asset!(app, "shaders/ring.wgsl");
    embedded_asset!(app, "shaders/figure.wgsl");
    embedded_asset!(app, "shaders/portals.wgsl");
    embedded_asset!(app, "shaders/solid.wgsl");
    embedded_asset!(app, "drum/shaders/drum.wgsl");
    embedded_asset!(app, "drum/shaders/columns.wgsl");
}
