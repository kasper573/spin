//! The whole application: bevy's defaults plus every plugin of ours, on the browser page.
use bevy::prelude::*;

use crate::core::web;
use crate::systems::{
    aim::AimPlugin, controls::ControlsPlugin, drum::DrumPlugin, hud::HudPlugin,
    persistence::PersistencePlugin, rafts::RaftsPlugin, scene::ScenePlugin,
    settings::SettingsPlugin, sim::SimulationPlugin, testing::TestingPlugin, water::WaterPlugin,
};

pub fn build() -> App {
    web::install_script_hooks();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(web::primary_window()),
        ..default()
    }));
    super::embed_shaders(&mut app);
    app.add_plugins((
        ScenePlugin,
        SimulationPlugin,
        SettingsPlugin,
        DrumPlugin,
        WaterPlugin,
        RaftsPlugin,
        AimPlugin,
        ControlsPlugin,
        HudPlugin,
        PersistencePlugin,
        TestingPlugin,
    ))
    .add_systems(PreUpdate, sync_window);
    app
}

fn sync_window(mut windows: Query<&mut Window>) {
    for mut window in &mut windows {
        web::sync_window(&mut window);
    }
}
