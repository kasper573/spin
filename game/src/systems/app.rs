//! The whole application, assembled for whichever host installs a platform adapter.
use bevy::prelude::*;

use crate::core::platform::{ClientPlatform, Platform};
use crate::systems::{
    aim::AimPlugin, controls::ControlsPlugin, drum::DrumPlugin, hud::HudPlugin,
    persistence::PersistencePlugin, rafts::RaftsPlugin, scene::ScenePlugin,
    settings::SettingsPlugin, sim::SimulationPlugin, testing::TestingPlugin, water::WaterPlugin,
};

pub fn build(platform: impl Platform, window: Window) -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(window),
        ..default()
    }))
    .insert_resource(ClientPlatform(Box::new(platform)));
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

fn sync_window(platform: Res<ClientPlatform>, mut windows: Query<&mut Window>) {
    for mut window in &mut windows {
        platform.0.sync_window(&mut window);
    }
}
