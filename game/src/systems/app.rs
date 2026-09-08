//! The whole application: bevy's defaults plus every plugin of ours, on the browser page or
//! headless for the bench and the tests.
use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use crate::core::fluid::FluidPlugin;
use crate::core::web;
use crate::systems::drum::{HALF_WIDTH, RADIUS};
use crate::systems::{
    aim::AimPlugin, controls::ControlsPlugin, drum::DrumPlugin, hud::HudPlugin,
    persistence::PersistencePlugin, player::PlayerPlugin, rafts::RaftsPlugin, scene::ScenePlugin,
    settings::SettingsPlugin, sim::SimulationPlugin, testing::TestingPlugin,
    thrusters::ThrustersPlugin, water::WaterPlugin,
};

pub fn build() -> App {
    web::install_script_hooks();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(web::primary_window()),
        ..default()
    }));
    simulation(&mut app);
    app.add_plugins((ControlsPlugin, HudPlugin, PersistencePlugin, TestingPlugin))
        .add_systems(PreUpdate, sync_window);
    app
}

/// The simulation without a window or the page: rendering still runs, into nothing.
pub fn build_headless() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>(),
    );
    simulation(&mut app);
    app
}

fn simulation(app: &mut App) {
    super::embed_shaders(app);
    app.add_plugins((
        ScenePlugin,
        SimulationPlugin,
        SettingsPlugin,
        DrumPlugin,
        FluidPlugin {
            min: [-RADIUS, -HALF_WIDTH, -RADIUS],
            max: [RADIUS, HALF_WIDTH, RADIUS],
        },
        WaterPlugin,
        RaftsPlugin,
        PlayerPlugin,
        ThrustersPlugin,
        AimPlugin,
    ));
}

fn sync_window(mut windows: Query<&mut Window>) {
    for mut window in &mut windows {
        web::sync_window(&mut window);
    }
}
