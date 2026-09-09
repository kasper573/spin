//! The whole application: bevy's defaults plus every plugin of ours, on the browser page or
//! headless for the bench and the tests.
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::settings::RenderCreation;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use crate::core::fluid::FluidPlugin;
use crate::core::web;
use crate::systems::{
    aim::AimPlugin, controls::ControlsPlugin, drum::DrumPlugin, hud::HudPlugin,
    persistence::PersistencePlugin, player::PlayerPlugin, scene::ScenePlugin,
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
    app.add_plugins((
        ControlsPlugin,
        HudPlugin,
        PersistencePlugin,
        TestingPlugin,
        RenderDiagnosticsPlugin,
    ))
    .add_systems(PreUpdate, sync_window);
    app
}

/// The simulation without a window or the page: rendering still runs, into nothing, on the
/// calling thread and with the GPU device given.
pub fn build_headless(render: RenderCreation) -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                render_creation: render,
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
        FluidPlugin,
        WaterPlugin,
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
