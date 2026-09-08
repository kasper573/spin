//! Text-only overlay: every key binding with its current value, the crosshair, and live readouts.
use bevy::prelude::*;

use crate::core::fly_camera::FlyCamera;
use crate::systems::controls::{Action, Controls};
use crate::systems::settings::{Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameRate>()
            .add_systems(Startup, spawn)
            .add_systems(Update, refresh.in_set(SimSet::Observe));
    }
}

/// Smoothed frames per second.
#[derive(Resource, Default)]
pub struct FrameRate(pub f32);

#[derive(Component)]
struct HudText;

fn spawn(mut commands: Commands) {
    let font = TextFont {
        font_size: FontSize::Px(13.0),
        ..default()
    };
    commands.spawn((
        HudText,
        Text::new(""),
        font.clone(),
        TextColor(Color::srgba(0.92, 0.95, 1.0, 0.92)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new("+"),
                TextFont {
                    font_size: FontSize::Px(22.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
            ));
        });
}

fn refresh(
    time: Res<Time>,
    mut rate: ResMut<FrameRate>,
    settings: Res<Settings>,
    sim: Res<Simulation>,
    controls: Res<Controls>,
    cameras: Query<&FlyCamera>,
    mut texts: Query<&mut Text, With<HudText>>,
) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        rate.0 += (1.0 / dt - rate.0) * 0.05;
    }
    let speed = cameras.single().map(|c| c.speed.0).unwrap_or(0.0);
    let text = hud_text(&settings, &sim, &controls, speed, rate.0);
    for mut t in &mut texts {
        if t.0 != text {
            t.0 = text.clone();
        }
    }
}

pub fn hud_text(
    settings: &Settings,
    sim: &Simulation,
    controls: &Controls,
    fly_speed: f32,
    fps: f32,
) -> String {
    let mut out = String::new();
    out.push_str("SPIN GRAVITY WHEEL\n");
    out.push_str(if controls.active {
        "Esc releases the mouse\n\n"
    } else {
        "click the view to take control\n\n"
    });
    out.push_str(&format!(
        "fly      WASD move · Space/Shift up/down · Q/E roll · wheel speed {fly_speed:.1} m/s\n"
    ));
    out.push_str("mouse    LMB water · RMB raft · MMB raise land (Ctrl lowers)\n\n");
    for dial in Dial::ALL {
        out.push_str(&format!(
            "{:<8} {:<15} {}\n",
            dial.key_label(),
            dial.label(),
            dial.value_text(settings)
        ));
    }
    out.push_str("         hold Ctrl with F1-F7 to decrease\n\n");
    for toggle in Toggle::ALL {
        out.push_str(&format!(
            "{:<8} {:<15} {}\n",
            toggle.key_label(),
            toggle.label(),
            if toggle.get(settings) { "on" } else { "off" }
        ));
    }
    out.push('\n');
    for action in Action::ALL {
        out.push_str(&format!("{:<8} {}\n", action.key_label(), action.label()));
    }
    out.push_str(&format!(
        "\nwater {:.0} L · rafts {} · spin {:.2} rad/s · {:.0} fps · sim {:.0}%",
        sim.water().0,
        sim.rafts.len(),
        sim.drum.spin.0,
        fps,
        sim.rate * 100.0
    ));
    out
}
