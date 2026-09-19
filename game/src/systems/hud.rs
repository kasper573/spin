//! The overlay: the crosshair, in the bottom left corner a square for each tool on the belt,
//! in the order of their number keys, the one that is out outlined in orange, and in the top
//! left every key binding that is not a tool's, with its current value, over the live
//! readouts, for as long as that list is wanted. What a tool is worked with is for the tool
//! itself to show.
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::core::fluid::Fluid;
use crate::core::units::Seconds;
use crate::systems::aim::Aim;
use crate::systems::controls::ClearAction;
use crate::systems::settings::{Action, Dial, Settings, Toggle};
use crate::systems::sim::{SimSet, Simulation};
use crate::systems::tools::Toolbelt;

/// How far up the view the tools' squares reach from its bottom edge, as a share of its
/// height.
pub const TOOLS_REACH: f32 = (MARGIN + SQUARE) / 100.0;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameRate>()
            .init_resource::<FrameWindow>()
            .add_systems(Startup, spawn)
            .add_systems(Update, (refresh, highlight).in_set(SimSet::Observe));
    }
}

/// The last full second of frames: how many there were, and the longest one. An average alone
/// would hide a stall among quick frames.
#[derive(Resource, Default, Clone, Copy)]
pub struct FrameRate {
    pub fps: f32,
    pub worst: Seconds,
}

/// The tools' squares, in hundredths of the view's height: how far they keep from its
/// edges and from each other, how big they are and how heavy their outline, and how much
/// room they leave round their icons.
const MARGIN: f32 = 2.4;
const SQUARE: f32 = 6.6;
const GAP: f32 = 0.9;
const OUTLINE: f32 = 0.3;
const INSET: f32 = 0.9;
const KEY_LETTERS: f32 = 1.5;
const CROSSHAIR: f32 = 3.0;
/// How finely an icon is drawn, and how many samples across each of its pixels is averaged
/// over.
const ICON_PIXELS: u32 = 64;
const ICON_SAMPLES: u32 = 4;
const PUT_AWAY: Color = Color::srgba(1.0, 1.0, 1.0, 0.85);
const OUT: Color = Color::srgb(1.0, 0.45, 0.04);

/// The second of frames being gathered.
#[derive(Resource, Default)]
struct FrameWindow {
    seconds: f32,
    frames: u32,
    worst: f32,
}

#[derive(Component)]
struct HudText;

/// The square of the tool in a slot.
#[derive(Component)]
struct ToolSquare(usize);

fn spawn(mut commands: Commands, belt: Res<Toolbelt>, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        HudText,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(Color::srgba(0.92, 0.95, 1.0, 0.92)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            width: percent(100),
            height: percent(100),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        children![(
            Text::new("+"),
            TextFont {
                font_size: FontSize::Vh(CROSSHAIR),
                ..default()
            },
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        )],
    ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: vh(MARGIN),
            bottom: vh(MARGIN),
            column_gap: vh(GAP),
            ..default()
        })
        .with_children(|squares| {
            for (slot, tool) in belt.slots().iter().enumerate() {
                squares.spawn((
                    ToolSquare(slot),
                    Node {
                        width: vh(SQUARE),
                        height: vh(SQUARE),
                        border: UiRect::all(vh(OUTLINE)),
                        padding: UiRect::all(vh(INSET)),
                        ..default()
                    },
                    BorderColor::all(PUT_AWAY),
                    BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.25)),
                    children![
                        (
                            ImageNode::new(images.add(icon(tool.icon))),
                            Node {
                                width: percent(100),
                                height: percent(100),
                                ..default()
                            },
                        ),
                        (
                            Text::new((slot + 1).to_string()),
                            TextFont {
                                font_size: FontSize::Vh(KEY_LETTERS),
                                ..default()
                            },
                            TextColor(PUT_AWAY),
                            Node {
                                position_type: PositionType::Absolute,
                                left: vh(OUTLINE),
                                top: vh(0.0),
                                ..default()
                            },
                        ),
                    ],
                ));
            }
        });
}

/// A tool's icon as a picture: white wherever the icon covers, each pixel as opaque as the
/// share of it that is covered.
fn icon(covers: fn(Vec2) -> f32) -> Image {
    let across = ICON_PIXELS * ICON_SAMPLES;
    let mut data = Vec::with_capacity((ICON_PIXELS * ICON_PIXELS * 4) as usize);
    for y in 0..ICON_PIXELS {
        for x in 0..ICON_PIXELS {
            let mut covered = 0.0;
            for sy in 0..ICON_SAMPLES {
                for sx in 0..ICON_SAMPLES {
                    let at = UVec2::new(x * ICON_SAMPLES + sx, y * ICON_SAMPLES + sy);
                    covered += covers((at.as_vec2() + 0.5) / across as f32);
                }
            }
            let share = covered / (ICON_SAMPLES * ICON_SAMPLES) as f32;
            data.extend([255, 255, 255, (share * 255.0).round() as u8]);
        }
    }
    Image::new(
        Extent3d {
            width: ICON_PIXELS,
            height: ICON_PIXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn highlight(belt: Res<Toolbelt>, mut squares: Query<(&ToolSquare, &mut BorderColor)>) {
    for (ToolSquare(slot), mut outline) in &mut squares {
        let colour = if belt.wielded() == Some(*slot) {
            OUT
        } else {
            PUT_AWAY
        };
        outline.set_if_neq(BorderColor::all(colour));
    }
}

fn hud_text(
    settings: &Settings,
    sim: &Simulation,
    fluid: &Fluid,
    engaged: bool,
    held: Option<Dial>,
    rate: FrameRate,
) -> String {
    if !settings.help {
        return format!("{} help", Toggle::Help.key_label());
    }
    let mut out = String::from("SPIN GRAVITY WHEEL\n");
    out.push_str(if engaged {
        "Esc releases the mouse\n\n"
    } else {
        "click the view to take control\n\n"
    });
    out.push_str("thrust   W/S fore/aft | A/D left/right | Space/Shift up/down | Q/E roll\n");
    out.push_str("mouse    pitch/yaw\n\n");
    out.push_str("hold a key and turn the mouse wheel to adjust:\n");
    for dial in Dial::ALL {
        let Some(key) = dial.key() else {
            continue;
        };
        let mark = if held == Some(dial) { '>' } else { ' ' };
        out.push_str(&format!(
            "{mark}{:<7} {:<15} {}\n",
            format!("{key:?}"),
            dial.label(),
            dial.value_text(settings)
        ));
    }
    out.push('\n');
    for toggle in Toggle::ALL {
        out.push_str(&format!(
            "{:<8} {:<15} {}\n",
            toggle.key_label(),
            toggle.label(),
            if toggle.get(settings) { "on" } else { "off" }
        ));
    }
    for action in Action::ALL {
        out.push_str(&format!("{:<8} {}\n", action.key_label(), action.label()));
    }
    out.push('\n');
    for action in ClearAction::ALL {
        out.push_str(&format!("{:<11} {}\n", action.key_label(), action.label()));
    }
    let footing = sim.footing();
    let footing = if footing.airborne {
        "airborne".to_owned()
    } else if !sim.avatar().solid {
        "ghost".to_owned()
    } else {
        format!(
            "weight {:.2} g | ground speed {:.1} m/s",
            footing.weight, footing.ground_speed
        )
    };
    out.push_str(&format!(
        "\n{footing}\nwater {:.1} m3 | spin {:.3} rad/s | {:.0} fps, worst {:.0} ms | sim {:.0}%",
        fluid.litres().0 / 1000.0,
        sim.drum.spin.0,
        rate.fps,
        rate.worst.0 * 1000.0,
        sim.rate * 100.0
    ));
    out
}

#[allow(clippy::too_many_arguments)]
fn refresh(
    time: Res<Time>,
    mut window: ResMut<FrameWindow>,
    mut rate: ResMut<FrameRate>,
    settings: Res<Settings>,
    sim: Res<Simulation>,
    fluid: Res<Fluid>,
    aim: Res<Aim>,
    keys: Res<ButtonInput<KeyCode>>,
    mut texts: Query<&mut Text, With<HudText>>,
) {
    let dt = time.delta_secs();
    window.seconds += dt;
    window.frames += 1;
    window.worst = window.worst.max(dt);
    if window.seconds >= 1.0 {
        *rate = FrameRate {
            fps: window.frames as f32 / window.seconds,
            worst: Seconds(window.worst),
        };
        *window = FrameWindow::default();
    }
    let text = hud_text(
        &settings,
        &sim,
        &fluid,
        aim.engaged,
        Dial::held(&keys),
        *rate,
    );
    for mut t in &mut texts {
        if t.0 != text {
            t.0 = text.clone();
        }
    }
}
