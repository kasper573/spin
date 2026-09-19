//! A screen set in a tool's body that shows one of the settings' dials as the tool's own
//! instrument: its value writ large over its unit, and nothing else. It is laid out as UI and
//! drawn into a picture of its own, which a panel in the tool gives off as light, the way a
//! screen does.
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

use crate::systems::settings::{Dial, Settings};
use crate::systems::sim::SimSet;
use crate::systems::tools::{Carried, Workbench};

impl Workbench<'_, '_, '_> {
    /// A screen this big for a dial, lit in this colour, its face looking along its own +z.
    pub fn dial_screen(&mut self, dial: Dial, accent: Color, size: Vec2, at: Transform) {
        let mut picture = Image::new_fill(
            Extent3d {
                width: PIXELS_ACROSS,
                height: (PIXELS_ACROSS as f32 * size.y / size.x) as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        picture.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT;
        let picture = self.images.add(picture);
        let camera = self
            .commands
            .spawn((
                Screen(self.slot),
                Camera2d,
                Camera {
                    order: -1,
                    is_active: false,
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(picture.clone().into()),
                Msaa::Off,
            ))
            .id();
        let lettering = |size: f32| TextFont {
            font_size: FontSize::Vh(size),
            ..default()
        };
        self.commands.spawn((
            UiTargetCamera(camera),
            Node {
                width: percent(100),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::BLACK.mix(&accent, 0.06)),
            children![
                (
                    DialValue(dial),
                    Text::default(),
                    lettering(LARGE_LETTERS),
                    TextColor(Color::WHITE),
                ),
                (
                    Text::new(dial.unit()),
                    lettering(SMALL_LETTERS),
                    TextColor(accent),
                ),
            ],
        ));
        let mut panel = self.finish(StandardMaterial {
            base_color: Color::BLACK,
            emissive: LinearRgba::WHITE * LUMINANCE,
            emissive_exposure_weight: 1.0,
            emissive_texture: Some(picture),
            perceptual_roughness: 0.15,
            ..default()
        });
        // a mirror is too far off to read the screen in: it shows its light, all in one
        panel.mirrored.glow = Color::WHITE.mix(&accent, 0.5).to_linear() * (LUMINANCE * LIT);
        self.part(Rectangle::new(size.x, size.y), &panel, at);
    }
}

pub struct ScreenPlugin;

impl Plugin for ScreenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (read.run_if(resource_changed::<Settings>), switch_on).in_set(SimSet::Observe),
        );
    }
}

/// How finely a screen is drawn, and how tall its lettering stands, as shares of its height:
/// the value as tall as leaves room across the screen for the longest a dial runs to.
const PIXELS_ACROSS: u32 = 256;
const SMALL_LETTERS: f32 = 24.0;
const LARGE_LETTERS: f32 = 54.0;
/// What white on a screen gives off: enough to read in full sun. About this much of a screen
/// is lit, lettering and ground taken together.
const LUMINANCE: f32 = 60_000.0;
const LIT: f32 = 0.15;

/// The camera that draws the screen of the tool in a slot.
#[derive(Component)]
struct Screen(usize);

/// The lettering that shows a dial's value.
#[derive(Component)]
struct DialValue(Dial);

fn read(settings: Res<Settings>, mut values: Query<(&DialValue, &mut Text)>) {
    for (DialValue(dial), mut text) in &mut values {
        text.0 = dial.number(&settings);
    }
}

/// A screen is only drawn while its tool is out to be looked at.
fn switch_on(carried: Res<Carried>, mut screens: Query<(&Screen, &mut Camera)>) {
    for (Screen(slot), mut camera) in &mut screens {
        let on = carried.slot == Some(*slot);
        if camera.is_active != on {
            camera.is_active = on;
        }
    }
}
