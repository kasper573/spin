//! The portals as they are seen: what the surfaces they are let into are told of them, so that
//! the ground and the glass can draw them as parts of themselves (see `portals.wgsl`), what is
//! seen through them (see `eyes`), in which whatever stands in the ring is seen too (see
//! `echoes`), and the light their burning rims throw on what is round them.
mod echoes;
mod eyes;
mod lights;
mod solids;

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use crate::core::units::Seconds;
use crate::systems::drum::{Drum, DrumSurface, FLAME_BAND, MouthColour, MouthCoords};
use crate::systems::scene::Vantage;

pub use eyes::Pictures;
pub use solids::{SolidCopies, SolidMaterial, solid};

/// The colour a portal of each kind burns in.
pub fn portal_colour(colour: MouthColour) -> Color {
    match colour {
        MouthColour::Blue => Color::srgb(0.05, 0.45, 1.0),
        MouthColour::Orange => Color::srgb(1.0, 0.42, 0.03),
    }
}

/// How bright a portal's colour is at its bottom, its top being one: a portal shows which way
/// up it is by being brighter toward its top.
pub const DARKEST: f64 = 0.12;

/// How bright a portal's colour is `up` its mouth, in radii from its middle.
pub fn grade(up: f64) -> f64 {
    let share = (0.5 + 0.5 * up / (1.0 + FLAME_BAND)).clamp(0.0, 1.0);
    DARKEST + (1.0 - DARKEST) * share
}

/// A mouth as the shaders take it; `portals.wgsl` says what rides along in each fourth place.
#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct MouthUniform {
    origin: Vec4,
    spinward: Vec4,
    upward: Vec4,
    inward: Vec4,
    colour: Vec4,
    centre: Vec4,
    through: [Vec4; 3],
    found: Mat4,
}

/// The pair of mouths as the shaders take them. A material that draws them keeps them at
/// binding 11, which is where `portals.wgsl` reads them from.
#[derive(ShaderType, Clone, Debug, Default)]
pub struct MouthsUniform {
    mouths: [MouthUniform; 2],
    shape: Vec4,
    look: Vec4,
    about: Vec4,
}

impl MouthsUniform {
    /// The drum's mouths as they lie about a vantage at a time.
    pub fn of(drum: &Drum, vantage: &Vantage, found: [Mat4; 2], time: Seconds) -> MouthsUniform {
        let mouths = [
            (MouthColour::Blue, found[0]),
            (MouthColour::Orange, found[1]),
        ]
        .map(|(colour, found)| {
            let Some(mouth) = drum.mouths.get(colour) else {
                return MouthUniform::default();
            };
            let seat = drum.mouth_seat(mouth);
            let on = match mouth.surface() {
                DrumSurface::Wall => 1.0,
                DrumSurface::Cap(_) => 2.0,
            };
            let (sin, cos) = mouth.roll.0.sin_cos();
            let frame = vantage.frame;
            let along = |v: [f64; 3], w: f64| Vec3::from(v.map(|x| x as f32)).extend(w as f32);
            let laid = |v: [f64; 3], w: f64| along(frame.vector(v), w);
            // the rows of the turn that takes what goes in at this mouth out of the other
            let through = drum.mouth_view(colour).map_or([Vec4::ZERO; 3], |view| {
                let turned = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    .map(|axis| frame.vector(view.turned(frame.vector_back(axis))));
                [0, 1, 2].map(|row| along([turned[0][row], turned[1][row], turned[2][row]], 0.0))
            });
            MouthUniform {
                origin: vantage.local(seat.origin).extend(on),
                spinward: laid(seat.spinward, sin),
                upward: laid(seat.upward, cos),
                inward: laid(seat.inward, 0.0),
                colour: portal_colour(colour).to_linear().to_vec4(),
                centre: vantage
                    .local(drum.mouth_point(mouth, MouthCoords::default()))
                    .extend(0.0),
                through,
                found,
            }
        });
        MouthsUniform {
            mouths,
            shape: Vec4::new(
                drum.ring.radius.0,
                drum.mouths.radius() as f32,
                drum.mouths.fill() as f32,
                time.0 % FIRE_REPEATS.0,
            ),
            look: Vec4::new(
                FLAME_BAND as f32,
                DARKEST as f32,
                drum.ring.half_width.0,
                0.0,
            ),
            about: {
                let [x, y, z] = vantage.viewpoint.origin;
                Vec4::new(x as f32, (y + vantage.frame.site.y) as f32, z as f32, 0.0)
            },
        }
    }
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            eyes::PortalEyesPlugin,
            echoes::EchoPlugin,
            lights::RimLightPlugin,
            solids::SolidsPlugin,
        ));
    }
}

/// The fire is drawn from the time, which is kept small enough to be told apart from one frame
/// to the next however long the world has run: it is wound back this often, which is a whole
/// number of the runs the fire is carried out in.
const FIRE_REPEATS: Seconds = Seconds(3600.0);
