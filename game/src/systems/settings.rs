//! User-adjustable parameters and the keys that drive them. Every dial is a value with a range and
//! a step; every toggle flips a flag; every action is a single press.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::avatar;
use crate::core::sheet::Sheet;
use crate::core::units::{
    EARTH_GRAVITY, KilogramsPerCubicMetre, LitresPerSecond, Metres, MetresPerSecondSquared,
    RadiansPerSecond,
};
use crate::systems::air::Air;
use crate::systems::drum::{DEFAULT_RING, Ring, SheetWindow};
use crate::systems::sim::{SimSet, Simulation, standing_gravity, standing_spin};

/// How many of a dial's fine steps it takes before the steps grow tenfold.
const FINE_STEPS: f32 = 100.0;

#[derive(Resource, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub spin: RadiansPerSecond,
    /// How much water the water tool pours.
    pub flow: LitresPerSecond,
    /// How much land the land tool puts down, or takes up.
    pub build: LitresPerSecond,
    /// How wide the portal tool makes its portals.
    pub portal: Metres,
    pub air: bool,
    /// Whether the avatar is solid to the drum and the water, or a ghost.
    pub collisions: bool,
    /// Whether the HUD lists the keys and the readouts, or only how to bring them back.
    pub help: bool,
    /// The ring's size across and along its axis.
    pub diameter: Metres,
    pub width: Metres,
    /// Peak acceleration of each of the avatar's thrusters.
    pub thrust: MetresPerSecondSquared,
}

impl Default for Settings {
    fn default() -> Self {
        let spin = standing_spin(DEFAULT_RING);
        Settings {
            spin,
            flow: LitresPerSecond(20_000.0),
            build: LitresPerSecond(4_000.0),
            portal: Metres(2.5),
            air: true,
            collisions: true,
            help: true,
            diameter: Metres(DEFAULT_RING.radius.0 * 2.0),
            width: Metres(DEFAULT_RING.half_width.0 * 2.0),
            thrust: avatar::equalized_thrust(standing_gravity(spin, DEFAULT_RING)),
        }
    }
}

impl Settings {
    /// Every dial clamped to its range; anything not finite falls back to the default.
    pub fn sanitized(mut self) -> Settings {
        let defaults = Settings::default();
        for dial in Dial::ALL {
            let value = dial.get(&self);
            if !(dial.min()..=dial.max()).contains(&value) {
                let value = if value.is_finite() {
                    value.clamp(dial.min(), dial.max())
                } else {
                    dial.get(&defaults)
                };
                dial.set(&mut self, value);
            }
        }
        self
    }

    /// The ring these settings ask for.
    pub fn ring(&self) -> Ring {
        Ring {
            radius: Metres(self.diameter.0 / 2.0),
            half_width: Metres(self.width.0 / 2.0),
        }
    }

    /// The gravity the avatar stands under with this spin on this ring.
    pub fn standing_gravity(&self) -> MetresPerSecondSquared {
        standing_gravity(self.spin, self.ring())
    }

    /// Set the thrusters' power to what the standing gravity calls for, as the initial state
    /// does, so that after changing the ring or its spin the game plays as it did at the start.
    pub fn equalize_thrust(&mut self) {
        self.thrust = avatar::equalized_thrust(self.standing_gravity());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dial {
    Spin,
    Flow,
    Build,
    Portal,
    Diameter,
    Width,
    Thrust,
}

impl Dial {
    pub const ALL: [Dial; 7] = [
        Dial::Spin,
        Dial::Flow,
        Dial::Build,
        Dial::Portal,
        Dial::Diameter,
        Dial::Width,
        Dial::Thrust,
    ];

    /// The key that, held, puts the mouse wheel on the dial. The dials a tool carries have none:
    /// they are turned with the tool in hand.
    pub fn key(self) -> Option<KeyCode> {
        match self {
            Dial::Spin => Some(KeyCode::F1),
            Dial::Diameter => Some(KeyCode::F4),
            Dial::Width => Some(KeyCode::F5),
            Dial::Thrust => Some(KeyCode::F6),
            Dial::Flow | Dial::Build | Dial::Portal => None,
        }
    }

    /// The dial whose key is held, if one is.
    pub fn held(keys: &ButtonInput<KeyCode>) -> Option<Dial> {
        Dial::ALL
            .into_iter()
            .find(|dial| dial.key().is_some_and(|key| keys.pressed(key)))
    }

    pub fn label(self) -> &'static str {
        match self {
            Dial::Spin => "spin",
            Dial::Flow => "flow",
            Dial::Build => "build",
            Dial::Portal => "portal",
            Dial::Diameter => "ring diameter",
            Dial::Width => "ring width",
            Dial::Thrust => "thruster power",
        }
    }

    pub fn min(self) -> f32 {
        match self {
            Dial::Spin | Dial::Flow | Dial::Build | Dial::Thrust => 0.0,
            Dial::Diameter => 6.0,
            Dial::Width => 2.0,
            Dial::Portal => 0.5,
        }
    }

    /// The rates run up to 999 of their unit, and the ring's size has no
    /// end, nor has a portal's: the simulation represents any size, and whether a portal fits
    /// where it is wanted is for the place to say.
    pub fn max(self) -> f32 {
        match self {
            Dial::Spin | Dial::Thrust => 999.0,
            Dial::Flow | Dial::Build => 999_000.0,
            Dial::Diameter | Dial::Width | Dial::Portal => f32::MAX,
        }
    }

    /// The finest step of the dial, taken near zero.
    pub fn step(self) -> f32 {
        match self {
            Dial::Spin => 0.05,
            Dial::Flow => 5000.0,
            Dial::Build => 500.0,
            Dial::Diameter | Dial::Width => 1.0,
            Dial::Portal => 0.1,
            Dial::Thrust => 0.5,
        }
    }

    /// The step taken at a value: ten times coarser for every ten times the value grows past
    /// `FINE_STEPS` steps, so a dial that runs to 999 still turns through its range in a few
    /// hundred clicks without losing the fine steps near zero.
    pub fn step_at(self, value: f32) -> f32 {
        let fine = self.step();
        let coarse = (value.abs() / (fine * FINE_STEPS)).log10().floor() + 1.0;
        fine * 10f32.powf(coarse.max(0.0))
    }

    pub fn get(self, s: &Settings) -> f32 {
        match self {
            Dial::Spin => s.spin.0,
            Dial::Flow => s.flow.0,
            Dial::Build => s.build.0,
            Dial::Portal => s.portal.0,
            Dial::Diameter => s.diameter.0,
            Dial::Width => s.width.0,
            Dial::Thrust => s.thrust.0 as f32,
        }
    }

    pub fn set(self, s: &mut Settings, value: f32) {
        let value = value.clamp(self.min(), self.max());
        match self {
            Dial::Spin => s.spin = RadiansPerSecond(value),
            Dial::Flow => s.flow = LitresPerSecond(value),
            Dial::Build => s.build = LitresPerSecond(value),
            Dial::Portal => s.portal = Metres(value),
            Dial::Diameter => s.diameter = Metres(value),
            Dial::Width => s.width = Metres(value),
            Dial::Thrust => s.thrust = MetresPerSecondSquared(value as f64),
        }
    }

    /// Move one step up (`direction` > 0) or down, snapping to the grid of the step taken.
    pub fn adjust(self, s: &mut Settings, direction: i32) {
        let value = self.get(s);
        let step = if direction < 0 {
            self.step_at(value - self.step_at(value) * 0.5)
        } else {
            self.step_at(value)
        };
        let steps = (value / step).round() + direction as f32;
        self.set(s, steps * step);
    }

    /// The dial's value as a number of its unit.
    pub fn number(self, s: &Settings) -> String {
        let v = self.get(s);
        match self {
            Dial::Spin => format!("{v:.3}"),
            Dial::Flow | Dial::Build => format!("{:.1}", v / 1000.0),
            Dial::Thrust | Dial::Portal => format!("{v:.1}"),
            Dial::Diameter | Dial::Width => format!("{v:.0}"),
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Dial::Spin => "rad/s",
            Dial::Flow | Dial::Build => "m3/s",
            Dial::Diameter | Dial::Width | Dial::Portal => "m",
            Dial::Thrust => "m/s2",
        }
    }

    pub fn value_text(self, s: &Settings) -> String {
        let standing = s.standing_gravity().0;
        let note = match self {
            Dial::Spin => format!(" ({:.2} g standing)", standing / EARTH_GRAVITY.0),
            Dial::Thrust if standing > 0.01 => {
                format!(" ({:.2} g standing)", s.thrust.0 / standing)
            }
            _ => String::new(),
        };
        format!("{} {}", self.number(s), self.unit())
            .trim_end()
            .to_owned()
            + &note
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Toggle {
    Air,
    Collisions,
    Help,
}

impl Toggle {
    pub const ALL: [Toggle; 3] = [Toggle::Air, Toggle::Collisions, Toggle::Help];

    pub fn key(self) -> KeyCode {
        match self {
            Toggle::Air => KeyCode::KeyG,
            Toggle::Collisions => KeyCode::Enter,
            Toggle::Help => KeyCode::KeyH,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Toggle::Air => "G",
            Toggle::Collisions => "Enter",
            Toggle::Help => "H",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Toggle::Air => "air drag",
            Toggle::Collisions => "player collisions",
            Toggle::Help => "this list",
        }
    }

    pub fn get(self, s: &Settings) -> bool {
        match self {
            Toggle::Air => s.air,
            Toggle::Collisions => s.collisions,
            Toggle::Help => s.help,
        }
    }

    pub fn flip(self, s: &mut Settings) {
        match self {
            Toggle::Air => s.air = !s.air,
            Toggle::Collisions => s.collisions = !s.collisions,
            Toggle::Help => s.help = !s.help,
        }
    }
}

/// Single presses that set something rather than switch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    EqualizeThrust,
}

impl Action {
    pub const ALL: [Action; 1] = [Action::EqualizeThrust];

    pub fn key(self) -> KeyCode {
        match self {
            Action::EqualizeThrust => KeyCode::KeyT,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Action::EqualizeThrust => "T",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Action::EqualizeThrust => "equalize thruster power to gravity",
        }
    }

    pub fn apply(self, s: &mut Settings) {
        match self {
            Action::EqualizeThrust => s.equalize_thrust(),
        }
    }
}

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Settings>().add_systems(
            Update,
            apply
                .in_set(SimSet::Command)
                .run_if(resource_changed::<Settings>),
        );
    }
}

fn apply(
    settings: Res<Settings>,
    air: Res<Air>,
    mut sim: ResMut<Simulation>,
    (mut sheet, mut window): (ResMut<Sheet>, ResMut<SheetWindow>),
) {
    sim.resize(settings.ring(), (&mut sheet, &mut window));
    sim.drum.target_spin = settings.spin;
    sim.thrusters.power = settings.thrust;
    let air_density = if settings.air {
        air.density()
    } else {
        KilogramsPerCubicMetre(0.0)
    };
    sim.params.air_density = air_density;
    sim.body_params.air_density = air_density;
    sim.avatar_mut().solid = settings.collisions;
}
