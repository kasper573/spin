//! User-adjustable parameters and the keys that drive them. Every dial is a value with a range and
//! a step; every toggle flips a flag; every action is a single press.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::avatar;
use crate::core::units::{
    EARTH_GRAVITY, LitresPerSecond, Metres, MetresPerSecondSquared, RadiansPerSecond,
};
use crate::systems::drum::{DEFAULT_RING, Ring};
use crate::systems::sim::{SimSet, Simulation, standing_gravity, standing_spin};

/// How many of a dial's fine steps it takes before the steps grow tenfold.
const FINE_STEPS: f32 = 100.0;

#[derive(Resource, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub spin: RadiansPerSecond,
    pub flow: LitresPerSecond,
    pub viscosity: f32,
    pub wall_friction: f32,
    pub air: bool,
    /// Whether the avatar is solid to the drum and the water, or a ghost.
    pub collisions: bool,
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
            viscosity: 0.15,
            wall_friction: 0.5,
            air: true,
            collisions: true,
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
    Viscosity,
    WallFriction,
    Diameter,
    Width,
    Thrust,
}

impl Dial {
    pub const ALL: [Dial; 7] = [
        Dial::Spin,
        Dial::Flow,
        Dial::Viscosity,
        Dial::WallFriction,
        Dial::Diameter,
        Dial::Width,
        Dial::Thrust,
    ];

    pub fn key(self) -> KeyCode {
        match self {
            Dial::Spin => KeyCode::F1,
            Dial::Flow => KeyCode::F2,
            Dial::Viscosity => KeyCode::F3,
            Dial::WallFriction => KeyCode::F4,
            Dial::Diameter => KeyCode::F5,
            Dial::Width => KeyCode::F6,
            Dial::Thrust => KeyCode::F7,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Dial::Spin => "F1",
            Dial::Flow => "F2",
            Dial::Viscosity => "F3",
            Dial::WallFriction => "F4",
            Dial::Diameter => "F5",
            Dial::Width => "F6",
            Dial::Thrust => "F7",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Dial::Spin => "spin",
            Dial::Flow => "flow",
            Dial::Viscosity => "viscosity",
            Dial::WallFriction => "wall friction",
            Dial::Diameter => "ring diameter",
            Dial::Width => "ring width",
            Dial::Thrust => "thruster power",
        }
    }

    pub fn min(self) -> f32 {
        match self {
            Dial::Spin | Dial::Flow | Dial::Viscosity | Dial::WallFriction | Dial::Thrust => 0.0,
            Dial::Diameter => 6.0,
            Dial::Width => 2.0,
        }
    }

    /// The rates run up to 999 of their unit, the fractions to one, and the ring's size has no
    /// end: the simulation represents any size.
    pub fn max(self) -> f32 {
        match self {
            Dial::Spin | Dial::Thrust => 999.0,
            Dial::Flow => 999_000.0,
            Dial::Viscosity | Dial::WallFriction => 1.0,
            Dial::Diameter | Dial::Width => f32::MAX,
        }
    }

    /// The finest step of the dial, taken near zero.
    pub fn step(self) -> f32 {
        match self {
            Dial::Spin => 0.05,
            Dial::Flow => 5000.0,
            Dial::Viscosity | Dial::WallFriction => 0.05,
            Dial::Diameter | Dial::Width => 1.0,
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
            Dial::Viscosity => s.viscosity,
            Dial::WallFriction => s.wall_friction,
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
            Dial::Viscosity => s.viscosity = value,
            Dial::WallFriction => s.wall_friction = value,
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

    pub fn value_text(self, s: &Settings) -> String {
        let v = self.get(s);
        match self {
            Dial::Spin => format!(
                "{v:.3} rad/s ({:.2} g standing)",
                s.standing_gravity().0 / EARTH_GRAVITY.0
            ),
            Dial::Flow => format!("{:.1} m3/s", v / 1000.0),
            Dial::Viscosity | Dial::WallFriction => format!("{v:.2}"),
            Dial::Diameter | Dial::Width => format!("{v:.0} m"),
            Dial::Thrust => {
                let standing = s.standing_gravity().0;
                if standing > 0.01 {
                    format!("{v:.1} m/s2 ({:.2} g standing)", s.thrust.0 / standing)
                } else {
                    format!("{v:.1} m/s2")
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Toggle {
    Air,
    Collisions,
}

impl Toggle {
    pub const ALL: [Toggle; 2] = [Toggle::Air, Toggle::Collisions];

    pub fn key(self) -> KeyCode {
        match self {
            Toggle::Air => KeyCode::KeyG,
            Toggle::Collisions => KeyCode::Enter,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Toggle::Air => "G",
            Toggle::Collisions => "Enter",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Toggle::Air => "air drag",
            Toggle::Collisions => "player collisions",
        }
    }

    pub fn get(self, s: &Settings) -> bool {
        match self {
            Toggle::Air => s.air,
            Toggle::Collisions => s.collisions,
        }
    }

    pub fn flip(self, s: &mut Settings) {
        match self {
            Toggle::Air => s.air = !s.air,
            Toggle::Collisions => s.collisions = !s.collisions,
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

fn apply(settings: Res<Settings>, mut sim: ResMut<Simulation>) {
    sim.resize(settings.ring());
    sim.drum.target_spin = settings.spin;
    sim.thrusters.power = settings.thrust;
    sim.params.viscosity = settings.viscosity;
    sim.params.wall_friction = settings.wall_friction;
    sim.params.air = settings.air;
    sim.body_params.air = settings.air;
    sim.avatar_mut().solid = settings.collisions;
}
