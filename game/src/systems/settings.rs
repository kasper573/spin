//! User-adjustable parameters and the keys that drive them. Every dial is a value with a range and
//! a step; every toggle flips a flag.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::units::{
    EARTH_GRAVITY, LitresPerSecond, Metres, MetresPerSecond, RadiansPerSecond,
};
use crate::systems::sim::{SimSet, Simulation, standing_gravity, standing_spin};

#[derive(Resource, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub spin: RadiansPerSecond,
    pub flow: LitresPerSecond,
    pub viscosity: f32,
    pub wall_friction: f32,
    pub raft_friction: f32,
    pub air: bool,
    /// Whether the avatar is solid to the drum, the water and the rafts, or a ghost.
    pub collisions: bool,
    pub brush_size: Metres,
    pub brush_rate: MetresPerSecond,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            spin: standing_spin(),
            flow: LitresPerSecond(20_000.0),
            viscosity: 0.15,
            wall_friction: 0.5,
            raft_friction: 0.45,
            air: true,
            collisions: true,
            brush_size: Metres(2.0),
            brush_rate: MetresPerSecond(1.0),
        }
    }
}

impl Settings {
    /// Every dial clamped to its range; anything not finite falls back to the default.
    pub fn sanitized(mut self) -> Settings {
        let defaults = Settings::default();
        for dial in Dial::ALL {
            let value = dial.get(&self);
            let value = if value.is_finite() {
                value.clamp(dial.min(), dial.max())
            } else {
                dial.get(&defaults)
            };
            dial.set(&mut self, value);
        }
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dial {
    Spin,
    Flow,
    Viscosity,
    WallFriction,
    RaftFriction,
    BrushSize,
    BrushRate,
}

impl Dial {
    pub const ALL: [Dial; 7] = [
        Dial::Spin,
        Dial::Flow,
        Dial::Viscosity,
        Dial::WallFriction,
        Dial::RaftFriction,
        Dial::BrushSize,
        Dial::BrushRate,
    ];

    pub fn key(self) -> KeyCode {
        match self {
            Dial::Spin => KeyCode::F1,
            Dial::Flow => KeyCode::F2,
            Dial::Viscosity => KeyCode::F3,
            Dial::WallFriction => KeyCode::F4,
            Dial::RaftFriction => KeyCode::F5,
            Dial::BrushSize => KeyCode::F6,
            Dial::BrushRate => KeyCode::F7,
        }
    }

    pub fn key_label(self) -> &'static str {
        match self {
            Dial::Spin => "F1",
            Dial::Flow => "F2",
            Dial::Viscosity => "F3",
            Dial::WallFriction => "F4",
            Dial::RaftFriction => "F5",
            Dial::BrushSize => "F6",
            Dial::BrushRate => "F7",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Dial::Spin => "spin",
            Dial::Flow => "flow",
            Dial::Viscosity => "viscosity",
            Dial::WallFriction => "wall friction",
            Dial::RaftFriction => "raft friction",
            Dial::BrushSize => "brush size",
            Dial::BrushRate => "brush rate",
        }
    }

    pub fn min(self) -> f32 {
        match self {
            Dial::Spin | Dial::Flow | Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => {
                0.0
            }
            Dial::BrushSize => 0.5,
            Dial::BrushRate => 0.1,
        }
    }

    pub fn max(self) -> f32 {
        match self {
            Dial::Spin => 2.0,
            Dial::Flow => 200_000.0,
            Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => 1.0,
            Dial::BrushSize => 8.0,
            Dial::BrushRate => 5.0,
        }
    }

    pub fn step(self) -> f32 {
        match self {
            Dial::Spin => 0.05,
            Dial::Flow => 5000.0,
            Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => 0.05,
            Dial::BrushSize => 0.5,
            Dial::BrushRate => 0.1,
        }
    }

    pub fn get(self, s: &Settings) -> f32 {
        match self {
            Dial::Spin => s.spin.0,
            Dial::Flow => s.flow.0,
            Dial::Viscosity => s.viscosity,
            Dial::WallFriction => s.wall_friction,
            Dial::RaftFriction => s.raft_friction,
            Dial::BrushSize => s.brush_size.0,
            Dial::BrushRate => s.brush_rate.0,
        }
    }

    pub fn set(self, s: &mut Settings, value: f32) {
        let value = value.clamp(self.min(), self.max());
        match self {
            Dial::Spin => s.spin = RadiansPerSecond(value),
            Dial::Flow => s.flow = LitresPerSecond(value),
            Dial::Viscosity => s.viscosity = value,
            Dial::WallFriction => s.wall_friction = value,
            Dial::RaftFriction => s.raft_friction = value,
            Dial::BrushSize => s.brush_size = Metres(value),
            Dial::BrushRate => s.brush_rate = MetresPerSecond(value),
        }
    }

    /// Move one step up (`direction` > 0) or down, snapping to the step grid.
    pub fn adjust(self, s: &mut Settings, direction: i32) {
        let steps = (self.get(s) / self.step()).round() + direction as f32;
        self.set(s, steps * self.step());
    }

    pub fn value_text(self, s: &Settings) -> String {
        let v = self.get(s);
        match self {
            Dial::Spin => format!(
                "{v:.3} rad/s ({:.2} g standing)",
                standing_gravity(s.spin).0 / EARTH_GRAVITY.0
            ),
            Dial::Flow => format!("{:.1} m3/s", v / 1000.0),
            Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => format!("{v:.2}"),
            Dial::BrushSize => format!("{v:.1} m"),
            Dial::BrushRate => format!("{v:.1} m/s"),
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
            Toggle::Collisions => "collisions",
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
    sim.drum.target_spin = settings.spin;
    sim.params.viscosity = settings.viscosity;
    sim.params.wall_friction = settings.wall_friction;
    sim.params.air = settings.air;
    sim.set_raft_friction(settings.raft_friction as f64);
    sim.body_params.air = settings.air;
    sim.avatar_mut().solid = settings.collisions;
}
