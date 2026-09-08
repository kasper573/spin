//! User-adjustable parameters and the keys that drive them. Every dial is a value with a range and
//! a step; every toggle flips a flag.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::units::{LitresPerSecond, Metres, MetresPerSecond, RadiansPerSecond};
use crate::systems::sim::{SimSet, Simulation};

#[derive(Resource, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub spin: RadiansPerSecond,
    pub flow: LitresPerSecond,
    pub viscosity: f32,
    pub wall_friction: f32,
    pub raft_friction: f32,
    pub air: bool,
    /// Whether the shuttle is solid to the drum, the water and the rafts.
    pub collisions: bool,
    pub brush_size: Metres,
    pub brush_rate: MetresPerSecond,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            spin: RadiansPerSecond(0.0),
            flow: LitresPerSecond(500.0),
            viscosity: 0.15,
            wall_friction: 0.5,
            raft_friction: 0.45,
            air: true,
            collisions: false,
            brush_size: Metres(0.6),
            brush_rate: MetresPerSecond(0.8),
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
            Dial::BrushSize | Dial::BrushRate => 0.1,
        }
    }

    pub fn max(self) -> f32 {
        match self {
            Dial::Spin => 3.0,
            Dial::Flow => 2000.0,
            Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => 1.0,
            Dial::BrushSize => 2.0,
            Dial::BrushRate => 3.0,
        }
    }

    pub fn step(self) -> f32 {
        match self {
            Dial::Spin => 0.25,
            Dial::Flow => 100.0,
            Dial::Viscosity | Dial::WallFriction | Dial::RaftFriction => 0.05,
            Dial::BrushSize | Dial::BrushRate => 0.1,
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
            Dial::Spin => format!("{v:.2} rad/s"),
            Dial::Flow => format!("{v:.0} L/s"),
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
    sim.body_params.friction = settings.raft_friction as f64;
    sim.body_params.air = settings.air;
    sim.shuttle_mut().solid = settings.collisions;
}
