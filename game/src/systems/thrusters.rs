//! What the thrusters look and sound like. The widget is a three-axis cross in the bottom-left of
//! the view, in the frame the thrusters sit in: up, down, left and right arms at full length,
//! the forward and back arms receding diagonally, and a bent arm at each shoulder for the roll
//! pair. Each arm is a thruster where it is mounted, and fills from the centre outward as that
//! thruster spools up: pushing forward lights the arm at the back. Each thruster also has a
//! voice, the same jet for all of them, heard from where it sits around the head and as loud as
//! its level says.
//!
//! The cross is projected onto the image plane by hand rather than left to the camera: a solid
//! drawn off-axis under a wide lens skews toward the vanishing point.
use core::time::Duration;
use std::num::NonZero;

use bevy::audio::{
    AddAudioSource, AudioSinkPlayback, ChannelCount, Decodable, SampleRate, Source, Volume,
};
use bevy::prelude::*;

use crate::core::audio::{self, Placement, Voice};
use crate::core::avatar::{self, Thruster};
use crate::systems::controls::Controls;
use crate::systems::player::{Player, PlayerCamera};
use crate::systems::sim::{SimSet, Simulation};

/// How far in front of the eye the widget hangs.
const DEPTH: f32 = 1.0;
/// Arm length as a fraction of the view's half height at that depth.
const ARM: f32 = 0.17;
/// The forward arm recedes into the view along this screen direction, at this fraction of the
/// other arms' length; the back arm comes out the opposite way.
const RECEDING: Vec2 = Vec2::new(0.707, 0.707);
const RECEDING_LENGTH: f32 = 0.85;
/// The roll arms' radius as a multiple of the arm length, and the arc each sweeps in degrees
/// counter-clockwise from the cross's right, so the right one runs from below its shoulder up
/// and over toward the left, and the left one mirrors it.
const SHOULDER: f32 = 1.3;
const SHOULDER_ARC: (f32, f32) = (-60.0, 75.0);
const ARC_SEGMENTS: usize = 24;
const STEREO: ChannelCount = NonZero::new(2).unwrap();
const IDLE: Color = Color::srgba(1.0, 1.0, 1.0, 0.9);
const FIRING: Color = Color::srgb(1.0, 0.32, 0.04);

#[derive(Default, Reflect, GizmoConfigGroup)]
struct ArmGizmos;

#[derive(Default, Reflect, GizmoConfigGroup)]
struct FillGizmos;

/// A thruster's voice as an audio asset.
#[derive(Asset, TypePath)]
struct ThrusterSound(Thruster);

/// The voice as the two ears hear it, a left sample then a right one.
struct ThrusterDecoder {
    voice: Voice,
    placement: Placement,
    right: Option<f32>,
}

/// The entity playing one thruster's voice.
#[derive(Component)]
struct Speaker(Thruster);

pub struct ThrustersPlugin;

impl Plugin for ThrustersPlugin {
    fn build(&self, app: &mut App) {
        app.insert_gizmo_config(ArmGizmos, on_top(2.0))
            .insert_gizmo_config(FillGizmos, on_top(6.0))
            .add_audio_source::<ThrusterSound>()
            .add_systems(Update, (draw, speak).in_set(SimSet::Observe));
    }
}

fn on_top(width: f32) -> GizmoConfig {
    GizmoConfig {
        depth_bias: -1.0,
        line: GizmoLineConfig { width, ..default() },
        ..default()
    }
}

fn draw(
    player: Res<Player>,
    sim: Res<Simulation>,
    cameras: Query<&Projection, With<PlayerCamera>>,
    mut arms: Gizmos<ArmGizmos>,
    mut fills: Gizmos<FillGizmos>,
) {
    let Some(Projection::Perspective(lens)) = cameras.iter().next() else {
        return;
    };
    let view = player.view(sim.avatar());
    let half_height = DEPTH * (lens.fov / 2.0).tan();
    let half_width = half_height * lens.aspect_ratio;
    let arm = ARM * half_height;
    let margin = (SHOULDER + 0.6) * arm;
    let centre = view.translation + view.forward() * DEPTH
        - view.right() * (half_width - margin)
        - view.up() * (half_height - margin);
    let place = |p: Vec2| centre + view.right() * (p.x * arm) + view.up() * (p.y * arm);
    for thruster in Thruster::ALL {
        let mount = thruster.mount().map(|m| (m / avatar::RADIUS) as f32);
        let points = match thruster {
            Thruster::RollLeft | Thruster::RollRight => shoulder(mount[0]),
            _ => vec![Vec2::ZERO, screen(mount)],
        };
        arms.linestrip(points.iter().map(|p| place(*p)), IDLE);
        let level = sim.thrusters.level(thruster) as f32;
        if level > 0.0 {
            fills.linestrip(fill(&points, level).into_iter().map(place), FIRING);
        }
    }
    fills.sphere(Isometry3d::from_translation(centre), arm * 0.06, IDLE);
}

/// Where a point of the hull in the thrusters' frame (x right, y up, z back) lands on the widget.
fn screen(direction: [f32; 3]) -> Vec2 {
    Vec2::new(direction[0], direction[1]) - RECEDING * (direction[2] * RECEDING_LENGTH)
}

/// The bent arm at a shoulder (`side` 1 right, -1 left): an arc from below the shoulder up and
/// over toward the other side, the way that shoulder's roll thruster lifts it.
fn shoulder(side: f32) -> Vec<Vec2> {
    (0..=ARC_SEGMENTS)
        .map(|i| {
            let t = i as f32 / ARC_SEGMENTS as f32;
            let a = (SHOULDER_ARC.0 + (SHOULDER_ARC.1 - SHOULDER_ARC.0) * t).to_radians();
            Vec2::new(side * a.cos() * SHOULDER, a.sin() * SHOULDER)
        })
        .collect()
}

/// The first `level` of a polyline, by length, from its start.
fn fill(points: &[Vec2], level: f32) -> Vec<Vec2> {
    let total: f32 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
    let mut left = total * level.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(points.len());
    out.extend(points.first());
    for w in points.windows(2) {
        let len = w[0].distance(w[1]);
        if left <= len {
            out.push(w[0].lerp(w[1], left / len.max(1e-9)));
            break;
        }
        out.push(w[1]);
        left -= len;
    }
    out
}

/// Start the voices once the viewer has taken control (a browser only lets sound start after a
/// click; headless there are no controls and no voices), then keep each one as loud as its
/// thruster's level says.
fn speak(
    mut commands: Commands,
    controls: Option<Res<Controls>>,
    sim: Res<Simulation>,
    mut sounds: ResMut<Assets<ThrusterSound>>,
    mut speakers: Query<(&Speaker, &mut AudioSink)>,
    mut started: Local<bool>,
) {
    if !*started && controls.is_some_and(|controls| controls.active) {
        *started = true;
        for thruster in Thruster::ALL {
            commands.spawn((
                Speaker(thruster),
                AudioPlayer(sounds.add(ThrusterSound(thruster))),
                PlaybackSettings {
                    volume: Volume::Linear(0.0),
                    ..PlaybackSettings::LOOP
                },
            ));
        }
    }
    for (speaker, mut sink) in &mut speakers {
        sink.set_volume(Volume::Linear(audio::gain(sim.thrusters.level(speaker.0))));
    }
}

impl Decodable for ThrusterSound {
    type Decoder = ThrusterDecoder;

    fn decoder(&self) -> ThrusterDecoder {
        ThrusterDecoder {
            voice: Voice::new(self.0 as u64),
            placement: Placement::around(self.0.mount()),
            right: None,
        }
    }
}

impl Iterator for ThrusterDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if let Some(right) = self.right.take() {
            return Some(right);
        }
        let [left, right] = self.placement.hear(self.voice.sample());
        self.right = Some(right);
        Some(left)
    }
}

impl Source for ThrusterDecoder {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        STEREO
    }

    fn sample_rate(&self) -> SampleRate {
        audio::SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}
