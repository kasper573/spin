//! What the thrusters look and sound like. The widget is a three-axis cross in the bottom-left of
//! the view, in the frame the thrusters sit in: up, down, left and right arms at full length,
//! the forward and back arms receding diagonally. Each arm is a linear thruster where it is
//! mounted, and fills from the centre outward as that thruster spools up: pushing forward lights
//! the arm at the back. Around the cross, three rings form a ball, one about each axis, for the
//! turning pairs: as the hull turns, an arrow grows along that axis's ring from its front the way
//! the hull is turning, as far round as the pair's net level says. Each thruster also has a
//! voice, a jet for the pushing ones and a lighter puff for the turning ones, heard from where it
//! sits around the head and as loud as its level says.
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
/// The rings' radius as a multiple of the arm length, how far past them the widget keeps from
/// the view's edge, how far round a ring an arrow reaches at full turn, and the size of its head
/// as a fraction of the arm length.
const RING: f32 = 1.25;
const RING_MARGIN: f32 = 0.6;
const RING_SEGMENTS: usize = 48;
const FULL_TURN: f32 = 90.0;
const ARROWHEAD: f32 = 0.1;
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
    let margin = (RING + RING_MARGIN) * arm;
    let centre = view.translation + view.forward() * DEPTH
        - view.right() * (half_width - margin)
        - view.up() * (half_height - margin);
    let place = |p: Vec2| centre + view.right() * (p.x * arm) + view.up() * (p.y * arm);
    for thruster in Thruster::ALL {
        if thruster.turns() {
            continue;
        }
        let mount = thruster.mount().map(|m| (m / avatar::RADIUS) as f32);
        let points = [Vec2::ZERO, screen(mount)];
        arms.linestrip(points.iter().map(|p| place(*p)), IDLE);
        let level = sim.thrusters.level(thruster) as f32;
        if level > 0.0 {
            fills.linestrip(fill(&points, level).into_iter().map(place), FIRING);
        }
    }
    for (axis, turn) in [
        (Axis::Pitch, sim.thrusters.pitch() as f32),
        (Axis::Yaw, sim.thrusters.yaw() as f32),
        (Axis::Roll, sim.thrusters.roll() as f32),
    ] {
        arms.linestrip(ring(axis, 0.0, 360.0).into_iter().map(place), IDLE);
        if turn != 0.0 {
            fills.linestrip(arrow(axis, turn).into_iter().map(place), FIRING);
        }
    }
}

/// The hull's turning axes, each with a ring around it: pitch about the right axis, yaw about
/// the up axis and roll about the back axis, turning the right-hand way about each.
#[derive(Clone, Copy)]
enum Axis {
    Pitch,
    Yaw,
    Roll,
}

impl Axis {
    /// The axis in the thrusters' frame (x right, y up, z back).
    fn direction(self) -> Vec3 {
        match self {
            Axis::Pitch => Vec3::X,
            Axis::Yaw => Vec3::Y,
            Axis::Roll => Vec3::Z,
        }
    }

    /// Where an arrow along the ring starts: straight ahead for the pitch and yaw rings, the top
    /// of the head for the roll ring.
    fn front(self) -> Vec3 {
        match self {
            Axis::Pitch | Axis::Yaw => Vec3::NEG_Z,
            Axis::Roll => Vec3::Y,
        }
    }
}

/// A stretch of an axis's ring, by angle turned about the axis from the ring's front in degrees.
fn ring(axis: Axis, from: f32, to: f32) -> Vec<Vec2> {
    let segments = ((to - from).abs() / 360.0 * RING_SEGMENTS as f32)
        .ceil()
        .max(1.0) as usize;
    (0..=segments)
        .map(|i| {
            let a = (from + (to - from) * i as f32 / segments as f32).to_radians();
            let p = Quat::from_axis_angle(axis.direction(), a) * axis.front() * RING;
            screen(p.to_array())
        })
        .collect()
}

/// An arrow from the front of a ring along it, as far round as `turn` says (-1 to 1) and with
/// its head at the far end, pointing the way the hull is turning.
fn arrow(axis: Axis, turn: f32) -> Vec<Vec2> {
    let mut shaft = ring(axis, 0.0, turn.clamp(-1.0, 1.0) * FULL_TURN);
    let tip = shaft[shaft.len() - 1];
    let along = (tip - shaft[shaft.len() - 2]).normalize_or_zero();
    let across = along.perp();
    shaft.push(tip - along * ARROWHEAD + across * ARROWHEAD * 0.6);
    shaft.push(tip);
    shaft.push(tip - along * ARROWHEAD - across * ARROWHEAD * 0.6);
    shaft
}

/// Where a point of the hull in the thrusters' frame (x right, y up, z back) lands on the widget.
fn screen(direction: [f32; 3]) -> Vec2 {
    Vec2::new(direction[0], direction[1]) - RECEDING * (direction[2] * RECEDING_LENGTH)
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
        let seed = self.0 as u64;
        ThrusterDecoder {
            voice: if self.0.turns() {
                Voice::puff(seed)
            } else {
                Voice::new(seed)
            },
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
