//! Seeing through the portals. What lies beyond a mouth is drawn by a camera of its own, which
//! looks from where the viewer's eye is once it is carried through the pair (see `MouthSight`)
//! and sees nothing that is not past the far mouth. It draws a vantage of its own: the ring
//! laid about the far mouth, so that what is seen there is drawn as exactly as what is round
//! the viewer however far round the ring it is. Its picture is what the surface a mouth is let
//! into shows where the mouth is open.
//!
//! A mouth seen through a mouth shows the picture of the frame before, since a picture cannot
//! be drawn into while it is read, so each camera has two pictures and draws into them in turn.
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget, SubCameraView, Viewport};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::ShadowFilteringMethod;
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureFormat};

use crate::core::math::{Quatd, Vec3d, dot, norm, quat_conjugate, quat_rotate};
use crate::systems::drum::{Drum, FLAME_BAND, Mouth, MouthAnchor, MouthColour, MouthCoords, Site};
use crate::systems::player::PlayerCamera;
use crate::systems::scene::{
    NEAR, SEEN_THROUGH_LAYERS, SeenFrom, SettleVantages, VANTAGES, Vantage, Vantages,
};
use crate::systems::sim::Simulation;

/// The pictures of what lies beyond each mouth, and where in them each vantage finds what a
/// point of a mouth shows.
#[derive(Resource)]
pub struct Pictures {
    /// The pictures drawn this frame, and those drawn the frame before.
    fresh: [Handle<Image>; 2],
    stale: [Handle<Image>; 2],
    found: [[Mat4; 2]; VANTAGES],
}

impl Pictures {
    /// The pictures a vantage reads: the viewer's own reads this frame's, which are drawn
    /// before it is; one beyond a mouth reads the last frame's.
    pub fn read_from(&self, vantage: usize) -> [Option<Handle<Image>>; 2] {
        let pictures = if vantage == 0 {
            &self.fresh
        } else {
            &self.stale
        };
        pictures.clone().map(Some)
    }

    /// For each mouth, what takes a point drawn for a vantage to where the mouth's picture
    /// shows what is seen through the mouth there, as the picture's camera clips it.
    pub fn found_from(&self, vantage: usize) -> [Mat4; 2] {
        self.found[vantage]
    }
}

pub struct PortalEyesPlugin;

impl Plugin for PortalEyesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(
            Update,
            (size, lean_in, look_through).chain().in_set(SettleVantages),
        );
    }
}

/// The camera that draws what is seen through the mouth of a colour.
#[derive(Component)]
struct PortalEye(MouthColour);

/// A mouth is looked through only while it takes up more of the view than this, in radians
/// across, and the eye is never taken to be closer to it than this when choosing how finely
/// to draw what is beyond it.
const SMALLEST: f64 = 0.002;
const NEAREST: f64 = 1.0;
/// The plane nothing short of is seen is kept this far past the far mouth's surface, and the
/// eye is this far short of it at least, or what is seen is simply what is before the eye.
const PAST: f64 = 0.01;
const SHORT: f32 = 0.02;
/// How many points of a mouth's rim its window in the view is found from, and how much wider
/// than the rim they are taken, the rim being round and the points few.
const WINDOW_POINTS: usize = 24;
const WINDOW_ROOM: f64 = 1.05;
/// Over a mouth, the nearest the eye sees anything is this share of its height over the
/// mouth, down to this.
const LEANT: f32 = 0.25;
const CLOSEST: f32 = 0.002;

fn picture(images: &mut Assets<Image>) -> Handle<Image> {
    images.add(Image::new_target_texture(
        1,
        1,
        TextureFormat::Rgba16Float,
        None,
    ))
}

fn spawn(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let pictures = Pictures {
        fresh: [(); 2].map(|()| picture(&mut images)),
        stale: [(); 2].map(|()| picture(&mut images)),
        found: [[Mat4::IDENTITY; 2]; VANTAGES],
    };
    for (i, colour) in MouthColour::BOTH.into_iter().enumerate() {
        commands.spawn((
            PortalEye(colour),
            SeenFrom(i + 1),
            RenderLayers::layer(i + 1),
            Camera3d::default(),
            Camera {
                order: i as isize - 2,
                is_active: false,
                ..default()
            },
            RenderTarget::Image(ImageRenderTarget {
                handle: pictures.fresh[i].clone(),
                scale_factor: 1.0,
            }),
            Hdr,
            Exposure::SUNLIGHT,
            Tonemapping::None,
            ScreenSpaceTransmission {
                steps: SEEN_THROUGH_LAYERS,
                ..default()
            },
            Msaa::Sample4,
            DepthPrepass,
            ShadowFilteringMethod::Gaussian,
        ));
    }
    commands.insert_resource(pictures);
}

/// The pictures are as large as the view they are shown in.
fn size(
    pictures: Res<Pictures>,
    mut images: ResMut<Assets<Image>>,
    player: Query<&Camera, With<PlayerCamera>>,
) {
    let Some(wanted) = player.iter().find_map(Camera::physical_target_size) else {
        return;
    };
    for handle in pictures.fresh.iter().chain(&pictures.stale) {
        if images
            .get(handle)
            .is_some_and(|image| image.size() != wanted)
            && let Some(mut image) = images.get_mut(handle)
        {
            image.resize(Extent3d {
                width: wanted.x,
                height: wanted.y,
                depth_or_array_layers: 1,
            });
        }
    }
}

/// An eye about to go through a mouth comes as close to the mouth's surface as it likes before
/// it does, and the surface is what shows it the far side until then: so the nearest the eye
/// sees anything is kept within its distance from a mouth it is over.
fn lean_in(sim: Res<Simulation>, mut player: Query<&mut Projection, With<PlayerCamera>>) {
    let drum = &sim.drum;
    let (eye, _) = sim.eye();
    let rim = drum.mouths.radius() * (1.0 + FLAME_BAND);
    let clear = drum
        .mouths
        .standing()
        .map(|(_, mouth)| drum.mouth_coords(mouth, eye))
        .filter(|over| over.across() < rim && over.h > 0.0)
        .map(|over| over.h)
        .fold(f64::INFINITY, f64::min);
    let near = (clear * LEANT as f64).clamp(CLOSEST as f64, NEAR as f64) as f32;
    for mut projection in &mut player {
        if let Projection::Perspective(lens) = &mut *projection
            && lens.near != near
        {
            lens.near = near;
            lens.near_clip_plane = Vec4::new(0.0, 0.0, -1.0, -near);
        }
    }
}

/// The site of the wall a mouth is nearest, which what is seen beyond it is drawn about.
fn site_of(drum: &Drum, mouth: &Mouth) -> Site {
    match mouth.anchor {
        MouthAnchor::Wall { round, along } => Site::on(round, along, drum.ring),
        MouthAnchor::Cap { side, round, .. } => Site::on(
            round,
            side.sign() * drum.ring.half_width.0 as f64,
            drum.ring,
        ),
    }
}

/// Settle the vantage beyond each mouth that is seen through, aim its camera, and work out
/// where every vantage finds the pictures.
fn look_through(
    sim: Res<Simulation>,
    mut vantages: ResMut<Vantages>,
    mut pictures: ResMut<Pictures>,
    mut standoffs: Local<[Option<f64>; 2]>,
    player: Query<(&Projection, &Camera), With<PlayerCamera>>,
    mut eyes: Query<
        (
            &PortalEye,
            &mut Camera,
            &mut RenderTarget,
            &mut Projection,
            &mut Transform,
        ),
        Without<PlayerCamera>,
    >,
) {
    let Some((Projection::Perspective(lens), shown_in)) = player.iter().next() else {
        return;
    };
    let shown = shown_in.physical_target_size();
    let pictures = &mut *pictures;
    std::mem::swap(&mut pictures.fresh, &mut pictures.stale);
    let drum = &sim.drum;
    let (eye, attitude) = sim.eye();
    let mut sights = [None; 2];
    for (i, colour) in MouthColour::BOTH.into_iter().enumerate() {
        let standoff = &mut standoffs[i];
        let pair = drum.mouths.get(colour).zip(drum.mouths.get(colour.other()));
        vantages.0[i + 1] = pair.map(|(near, far)| {
            let reach = (norm(&between(
                drum.mouth_point(near, MouthCoords::default()),
                eye,
            )) - drum.mouths.radius())
            .max(NEAREST);
            let kept = standoff.filter(|kept| (kept / 2.0..kept * 4.0).contains(&reach));
            let drawn_for = *standoff.insert(kept.unwrap_or(reach));
            let centre = drum.mouth_point(far, MouthCoords::default());
            Vantage {
                enclosed: true,
                ..Vantage::about(drum, site_of(drum, far), centre, drawn_for)
            }
        });
        if pair.is_none() {
            *standoff = None;
        }
        sights[i] = drum
            .mouth_sight(colour, eye)
            .filter(|_| seen_through(drum, colour, eye, attitude, lens))
            .zip(vantages.0[i + 1]);
    }
    for (PortalEye(colour), mut camera, mut target, mut projection, mut transform) in &mut eyes {
        let i = MouthColour::BOTH
            .iter()
            .position(|c| c == colour)
            .unwrap_or(0);
        let active = sights[i].is_some();
        if camera.is_active != active {
            camera.is_active = active;
        }
        let Some((sight, vantage)) = sights[i] else {
            continue;
        };
        *target = RenderTarget::Image(ImageRenderTarget {
            handle: pictures.fresh[i].clone(),
            scale_factor: 1.0,
        });
        let window = drum
            .mouths
            .get(*colour)
            .zip(shown)
            .and_then(|(near, shown)| window_on(drum, near, eye, attitude, lens, shown));
        let (viewport, sub_view) = match (window, shown) {
            (Some((offset, size)), Some(full_size)) => (
                Some(Viewport {
                    physical_position: offset,
                    physical_size: size,
                    ..default()
                }),
                Some(SubCameraView {
                    full_size,
                    offset: offset.as_vec2(),
                    size,
                }),
            ),
            _ => (None, None),
        };
        if camera.sub_camera_view != sub_view {
            camera.viewport = viewport;
            camera.sub_camera_view = sub_view;
        }
        let pose = vantage.pose(sight.point(eye), sight.attitude(attitude));
        *transform = pose;
        let inward = Vec3::from(vantage.frame.vector(sight.inward).map(|c| c as f32));
        let past = [0, 1, 2].map(|k| sight.threshold[k] + sight.inward[k] * PAST);
        let normal = pose.rotation.inverse() * inward;
        let offset = -inward.dot(vantage.local(past) - pose.translation);
        // a plane square to the view is taken for the lens's own near plane wherever it
        // lies, so one that is square to the view is made the lens's own
        let square = normal.x == 0.0 && normal.y == 0.0;
        let (near, near_clip_plane) = if offset >= -SHORT {
            (lens.near, Vec4::new(0.0, 0.0, -1.0, -lens.near))
        } else if square {
            (-offset, Vec4::new(0.0, 0.0, -1.0, offset))
        } else {
            (lens.near, normal.extend(offset))
        };
        *projection = Projection::Perspective(PerspectiveProjection {
            near,
            near_clip_plane,
            ..lens.clone()
        });
    }
    let clip = Mat4::perspective_infinite_reverse_rh(lens.fov, lens.aspect_ratio, lens.near);
    for (from, found) in vantages.0.iter().zip(&mut pictures.found) {
        for (sight, found) in sights.iter().zip(found) {
            let (Some(from), Some((sight, beyond))) = (from, sight) else {
                continue;
            };
            let looks_from = beyond.drawn(sight.point(eye));
            let back = quat_conjugate(&beyond.frame.attitude(sight.attitude(attitude)));
            let viewed = |x: Vec3d| {
                let at = beyond.drawn(sight.point(from.drawn_back(x)));
                quat_rotate(&back, &[0, 1, 2].map(|k| at[k] - looks_from[k]))
            };
            let origin = viewed([0.0; 3]);
            let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|axis| {
                let to = viewed(axis);
                Vec3::from([0, 1, 2].map(|k| (to[k] - origin[k]) as f32)).extend(0.0)
            });
            let view = Mat4::from_cols(
                axes[0],
                axes[1],
                axes[2],
                Vec3::from(origin.map(|c| c as f32)).extend(1.0),
            );
            *found = clip * view;
        }
    }
}

/// Whether the eye sees into the mouth of a colour: the pair stands open, the eye is before
/// the mouth, and the mouth is in its view and not too small in it to show anything.
fn seen_through(
    drum: &Drum,
    colour: MouthColour,
    eye: Vec3d,
    attitude: Quatd,
    lens: &PerspectiveProjection,
) -> bool {
    let Some(near) = drum.mouths.get(colour) else {
        return false;
    };
    let radius = drum.mouths.radius();
    if drum.mouths.fill() >= 1.0 || drum.mouth_coords(near, eye).h <= 0.0 {
        return false;
    }
    let to = between(eye, drum.mouth_point(near, MouthCoords::default()));
    let distance = norm(&to);
    if distance <= radius {
        return true;
    }
    let across = (radius / distance).asin();
    let ahead = quat_rotate(&attitude, &[0.0, 0.0, -1.0]);
    let off = (dot(&to, &ahead) / distance).clamp(-1.0, 1.0).acos();
    let corner =
        ((lens.fov as f64 / 2.0).tan() * (1.0 + (lens.aspect_ratio as f64).powi(2)).sqrt()).atan();
    across > SMALLEST && off < corner + across
}

/// The window of the view a mouth shows in, with its flames: what lies beyond the mouth is
/// seen nowhere else, so nowhere else is it drawn, and a mouth costs what it covers of the view
/// rather than a view of its own. None where the mouth comes too near the eye to have a window
/// short of the whole view.
fn window_on(
    drum: &Drum,
    near: &Mouth,
    eye: Vec3d,
    attitude: Quatd,
    lens: &PerspectiveProjection,
    shown: UVec2,
) -> Option<(UVec2, UVec2)> {
    let rim = drum.mouths.radius() * (1.0 + FLAME_BAND) * WINDOW_ROOM;
    let back = quat_conjugate(&attitude);
    let tan = (lens.fov as f64 / 2.0).tan();
    let (mut low, mut high) = ([f64::MAX; 2], [f64::MIN; 2]);
    for k in 0..WINDOW_POINTS {
        let turn = k as f64 / WINDOW_POINTS as f64 * std::f64::consts::TAU;
        let at = MouthCoords {
            u: rim * turn.cos(),
            v: rim * turn.sin(),
            h: 0.0,
        };
        let seen = quat_rotate(&back, &between(eye, drum.mouth_point(near, at)));
        if -seen[2] < 2.0 * lens.near as f64 {
            return None;
        }
        let across = [
            seen[0] / (-seen[2] * tan * lens.aspect_ratio as f64),
            -seen[1] / (-seen[2] * tan),
        ];
        for axis in 0..2 {
            low[axis] = low[axis].min(across[axis]);
            high[axis] = high[axis].max(across[axis]);
        }
    }
    let size = [shown.x as f64, shown.y as f64];
    let pixel =
        |across: f64, axis: usize| ((across + 1.0) / 2.0 * size[axis]).clamp(0.0, size[axis]);
    let from = [0, 1].map(|axis| pixel(low[axis], axis).floor() as u32);
    let to = [0, 1].map(|axis| pixel(high[axis], axis).ceil() as u32);
    (to[0] > from[0] && to[1] > from[1]).then(|| {
        (
            UVec2::new(from[0], from[1]),
            UVec2::new(to[0] - from[0], to[1] - from[1]),
        )
    })
}

fn between(from: Vec3d, to: Vec3d) -> Vec3d {
    [0, 1, 2].map(|k| to[k] - from[k])
}
