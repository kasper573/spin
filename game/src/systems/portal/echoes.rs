//! What stands in the ring as every vantage shows it. The ring itself is drawn for each
//! vantage by what draws it; whatever else is there, the viewer's own body and tools and every
//! lamp, is drawn once, about the viewer, and echoed here: a copy of it for each vantage beyond
//! a mouth, laid where that vantage has it, and a copy of each solid thing wherever a vantage
//! has what has gone in at a mouth come out of the other, so that a body half way through a
//! pair is seen whole, half of it at each mouth, the surfaces hiding the rest. A lamp before a
//! mouth shines out of the other as it would from where it is seen through there: from behind
//! that mouth's surface, in the cone that what is open of the mouth lets out.
use bevy::camera::visibility::VisibilitySystems;
use bevy::light::NotShadowCaster;
use bevy::light::cluster::GlobalClusterSettings;
use bevy::math::{Affine3A, DAffine3, DMat3, DVec3};
use bevy::prelude::*;

use crate::core::math::Vec3d;
use crate::systems::drum::{MouthColour, MouthCoords};
use crate::systems::portal::{SolidCopies, SolidMaterial};
use crate::systems::scene::{SeenFrom, VANTAGES, Vantages};
use crate::systems::sim::Simulation;

pub struct EchoPlugin;

impl Plugin for EchoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, make_room_for_lamps)
            .add_systems(Update, (echo_solids, echo_lamps, forget))
            .add_systems(
                PostUpdate,
                place
                    .after(TransformSystems::Propagate)
                    .after(VisibilitySystems::VisibilityPropagate)
                    .before(VisibilitySystems::CheckVisibility),
            );
    }
}

/// A copy of something drawn about the viewer, for the vantage it is `SeenFrom`: as it stands,
/// or as it comes out of the other mouth once it has gone in at the one of this colour.
#[derive(Component, Clone, Copy)]
struct Echo {
    of: Entity,
    through: Option<MouthColour>,
}

/// Something is echoed out of a mouth only while it is this close to the other, in the
/// mouth's radii: nothing farther off reaches through.
const REACHES: f64 = 3.0;
/// How many points of a mouth's rim the cone of a lamp shining out of it is put round.
const RIM: usize = 8;

/// The ways something is seen: as it stands, or out of the other mouth having gone in at one.
const WAYS: [Option<MouthColour>; 3] = [None, Some(MouthColour::Blue), Some(MouthColour::Orange)];

type Unseen = (Without<SeenFrom>, Without<Echo>);

/// Every vantage and way through that a solid thing is echoed for: all but the viewer's own
/// sight of it as it stands, which is the thing itself.
fn echoes(of: Entity) -> impl Iterator<Item = (SeenFrom, Echo)> {
    (0..VANTAGES).flat_map(move |seen| {
        WAYS.into_iter()
            .filter(move |through| seen > 0 || through.is_some())
            .map(move |through| (SeenFrom(seen), Echo { of, through }))
    })
}

/// Every lamp may light a view three times over, as itself and out of either mouth, and the
/// lists the lamps are sorted into for a view are made to hold as many from the start: grown
/// only once they overflow, they would leave the frames until then wrongly lit.
fn make_room_for_lamps(settings: Option<ResMut<GlobalClusterSettings>>) {
    let ways = WAYS.len();
    if let Some(gpu) = settings.and_then(|s| s.into_inner().gpu_clustering.as_mut()) {
        gpu.initial_z_slice_list_capacity *= ways;
        gpu.initial_index_list_capacity *= ways;
    }
}

#[allow(clippy::type_complexity)]
fn echo_solids(
    mut commands: Commands,
    mut copies: ResMut<SolidCopies>,
    mut materials: ResMut<Assets<SolidMaterial>>,
    solids: Query<
        (
            Entity,
            &Mesh3d,
            &MeshMaterial3d<SolidMaterial>,
            Has<NotShadowCaster>,
        ),
        (Added<Mesh3d>, Unseen),
    >,
) {
    for (of, mesh, material, shadowless) in &solids {
        for (seen, echo) in echoes(of) {
            let mut copy = commands.spawn((
                echo,
                seen,
                seen.layers(),
                mesh.clone(),
                MeshMaterial3d(copies.seen_from(&material.0, seen.0, &mut materials)),
                Transform::default(),
            ));
            if shadowless {
                copy.insert(NotShadowCaster);
            }
        }
    }
}

fn echo_lamps(
    mut commands: Commands,
    lamps: Query<(Entity, &PointLight), (Added<PointLight>, Unseen)>,
) {
    for (of, lamp) in &lamps {
        for (seen, echo) in echoes(of) {
            let mut copy = commands.spawn((echo, seen, seen.layers(), Transform::default()));
            match echo.through {
                None => copy.insert(*lamp),
                Some(_) => copy.insert(SpotLight::default()),
            };
        }
    }
}

fn forget(mut commands: Commands, echoes: Query<(Entity, &Echo)>, all: Query<()>) {
    for (entity, echo) in &echoes {
        if !all.contains(echo.of) {
            commands.entity(entity).despawn();
        }
    }
}

/// Lay every echo where its vantage has what it echoes, as that is now, and show it while
/// that is shown and its vantage is looked from.
#[allow(clippy::type_complexity)]
fn place(
    sim: Res<Simulation>,
    vantages: Res<Vantages>,
    echoed: Query<(&GlobalTransform, &InheritedVisibility, Option<&PointLight>), Without<Echo>>,
    mut echoes: Query<(
        &Echo,
        &SeenFrom,
        &mut GlobalTransform,
        &mut InheritedVisibility,
        Option<&mut PointLight>,
        Option<&mut SpotLight>,
    )>,
) {
    let Some(own) = vantages.0[0] else {
        return;
    };
    let drum = &sim.drum;
    let laid = |seen: usize, through: Option<MouthColour>| -> Option<(DAffine3, Option<Vec3d>)> {
        let vantage = vantages.0[seen]?;
        let near = match through {
            Some(colour) => {
                Some(drum.mouth_point(drum.mouths.get(colour)?, MouthCoords::default()))
            }
            None => None,
        };
        let sight = match through {
            Some(colour) => Some(drum.mouth_sight(colour, near?)?),
            None => None,
        };
        let carried = |x: Vec3d| {
            let p = own.drawn_back(x);
            vantage.drawn(sight.map_or(p, |sight| sight.point(p)))
        };
        let origin = DVec3::from_array(carried([0.0; 3]));
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .map(|axis| DVec3::from_array(carried(axis)) - origin);
        let map = DAffine3 {
            matrix3: DMat3::from_cols(axes[0], axes[1], axes[2]),
            translation: origin,
        };
        Some((map, near))
    };
    let maps: [[_; 3]; VANTAGES] =
        std::array::from_fn(|seen| WAYS.map(|through| laid(seen, through)));
    let reach = REACHES * drum.mouths.radius();
    let open = drum.mouths.passable() > 0.0;
    // what is open of each mouth as every vantage has it drawn, for a lamp to shine out of
    let rims: [[_; 2]; VANTAGES] = std::array::from_fn(|seen| {
        MouthColour::BOTH.map(|colour| {
            let (vantage, mouth) = (vantages.0[seen]?, drum.mouths.get(colour)?);
            let at = |u: f64, v: f64| {
                let coords = MouthCoords { u, v, h: 0.0 };
                DVec3::from_array(vantage.drawn(drum.mouth_point(mouth, coords)))
            };
            let passable = drum.mouths.passable();
            let rim = std::array::from_fn::<_, RIM, _>(|k| {
                let (sin, cos) = (k as f64 / RIM as f64 * std::f64::consts::TAU).sin_cos();
                at(cos * passable, sin * passable)
            });
            Some((at(0.0, 0.0), rim))
        })
    });
    for (echo, seen, mut transform, mut shown, lamp, cone) in &mut echoes {
        let way = WAYS.iter().position(|w| *w == echo.through).unwrap_or(0);
        let (Ok((of, of_shown, of_lamp)), Some((map, near))) =
            (echoed.get(echo.of), maps[seen.0][way])
        else {
            *shown = InheritedVisibility::HIDDEN;
            continue;
        };
        let at = of.translation().as_dvec3();
        let reach = of_lamp.map_or(reach, |lamp| lamp.range as f64);
        let reaches =
            near.is_none_or(|near| open && DVec3::from_array(own.drawn(near)).distance(at) < reach);
        *shown = if of_shown.get() && reaches {
            InheritedVisibility::VISIBLE
        } else {
            InheritedVisibility::HIDDEN
        };
        let affine = of.affine();
        let laid = map
            * DAffine3 {
                matrix3: affine.matrix3.as_dmat3(),
                translation: affine.translation.as_dvec3(),
            };
        *transform = GlobalTransform::from(Affine3A::from_mat3_translation(
            laid.matrix3.as_mat3(),
            laid.translation.as_vec3(),
        ));
        if let (Some(mut lamp), Some(of_lamp)) = (lamp, of_lamp) {
            *lamp = *of_lamp;
        }
        let out_of = echo.through.map(MouthColour::other);
        let rim = out_of.and_then(|colour| rims[seen.0][colour as usize]);
        if let (Some(mut cone), Some(of_lamp), Some((middle, rim))) = (cone, of_lamp, rim) {
            let from = laid.translation;
            let axis = (middle - from).normalize_or_zero();
            let widest = rim
                .iter()
                .map(|p| (p - from).normalize_or_zero().dot(axis))
                .fold(1.0, f64::min);
            let before = echo
                .through
                .and_then(|colour| drum.mouths.get(colour))
                .is_some_and(|mouth| {
                    drum.mouth_coords(mouth, own.drawn_back(at.to_array())).h > 0.0
                });
            // a lamp still before the one mouth shines from behind the other, out through it
            if !before || axis == DVec3::ZERO || widest <= 0.0 {
                *shown = InheritedVisibility::HIDDEN;
                continue;
            }
            let outer_angle = widest.acos() as f32;
            *cone = SpotLight {
                color: of_lamp.color,
                intensity: of_lamp.intensity,
                range: of_lamp.range,
                radius: of_lamp.radius,
                inner_angle: outer_angle,
                outer_angle,
                ..default()
            };
            let turned =
                Transform::from_translation(from.as_vec3()).looking_to(axis.as_vec3(), Vec3::Y);
            *transform = GlobalTransform::from(turned);
        }
    }
}
