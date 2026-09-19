//! What the water and the glass mirror of the viewer's own figure. A mirror shows what is on
//! screen, found by marching its mirrored ray across the picture, and the viewer's body never
//! is: the eye sits inside it, and sees the tool before it only from behind. So whatever wants
//! to be seen in a mirror says so part by part, each as the solid it is and in the finish it
//! has, and the mirrors cast their rays against those before they look for anything else.
use bevy::camera::visibility::VisibilitySystems;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// The most parts the mirrors are told of; any more go unmirrored.
pub const MOST_PARTS: usize = 32;
/// The most corners the outline of a prism may have.
pub const MOST_CORNERS: usize = 8;

/// A solid as the mirrors are told of it, in a frame of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Solid {
    /// A flat outline in the xy plane that bulges nowhere inward, its corners given
    /// anticlockwise, drawn out this far either way along z.
    Prism {
        outline: [Vec2; MOST_CORNERS],
        corners: usize,
        half_depth: f32,
    },
    /// Round about the y axis, this thick, running this far either way along it.
    Round {
        radius: f32,
        half_length: f32,
        ends: Ends,
    },
}

/// How a round solid ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ends {
    /// Cut off flat, and bored through along its axis this wide, if at all.
    Flat { bore: f32 },
    /// In half a ball.
    Round,
}

/// A shape the mirrors can be told of.
pub trait MirroredSolid {
    /// The solid the shape is, and the turn that lays the solid's frame into the shape's own.
    fn solid(&self) -> (Solid, Quat);
}

/// A part of something as mirrors show it: the solid it is, laid into the frame of the entity
/// it is on by this turn, and the finish it has.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Mirrored {
    pub solid: Solid,
    pub turn: Quat,
    pub finish: MirroredFinish,
}

/// What a part is finished in, as mirrors show it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirroredFinish {
    pub colour: Color,
    pub metallic: f32,
    pub roughness: f32,
    /// The luminance it gives off of itself.
    pub glow: LinearRgba,
}

impl Mirrored {
    pub fn of(shape: &impl MirroredSolid, finish: MirroredFinish) -> Mirrored {
        let (solid, turn) = shape.solid();
        Mirrored {
            solid,
            turn,
            finish,
        }
    }
}

impl MirroredFinish {
    /// A material as the mirrors are to show it. What it gives off counts only if the eye's
    /// exposure is applied to it, as it is to everything the mirrors show.
    pub fn of(material: &StandardMaterial) -> MirroredFinish {
        MirroredFinish {
            colour: material.base_color,
            metallic: material.metallic,
            roughness: material.perceptual_roughness,
            glow: material.emissive * material.emissive_exposure_weight,
        }
    }
}

/// Everything mirrored, as it stands this frame about the point the scene is drawn about.
#[derive(Resource, Clone, Default)]
pub struct Figure(pub FigureUniform);

/// A part as the shaders take it; `figure.wgsl` says what rides along in each fourth place.
#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct PartUniform {
    x: Vec4,
    y: Vec4,
    z: Vec4,
    at: Vec4,
    colour: Vec4,
    glow: Vec4,
    outline: [Vec4; MOST_CORNERS / 2],
}

/// The figure as the shaders take it. A material that mirrors it keeps it at binding 10,
/// which is where `figure.wgsl` reads it from.
#[derive(ShaderType, Clone, Debug, Default)]
pub struct FigureUniform {
    parts: [PartUniform; MOST_PARTS],
    /// A sphere round all the parts: its centre and radius.
    within: Vec4,
    count: u32,
}

/// When the figure is gathered: once everything has been put where it is drawn this frame.
/// Whatever hands the figure on to a shader does so after it.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FigureGathered;

pub struct FigurePlugin;

impl Plugin for FigurePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Figure>().add_systems(
            PostUpdate,
            gather
                .in_set(FigureGathered)
                .after(TransformSystems::Propagate)
                .after(VisibilitySystems::VisibilityPropagate),
        );
    }
}

impl MirroredSolid for Cuboid {
    fn solid(&self) -> (Solid, Quat) {
        let (outline, corners) = rectangle(self.half_size.truncate());
        let half_depth = self.half_size.z;
        (
            Solid::Prism {
                outline,
                corners,
                half_depth,
            },
            Quat::IDENTITY,
        )
    }
}

impl MirroredSolid for Rectangle {
    fn solid(&self) -> (Solid, Quat) {
        let (outline, corners) = rectangle(self.half_size);
        (
            Solid::Prism {
                outline,
                corners,
                half_depth: SHEET,
            },
            Quat::IDENTITY,
        )
    }
}

impl MirroredSolid for Extrusion<ConvexPolygon> {
    fn solid(&self) -> (Solid, Quat) {
        prism(self.base_shape.vertices(), self.half_depth)
    }
}

impl MirroredSolid for Extrusion<Triangle2d> {
    fn solid(&self) -> (Solid, Quat) {
        prism(&self.base_shape.vertices, self.half_depth)
    }
}

impl MirroredSolid for Extrusion<Annulus> {
    fn solid(&self) -> (Solid, Quat) {
        (
            Solid::Round {
                radius: self.base_shape.outer_circle.radius,
                half_length: self.half_depth,
                ends: Ends::Flat {
                    bore: self.base_shape.inner_circle.radius,
                },
            },
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        )
    }
}

impl MirroredSolid for Torus {
    /// A ring, which its tube is thin enough to pass for cut square.
    fn solid(&self) -> (Solid, Quat) {
        (
            Solid::Round {
                radius: self.major_radius + self.minor_radius,
                half_length: self.minor_radius,
                ends: Ends::Flat {
                    bore: self.major_radius - self.minor_radius,
                },
            },
            Quat::IDENTITY,
        )
    }
}

impl MirroredSolid for Cylinder {
    fn solid(&self) -> (Solid, Quat) {
        (
            Solid::Round {
                radius: self.radius,
                half_length: self.half_height,
                ends: Ends::Flat { bore: 0.0 },
            },
            Quat::IDENTITY,
        )
    }
}

impl MirroredSolid for Capsule3d {
    fn solid(&self) -> (Solid, Quat) {
        (
            Solid::Round {
                radius: self.radius,
                half_length: self.half_length,
                ends: Ends::Round,
            },
            Quat::IDENTITY,
        )
    }
}

impl MirroredSolid for Sphere {
    fn solid(&self) -> (Solid, Quat) {
        (
            Solid::Round {
                radius: self.radius,
                half_length: 0.0,
                ends: Ends::Round,
            },
            Quat::IDENTITY,
        )
    }
}

/// Half as thick as a flat sheet is taken to be.
const SHEET: f32 = 0.0005;

fn rectangle(half: Vec2) -> ([Vec2; MOST_CORNERS], usize) {
    let mut outline = [Vec2::ZERO; MOST_CORNERS];
    let corners = [
        Vec2::new(-half.x, -half.y),
        Vec2::new(half.x, -half.y),
        half,
        Vec2::new(-half.x, half.y),
    ];
    outline[..corners.len()].copy_from_slice(&corners);
    (outline, corners.len())
}

/// The prism on an outline that bulges nowhere inward, whichever way round it is given. Of
/// more corners than the mirrors take, the first are kept.
fn prism(vertices: &[Vec2], half_depth: f32) -> (Solid, Quat) {
    let corners = vertices.len().min(MOST_CORNERS);
    let mut outline = [Vec2::ZERO; MOST_CORNERS];
    outline[..corners].copy_from_slice(&vertices[..corners]);
    let twice_the_area: f32 = (0..corners)
        .map(|i| outline[i].perp_dot(outline[(i + 1) % corners]))
        .sum();
    if twice_the_area < 0.0 {
        outline[..corners].reverse();
    }
    (
        Solid::Prism {
            outline,
            corners,
            half_depth,
        },
        Quat::IDENTITY,
    )
}

impl Solid {
    /// How far the solid reaches from the middle of its frame.
    fn reach(&self) -> f32 {
        match *self {
            Solid::Prism {
                outline,
                corners,
                half_depth,
            } => outline[..corners]
                .iter()
                .map(|corner| corner.extend(half_depth).length())
                .fold(0.0, f32::max),
            Solid::Round {
                radius,
                half_length,
                ends: Ends::Flat { .. },
            } => Vec2::new(radius, half_length).length(),
            Solid::Round {
                radius,
                half_length,
                ends: Ends::Round,
            } => radius + half_length,
        }
    }
}

fn gather(
    mut figure: ResMut<Figure>,
    parts: Query<(&Mirrored, &GlobalTransform, &InheritedVisibility)>,
) {
    let mut uniform = FigureUniform::default();
    let (mut low, mut high) = (Vec3::MAX, Vec3::MIN);
    for (part, transform, _) in parts
        .iter()
        .filter(|(.., visible)| visible.get())
        .take(MOST_PARTS)
    {
        let (_, rotation, at) = transform.to_scale_rotation_translation();
        let frame = rotation * part.turn;
        let reach = part.solid.reach();
        low = low.min(at - reach);
        high = high.max(at + reach);
        let mut outline = [Vec4::ZERO; MOST_CORNERS / 2];
        let (corners, length, radius, bore) = match part.solid {
            Solid::Prism {
                outline: corners,
                corners: count,
                half_depth,
            } => {
                for (pair, packed) in corners.chunks(2).zip(&mut outline) {
                    *packed = Vec4::new(pair[0].x, pair[0].y, pair[1].x, pair[1].y);
                }
                (count as f32, half_depth, 0.0, 0.0)
            }
            Solid::Round {
                radius,
                half_length,
                ends,
            } => {
                let bore = match ends {
                    Ends::Flat { bore } => bore,
                    Ends::Round => -1.0,
                };
                (0.0, half_length, radius, bore)
            }
        };
        uniform.parts[uniform.count as usize] = PartUniform {
            x: (frame * Vec3::X).extend(corners),
            y: (frame * Vec3::Y).extend(length),
            z: (frame * Vec3::Z).extend(radius),
            at: at.extend(bore),
            colour: part
                .finish
                .colour
                .to_linear()
                .to_vec3()
                .extend(part.finish.metallic),
            glow: part.finish.glow.to_vec3().extend(part.finish.roughness),
            outline,
        };
        uniform.count += 1;
    }
    if uniform.count > 0 {
        uniform.within = ((low + high) / 2.0).extend((high - low).length() / 2.0);
    }
    figure.0 = uniform;
}
