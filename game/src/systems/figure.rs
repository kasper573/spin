//! What the water and the glass mirror of the viewer's own figure. A mirror shows what is on
//! screen, found by marching its mirrored ray across the picture, and the viewer's body never
//! is: the eye sits inside it. So whatever wants to be seen in a mirror says so part by part,
//! as limbs with rounded ends, and the mirrors cast their rays against those before they look
//! for anything else.
use bevy::camera::visibility::VisibilitySystems;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// The most limbs the mirrors are told of; any more go unmirrored.
pub const MOST_LIMBS: usize = 16;

/// A part of something as mirrors show it: a limb with round ends between two points of the
/// entity's own frame, this thick, of this colour, and glowing of itself with this luminance.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Mirrored {
    pub from: Vec3,
    pub to: Vec3,
    pub radius: f32,
    pub colour: Color,
    pub glow: LinearRgba,
}

impl Mirrored {
    /// A limb that only takes the light that falls on it.
    pub fn matte(from: Vec3, to: Vec3, radius: f32, colour: Color) -> Mirrored {
        Mirrored {
            from,
            to,
            radius,
            colour,
            glow: LinearRgba::NONE,
        }
    }
}

/// Everything mirrored, as it stands this frame about the point the scene is drawn about.
#[derive(Resource, Clone, Default)]
pub struct Figure(pub FigureUniform);

#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct LimbUniform {
    /// One end of the limb, and its radius.
    from: Vec4,
    to: Vec4,
    colour: Vec4,
    glow: Vec4,
}

/// The figure as the shaders take it. A material that mirrors it keeps it at binding 10,
/// which is where `figure.wgsl` reads it from.
#[derive(ShaderType, Clone, Debug, Default)]
pub struct FigureUniform {
    limbs: [LimbUniform; MOST_LIMBS],
    /// A sphere round all the limbs: its centre and radius.
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

fn gather(
    mut figure: ResMut<Figure>,
    limbs: Query<(&Mirrored, &GlobalTransform, &InheritedVisibility)>,
) {
    let mut uniform = FigureUniform::default();
    let (mut low, mut high) = (Vec3::MAX, Vec3::MIN);
    for (limb, transform, _) in limbs
        .iter()
        .filter(|(.., visible)| visible.get())
        .take(MOST_LIMBS)
    {
        let from = transform.transform_point(limb.from);
        let to = transform.transform_point(limb.to);
        low = low.min(from.min(to) - limb.radius);
        high = high.max(from.max(to) + limb.radius);
        uniform.limbs[uniform.count as usize] = LimbUniform {
            from: from.extend(limb.radius),
            to: to.extend(0.0),
            colour: limb.colour.to_linear().to_vec4(),
            glow: limb.glow.to_vec4(),
        };
        uniform.count += 1;
    }
    if uniform.count > 0 {
        uniform.within = ((low + high) / 2.0).extend((high - low).length() / 2.0);
    }
    figure.0 = uniform;
}
