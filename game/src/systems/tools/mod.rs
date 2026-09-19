//! The tools the viewer works the ring with. One is out at a time, or none: a tool's number
//! key brings it out, putting away whichever was, and the key of the tool that is out puts
//! that away. What a tool is and does is its own business. All they share is that only the
//! tool that is out is given any input, that it is a thing in the world, floating out in
//! front of the avatar's body where the eye sees it, and that it has an icon on the HUD.
use std::any::TypeId;

use bevy::ecs::component::Mutable;
use bevy::ecs::system::ScheduleSystem;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::core::avatar::{self, WALK_SPEED};
use crate::core::math::{mat3mul, quat_conjugate, quat_rotate};
use crate::core::units::{Metres, Seconds};
use crate::systems::aim::{self, Aim};
use crate::systems::figure::Mirrored;
use crate::systems::scene::Viewpoint;
use crate::systems::sim::{SimSet, Simulation};

pub mod land_tool;
pub mod muzzle;
pub mod screen;
pub mod water_tool;

/// Something the viewer can bring out and work the ring with. The resource is the tool's
/// state while it is being worked, and its default the tool at rest, which is what it is put
/// back to whenever it is put away or the viewer leaves the controls.
pub trait Tool: Resource<Mutability = Mutable> + Default + PartialEq {
    /// How far ahead of its handle the tool reaches.
    const REACH: Metres;

    /// The tool's icon on the HUD: how much of a point of the unit square it covers, y down.
    fn icon(at: Vec2) -> f32;

    /// Puts the tool together, in metres: x to the right, y up, and the muzzle toward -z.
    fn model(bench: &mut Workbench);
}

pub trait ToolApp {
    /// Hangs a tool on the belt, under the next number key, with the systems that work it:
    /// they run only while the tool is out and the viewer at the controls.
    fn add_tool<T: Tool, M>(
        &mut self,
        operate: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self;
}

/// When the tools take their input. Whatever has first call on a key, a button or the wheel
/// runs before it and uses the input up.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ToolInput;

/// The tools there are, in the order of their number keys, and which of them is out. The
/// first one hung on it starts out.
#[derive(Resource, Default)]
pub struct Toolbelt {
    slots: Vec<ToolSlot>,
    wielded: Option<usize>,
}

pub struct ToolSlot {
    tool: TypeId,
    reach: Metres,
    pub icon: fn(Vec2) -> f32,
}

impl Toolbelt {
    pub fn slots(&self) -> &[ToolSlot] {
        &self.slots
    }

    /// The slot of the tool that is out.
    pub fn wielded(&self) -> Option<usize> {
        self.wielded
    }

    pub fn wields<T: Tool>(&self) -> bool {
        self.wielded.is_some() && self.wielded == self.slot_of::<T>()
    }

    /// The key that brings out the tool in a slot, and puts it away again.
    pub fn key(slot: usize) -> KeyCode {
        NUMBER_KEYS[slot]
    }

    /// What a tool's key does: brings the tool out, or puts it away if it is the one out.
    pub fn toggle(&mut self, slot: usize) {
        self.wielded = (self.wielded != Some(slot)).then_some(slot);
    }

    pub fn put_away(&mut self) {
        self.wielded = None;
    }

    fn slot_of<T: Tool>(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.tool == TypeId::of::<T>())
    }
}

/// What a tool is put together with.
pub struct Workbench<'a, 'w, 's> {
    pub commands: &'a mut Commands<'w, 's>,
    pub meshes: &'a mut Assets<Mesh>,
    pub materials: &'a mut Assets<StandardMaterial>,
    pub images: &'a mut Assets<Image>,
    model: Entity,
    slot: usize,
    handle: Vec3,
}

/// The turn that lays something made standing up, along y, down along the barrel instead.
pub fn laid_along() -> Quat {
    Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

/// The turn that stands a side view of the tool, drawn with x ahead and y up, up in it.
pub fn side_on() -> Quat {
    Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
}

/// Where something flat lies on a slope of the tool's side view, drawn with x ahead and y up:
/// midway between the slope's top and its foot, lifted this far off it, facing up and back
/// off it with its own top toward the slope's.
pub fn lying_on(top: Vec2, foot: Vec2, lifted: f32) -> Transform {
    let middle = (top + foot) / 2.0;
    let up_it = (top - foot).normalize();
    let off_it = Vec3::new(0.0, up_it.x, up_it.y);
    Transform::from_translation(Vec3::new(0.0, middle.y, -middle.x) + off_it * lifted)
        .with_rotation(Quat::from_rotation_x(
            up_it.to_angle() - std::f32::consts::FRAC_PI_2,
        ))
}

impl Workbench<'_, '_, '_> {
    /// Where the middle of the tool's handle is, which is the point of it that is kept at
    /// its place before the eye. Every part that follows is placed about it.
    pub fn handle_at(&mut self, handle: Vec3) {
        self.handle = handle;
    }

    pub fn finish(&mut self, material: StandardMaterial) -> Handle<StandardMaterial> {
        self.materials.add(material)
    }

    pub fn part(
        &mut self,
        mesh: impl Into<Mesh>,
        material: &Handle<StandardMaterial>,
        at: Transform,
    ) -> Entity {
        let mesh = self.meshes.add(mesh);
        self.commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                at.with_translation(at.translation - self.handle),
                ChildOf(self.model),
            ))
            .id()
    }

    /// What the water and the glass are to show of the tool where they mirror it: its parts
    /// are too fine for them, so it tells them of its bulk apart.
    pub fn mirrored(&mut self, limb: Mirrored) {
        self.commands.spawn((
            limb,
            Transform::from_translation(-self.handle),
            Visibility::default(),
            ChildOf(self.model),
        ));
    }
}

pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Toolbelt>()
            .init_resource::<Carried>()
            .configure_sets(Update, ToolInput.in_set(SimSet::Command))
            .add_systems(Update, switch.in_set(ToolInput))
            .add_systems(Update, carry.in_set(SimSet::Observe))
            .add_plugins((
                muzzle::MuzzlePlugin,
                screen::ScreenPlugin,
                water_tool::WaterToolPlugin,
                land_tool::LandToolPlugin,
            ));
    }
}

impl ToolApp for App {
    fn add_tool<T: Tool, M>(
        &mut self,
        operate: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        let mut belt = self.world_mut().resource_mut::<Toolbelt>();
        assert!(
            belt.slots.len() < NUMBER_KEYS.len(),
            "there are only {} number keys to bring tools out with",
            NUMBER_KEYS.len()
        );
        belt.slots.push(ToolSlot {
            tool: TypeId::of::<T>(),
            reach: T::REACH,
            icon: T::icon,
        });
        belt.wielded = Some(0);
        self.init_resource::<T>()
            .add_systems(Startup, assemble::<T>)
            .add_systems(
                Update,
                (
                    operate.run_if(operating::<T>),
                    rest::<T>.run_if(not(operating::<T>)),
                )
                    .after(switch)
                    .in_set(ToolInput),
            )
    }
}

const NUMBER_KEYS: [KeyCode; 9] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

/// Where a tool's handle floats from the eye, in metres: to the right of it, below it and
/// ahead of it, with the tool turned in a little toward what the eye looks at.
const PLACE: Vec3 = Vec3::new(0.25, -0.34, -0.54);
const TOE_IN: f32 = 0.05;
/// Put away, a tool hangs this far from its place, tipped this far down, out of the eye's
/// sight, and it takes this long to come up from there or go down to it.
const AWAY: Vec3 = Vec3::new(0.0, -0.45, 0.2);
const TIPPED: f32 = 0.9;
const DRAW_TIME: Seconds = Seconds(0.18);
/// The tool trails the turning of the view by this much of a metre for every radian a second,
/// catching up at this rate, and swings this far with every stride.
const SWAY: f32 = 0.012;
const SWAY_RATE: f32 = 12.0;
const STRIDE: Metres = Metres(1.5);
const BOB: Metres = Metres(0.006);
/// A tool has nothing to stop it against the ground or the glass, so it is drawn back to keep
/// its far end this clear of whatever is ahead of it, as far back as this and at this rate.
const CLEARANCE: Metres = Metres(0.06);
const MOST_DRAWN_BACK: Metres = Metres(0.6);
const DRAW_BACK_RATE: f32 = 18.0;

/// The model of the tool in a slot.
#[derive(Component)]
struct Held(usize);

/// The tool seen before the eye, which lags the one the belt has out by the time it takes
/// the one to go down out of sight and the other to come up, and how it floats there.
#[derive(Resource, Default)]
struct Carried {
    slot: Option<usize>,
    /// How far up from out of sight it has come, 0 to 1.
    raised: f32,
    sway: Vec2,
    /// How far the avatar has walked, in strides.
    strides: f32,
    drawn_back: f32,
}

fn operating<T: Tool>(belt: Res<Toolbelt>, aim: Res<Aim>) -> bool {
    aim.engaged && belt.wields::<T>()
}

fn rest<T: Tool>(mut tool: ResMut<T>) {
    tool.set_if_neq(T::default());
}

fn switch(keys: Res<ButtonInput<KeyCode>>, mut belt: ResMut<Toolbelt>) {
    for slot in 0..belt.slots.len() {
        if keys.just_pressed(Toolbelt::key(slot)) {
            belt.toggle(slot);
        }
    }
}

fn assemble<T: Tool>(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    belt: Res<Toolbelt>,
) {
    let Some(slot) = belt.slot_of::<T>() else {
        return;
    };
    let model = commands
        .spawn((Held(slot), Transform::default(), Visibility::Hidden))
        .id();
    T::model(&mut Workbench {
        commands: &mut commands,
        meshes: &mut meshes,
        materials: &mut materials,
        images: &mut images,
        model,
        slot,
        handle: Vec3::ZERO,
    });
}

/// Take the tool that was put away down out of sight, bring the one that is out up in its
/// stead, and keep it at its place before the eye as the eye moves: trailing the view a
/// little as it turns, swinging with the avatar's stride, and drawn back from whatever
/// stands in its way.
fn carry(
    time: Res<Time>,
    belt: Res<Toolbelt>,
    sim: Res<Simulation>,
    viewpoint: Res<Viewpoint>,
    mut carried: ResMut<Carried>,
    mut models: Query<(&Held, &mut Transform, &mut Visibility)>,
) {
    let dt = time.delta_secs();
    let step = dt / DRAW_TIME.0;
    if carried.slot == belt.wielded {
        carried.raised = (carried.raised + step).min(1.0);
    } else {
        carried.raised = (carried.raised - step).max(0.0);
        if carried.raised == 0.0 {
            carried.slot = belt.wielded;
        }
    }
    let Some(slot) = carried.slot else {
        for (_, _, mut visibility) in &mut models {
            visibility.set_if_neq(Visibility::Hidden);
        }
        return;
    };

    let hull = sim.avatar();
    let turning = quat_rotate(&quat_conjugate(&hull.q), &hull.w).map(|w| w as f32);
    let trailing = Vec2::new(turning[1], -turning[0]) * SWAY;
    carried.sway = carried.sway.lerp(trailing, (SWAY_RATE * dt).min(1.0));
    let footing = sim.footing();
    let walking = if footing.weight > 0.0 {
        (footing.ground_speed / WALK_SPEED.0).min(1.0)
    } else {
        0.0
    };
    carried.strides = (carried.strides + footing.ground_speed * walking * dt / STRIDE.0).fract();
    let swing = carried.strides * std::f32::consts::TAU;
    let bob = Vec2::new(swing.cos(), -swing.sin().abs()) * BOB.0 * walking;

    let beside = DVec3::from_array(avatar::eye_offset()) + PLACE.with_z(0.0).as_dvec3();
    let from = DVec3::from_array(hull.to_world(&beside.to_array()));
    let ahead = DVec3::from_array(mat3mul(&hull.m, &[0.0, 0.0, -1.0]));
    let clear = aim::cast(from, ahead, &sim.drum)
        .map_or(f32::INFINITY, |hit| hit.point.distance(from) as f32);
    let in_the_way =
        (belt.slots[slot].reach.0 - PLACE.z + CLEARANCE.0 - clear).clamp(0.0, MOST_DRAWN_BACK.0);
    carried.drawn_back += (in_the_way - carried.drawn_back) * (DRAW_BACK_RATE * dt).min(1.0);

    let away = 1.0 - carried.raised * carried.raised * (3.0 - 2.0 * carried.raised);
    let floating = Transform {
        translation: PLACE + (carried.sway + bob).extend(carried.drawn_back) + AWAY * away,
        rotation: Quat::from_rotation_y(TOE_IN + carried.sway.x)
            * Quat::from_rotation_x(-TIPPED * away),
        scale: Vec3::ONE,
    };
    let placed = viewpoint.view(hull) * floating;
    for (held, mut transform, mut visibility) in &mut models {
        if held.0 == slot {
            *transform = placed;
            visibility.set_if_neq(Visibility::Visible);
        } else {
            visibility.set_if_neq(Visibility::Hidden);
        }
    }
}
