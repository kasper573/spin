//! The container the fluid and bodies live in, seen only through its walls. Bodies collide with
//! it on the CPU through [`Vessel`]; the water meets it on the GPU through a shader module named
//! `vessel` (see `particles.wgsl` for the functions it must define) bound at group 1, whose
//! layout and per-frame bind group the vessel's plugin provides.
use bevy::prelude::*;
use bevy::render::render_resource::{BindGroup, BindGroupLayoutDescriptor};

/// The layout of bind group 1 of every fluid kernel, inserted into the render app by the vessel.
#[derive(Resource, Clone)]
pub struct VesselLayout(pub BindGroupLayoutDescriptor);

/// Bind group 1 for this frame: one dynamic offset per substep of the frame, then one more for
/// the vessel as it is after them.
#[derive(Resource)]
pub struct VesselBinding {
    pub bind_group: BindGroup,
    pub offsets: Vec<u32>,
}

pub const MAX_CONTACT_NORMALS: usize = 2;

/// Walls a point was pushed away from, as inward unit normals.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Contact {
    pub normals: [[f32; 3]; MAX_CONTACT_NORMALS],
    pub count: u8,
}

impl Contact {
    pub fn push(&mut self, normal: [f32; 3]) {
        if (self.count as usize) < MAX_CONTACT_NORMALS {
            self.normals[self.count as usize] = normal;
            self.count += 1;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f32; 3]> {
        self.normals[..self.count as usize].iter()
    }
}

/// How deep something sits inside a wall and which way is out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penetration {
    pub depth: f64,
    pub normal: [f64; 3],
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Penetrations {
    items: [Option<Penetration>; MAX_CONTACT_NORMALS],
}

impl Penetrations {
    pub fn push(&mut self, penetration: Penetration) {
        if let Some(slot) = self.items.iter_mut().find(|s| s.is_none()) {
            *slot = Some(penetration);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Penetration> + '_ {
        self.items.iter().flatten().copied()
    }
}

pub trait Vessel {
    /// Move a point that left the vessel back inside, `margin` away from the walls.
    fn confine(&self, p: &mut [f32; 3], margin: f32) -> Contact;
    /// Every wall a point inside the vessel is currently inside of.
    fn penetrations(&self, p: [f64; 3]) -> Penetrations;
    /// Every wall a sphere overlaps, from whichever side of the wall it is on.
    fn sphere_penetrations(&self, centre: [f64; 3], radius: f64) -> Penetrations;
    /// Velocity of the wall material at a point.
    fn wall_velocity(&self, p: [f64; 3]) -> [f64; 3];
    /// Velocity of the air at a point, or none where there is no air.
    fn air_velocity(&self, p: [f64; 3]) -> Option<[f64; 3]>;
    /// Angular velocity of the vessel as a whole.
    fn angular_velocity(&self) -> [f64; 3];
}
