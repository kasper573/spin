//! The container the fluid and bodies live in, seen only through its walls. Everything is
//! simulated in the vessel's own frame, the one its walls stand still in: the bodies about a
//! point of the vessel near them, with the vessel's motion felt as the accelerations its frame
//! gives them, so their coordinates stay small however far the vessel reaches; the water about
//! the vessel's own centre, on the GPU, through a shader module named `vessel` (see
//! `particles.wgsl` for the functions it must define) bound at group 1, whose layout and
//! per-frame bind group the vessel's plugin provides.
use bevy::prelude::*;
use bevy::render::render_resource::{BindGroup, BindGroupLayoutDescriptor};

use crate::core::math::{Quatd, Vec3d, quat_conjugate, quat_rotate};

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

/// Where the bodies' frame sits in the water's: both are fixed to the vessel, the water's about
/// the vessel's centre and the bodies' about a point of the vessel, turned so.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFrame {
    /// The bodies' origin in the water's frame.
    pub origin: Vec3d,
    /// Turns a vector of the bodies' frame into the water's.
    pub rotation: Quatd,
}

impl WaterFrame {
    pub fn to_water(&self, p: Vec3d) -> Vec3d {
        let r = quat_rotate(&self.rotation, &p);
        [
            self.origin[0] + r[0],
            self.origin[1] + r[1],
            self.origin[2] + r[2],
        ]
    }

    pub fn vector_to_water(&self, v: Vec3d) -> Vec3d {
        quat_rotate(&self.rotation, &v)
    }

    pub fn vector_from_water(&self, v: Vec3d) -> Vec3d {
        quat_rotate(&quat_conjugate(&self.rotation), &v)
    }
}

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

/// The vessel as the bodies meet it, in their frame.
pub trait Vessel {
    /// Where the bodies' frame sits in the water's.
    fn water_frame(&self) -> WaterFrame;
    /// The frame's own angular velocity at the end of the substep being taken, how fast it
    /// changed over that substep, and the point of the frame it turns about.
    fn angular_velocity(&self) -> Vec3d;
    fn angular_acceleration(&self) -> Vec3d;
    fn pivot(&self) -> Vec3d;
    /// Whether the vessel's air, which rests in its frame, is at `p`.
    fn has_air(&self, p: Vec3d) -> bool;
    /// The velocity, in the frame, of something at rest among the stars at `p`.
    fn star_velocity(&self, p: Vec3d) -> Vec3d;
    /// Every wall a point inside the vessel is currently inside of.
    fn penetrations(&self, p: Vec3d) -> Penetrations;
    /// Every wall a sphere overlaps, from whichever side of the wall it is on.
    fn sphere_penetrations(&self, centre: Vec3d, radius: f64) -> Penetrations;
}
