//! The glass drum: a solid cylinder spinning about its axis, with a sculptable landscape on the
//! inside of its floor that starts as a layer of ground around the whole ring. Implements the
//! vessel the fluid and the bodies live in. Its size is a setting: the ring can be made wider
//! or narrower, and what was sculpted keeps its shape and its place on the wall.
//!
//! The bodies live in the drum's own turning frame, about a site on its wall that follows the
//! viewer: x points out through the glass, y along the axis and z spinward round the ring, and
//! everything is measured from there. The drum's geometry is worked out from those small
//! coordinates and its radius without ever forming a large number, so a ring of any size is
//! exact where the viewer is. The water lives in a frame of the same kind about a site of its
//! own, where it was first put, so it is exact where it is.
mod gpu;
mod landscape;
mod render;

pub use gpu::{DrumFrame, DrumUniform};
pub use landscape::{CELL, Flood, Grid, Ground, Landscape, PATCH, Patch, Place, Round};
pub use render::{DrumPlugin, bed_albedo, chord, ground_albedo, slack};

use serde::{Deserialize, Serialize};

use crate::core::math::{Vec3d, quat_about_y, rotate_y};
use crate::core::units::{Metres, Radians, RadiansPerSecond, RadiansPerSecondSquared};
use crate::core::vessel::{Penetration, Penetrations, Vessel, WaterFrame};

/// The ring as it starts out.
pub const DEFAULT_RING: Ring = Ring {
    radius: Metres(10.5),
    half_width: Metres(6.0),
};
/// The ground that covers the glass all the way round in the initial state.
pub const GROUND_DEPTH: Metres = Metres(0.5);

/// The glass shell's thickness, felt only from outside.
pub const GLASS_THICKNESS: f64 = 0.1;
/// Maximum spin-up acceleration of the drum (rad/s²).
const SPIN_ACCEL: f64 = 0.6;

/// The drum's size: how far the glass is from the axis and how far each cap is from the middle.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub radius: Metres,
    pub half_width: Metres,
}

impl Ring {
    /// Distance from the axis to the top of the initial ground.
    pub fn floor_radius(self) -> Metres {
        Metres(self.radius.0 - GROUND_DEPTH.0)
    }

    /// The landscape may never be raised closer to the axis than this.
    pub fn max_height(self) -> Metres {
        Metres((self.radius.0 - 1.0).max(0.0))
    }

    /// How far along the axis from the middle something of this half-size may stand and still be
    /// clear of both caps. A drum narrower than the thing is wide leaves it nowhere but the
    /// middle, which is as clear of the caps as it can be got.
    pub fn room_along(self, margin: f64) -> f64 {
        (self.half_width.0 as f64 - margin).max(0.0)
    }

    /// Half the size of the box the drum fits in: radius across, half width along the axis.
    pub fn extent(self) -> [f32; 3] {
        [self.radius.0, self.half_width.0, self.radius.0]
    }

    /// How far the drum reaches from its centre along any axis.
    pub fn reach(self) -> Metres {
        Metres(self.radius.0.max(self.half_width.0))
    }
}

/// The patterns fixed to the wheel: the glass is gridded into panes this size and the ground
/// tiled in squares this size, both fixed to the wheel and closing on themselves round it.
pub const PANE: f64 = 1.0;
pub const TILE: f64 = 2.0;

/// A point of the drum's wall: where it is round the ring, exactly, and as a wheel angle, its
/// place along the axis, and where it lies in the cap grid fixed to the wheel: the grid's
/// offset from it along the frame's outward and spinward axes, within one pane, kept exactly as
/// the site moves so that the grid never shifts however large the ring.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct Site {
    pub round: Round,
    #[serde(skip)]
    pub phi: f64,
    pub y: f64,
    #[serde(skip)]
    pub cap: [f64; 2],
}

impl Site {
    /// The site at a wheel angle and axial place.
    pub fn at(phi: f64, y: f64, ring: Ring) -> Site {
        Site::on(Round::at_angle(phi, Grid::of(ring)), y, ring)
    }

    /// The site at a place round the ring and along it, with its place in the cap grid worked
    /// out from its angle: exact enough to lay the grid unless the ring is astronomically
    /// large, when the grid is laid afresh from here.
    pub fn on(round: Round, y: f64, ring: Ring) -> Site {
        let radius = ring.radius.0 as f64;
        let phi = round.angle(Grid::of(ring));
        let (sin, cos) = phi.sin_cos();
        Site {
            round,
            phi,
            y,
            cap: [
                (radius * cos).rem_euclid(PANE),
                (radius * sin).rem_euclid(PANE),
            ],
        }
    }

    /// How far round the ring the site is from wheel angle zero, which is where it lies in the
    /// patterns laid round the ring.
    pub fn arc(&self, ring: Ring) -> f64 {
        self.round.arc(Grid::of(ring))
    }
}

/// How the site moved: the arc it went round the ring, spinward, and how far along the axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shift {
    pub arc: f64,
    pub axial: f64,
}

pub struct Drum {
    pub ring: Ring,
    pub spin: RadiansPerSecond,
    pub target_spin: RadiansPerSecond,
    /// How fast the spin changed over the last step.
    pub spin_rate: RadiansPerSecondSquared,
    /// How far the drum has turned from the world's frame; the glass at wheel angle φ sits at
    /// world angle φ − angle.
    pub angle: Radians,
    /// The point of the wall the bodies' frame sits at, and the one the water's frame sits at.
    pub site: Site,
    pub water: Site,
    pub landscape: Landscape,
}

impl Default for Drum {
    fn default() -> Self {
        Drum::new(DEFAULT_RING)
    }
}

impl Drum {
    /// A still drum of this size with the initial ground all the way round.
    pub fn new(ring: Ring) -> Drum {
        Drum {
            ring,
            spin: RadiansPerSecond(0.0),
            target_spin: RadiansPerSecond(0.0),
            spin_rate: RadiansPerSecondSquared(0.0),
            angle: Radians(0.0),
            site: Site::at(0.0, 0.0, ring),
            water: Site::at(0.0, 0.0, ring),
            landscape: Landscape::flat(ring, GROUND_DEPTH),
        }
    }

    /// Make the drum another size. The site stays at its angle round the ring, and the ground
    /// and the water's site keep where they were from it; returns how a point of the frame
    /// that stays where it was about the axis is carried, which the frame's shift along the
    /// axis and its wall's move toward or away from the axis make of it.
    pub fn resize(&mut self, ring: Ring) -> Vec3d {
        let old = self.landscape.grid();
        let (before, from) = (self.ring.radius.0 as f64, self.site);
        let half_width = ring.half_width.0 as f64;
        self.ring = ring;
        self.site = Site::at(from.phi, from.y.clamp(-half_width, half_width), ring);
        let apart = old.short_way(self.water.round.cell as i128 - from.round.cell as i128);
        let water = Round {
            cell: Grid::of(ring).wrap(self.site.round.cell as i128 + apart as i128),
            across: self.water.round.across,
        };
        self.water = Site::on(water, self.water.y.clamp(-half_width, half_width), ring);
        self.landscape.resize(ring, from.round, self.site.round);
        [before - ring.radius.0 as f64, from.y - self.site.y, 0.0]
    }

    /// Spin up or down toward the target and turn; the angle is kept to one turn so that its
    /// single-precision copies stay exact however long the drum has been running.
    pub fn advance(&mut self, dt: f64) {
        let d = (self.target_spin.0 - self.spin.0) as f64;
        let max = SPIN_ACCEL * dt;
        let change = d.clamp(-max, max);
        let before = self.spin.0 as f64;
        self.spin.0 += change as f32;
        self.spin_rate = RadiansPerSecondSquared(if dt > 0.0 { (change / dt) as f32 } else { 0.0 });
        let turned = (before + self.spin.0 as f64) / 2.0 * dt;
        self.angle.0 = (self.angle.0 + turned).rem_euclid(std::f64::consts::TAU);
    }

    /// Move the site to the wall below a point of the frame, and say how the frame moved: the
    /// new frame is the old one carried along the wall by the shift and turned to face out
    /// there, which is what everything measured in the old frame must be put through.
    pub fn resite(&mut self, below: Vec3d) -> Shift {
        let turn = self.turn_to(below);
        let radius = self.ring.radius.0 as f64;
        let half_width = self.ring.half_width.0 as f64;
        let y = (self.site.y + below[1]).clamp(-half_width, half_width);
        let shift = Shift {
            arc: turn * radius,
            axial: y - self.site.y,
        };
        let grid = self.landscape.grid();
        let site = &mut self.site;
        // the site's chord moves the cap grid by the chord turned to the middle of the turn
        let chord = 2.0 * radius * (turn / 2.0).sin();
        let (sin_mid, cos_mid) = (site.phi + turn / 2.0).sin_cos();
        site.cap = [
            (site.cap[0] - chord * sin_mid).rem_euclid(PANE),
            (site.cap[1] + chord * cos_mid).rem_euclid(PANE),
        ];
        site.round = site.round.on(shift.arc, grid);
        site.phi = site.round.angle(grid);
        site.y = y;
        shift
    }

    /// A point of the old frame in the new one after a shift of the site, and the same for a
    /// vector.
    pub fn carried(&self, p: Vec3d, shift: Shift) -> Vec3d {
        let turn = shift.arc / self.ring.radius.0 as f64;
        let origin = self.wall_point(turn, shift.axial);
        rotate_y(
            &[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]],
            turn,
        )
    }

    pub fn carried_vector(&self, v: Vec3d, shift: Shift) -> Vec3d {
        rotate_y(&v, shift.arc / self.ring.radius.0 as f64)
    }

    /// The angle round the ring from the site to a point of the frame.
    pub fn turn_to(&self, p: Vec3d) -> f64 {
        p[2].atan2(self.ring.radius.0 as f64 + p[0])
    }

    /// The point of the ground a point of the frame stands over.
    pub fn place(&self, p: Vec3d) -> Place {
        self.place_from(&self.site, p)
    }

    /// How high the ground stands over the glass under a point of the frame.
    pub fn ground(&self, p: Vec3d) -> f64 {
        self.landscape.sample(self.place(p)).0
    }

    /// Which way the ground under a point of the frame faces, inward.
    pub fn ground_normal(&self, p: Vec3d) -> Vec3d {
        self.terrain_penetration(p, 0.0).1
    }

    /// Raise (or, by a negative amount, lower) the ground within `radius` of a point of the
    /// frame.
    pub fn sculpt(&mut self, p: Vec3d, radius: f64, amount: f64) {
        let at = self.place(p);
        self.landscape.sculpt(at, radius, amount);
    }

    /// Put the water's site on the wall under a point of the frame, which is only done while
    /// there is no water to be carried with it.
    pub fn settle_water(&mut self, p: Vec3d) {
        let half_width = self.ring.half_width.0 as f64;
        let at = self.place(p);
        self.water = Site::on(at.round, at.along.clamp(-half_width, half_width), self.ring);
    }

    /// A point of the wall, `turn` round the ring from the site and `axial` along the axis
    /// from it, in the frame: exact however large the ring, since the wall curves away from
    /// the site by a sagitta that is small when the turn is.
    pub fn wall_point(&self, turn: f64, axial: f64) -> Vec3d {
        let radius = self.ring.radius.0 as f64;
        let half = (turn / 2.0).sin();
        [-2.0 * radius * half * half, axial, radius * turn.sin()]
    }

    /// How far a point of the frame is inside the glass, and the direction out through it.
    pub fn depth_and_outward(&self, p: Vec3d) -> (f64, Vec3d) {
        let radius = self.ring.radius.0 as f64;
        let a = radius + p[0];
        let r = (a * a + p[2] * p[2]).sqrt();
        let outward = if r > 0.0 {
            [a / r, 0.0, p[2] / r]
        } else {
            [1.0, 0.0, 0.0]
        };
        let over = (2.0 * radius * p[0] + p[0] * p[0] + p[2] * p[2]) / (r + radius);
        (-over, outward)
    }

    /// How far a point of the frame is inside the glass.
    pub fn height_above_glass(&self, p: Vec3d) -> f64 {
        self.depth_and_outward(p).0
    }

    /// Whether a point of the frame is within the drum at all: inside the glass it turns in,
    /// and between the discs that close its ends.
    pub fn holds(&self, p: Vec3d) -> bool {
        self.height_above_glass(p) > 0.0 && self.axial(p).abs() < self.ring.half_width.0 as f64
    }

    /// A point's place along the axis, from the middle of the drum.
    pub fn axial(&self, p: Vec3d) -> f64 {
        self.site.y + p[1]
    }

    /// Whether a point of the frame is in the air the drum encloses.
    pub fn encloses(&self, p: Vec3d) -> bool {
        self.height_above_glass(p) > 0.0 && self.axial(p).abs() < self.ring.half_width.0 as f64
    }

    /// A point of the frame in the water's frame.
    pub fn to_water(&self, p: Vec3d) -> Vec3d {
        self.water_frame().to_water(p)
    }

    /// A point of the water's frame in the bodies' frame.
    pub fn from_water(&self, w: Vec3d) -> Vec3d {
        let frame = self.water_frame();
        let o = frame.origin;
        frame.vector_from_water([w[0] - o[0], w[1] - o[1], w[2] - o[2]])
    }

    /// Pull a point of the water's frame inside the drum, `margin` clear of the caps and above
    /// the landscape.
    pub fn place_inside(&self, p: Vec3d, margin: f64) -> Vec3d {
        let room = self.ring.room_along(margin);
        let along = (self.water.y + p[1]).clamp(-room, room);
        let q = [p[0], along - self.water.y, p[2]];
        let (height, outward) = self.depth_and_outward(q);
        let ground = self.landscape.sample(self.place_from(&self.water, q)).0;
        let lift = ground + margin - height;
        if lift > 0.0 {
            [q[0] - outward[0] * lift, q[1], q[2] - outward[2] * lift]
        } else {
            q
        }
    }

    /// Pull a sphere's centre inside the drum, clear of the caps and no deeper than the initial
    /// ground, keeping its bearing from the axis.
    pub fn place_sphere_inside(&self, c: Vec3d, radius: f64) -> Vec3d {
        let room = self.ring.room_along(radius);
        let y = self.axial(c).clamp(-room, room) - self.site.y;
        let (height, outward) = self.depth_and_outward(c);
        let least = GROUND_DEPTH.0 as f64 + radius;
        let lift = (least - height).max(0.0);
        [c[0] - outward[0] * lift, y, c[2] - outward[2] * lift]
    }

    /// Signed penetration of a point of the frame into the terrain (positive = inside) and the
    /// inward normal, with the terrain's surface shifted inward by `margin`.
    fn terrain_penetration(&self, p: Vec3d, margin: f64) -> (f64, Vec3d) {
        let radius = self.ring.radius.0 as f64;
        let (height, outward) = self.depth_and_outward(p);
        let a = radius + p[0];
        let r2 = a * a + p[2] * p[2];
        if r2 < 1e-12 {
            return (-radius, [0.0; 3]);
        }
        let (h, round, dy) = self.landscape.sample(self.place(p));
        let dphi = round * radius;
        let f = -height + h + margin;
        let g = [
            outward[0] - dphi * p[2] / r2,
            dy,
            outward[2] + dphi * a / r2,
        ];
        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
        (f / len, [-g[0] / len, -g[1] / len, -g[2] / len])
    }

    /// Whether a sphere belongs to the inside of the drum rather than the outside of it. A
    /// sphere still touching the hull belongs to the inside: the hull is a shell of some
    /// thickness and a sphere has some size, so one whose centre has passed the inner face is
    /// still half in the room it came from, and what it wants is to be put back in it. Judging
    /// that by the centre alone makes the hull a trapdoor, since a body that crosses the face
    /// between one step and the next is then met by the outer surface, which throws it clear.
    fn holds_sphere(&self, c: Vec3d, radius: f64) -> bool {
        let clear = GLASS_THICKNESS + radius;
        self.height_above_glass(c) > -clear
            && self.axial(c).abs() < self.ring.half_width.0 as f64 + clear
    }

    /// A sphere inside the drum against the rim, the landscape and the caps.
    fn inner_sphere_penetrations(&self, c: Vec3d, radius: f64) -> Penetrations {
        let mut out = Penetrations::default();
        let (depth, normal) = if self.landscape.is_empty() {
            let (height, outward) = self.depth_and_outward(c);
            (radius - height, outward.map(|x| -x))
        } else {
            self.terrain_penetration(c, radius)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let cap = self.ring.room_along(radius);
        let y = self.axial(c);
        if y > cap {
            out.push(Penetration {
                depth: y - cap,
                normal: [0.0, -1.0, 0.0],
            });
        } else if y < -cap {
            out.push(Penetration {
                depth: -cap - y,
                normal: [0.0, 1.0, 0.0],
            });
        }
        out
    }

    /// A sphere outside the drum against the outer surface of the glass shell.
    fn outer_sphere_penetrations(&self, c: Vec3d, radius: f64) -> Penetrations {
        let mut out = Penetrations::default();
        let outer_half = self.ring.half_width.0 as f64 + GLASS_THICKNESS;
        let (height, radial) = self.depth_and_outward(c);
        let y = self.axial(c);
        let axial = [0.0, y.signum(), 0.0];
        let (dr, dy) = (-height - GLASS_THICKNESS, y.abs() - outer_half);
        let (distance, normal) = if dr > 0.0 && dy > 0.0 {
            let d = (dr * dr + dy * dy).sqrt();
            (
                d,
                [radial[0] * dr / d, axial[1] * dy / d, radial[2] * dr / d],
            )
        } else if dr > 0.0 {
            (dr, radial)
        } else if dy > 0.0 {
            (dy, axial)
        } else if dr > dy {
            (dr, radial)
        } else {
            (dy, axial)
        };
        if distance < radius {
            out.push(Penetration {
                depth: radius - distance,
                normal,
            });
        }
        out
    }
}

impl Drum {
    /// The point of the ground a point of a frame on the wall at `site` stands over.
    fn place_from(&self, site: &Site, p: Vec3d) -> Place {
        let radius = self.ring.radius.0 as f64;
        Place {
            round: site
                .round
                .on(self.turn_to(p) * radius, self.landscape.grid()),
            along: site.y + p[1],
        }
    }
}

impl Vessel for Drum {
    fn water_frame(&self) -> WaterFrame {
        let arc = self
            .water
            .round
            .arc_to(self.site.round, self.landscape.grid());
        let turn = arc / self.ring.radius.0 as f64;
        WaterFrame {
            origin: self.wall_point(turn, self.site.y - self.water.y),
            rotation: quat_about_y(-turn),
        }
    }

    fn angular_velocity(&self) -> Vec3d {
        [0.0, self.spin.0 as f64, 0.0]
    }

    fn angular_acceleration(&self) -> Vec3d {
        [0.0, self.spin_rate.0 as f64, 0.0]
    }

    fn pivot(&self) -> Vec3d {
        [-(self.ring.radius.0 as f64), -self.site.y, 0.0]
    }

    fn has_air(&self, p: Vec3d) -> bool {
        self.encloses(p)
    }

    fn star_velocity(&self, p: Vec3d) -> Vec3d {
        let w = self.spin.0 as f64;
        [-w * p[2], 0.0, w * (self.ring.radius.0 as f64 + p[0])]
    }

    fn penetrations(&self, p: Vec3d) -> Penetrations {
        let mut out = Penetrations::default();
        let (depth, normal) = if self.landscape.is_empty() {
            let (height, outward) = self.depth_and_outward(p);
            (-height, outward.map(|x| -x))
        } else {
            self.terrain_penetration(p, 0.0)
        };
        if depth > 0.0 {
            out.push(Penetration { depth, normal });
        }
        let hw = self.ring.half_width.0 as f64;
        let y = self.axial(p);
        if y > hw {
            out.push(Penetration {
                depth: y - hw,
                normal: [0.0, -1.0, 0.0],
            });
        } else if y < -hw {
            out.push(Penetration {
                depth: -hw - y,
                normal: [0.0, 1.0, 0.0],
            });
        }
        out
    }

    fn sphere_penetrations(&self, centre: Vec3d, radius: f64) -> Penetrations {
        if self.holds_sphere(centre, radius) {
            self.inner_sphere_penetrations(centre, radius)
        } else {
            self.outer_sphere_penetrations(centre, radius)
        }
    }
}
