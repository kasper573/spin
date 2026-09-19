//! The mouths of a pair of portals let into the drum's walls: a blue one and an orange one, each
//! a disc wrapped onto the surface it was put on, which is the ground round the wall or one of
//! the caps. One alone is only paint. Two are one opening: the drum is joined to itself there,
//! each point of the one mouth being the matching point of the other, so that what goes in at
//! the one comes out of the other.
//!
//! A mouth is measured in a chart of its own: `u` and `v` across it from its centre, `v` toward
//! its top, walked along the wall the way the ground is laid, and `h` how high over its surface,
//! which on the wall is the ground whatever shape that has. The point `h` behind the one mouth
//! is the point `h` before the other, the same way up and mirrored left to right, as two faces
//! that look at each other are. Vectors are carried by the frames the two surfaces have at those
//! points, which is a turn and nothing else, so whatever goes through keeps its speed exactly
//! and comes out as far over the ground as it went under it.
use std::f64::consts::{FRAC_PI_2, TAU};

use serde::{Deserialize, Serialize};

use super::{Drum, Place, Round};
use crate::core::math::{
    Quatd, Vec3d, cross, dot, norm, quat_conjugate, quat_from_basis, quat_mul,
};
use crate::core::units::{Metres, Radians, Seconds};
use crate::core::vessel::{Passage, Penetration};

/// How long a pair of mouths takes to open once both stand, and one to close once it is alone.
pub const OPENING: Seconds = Seconds(0.6);
/// How far past its rim a mouth's flames reach, as a share of its radius: the room a mouth
/// needs round it on the surface it is put on.
pub const FLAME_BAND: f64 = 0.18;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouthColour {
    Blue,
    Orange,
}

impl MouthColour {
    pub const BOTH: [MouthColour; 2] = [MouthColour::Blue, MouthColour::Orange];

    pub fn other(self) -> MouthColour {
        match self {
            MouthColour::Blue => MouthColour::Orange,
            MouthColour::Orange => MouthColour::Blue,
        }
    }
}

/// The caps that close the drum: the one low along the axis and the one high along it.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapSide {
    Low,
    High,
}

impl CapSide {
    /// Which way along the axis the cap lies from the middle of the drum.
    pub fn sign(self) -> f64 {
        match self {
            CapSide::Low => -1.0,
            CapSide::High => 1.0,
        }
    }
}

/// The inside surfaces of the drum: the wall with the ground on it, and the caps.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrumSurface {
    Wall,
    Cap(CapSide),
}

/// Where a mouth's centre is, fixed to the wheel: round the ring and along it on the wall, or
/// round the ring and in from the glass on a cap.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum MouthAnchor {
    Wall {
        round: Round,
        along: f64,
    },
    Cap {
        side: CapSide,
        round: Round,
        depth: f64,
    },
}

/// A mouth: where it is, and how far its top is turned from the way its surface lies, which is
/// along the axis on the wall and toward the axis on a cap.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Mouth {
    pub anchor: MouthAnchor,
    pub roll: Radians,
}

impl Mouth {
    pub fn surface(&self) -> DrumSurface {
        match self.anchor {
            MouthAnchor::Wall { .. } => DrumSurface::Wall,
            MouthAnchor::Cap { side, .. } => DrumSurface::Cap(side),
        }
    }
}

/// A point in a mouth's chart.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouthCoords {
    pub u: f64,
    pub v: f64,
    pub h: f64,
}

impl MouthCoords {
    /// How far from the mouth's centre the point is across it.
    pub fn across(self) -> f64 {
        self.u.hypot(self.v)
    }

    /// The same point as the other mouth of the pair has it.
    fn through(self) -> MouthCoords {
        MouthCoords {
            u: -self.u,
            v: self.v,
            h: -self.h,
        }
    }
}

/// How a mouth's surface lies where the mouth is, in the frame: the point its chart is
/// measured from, which on the wall is the glass under its centre and on a cap its centre, and
/// the way the surface runs there before the mouth's roll, spinward and then along the axis
/// or toward it, and off it into the room.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MouthSeat {
    pub origin: Vec3d,
    pub spinward: Vec3d,
    pub upward: Vec3d,
    pub inward: Vec3d,
}

/// How what lies beyond a mouth is seen by an eye before it. Each ray that goes in at a point
/// of the mouth comes out of the other where that point is joined to, turned as the two
/// surfaces lie there, which no one picture shows where the surfaces are not flat. What is
/// drawn is what one rigid move of the eye shows: the move that lays the two surfaces on each
/// other at the point of the mouth nearest the eye, so that an eye going through sees, the
/// moment before it does, just what it sees the moment after.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MouthSight {
    /// The point of the near mouth the move is fitted at, which `passage` has come out.
    pub pivot: Vec3d,
    pub passage: Passage,
    /// A point of the plane that nothing of the far mouth's own surface rises over, and the
    /// way off it into the room: only what is beyond that plane is seen.
    pub threshold: Vec3d,
    pub inward: Vec3d,
}

impl MouthSight {
    /// Where a point before the near mouth is seen from beyond the far one.
    pub fn point(&self, p: Vec3d) -> Vec3d {
        let off = self.passage.turned([0, 1, 2].map(|k| p[k] - self.pivot[k]));
        [0, 1, 2].map(|k| self.passage.point[k] + off[k])
    }

    pub fn attitude(&self, q: Quatd) -> Quatd {
        quat_mul(&self.passage.turn, &q)
    }
}

/// What came of fitting a mouth where it was wanted: the mouth where it fits, with the other
/// mouth as it fits at the pair's new size, or the mouth as it was wanted, which does not fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MouthFit {
    Fits { mouth: Mouth, other: Option<Mouth> },
    Refused(Mouth),
}

impl MouthFit {
    /// The mouth the fit is of, where it would go.
    pub fn mouth(&self) -> Mouth {
        match *self {
            MouthFit::Fits { mouth, .. } | MouthFit::Refused(mouth) => mouth,
        }
    }
}

/// The pair of mouths: whichever of them stand, the one size they share, and how far they are
/// from open.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Mouths {
    blue: Option<Mouth>,
    orange: Option<Mouth>,
    diameter: Metres,
    /// How far from open the mouths are, one being shut: not kept, since a pair found standing
    /// is open and a mouth found alone is shut.
    #[serde(skip)]
    shut: f64,
}

impl Default for Mouths {
    fn default() -> Self {
        Mouths {
            blue: None,
            orange: None,
            diameter: Metres(0.0),
            shut: 1.0,
        }
    }
}

impl Mouths {
    pub fn get(&self, colour: MouthColour) -> Option<&Mouth> {
        match colour {
            MouthColour::Blue => self.blue.as_ref(),
            MouthColour::Orange => self.orange.as_ref(),
        }
    }

    /// The mouths that stand.
    pub fn standing(&self) -> impl Iterator<Item = (MouthColour, &Mouth)> {
        MouthColour::BOTH
            .into_iter()
            .filter_map(|colour| Some((colour, self.get(colour)?)))
    }

    pub fn is_empty(&self) -> bool {
        self.blue.is_none() && self.orange.is_none()
    }

    pub fn diameter(&self) -> Metres {
        self.diameter
    }

    pub fn radius(&self) -> f64 {
        self.diameter.0 as f64 / 2.0
    }

    /// How much of each mouth is filled in with its colour, from its rim inward: all of it
    /// while it is alone, none once the pair is open.
    pub fn fill(&self) -> f64 {
        self.shut * self.shut * (3.0 - 2.0 * self.shut)
    }

    /// How far from a mouth's centre it can be gone through: only a pair can be, and only
    /// where it is not filled in, so what is seen to open is what opens.
    pub fn passable(&self) -> f64 {
        if self.is_pair() {
            self.radius() * (1.0 - self.fill())
        } else {
            0.0
        }
    }

    pub fn remove(&mut self, colour: MouthColour) {
        *self.slot(colour) = None;
    }

    pub fn clear(&mut self) {
        *self = Mouths::default();
    }

    /// As found in a save: a pair open, one alone shut.
    fn settled(mut self) -> Mouths {
        self.shut = if self.is_pair() { 0.0 } else { 1.0 };
        self
    }

    pub fn is_pair(&self) -> bool {
        self.blue.is_some() && self.orange.is_some()
    }

    fn slot(&mut self, colour: MouthColour) -> &mut Option<Mouth> {
        match colour {
            MouthColour::Blue => &mut self.blue,
            MouthColour::Orange => &mut self.orange,
        }
    }

    /// Toward open while both stand, toward shut otherwise.
    pub(super) fn advance(&mut self, dt: f64) {
        let step = dt / OPENING.0 as f64;
        self.shut = if self.is_pair() {
            (self.shut - step).max(0.0)
        } else {
            (self.shut + step).min(1.0)
        };
    }
}

impl Drum {
    /// The mouth a fit found room for takes its place, shut, to open from there; the pair is
    /// the size it was fitted at.
    pub fn put_mouth(&mut self, colour: MouthColour, fit: MouthFit, diameter: Metres) {
        let MouthFit::Fits { mouth, other } = fit else {
            return;
        };
        *self.mouths.slot(colour) = Some(mouth);
        *self.mouths.slot(colour.other()) = other;
        self.mouths.diameter = diameter;
        self.mouths.shut = 1.0;
    }

    /// A mouth centred on a point of a surface, its top turned toward `up` as far as the
    /// surface allows, or, where `up` stands straight off the surface, with `right` to its
    /// right instead. Not fitted: see [`Drum::fit_mouth`].
    pub fn mouth_at(&self, at: Vec3d, surface: DrumSurface, up: Vec3d, right: Vec3d) -> Mouth {
        let anchor = self.mouth_anchor(at, surface);
        let flat = Mouth {
            anchor,
            roll: Radians(0.0),
        };
        let [t1, t2, n] = self.mouth_frame(&flat, MouthCoords::default());
        let laid = |v: Vec3d| {
            let off = dot(&v, &n);
            [v[0] - n[0] * off, v[1] - n[1] * off, v[2] - n[2] * off]
        };
        let top = laid(up);
        let top = if norm(&top) > STANDS_OFF * norm(&up).max(f64::MIN_POSITIVE) {
            top
        } else {
            cross(&n, &laid(right))
        };
        Mouth {
            anchor,
            roll: Radians((-dot(&top, &t1)).atan2(dot(&top, &t2))),
        }
    }

    /// Where a point of a surface is, fixed to the wheel.
    pub fn mouth_anchor(&self, at: Vec3d, surface: DrumSurface) -> MouthAnchor {
        let radius = self.ring.radius.0 as f64;
        let round = self
            .site
            .round
            .on(self.turn_to(at) * radius, self.landscape.grid());
        match surface {
            DrumSurface::Wall => MouthAnchor::Wall {
                round,
                along: self.axial(at),
            },
            DrumSurface::Cap(side) => MouthAnchor::Cap {
                side,
                round,
                depth: self.height_above_glass(at),
            },
        }
    }

    /// Find room for a mouth of `colour` where one is wanted, at the size the pair is to have
    /// from now on: clear of the edges of its surface, of the ground where it is on a cap, and
    /// of the other mouth, which takes the new size where it stands. It is moved no further
    /// than that needs, and refused where that cannot be had.
    pub fn fit_mouth(&self, colour: MouthColour, wanted: Mouth, diameter: Metres) -> MouthFit {
        let rim = diameter.0 as f64 / 2.0 * (1.0 + FLAME_BAND);
        let other = match self.mouths.get(colour.other()) {
            Some(other) => match self.clear_of_edges(*other, rim) {
                Some(other) => Some(other),
                None => return MouthFit::Refused(wanted),
            },
            None => None,
        };
        let Some(mouth) = self.clear_of_edges(wanted, rim) else {
            return MouthFit::Refused(wanted);
        };
        let Some(other) = other else {
            return MouthFit::Fits { mouth, other: None };
        };
        let mouth = if self.apart(&mouth, &other) < 2.0 * rim {
            self.pushed_off(mouth, &other, 2.0 * rim)
                .and_then(|pushed| self.clear_of_edges(pushed, rim))
                .filter(|pushed| self.apart(pushed, &other) >= 2.0 * rim * (1.0 - 1e-9))
        } else {
            Some(mouth)
        };
        match mouth {
            Some(mouth) => MouthFit::Fits {
                mouth,
                other: Some(other),
            },
            None => MouthFit::Refused(wanted),
        }
    }

    /// The point of the frame at a place in a mouth's chart.
    pub fn mouth_point(&self, mouth: &Mouth, at: MouthCoords) -> Vec3d {
        let (s, a) = unrolled(mouth.roll, at.u, at.v);
        match self.seat(mouth) {
            Seat::Wall { turn, y } => {
                let radius = self.ring.radius.0 as f64;
                let half_width = self.ring.half_width.0 as f64;
                let turn = turn + s / radius;
                let along = (self.site.y + y + a).clamp(-half_width, half_width);
                let ground = self.landscape.sample(Place {
                    round: self.site.round.on(turn * radius, self.landscape.grid()),
                    along,
                });
                let on_glass = self.wall_point(turn, along - self.site.y);
                let inward = inward_at(turn);
                let height = ground.0 + at.h;
                [
                    on_glass[0] + inward[0] * height,
                    on_glass[1],
                    on_glass[2] + inward[2] * height,
                ]
            }
            Seat::Cap { centre, t1, t2, n } => {
                [0, 1, 2].map(|k| centre[k] + t1[k] * s + t2[k] * a + n[k] * at.h)
            }
        }
    }

    /// A point of the frame in a mouth's chart.
    pub fn mouth_coords(&self, mouth: &Mouth, p: Vec3d) -> MouthCoords {
        let (s, a, h) = match self.seat(mouth) {
            Seat::Wall { turn, y } => {
                let radius = self.ring.radius.0 as f64;
                let round = (self.turn_to(p) - turn + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0;
                (
                    round * radius,
                    p[1] - y,
                    self.height_above_glass(p) - self.ground(p),
                )
            }
            Seat::Cap { centre, t1, t2, n } => {
                let q = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                (dot(&q, &t1), dot(&q, &t2), dot(&q, &n))
            }
        };
        let (sin, cos) = mouth.roll.0.sin_cos();
        MouthCoords {
            u: s * cos + a * sin,
            v: -s * sin + a * cos,
            h,
        }
    }

    /// The frame a mouth's surface has at a place in its chart: to the right across the mouth
    /// as one faces it, toward its top, and off the surface into the room. On the wall that is
    /// off the glass rather than off the ground, so that it is a frame however the ground lies.
    pub fn mouth_frame(&self, mouth: &Mouth, at: MouthCoords) -> [Vec3d; 3] {
        let (t1, t2, n) = match self.seat(mouth) {
            Seat::Wall { turn, .. } => {
                let (s, _) = unrolled(mouth.roll, at.u, at.v);
                let turn = turn + s / self.ring.radius.0 as f64;
                let (sin, cos) = turn.sin_cos();
                ([-sin, 0.0, cos], [0.0, 1.0, 0.0], inward_at(turn))
            }
            Seat::Cap { t1, t2, n, .. } => (t1, t2, n),
        };
        let (sin, cos) = mouth.roll.0.sin_cos();
        [
            [0, 1, 2].map(|k| t1[k] * cos + t2[k] * sin),
            [0, 1, 2].map(|k| -t1[k] * sin + t2[k] * cos),
            n,
        ]
    }

    /// How a mouth's surface lies where the mouth is.
    pub fn mouth_seat(&self, mouth: &Mouth) -> MouthSeat {
        match self.seat(mouth) {
            Seat::Wall { turn, y } => {
                let (sin, cos) = turn.sin_cos();
                MouthSeat {
                    origin: self.wall_point(turn, y),
                    spinward: [-sin, 0.0, cos],
                    upward: [0.0, 1.0, 0.0],
                    inward: inward_at(turn),
                }
            }
            Seat::Cap { centre, t1, t2, n } => MouthSeat {
                origin: centre,
                spinward: t1,
                upward: t2,
                inward: n,
            },
        }
    }

    /// The rim of a disc of `radius` about a mouth's centre, wrapped onto its surface and held
    /// `lift` above it, with as many points as the ground it crosses has features to follow.
    pub fn mouth_rim(&self, mouth: &Mouth, radius: f64, lift: f64) -> Vec<Vec3d> {
        let grid = self.landscape.grid();
        let feature = grid.arc.min(grid.along);
        let points =
            ((TAU * radius / feature * RIM_PER_FEATURE) as usize).clamp(RIM_LEAST, RIM_MOST);
        (0..points)
            .map(|k| {
                let (sin, cos) = (k as f64 / points as f64 * TAU).sin_cos();
                self.mouth_point(
                    mouth,
                    MouthCoords {
                        u: cos * radius,
                        v: sin * radius,
                        h: lift,
                    },
                )
            })
            .collect()
    }

    /// The mouth a point of a surface lies on, if it lies on one.
    pub fn mouth_under(&self, p: Vec3d, surface: DrumSurface) -> Option<MouthColour> {
        let radius = self.mouths.radius();
        self.mouths
            .standing()
            .filter(|(_, mouth)| mouth.surface() == surface)
            .find(|(_, mouth)| self.mouth_coords(mouth, p).across() <= radius)
            .map(|(colour, _)| colour)
    }

    /// Where a point reached from `from` really is, when the way to it led through the pair of
    /// mouths: `from` is in the room before a mouth, and the point behind what is open of it.
    ///
    /// The velocity something has in the drum's frame is carried through as it is, turned.
    /// That is its momentum as the mouths have it, since both ride the drum: nothing is taken
    /// from the frame's own motion here, which among the stars is another at each mouth, and
    /// the difference is what the drum gives or takes through the mouths it holds.
    pub fn passage_through_mouths(&self, from: Vec3d, to: Vec3d) -> Option<Passage> {
        let passable = self.mouths.passable();
        if passable <= 0.0 || !self.holds(from) {
            return None;
        }
        MouthColour::BOTH.into_iter().find_map(|colour| {
            let (near, far) = (self.mouths.get(colour)?, self.mouths.get(colour.other())?);
            if self.mouth_coords(near, from).h < 0.0 {
                return None;
            }
            let behind = self.mouth_coords(near, to);
            (behind.h < 0.0 && behind.across() < passable).then(|| self.joined(near, far, behind))
        })
    }

    /// What a sphere over what is open of a mouth on a surface meets in place of the surface,
    /// which is not there: nothing, or the edge of the opening where it overlaps it. `None`
    /// where the sphere is over no opening of that surface, and meets the surface as ever.
    pub(super) fn open_edge(
        &self,
        c: Vec3d,
        radius: f64,
        surface: DrumSurface,
    ) -> Option<Option<Penetration>> {
        let passable = self.mouths.passable();
        if passable <= 0.0 {
            return None;
        }
        self.mouths
            .standing()
            .filter(|(_, mouth)| mouth.surface() == surface)
            .find_map(|(_, mouth)| {
                let at = self.mouth_coords(mouth, c);
                let across = at.across();
                (across < passable).then(|| {
                    let edge = self.mouth_point(
                        mouth,
                        MouthCoords {
                            u: at.u / across.max(f64::MIN_POSITIVE) * passable,
                            v: at.v / across.max(f64::MIN_POSITIVE) * passable,
                            h: 0.0,
                        },
                    );
                    let off = [c[0] - edge[0], c[1] - edge[1], c[2] - edge[2]];
                    let far = norm(&off);
                    (far < radius && far > 0.0).then(|| Penetration {
                        depth: radius - far,
                        normal: off.map(|x| x / far),
                    })
                })
            })
    }

    /// The rigid move that best carries what is before the one mouth to what is behind the
    /// other: their frames at their centres laid on each other. It is what the pair does
    /// exactly where both mouths are flat, and what is seen through a mouth is drawn by.
    pub fn mouth_view(&self, through: MouthColour) -> Option<Passage> {
        let (near, far) = (self.mouths.get(through)?, self.mouths.get(through.other())?);
        Some(self.joined(near, far, MouthCoords::default()))
    }

    /// How what lies beyond a mouth is seen from `eye`: see [`MouthSight`].
    pub fn mouth_sight(&self, through: MouthColour, eye: Vec3d) -> Option<MouthSight> {
        let (near, far) = (self.mouths.get(through)?, self.mouths.get(through.other())?);
        let radius = self.mouths.radius();
        let over = self.mouth_coords(near, eye);
        let within = (radius / over.across().max(f64::MIN_POSITIVE)).min(1.0);
        let nearest = MouthCoords {
            u: over.u * within,
            v: over.v * within,
            h: 0.0,
        };
        let centre = self.mouth_point(far, MouthCoords::default());
        let [_, _, inward] = self.mouth_frame(far, MouthCoords::default());
        let lift = (1..=SIGHT_RINGS)
            .flat_map(|ring| self.mouth_rim(far, radius * ring as f64 / SIGHT_RINGS as f64, 0.0))
            .map(|p| {
                dot(
                    &[p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]],
                    &inward,
                )
            })
            .fold(0.0, f64::max);
        Some(MouthSight {
            pivot: self.mouth_point(near, nearest),
            passage: self.joined(near, far, nearest),
            threshold: [0, 1, 2].map(|k| centre[k] + inward[k] * lift),
            inward,
        })
    }

    /// Take the mouths a save holds: a pair open and one alone shut, each where it fits on
    /// this drum, and none of what the save holds that is no place on it.
    pub fn load_mouths(&mut self, saved: &Mouths) {
        let grid = self.landscape.grid();
        let sound = |mouth: &Mouth| {
            let (MouthAnchor::Wall { round, along: span }
            | MouthAnchor::Cap {
                round, depth: span, ..
            }) = mouth.anchor;
            [round.across, span, mouth.roll.0]
                .iter()
                .all(|x| x.is_finite())
                && (0..grid.round).contains(&round.cell)
        };
        self.mouths = Mouths {
            blue: saved.blue.filter(sound),
            orange: saved.orange.filter(sound),
            diameter: Metres(if saved.diameter.0.is_finite() {
                saved.diameter.0.max(0.0)
            } else {
                0.0
            }),
            shut: 1.0,
        };
        self.refit_mouths();
        self.mouths = self.mouths.settled();
    }

    /// Keep what mouths still fit after the drum or the ground changed under them, where they
    /// fit now.
    pub(super) fn refit_mouths(&mut self) {
        let rim = self.mouths.radius() * (1.0 + FLAME_BAND);
        for colour in MouthColour::BOTH {
            let refitted = self
                .mouths
                .get(colour)
                .and_then(|mouth| self.clear_of_edges(*mouth, rim));
            *self.mouths.slot(colour) = refitted;
        }
        if let (Some(blue), Some(orange)) = (self.mouths.blue, self.mouths.orange)
            && self.apart(&blue, &orange) < 2.0 * rim * (1.0 - 1e-9)
        {
            self.mouths.orange = None;
        }
    }

    /// The mouths' anchors round a ring of another size: as far round from the site as they
    /// were, the way the ground is kept.
    pub(super) fn carry_mouths(&mut self, from: Round, old: super::Grid) {
        let grid = self.landscape.grid();
        let site = self.site.round;
        for colour in MouthColour::BOTH {
            if let Some(mouth) = self.mouths.slot(colour) {
                let (MouthAnchor::Wall { round, .. } | MouthAnchor::Cap { round, .. }) =
                    &mut mouth.anchor;
                *round = site.on(from.arc_to(*round, old), grid);
            }
        }
        self.refit_mouths();
    }
}

/// How short of its full length a viewer's up may fall when laid on a surface and still say
/// which way a mouth's top goes.
const STANDS_OFF: f64 = 0.2;
/// How many points a rim has, at least and at most, and how many per feature of the ground it
/// crosses.
const RIM_LEAST: usize = 48;
const RIM_MOST: usize = 512;
const RIM_PER_FEATURE: f64 = 2.0;
/// How many rings of a mouth's surface are looked over for its highest point.
const SIGHT_RINGS: usize = 3;
/// How many times a mouth on a cap is moved toward the axis to clear the ground before it is
/// given up on.
const CLEARINGS: usize = 8;

/// Where a mouth sits in the frame.
enum Seat {
    /// On the wall: how far round from the site, and where along the frame's axis.
    Wall { turn: f64, y: f64 },
    /// On a cap: its centre, the cap's own spinward and axisward there, and the way into the
    /// room.
    Cap {
        centre: Vec3d,
        t1: Vec3d,
        t2: Vec3d,
        n: Vec3d,
    },
}

fn inward_at(turn: f64) -> Vec3d {
    let (sin, cos) = turn.sin_cos();
    [-cos, 0.0, -sin]
}

fn unrolled(roll: Radians, u: f64, v: f64) -> (f64, f64) {
    let (sin, cos) = roll.0.sin_cos();
    (u * cos - v * sin, u * sin + v * cos)
}

impl Drum {
    fn seat(&self, mouth: &Mouth) -> Seat {
        let radius = self.ring.radius.0 as f64;
        let grid = self.landscape.grid();
        match mouth.anchor {
            MouthAnchor::Wall { round, along } => Seat::Wall {
                turn: self.site.round.arc_to(round, grid) / radius,
                y: along - self.site.y,
            },
            MouthAnchor::Cap { side, round, depth } => {
                let turn = self.site.round.arc_to(round, grid) / radius;
                let half_width = self.ring.half_width.0 as f64;
                let on_glass = self.wall_point(turn, side.sign() * half_width - self.site.y);
                let inward = inward_at(turn);
                let (sin, cos) = turn.sin_cos();
                let spinward = [-sin * side.sign(), 0.0, cos * side.sign()];
                Seat::Cap {
                    centre: [
                        on_glass[0] + inward[0] * depth,
                        on_glass[1],
                        on_glass[2] + inward[2] * depth,
                    ],
                    t1: spinward,
                    t2: inward,
                    n: [0.0, -side.sign(), 0.0],
                }
            }
        }
    }

    /// The point `at` behind `near` as `far` has it, and the turn between the two surfaces'
    /// frames there.
    fn joined(&self, near: &Mouth, far: &Mouth, at: MouthCoords) -> Passage {
        let out = at.through();
        let [eu, ev, n] = self.mouth_frame(near, at);
        let [fu, fv, fn_] = self.mouth_frame(far, out);
        let before = quat_from_basis(&eu, &ev, &n);
        let after = quat_from_basis(&fu.map(|x| -x), &fv, &fn_.map(|x| -x));
        Passage {
            point: self.mouth_point(far, out),
            turn: normalised(quat_mul(&after, &quat_conjugate(&before))),
        }
    }

    /// A mouth moved as little as keeps a disc of `rim` about it on its surface: between the
    /// caps on the wall, and above the ground on a cap.
    fn clear_of_edges(&self, mouth: Mouth, rim: f64) -> Option<Mouth> {
        let radius = self.ring.radius.0 as f64;
        let half_width = self.ring.half_width.0 as f64;
        match mouth.anchor {
            MouthAnchor::Wall { round, along } => {
                let room = half_width - rim;
                (room >= 0.0 && rim < FRAC_PI_2 * radius).then(|| Mouth {
                    anchor: MouthAnchor::Wall {
                        round,
                        along: along.clamp(-room, room),
                    },
                    ..mouth
                })
            }
            MouthAnchor::Cap { side, round, depth } => {
                let mut depth = depth.min(radius);
                for _ in 0..CLEARINGS {
                    let seated = Mouth {
                        anchor: MouthAnchor::Cap { side, round, depth },
                        ..mouth
                    };
                    let buried = self.buried(&seated, rim);
                    if buried <= 0.0 {
                        return Some(seated);
                    }
                    if depth >= radius {
                        return None;
                    }
                    depth = (depth + buried).min(radius);
                }
                None
            }
        }
    }

    /// How far under the ground the lowest point of a disc on a cap lies.
    fn buried(&self, mouth: &Mouth, rim: f64) -> f64 {
        self.mouth_rim(mouth, rim, 0.0)
            .into_iter()
            .map(|p| self.ground(p) - self.height_above_glass(p))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// How far apart two mouths' centres are: along the wall where both are on it, straight
    /// across otherwise.
    fn apart(&self, a: &Mouth, b: &Mouth) -> f64 {
        match (a.anchor, b.anchor) {
            (
                MouthAnchor::Wall { round, along },
                MouthAnchor::Wall {
                    round: other_round,
                    along: other_along,
                },
            ) => other_round
                .arc_to(round, self.landscape.grid())
                .hypot(along - other_along),
            _ => {
                let p = self.mouth_point(a, MouthCoords::default());
                let q = self.mouth_point(b, MouthCoords::default());
                norm(&[p[0] - q[0], p[1] - q[1], p[2] - q[2]])
            }
        }
    }

    /// A mouth moved straight away from another on the same surface until their centres are
    /// `apart`. Mouths on different surfaces are not moved off each other.
    fn pushed_off(&self, mouth: Mouth, other: &Mouth, apart: f64) -> Option<Mouth> {
        let grid = self.landscape.grid();
        match (mouth.anchor, other.anchor) {
            (
                MouthAnchor::Wall { round, along },
                MouthAnchor::Wall {
                    round: other_round,
                    along: other_along,
                },
            ) => {
                let room = self.ring.half_width.0 as f64 - apart / 2.0;
                let (arc, axial) = (other_round.arc_to(round, grid), along - other_along);
                let far = arc.hypot(axial);
                let (arc, axial) = if far > 0.0 {
                    (arc / far * apart, axial / far * apart)
                } else {
                    (apart, 0.0)
                };
                // held between the caps, the rest of the way is made up round the ring
                let along = (other_along + axial).clamp(-room.max(0.0), room.max(0.0));
                let axial = along - other_along;
                let round_by = (apart * apart - axial * axial).max(0.0).sqrt();
                let arc = if arc < 0.0 { -round_by } else { round_by };
                Some(Mouth {
                    anchor: MouthAnchor::Wall {
                        round: other_round.on(arc, grid),
                        along,
                    },
                    ..mouth
                })
            }
            (
                MouthAnchor::Cap { side, .. },
                MouthAnchor::Cap {
                    side: other_side, ..
                },
            ) if side == other_side => {
                let p = self.mouth_point(&mouth, MouthCoords::default());
                let q = self.mouth_point(other, MouthCoords::default());
                let d = [p[0] - q[0], 0.0, p[2] - q[2]];
                let far = norm(&d);
                let d = if far > 0.0 {
                    d.map(|x| x / far * apart)
                } else {
                    [0.0, 0.0, apart]
                };
                let at = [q[0] + d[0], p[1], q[2] + d[2]];
                let radius = self.ring.radius.0 as f64;
                Some(Mouth {
                    anchor: MouthAnchor::Cap {
                        side,
                        round: self.site.round.on(self.turn_to(at) * radius, grid),
                        depth: self.height_above_glass(at),
                    },
                    ..mouth
                })
            }
            _ => None,
        }
    }
}

fn normalised(q: Quatd) -> Quatd {
    let l = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    q.map(|x| x / l)
}
