use bevy::platform::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::Ring;
use crate::core::codec;
use crate::core::units::Metres;

/// How far apart the landscape's heights stand, round the ring and along its axis, on a ring
/// of any size: the finest feature the ground can hold.
pub const CELL: f64 = 0.25;
/// The landscape is kept in square patches this many cells a side, and only where it differs
/// from the depth of ground it was laid with.
pub const PATCH: i64 = 32;
/// The most patches the landscape counts round the ring or along it. Only a ring wider than a
/// light year reaches that, and its cells are wider than `CELL`.
const MOST_PATCHES: f64 = (1u64 << 55) as f64;
/// The most steps the search for the level a body of water stands at takes.
const FINDING_THE_LEVEL: u32 = 40;
/// Versions are counted across every landscape, so that none ever takes a number another has
/// had, and nothing worked out from one is taken for being up to date with another.
static VERSIONS: AtomicU64 = AtomicU64::new(0);

/// How the landscape's cells lie on a ring: a whole number of patches round it, so that the
/// cells close on themselves, and rows along the axis a cell apart from the middle out to past
/// the caps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// How many cells there are round the ring, and the arc each spans there.
    pub round: i64,
    pub arc: f64,
    /// How far apart the rows are, and how many there are either side of the middle one.
    pub along: f64,
    pub rows: i64,
}

impl Grid {
    pub fn of(ring: Ring) -> Grid {
        let circumference = TAU * ring.radius.0 as f64;
        let patches = (circumference / (PATCH as f64 * CELL))
            .round()
            .clamp(1.0, MOST_PATCHES);
        let round = patches as i64 * PATCH;
        let half_width = ring.half_width.0 as f64;
        let along = CELL.max(half_width / (MOST_PATCHES * PATCH as f64));
        Grid {
            round,
            arc: circumference / round as f64,
            along,
            rows: (half_width / along).ceil() as i64,
        }
    }

    /// How many patches there are round the ring.
    pub fn patches_round(self) -> i64 {
        self.round / PATCH
    }

    /// A number of cells round the ring, taken the short way.
    pub fn short_way(self, cells: i128) -> i64 {
        let round = self.round as i128;
        ((cells + round / 2).rem_euclid(round) - round / 2) as i64
    }

    /// A cell counted from wheel angle zero, however many times round the ring that is.
    pub fn wrap(self, cell: i128) -> i64 {
        cell.rem_euclid(self.round as i128) as i64
    }
}

/// A place round the ring, exact however large the ring is: the cell of the landscape it falls
/// in, counted spinward from wheel angle zero, and how far across that cell it lies, as a share
/// of the cell.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct Round {
    pub cell: i64,
    pub across: f64,
}

impl Round {
    pub fn at_angle(phi: f64, grid: Grid) -> Round {
        let cells = phi.rem_euclid(TAU) / TAU * grid.round as f64;
        let whole = cells.floor();
        Round {
            cell: grid.wrap(whole as i128),
            across: (cells - whole).clamp(0.0, 1.0),
        }
    }

    /// The place `arc` metres spinward of this one.
    pub fn on(self, arc: f64, grid: Grid) -> Round {
        let cells = self.across + arc / grid.arc;
        let whole = cells.floor();
        Round {
            cell: grid.wrap(self.cell as i128 + whole as i128),
            across: cells - whole,
        }
    }

    /// How far spinward `to` is from here, the short way round.
    pub fn arc_to(self, to: Round, grid: Grid) -> f64 {
        let cells = grid.short_way(to.cell as i128 - self.cell as i128);
        (cells as f64 + to.across - self.across) * grid.arc
    }

    /// This place as a wheel angle, as exact as an angle can be.
    pub fn angle(self, grid: Grid) -> f64 {
        (self.cell as f64 + self.across) / grid.round as f64 * TAU
    }

    /// How far this place is round the ring from wheel angle zero.
    pub fn arc(self, grid: Grid) -> f64 {
        (self.cell as f64 + self.across) * grid.arc
    }
}

/// A point of the ground: its place round the ring, and along the axis from the middle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub round: Round,
    pub along: f64,
}

/// How a body of water lies on a landscape: see [`Landscape::flooded`].
#[derive(Clone, Copy, Default, Debug)]
pub struct Flood {
    pub level: Metres,
    pub covered: f32,
    pub depth: Metres,
}

/// What of a landscape is kept: the depth it was laid with, and every patch where it differs
/// from that.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Ground {
    pub base: f32,
    pub patches: Vec<Patch>,
}

/// A patch of ground: which patch round the ring and along it, and its heights, round-major.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Patch {
    pub round: i64,
    pub along: i64,
    #[serde(with = "codec::f32s")]
    pub heights: Vec<f32>,
}

/// Sculptable terrain on the inside of the drum floor: a heightfield in the wheel's frame,
/// measured inward from the glass, with its heights a cell apart whatever size the ring is.
/// The ground is laid at one depth all round, and kept in patches only where it has been
/// sculpted away from that, so a ring of any size costs only what has been done to it. Height
/// zero means bare glass.
pub struct Landscape {
    ring: Ring,
    grid: Grid,
    base: f32,
    patches: HashMap<(i64, i64), Sculpted>,
    /// A number no landscape has had before, taken afresh on every change, so consumers can
    /// rebuild derived data lazily.
    version: u64,
}

/// A sculpted patch: its heights, round-major, and what is known of it, worked out afresh
/// whenever it or a patch beside it changes, so that what is known of the ground as a whole
/// costs only as much as there are patches.
struct Sculpted {
    heights: Box<[f32]>,
    /// The version of the landscape its heights last changed in.
    changed: u64,
    summary: Summary,
}

/// The heights of a patch's cells within the grid, lowest first, and the sum of all those
/// before each; and how steeply the ground rises anywhere in it or across its edges, per metre.
#[derive(Default)]
struct Summary {
    heights: Vec<f64>,
    sums: Vec<f64>,
    steepest: f64,
}

impl Landscape {
    /// Bare glass all the way round.
    pub fn new(ring: Ring) -> Self {
        Landscape {
            ring,
            grid: Grid::of(ring),
            base: 0.0,
            patches: HashMap::default(),
            version: next_version(),
        }
    }

    /// Ground of one depth everywhere.
    pub fn flat(ring: Ring, depth: Metres) -> Self {
        let mut land = Landscape::new(ring);
        land.flatten(depth);
        land
    }

    pub fn ring(&self) -> Ring {
        self.ring
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// The depth the ground keeps wherever it has not been sculpted.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Lay the ground round a ring of another size, as it lies about a place: every cell keeps
    /// how far round the ring it is from the one under `from`, which is now under `to`, and
    /// where it is along the axis. What the new ring has no room for is lost, and what it has
    /// more room for is laid as the rest of the ground is.
    pub fn resize(&mut self, ring: Ring, from: Round, to: Round) {
        let old = self.grid;
        let patches = std::mem::take(&mut self.patches);
        self.ring = ring;
        self.grid = Grid::of(ring);
        let max = ring.max_height().0;
        self.base = self.base.min(max);
        let room = self.grid.round / 2;
        for ((round, along), patch) in patches {
            for (k, height) in patch.heights.iter().enumerate() {
                let (cell, row) = cell_of(round, along, k);
                let offset = old.short_way(cell as i128 - from.cell as i128);
                if row.abs() > self.grid.rows || offset < -room || offset >= room {
                    continue;
                }
                let cell = self.grid.wrap(to.cell as i128 + offset as i128);
                self.set(cell, row, height.min(max));
            }
        }
        let base = self.base;
        self.patches
            .retain(|_, patch| patch.heights.iter().any(|h| *h != base));
        self.relaid();
    }

    /// Make the ground one depth everywhere.
    pub fn flatten(&mut self, depth: Metres) {
        self.base = depth.0.clamp(0.0, self.ring.max_height().0);
        self.patches.clear();
        self.version = next_version();
    }

    pub fn is_empty(&self) -> bool {
        self.base <= 0.0 && self.patches.is_empty()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn max_height(&self) -> f32 {
        self.summaries()
            .filter_map(|summary| summary.heights.last())
            .fold(self.base, |highest, h| highest.max(*h as f32))
    }

    /// The height of the ground at a cell round the ring and a row along it.
    pub fn height(&self, cell: i64, row: i64) -> f32 {
        let (key, k) = patch_of(cell, row);
        self.patches
            .get(&key)
            .map_or(self.base, |patch| patch.heights[k])
    }

    /// Every patch where the ground has been sculpted, by which patch round the ring and along
    /// it it is.
    pub fn patches(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.patches.keys().copied()
    }

    /// The heights of a sculpted patch, round-major.
    pub fn patch(&self, round: i64, along: i64) -> Option<&[f32]> {
        self.patches
            .get(&(round, along))
            .map(|patch| &patch.heights[..])
    }

    /// Every sculpted patch whose heights changed after the landscape's `version`, or all of
    /// them after 0, which no landscape ever has: by which patch round the ring and along it it
    /// is, and its heights, round-major.
    pub fn changed_since(&self, version: u64) -> impl Iterator<Item = ((i64, i64), &[f32])> + '_ {
        self.patches
            .iter()
            .filter(move |(_, patch)| patch.changed > version)
            .map(|(key, patch)| (*key, &patch.heights[..]))
    }

    /// How a body of water this big lies on the landscape once it has found its level: how
    /// high it stands over the glass, the share of the ground under it, and how deep it stands
    /// over that ground on average. Water settles into the hollows first, so what it hides is
    /// not its volume spread evenly: a little of it covers the whole of a flat ring and none
    /// of a steep one.
    pub fn flooded(&self, cubic_metres: f64) -> Flood {
        if cubic_metres <= 0.0 {
            return Flood::default();
        }
        let cells = self.grid.round as f64 * (2 * self.grid.rows + 1) as f64;
        let width = 2.0 * self.ring.half_width.0 as f64;
        let per_cell = TAU * self.ring.radius.0 as f64 * width / cells;
        let sculpted: usize = self.summaries().map(|summary| summary.heights.len()).sum();
        let laid = cells - sculpted as f64;
        let base = self.base as f64;
        let under = |level: f64| -> (f64, f64) {
            let (mut held, mut wet) = if base < level {
                (laid * (level - base), laid)
            } else {
                (0.0, 0.0)
            };
            for summary in self.summaries() {
                let k = summary.heights.partition_point(|h| *h < level);
                held += k as f64 * level - summary.sums[k];
                wet += k as f64;
            }
            (held * per_cell, wet)
        };
        // what a level holds grows faster the higher it stands, by a cell's worth for every
        // cell under it, so a step down at that rate from above never passes the level the
        // water finds, and lands on it once no height lies between the two
        let mut level = self.max_height() as f64 + cubic_metres / (per_cell * cells);
        let (mut held, mut wet) = under(level);
        for _ in 0..FINDING_THE_LEVEL {
            let next = level - (held - cubic_metres) / (wet * per_cell);
            if next.is_nan() || next >= level {
                break;
            }
            level = next;
            (held, wet) = under(level);
        }
        Flood {
            level: Metres(level as f32),
            covered: (wet / cells) as f32,
            depth: Metres(if wet > 0.0 {
                (held / (per_cell * wet)) as f32
            } else {
                0.0
            }),
        }
    }

    /// What of the landscape is worth keeping.
    pub fn ground(&self) -> Ground {
        let mut patches: Vec<Patch> = self
            .patches
            .iter()
            .map(|(&(round, along), patch)| Patch {
                round,
                along,
                heights: patch.heights.to_vec(),
            })
            .collect();
        patches.sort_by_key(|patch| (patch.round, patch.along));
        Ground {
            base: self.base,
            patches,
        }
    }

    /// Lay kept ground back down. Whatever does not fit this ring's grid is left out and
    /// whatever is not a height the ring can hold is made one.
    pub fn load(&mut self, ground: &Ground) {
        let max = self.ring.max_height().0;
        let height = |h: f32| {
            if h.is_finite() {
                h.clamp(0.0, max)
            } else {
                0.0
            }
        };
        self.base = height(ground.base);
        self.patches.clear();
        let cells = (PATCH * PATCH) as usize;
        for patch in &ground.patches {
            let rows = self.grid.rows;
            let fits = (0..self.grid.patches_round()).contains(&patch.round)
                && ((-rows).div_euclid(PATCH)..=rows.div_euclid(PATCH)).contains(&patch.along)
                && patch.heights.len() == cells;
            if fits {
                let heights: Box<[f32]> = patch.heights.iter().map(|h| height(*h)).collect();
                if heights.iter().any(|h| *h != self.base) {
                    self.patches.insert(
                        (patch.round, patch.along),
                        Sculpted {
                            heights,
                            changed: 0,
                            summary: Summary::default(),
                        },
                    );
                }
            }
        }
        self.relaid();
    }

    /// Raise (or lower, for negative `amount`) the terrain by up to `amount` metres inside a
    /// round brush of `radius` metres centred on a point of the ground.
    pub fn sculpt(&mut self, at: Place, radius: f64, amount: f64) {
        let grid = self.grid;
        let max = self.ring.max_height().0;
        let reach = (radius / grid.arc).ceil() as i64 + 1;
        let (first, last) = if 2 * reach + 1 >= grid.round {
            (-(grid.round / 2), grid.round - grid.round / 2 - 1)
        } else {
            (-reach, reach)
        };
        let lowest = ((at.along - radius) / grid.along).ceil() as i64;
        let highest = ((at.along + radius) / grid.along).floor() as i64;
        let rows = lowest.max(-grid.rows)..=highest.min(grid.rows);
        let mut touched = Vec::new();
        for offset in first..=last {
            let dx = (offset as f64 - at.round.across) * grid.arc;
            if dx.abs() >= radius {
                continue;
            }
            let cell = grid.wrap(at.round.cell as i128 + offset as i128);
            for row in rows.clone() {
                let dy = row as f64 * grid.along - at.along;
                let q = (dx * dx + dy * dy) / (radius * radius);
                if q >= 1.0 {
                    continue;
                }
                let w = (1.0 - q) * (1.0 - q);
                let before = self.height(cell, row);
                let after = (before + (amount * w) as f32).clamp(0.0, max);
                if after != before {
                    self.set(cell, row, after);
                    touched.push(patch_of(cell, row).0);
                }
            }
        }
        if touched.is_empty() {
            return;
        }
        touched.sort_unstable();
        touched.dedup();
        for key in &touched {
            if self
                .patches
                .get(key)
                .is_some_and(|patch| patch.heights.iter().all(|h| *h == self.base))
            {
                self.patches.remove(key);
            }
        }
        let mut around: Vec<(i64, i64)> = touched
            .iter()
            .flat_map(|&(round, along)| {
                let beside = |r: i64| self.grid.wrap(r as i128 * PATCH as i128) / PATCH;
                [
                    (round, along),
                    (beside(round - 1), along),
                    (beside(round + 1), along),
                    (round, along - 1),
                    (round, along + 1),
                ]
            })
            .collect();
        around.sort_unstable();
        around.dedup();
        self.summarize(&around);
        self.stamp(&touched);
    }

    /// Bilinear height at a point of the ground, and how fast it rises round the ring and along
    /// the axis, per metre.
    #[inline]
    pub fn sample(&self, at: Place) -> (f64, f64, f64) {
        let grid = self.grid;
        let (i0, fu) = (at.round.cell, at.round.across);
        let i1 = grid.wrap(i0 as i128 + 1);
        let v = (at.along / grid.along).clamp(-grid.rows as f64, grid.rows as f64 - 1e-9);
        let j0 = v.floor() as i64;
        let fv = v - j0 as f64;
        let j1 = j0 + 1;
        let h00 = self.height(i0, j0) as f64;
        let h10 = self.height(i1, j0) as f64;
        let h01 = self.height(i0, j1) as f64;
        let h11 = self.height(i1, j1) as f64;
        let h = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv;
        let round = ((h10 - h00) * (1.0 - fv) + (h11 - h01) * fv) / grid.arc;
        let along = ((h01 - h00) * (1.0 - fu) + (h11 - h10) * fu) / grid.along;
        (h, round, along)
    }

    /// How steeply the ground rises anywhere, per metre round the ring or along it.
    pub fn steepest(&self) -> f64 {
        self.summaries()
            .map(|summary| summary.steepest)
            .fold(0.0, f64::max)
    }

    fn summaries(&self) -> impl Iterator<Item = &Summary> + '_ {
        self.patches.values().map(|patch| &patch.summary)
    }

    /// Every patch has been laid afresh: work out what is known of each, and count all of them
    /// changed.
    fn relaid(&mut self) {
        let keys: Vec<(i64, i64)> = self.patches.keys().copied().collect();
        self.summarize(&keys);
        self.stamp(&keys);
    }

    /// Work out afresh what is known of these patches, and count the landscape changed.
    fn summarize(&mut self, keys: &[(i64, i64)]) {
        for key in keys {
            if let Some(summary) = self.summary_of(*key)
                && let Some(patch) = self.patches.get_mut(key)
            {
                patch.summary = summary;
            }
        }
        self.version = next_version();
    }

    fn summary_of(&self, (round, along): (i64, i64)) -> Option<Summary> {
        let grid = self.grid;
        let patch = self.patches.get(&(round, along))?;
        let mut heights = Vec::with_capacity(patch.heights.len());
        let mut steepest: f64 = 0.0;
        for (k, h) in patch.heights.iter().enumerate() {
            let (cell, row) = cell_of(round, along, k);
            if row.abs() <= grid.rows {
                heights.push(*h as f64);
            }
            for next in [-1i64, 1] {
                let other = self.height(grid.wrap(cell as i128 + next as i128), row);
                steepest = steepest.max((h - other).abs() as f64 / grid.arc);
                let other = self.height(cell, row + next);
                steepest = steepest.max((h - other).abs() as f64 / grid.along);
            }
        }
        heights.sort_by(f64::total_cmp);
        let mut sums = Vec::with_capacity(heights.len() + 1);
        let mut sum = 0.0;
        sums.push(sum);
        for h in &heights {
            sum += h;
            sums.push(sum);
        }
        Some(Summary {
            heights,
            sums,
            steepest,
        })
    }

    /// Count these patches' heights changed in the landscape's version.
    fn stamp(&mut self, keys: &[(i64, i64)]) {
        for key in keys {
            if let Some(patch) = self.patches.get_mut(key) {
                patch.changed = self.version;
            }
        }
    }

    fn set(&mut self, cell: i64, row: i64, height: f32) {
        let (key, k) = patch_of(cell, row);
        let base = self.base;
        self.patches
            .entry(key)
            .or_insert_with(|| Sculpted {
                heights: vec![base; (PATCH * PATCH) as usize].into_boxed_slice(),
                changed: 0,
                summary: Summary::default(),
            })
            .heights[k] = height;
    }
}

fn next_version() -> u64 {
    VERSIONS.fetch_add(1, Ordering::Relaxed) + 1
}

/// The patch a cell of the grid lies in, and where in the patch.
fn patch_of(cell: i64, row: i64) -> ((i64, i64), usize) {
    let key = (cell.div_euclid(PATCH), row.div_euclid(PATCH));
    let k = cell.rem_euclid(PATCH) * PATCH + row.rem_euclid(PATCH);
    (key, k as usize)
}

/// The cell of the grid at a place in a patch.
fn cell_of(round: i64, along: i64, k: usize) -> (i64, i64) {
    let k = k as i64;
    (round * PATCH + k / PATCH, along * PATCH + k % PATCH)
}
