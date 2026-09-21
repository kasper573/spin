//! The sheet of water on the drum's floor: which of the landscape's cells the sheet's cells
//! lie on, the floor's heights handed to it whenever they or the cells change, and the time
//! the drum's water has run handed to it frame by frame. A ring with no more cells round it
//! and along it than the sheet has is covered whole. On a wider one each of the sheet's cells
//! takes as many of the landscape's each way as has the sheet reach from cap to cap, so that
//! water ends nowhere but at the ring's own walls across it; round a ring too long for that,
//! the sheet lies about the water's site. Whenever the sheet comes to lie on other cells with
//! water on it, the water is carried to where the ground it lay on has gone.
use bevy::prelude::*;

use super::{Drum, Grid, GroundCarry, PATCH, Place};
use crate::core::math::{Vec3d, cross, dot};
use crate::core::rigid::{Body, BodyShape, WaterCoupling};
use crate::core::sheet::{SHEET_CELLS, SHEET_MOST_GRAINS, Sheet, SheetCarried, SheetLie};
use crate::core::sheet::{SHEET_WATCHED, SheetLanding, SheetWater};
use crate::core::units::{Litres, Metres, MetresPerSecond, Seconds};
use crate::systems::sim::Simulation;

/// What the sheet was last laid on: which of the landscape's cells, how those lie on the ring,
/// and the landscape's version.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct SheetWindow(Option<(WindowCells, Grid, u64)>);

/// Water poured on the drum's floor: the point of the drum's frame it comes down under, how
/// fast it comes down there, in the drum's frame, how wide it falls, and how much of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetPour {
    pub at: Vec3d,
    pub velocity: Vec3d,
    pub wide: Metres,
    pub water: Litres,
}

impl SheetWindow {
    /// Lay the sheet on the drum's floor as it is now, if it is not laid on it already. Water
    /// on the sheet goes where the ground under it went: `ground` says where, if the drum was
    /// made another size since the sheet was last laid.
    pub fn lay(&mut self, drum: &Drum, sheet: &mut Sheet, ground: Option<GroundCarry>) {
        let grid = drum.landscape.grid();
        let over = WindowCells::over(drum);
        let now = (over, grid, drum.landscape.version());
        if self.0 == Some(now) {
            return;
        }
        let carried = self
            .0
            .filter(|(was, ..)| !sheet.is_empty() && (*was != over || ground.is_some()))
            .map(|(was, old, _)| was.carried_to(over, [old, grid], ground));
        let lie = SheetLie {
            cells: over.cells,
            closed: over.closed,
            cell: [grid.arc, grid.along].map(|cell| Metres((cell * over.taken as f64) as f32)),
            radius: drum.ring.radius,
        };
        sheet.lay(lie, carried, |round, along| {
            let cell = grid.wrap(over.first.0 as i128 + round as i128 * over.taken as i128);
            drum.landscape
                .height(cell, over.first.1 + along as i64 * over.taken)
        });
        self.0 = Some(now);
    }

    /// Pour water on the floor under a point of the drum's frame, over a square this wide,
    /// coming down on it at `velocity` in the drum's frame, and say whether the sheet took it. While there is no water anywhere, the water's site is put
    /// where the first of it falls, and the sheet about it.
    pub fn pour(
        &mut self,
        drum: &mut Drum,
        sheet: &mut Sheet,
        dry: bool,
        poured: SheetPour,
    ) -> bool {
        let SheetPour {
            at,
            velocity,
            wide,
            water,
        } = poured;
        if dry && sheet.is_empty() {
            drum.settle_water(at);
            self.lay(drum, sheet, None);
        }
        let Some((over, grid, _)) = self.0 else {
            return false;
        };
        let Some(under) = over.cell_under(grid, drum.place(at)) else {
            return false;
        };
        let across = |axis: usize, cell: f64| {
            ((wide.0 as f64 / (cell * over.taken as f64)).round() as u32).clamp(1, over.cells[axis])
        };
        let cells = [across(0, grid.arc), across(1, grid.along)];
        let first = |axis: usize| {
            let first = under[axis] as i64 - (cells[axis] / 2) as i64;
            if axis == 0 && over.closed {
                first.rem_euclid(over.cells[0] as i64) as u32
            } else {
                first.clamp(0, (over.cells[axis] - cells[axis]) as i64) as u32
            }
        };
        let (_, outward) = drum.depth_and_outward(at);
        let spinward = [-outward[2], 0.0, outward[0]];
        let landing = SheetLanding {
            run: [
                MetresPerSecond(dot(&velocity, &spinward) as f32),
                MetresPerSecond(velocity[1] as f32),
            ],
            spread: MetresPerSecond(dot(&velocity, &outward).abs() as f32),
        };
        sheet.pour([first(0), first(1)], cells, water, landing)
    }

    /// Have the sheet report the water round a point of the drum's frame.
    fn watch(&self, drum: &Drum, sheet: &mut Sheet, at: Vec3d) {
        let Some((over, grid, _)) = self.0 else {
            return;
        };
        let Some(under) = over.cell_under(grid, drum.place(at)) else {
            return;
        };
        let half = (SHEET_WATCHED / 2) as i64;
        let first = |axis: usize| {
            let first = under[axis] as i64 - half;
            if axis == 0 && over.closed {
                first.rem_euclid(over.cells[0] as i64) as u32
            } else {
                first.clamp(0, (over.cells[axis] as i64 - 2 * half).max(0)) as u32
            }
        };
        sheet.watch([first(0), first(1)]);
    }

    /// How far under the surface of the sheet's water a point of the drum's frame is: less than
    /// nothing over it, and nothing where there is no water.
    pub fn sunk(&self, drum: &Drum, sheet: &Sheet, at: Vec3d) -> Metres {
        self.water_under(drum, sheet, at)
            .filter(|water| water.depth.0 > 0.0)
            .map_or(Metres(0.0), |water| {
                Metres(water.level.0 - drum.height_above_glass(at) as f32)
            })
    }

    /// What the sheet's water does to a body over a second taken as one substep: it bears up
    /// every sample of the hull that is under its surface with the weight of the water the
    /// sample displaces, and grips it with that water's mass.
    pub fn bearing(
        &self,
        drum: &Drum,
        sheet: &Sheet,
        body: &Body,
        shape: &BodyShape,
    ) -> Option<WaterCoupling> {
        let radius = drum.ring.radius.0 as f64;
        let mut felt = WaterCoupling {
            seconds: 1.0,
            substeps: 1.0,
            ..default()
        };
        for sample in &shape.samples {
            let at = body.to_world(&sample.map(f64::from));
            let Some(water) = self.water_under(drum, sheet, at) else {
                continue;
            };
            let (height, outward) = drum.depth_and_outward(at);
            if water.depth.0 <= 0.0 || height >= water.level.0 as f64 {
                continue;
            }
            let from_axis = radius - height;
            let turn = drum.spin.0 as f64 + water.run[0].0 as f64 / from_axis;
            let gripped = WATER_DENSITY * shape.volume_per_sample * body.inv_m;
            let lift = outward.map(|o| -o * gripped * turn * turn * from_axis);
            let arm = [at[0] - body.p[0], at[1] - body.p[1], at[2] - body.p[2]];
            let twist = cross(&arm, &lift);
            let spinward = [-outward[2], 0.0, outward[0]];
            for k in 0..3 {
                felt.buoyancy[k] += lift[k];
                felt.buoyancy_torque[k] += twist[k];
                felt.flow[k] += gripped * spinward[k] * water.run[0].0 as f64;
            }
            felt.flow[1] += gripped * water.run[1].0 as f64;
            felt.coupling += gripped;
            felt.wet += 1.0 / shape.samples.len() as f64;
        }
        (felt.coupling > 0.0).then_some(felt)
    }

    /// The sheet's water on the floor under a point of the drum's frame, as it was last reported,
    /// if it was reported there.
    fn water_under(&self, drum: &Drum, sheet: &Sheet, at: Vec3d) -> Option<SheetWater> {
        let lie = sheet.lie()?;
        let (over, grid, _) = self.0?;
        let under = drum.place(at);
        let round = (grid.wrap(under.round.cell as i128 - over.first.0 as i128) as f64
            + under.round.across)
            / over.taken as f64;
        let along = (under.along / grid.along - over.first.1 as f64) / over.taken as f64;
        sheet.watched().at(lie, [round, along])
    }
}

pub fn feed_sheet(
    sim: Res<Simulation>,
    mut sheet: ResMut<Sheet>,
    mut window: ResMut<SheetWindow>,
    mut fed: Local<Seconds>,
) {
    let drum = &sim.drum;
    window.lay(drum, &mut sheet, None);
    let Some((over, grid, _)) = window.0 else {
        return;
    };
    let from_site = grid.short_way(over.first.0 as i128 - drum.water.round.cell as i128) as f64
        - drum.water.round.across;
    sheet.place([
        Metres((from_site * grid.arc) as f32),
        Metres((over.first.1 as f64 * grid.along - drum.water.y) as f32),
    ]);
    window.watch(drum, &mut sheet, sim.avatar().p);
    let run = Seconds((sim.time.0 - fed.0).max(0.0));
    *fed = sim.time;
    sheet.advance(run, drum.spin);
}

/// Water's density, in kilograms a cubic metre.
const WATER_DENSITY: f64 = 1000.0;

/// The most of the landscape's cells one of the sheet's takes each way: the landscape's cells
/// round the ring come in patches this many long, so that many always divide them evenly.
const MOST_TAKEN: i64 = PATCH;
const _: () = assert!(MOST_TAKEN as u32 <= SHEET_MOST_GRAINS);

/// Which of the landscape's cells the sheet's lie on: the first of them round the ring and
/// along it, how many of the landscape's cells each of the sheet's takes each way, how many
/// cells the sheet has each way, and whether those round the ring close on themselves.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WindowCells {
    first: (i64, i64),
    taken: i64,
    cells: [u32; 2],
    closed: bool,
}

impl WindowCells {
    fn over(drum: &Drum) -> WindowCells {
        let grid = drum.landscape.grid();
        let [most_round, most_along] = SHEET_CELLS.map(i64::from);
        let rows = 2 * grid.rows;
        let taken = (rows as u64)
            .div_ceil(most_along as u64)
            .next_power_of_two()
            .min(MOST_TAKEN as u64) as i64;
        let closed = grid.round <= most_round * taken;
        let (first_round, round) = if closed {
            (0, grid.round / taken)
        } else {
            let site = drum.water.round.cell / taken * taken;
            let first = site as i128 - (most_round / 2 * taken) as i128;
            (grid.wrap(first), most_round)
        };
        let (first_row, along) = if rows <= most_along * taken {
            (-grid.rows, (rows + taken - 1) / taken)
        } else {
            let under = (drum.water.y / grid.along).floor() as i64;
            let reach = most_along * taken;
            let first = (under - reach / 2).clamp(-grid.rows, grid.rows - reach);
            (first, most_along)
        };
        WindowCells {
            first: (first_round, first_row),
            taken,
            cells: [round as u32, along as u32],
            closed,
        }
    }

    /// The sheet's cell under a point of the ground, if the sheet reaches there.
    fn cell_under(&self, grid: Grid, at: Place) -> Option<[u32; 2]> {
        let round = grid.wrap(at.round.cell as i128 - self.first.0 as i128) / self.taken;
        let along = ((at.along / grid.along).floor() as i64 - self.first.1).div_euclid(self.taken);
        (round < self.cells[0] as i64 && (0..self.cells[1] as i64).contains(&along))
            .then_some([round as u32, along as u32])
    }

    /// How water on a sheet that lay on these cells is carried to one laid on others: the
    /// landscape's cells are the grains, and each is the one it was when the ground was last
    /// carried, or the very same if it was not.
    fn carried_to(
        self,
        now: WindowCells,
        [old, new]: [Grid; 2],
        ground: Option<GroundCarry>,
    ) -> SheetCarried {
        let round = (0..now.cells[0] as i64 * now.taken)
            .map(|grain| {
                let cell = new.wrap(now.first.0 as i128 + grain as i128);
                let cell = ground.map_or(Some(cell), |ground| ground.out_of(cell))?;
                let was = old.wrap(cell as i128 - self.first.0 as i128);
                (was < self.cells[0] as i64 * self.taken).then_some(was as u32)
            })
            .collect();
        let along = (0..now.cells[1] as i64 * now.taken)
            .map(|grain| {
                let was = now.first.1 + grain - self.first.1;
                (0..self.cells[1] as i64 * self.taken)
                    .contains(&was)
                    .then_some(was as u32)
            })
            .collect();
        SheetCarried {
            grains: [self.taken as u32, now.taken as u32],
            was: [round, along],
            spread: (old.arc * old.along / (new.arc * new.along)) as f32,
        }
    }
}
