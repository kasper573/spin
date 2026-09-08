use super::H;

/// Counting-sort spatial grid over kernel-sized cells covering a fixed box; points outside the box
/// land in its edge cells.
pub struct Grid {
    min: [f32; 3],
    dims: [i32; 3],
    pub start: Vec<u32>,
    count: Vec<u32>,
    pub entries: Vec<u32>,
    cell: Vec<u32>,
}

impl Grid {
    pub fn new(cap: usize, min: [f32; 3], max: [f32; 3]) -> Self {
        let dims = [0, 1, 2].map(|a| ((max[a] - min[a]) / H).ceil() as i32 + 1);
        let cells = (dims[0] * dims[1] * dims[2]) as usize;
        Grid {
            min,
            dims,
            start: vec![0; cells + 1],
            count: vec![0; cells],
            entries: vec![0; cap],
            cell: vec![0; cap],
        }
    }

    #[inline]
    pub fn coords(&self, x: f32, y: f32, z: f32) -> [i32; 3] {
        let c = |v: f32, a: usize| {
            (((v - self.min[a]) * (1.0 / H)).floor() as i32).clamp(0, self.dims[a] - 1)
        };
        [c(x, 0), c(y, 1), c(z, 2)]
    }

    /// Index of a cell, or none if it lies outside the grid.
    #[inline]
    pub fn cell_index(&self, [ix, iy, iz]: [i32; 3]) -> Option<usize> {
        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.dims[0]
            || iy >= self.dims[1]
            || iz >= self.dims[2]
        {
            return None;
        }
        Some(((iz * self.dims[1] + iy) * self.dims[0] + ix) as usize)
    }

    pub fn build(&mut self, x: &[f32], y: &[f32], z: &[f32], n: usize) {
        self.count.fill(0);
        for i in 0..n {
            let c = self.cell_index(self.coords(x[i], y[i], z[i])).unwrap_or(0);
            self.cell[i] = c as u32;
            self.count[c] += 1;
        }
        self.start[0] = 0;
        for c in 0..self.count.len() {
            self.start[c + 1] = self.start[c] + self.count[c];
        }
        self.count.fill(0);
        for i in 0..n {
            let c = self.cell[i] as usize;
            self.entries[(self.start[c] + self.count[c]) as usize] = i as u32;
            self.count[c] += 1;
        }
    }
}
