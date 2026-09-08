//! Isosurface of a particle cloud: particles are splatted into a scalar grid and the surface is
//! extracted with naive surface nets (one vertex per sign-changing cell, one quad per crossing edge).

pub struct SurfaceMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// One value per vertex, the splatted particles' foam blended by kernel weight.
    pub foam: Vec<f32>,
    pub indices: Vec<u32>,
}

pub struct SurfaceGrid {
    origin: [f32; 3],
    cell: f32,
    dims: [usize; 3],
    density: Vec<f32>,
    foam: Vec<f32>,
    lo: [usize; 3],
    hi: [usize; 3],
    vertex_of_cell: Vec<u32>,
}

const NO_VERTEX: u32 = u32::MAX;

impl SurfaceGrid {
    /// A grid of `cell`-sized cubes covering the box from `min` to `max`.
    pub fn new(min: [f32; 3], max: [f32; 3], cell: f32) -> Self {
        let dims = [0, 1, 2].map(|a| ((max[a] - min[a]) / cell).ceil() as usize + 1);
        let corners = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1);
        SurfaceGrid {
            origin: min,
            cell,
            dims,
            density: vec![0.0; corners],
            foam: vec![0.0; corners],
            lo: dims,
            hi: [0; 3],
            vertex_of_cell: vec![NO_VERTEX; dims[0] * dims[1] * dims[2]],
        }
    }

    pub fn clear(&mut self) {
        self.density.fill(0.0);
        self.foam.fill(0.0);
        self.lo = self.dims;
        self.hi = [0; 3];
    }

    /// Add a particle's smooth kernel (radius `r`) to every grid corner it reaches.
    pub fn splat(&mut self, p: [f32; 3], foam: f32, r: f32) {
        let inv_r2 = 1.0 / (r * r);
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for a in 0..3 {
            let g = (p[a] - self.origin[a]) / self.cell;
            let reach = r / self.cell;
            lo[a] = (g - reach).ceil().max(0.0) as usize;
            hi[a] = ((g + reach).floor().max(0.0) as usize).min(self.dims[a]);
            if lo[a] > hi[a] {
                return;
            }
            self.lo[a] = self.lo[a].min(lo[a].saturating_sub(1));
            self.hi[a] = self.hi[a].max((hi[a] + 1).min(self.dims[a]));
        }
        let stride_y = self.dims[2] + 1;
        let stride_x = (self.dims[1] + 1) * stride_y;
        for i in lo[0]..=hi[0] {
            let dx = self.origin[0] + i as f32 * self.cell - p[0];
            for j in lo[1]..=hi[1] {
                let dy = self.origin[1] + j as f32 * self.cell - p[1];
                let dxy = dx * dx + dy * dy;
                if dxy * inv_r2 >= 1.0 {
                    continue;
                }
                let base = i * stride_x + j * stride_y;
                for k in lo[2]..=hi[2] {
                    let dz = self.origin[2] + k as f32 * self.cell - p[2];
                    let q = (dxy + dz * dz) * inv_r2;
                    if q < 1.0 {
                        let w = (1.0 - q) * (1.0 - q);
                        self.density[base + k] += w;
                        self.foam[base + k] += w * foam;
                    }
                }
            }
        }
    }

    pub fn extract(&mut self, iso: f32) -> SurfaceMesh {
        let mut mesh = SurfaceMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            foam: Vec::new(),
            indices: Vec::new(),
        };
        if self.lo[0] >= self.hi[0] || self.lo[1] >= self.hi[1] || self.lo[2] >= self.hi[2] {
            return mesh;
        }
        let [lo, hi] = [self.lo, self.hi];
        let stride_y = self.dims[2] + 1;
        let stride_x = (self.dims[1] + 1) * stride_y;
        let corner = |i: usize, j: usize, k: usize| i * stride_x + j * stride_y + k;
        let cell_index = |i: usize, j: usize, k: usize| (i * self.dims[1] + j) * self.dims[2] + k;
        let offsets: [[usize; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let ci = cell_index(i, j, k);
                    let d = offsets.map(|o| self.density[corner(i + o[0], j + o[1], k + o[2])]);
                    let mut mask = 0u8;
                    for (b, v) in d.iter().enumerate() {
                        if *v > iso {
                            mask |= 1 << b;
                        }
                    }
                    if mask == 0 || mask == 0xff {
                        self.vertex_of_cell[ci] = NO_VERTEX;
                        continue;
                    }
                    let mut sum = [0.0f32; 3];
                    let mut crossings = 0.0f32;
                    for (a, b) in EDGES {
                        if ((mask >> a) & 1) == ((mask >> b) & 1) {
                            continue;
                        }
                        let t = (iso - d[a]) / (d[b] - d[a]);
                        for axis in 0..3 {
                            let oa = offsets[a][axis] as f32;
                            let ob = offsets[b][axis] as f32;
                            sum[axis] += oa + (ob - oa) * t;
                        }
                        crossings += 1.0;
                    }
                    let base = [i, j, k];
                    let position = [0, 1, 2].map(|axis| {
                        self.origin[axis] + (base[axis] as f32 + sum[axis] / crossings) * self.cell
                    });
                    let gradient = [
                        (d[1] - d[0]) + (d[3] - d[2]) + (d[5] - d[4]) + (d[7] - d[6]),
                        (d[2] - d[0]) + (d[3] - d[1]) + (d[6] - d[4]) + (d[7] - d[5]),
                        (d[4] - d[0]) + (d[5] - d[1]) + (d[6] - d[2]) + (d[7] - d[3]),
                    ];
                    let len = (gradient[0] * gradient[0]
                        + gradient[1] * gradient[1]
                        + gradient[2] * gradient[2])
                        .sqrt()
                        .max(1e-9);
                    let normal = gradient.map(|g| -g / len);
                    let (mut foam, mut weight) = (0.0f32, 0.0f32);
                    for o in offsets {
                        let c = corner(i + o[0], j + o[1], k + o[2]);
                        foam += self.foam[c];
                        weight += self.density[c];
                    }
                    self.vertex_of_cell[ci] = mesh.positions.len() as u32;
                    mesh.positions.push(position);
                    mesh.normals.push(normal);
                    mesh.foam
                        .push(if weight > 0.0 { foam / weight } else { 0.0 });
                }
            }
        }
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let inside = self.density[corner(i, j, k)] > iso;
                    let quad = |mesh: &mut SurfaceMesh, cells: [usize; 4]| {
                        let v = cells.map(|c| self.vertex_of_cell[c]);
                        if v.contains(&NO_VERTEX) {
                            return;
                        }
                        let [a, b, c, d] = if inside { v } else { [v[0], v[3], v[2], v[1]] };
                        mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
                    };
                    if i < hi[0]
                        && (self.density[corner(i + 1, j, k)] > iso) != inside
                        && j > lo[1]
                        && k > lo[2]
                    {
                        quad(
                            &mut mesh,
                            [
                                cell_index(i, j, k),
                                cell_index(i, j - 1, k),
                                cell_index(i, j - 1, k - 1),
                                cell_index(i, j, k - 1),
                            ],
                        );
                    }
                    if j < hi[1]
                        && (self.density[corner(i, j + 1, k)] > iso) != inside
                        && i > lo[0]
                        && k > lo[2]
                    {
                        quad(
                            &mut mesh,
                            [
                                cell_index(i, j, k),
                                cell_index(i, j, k - 1),
                                cell_index(i - 1, j, k - 1),
                                cell_index(i - 1, j, k),
                            ],
                        );
                    }
                    if k < hi[2]
                        && (self.density[corner(i, j, k + 1)] > iso) != inside
                        && i > lo[0]
                        && j > lo[1]
                    {
                        quad(
                            &mut mesh,
                            [
                                cell_index(i, j, k),
                                cell_index(i - 1, j, k),
                                cell_index(i - 1, j - 1, k),
                                cell_index(i, j - 1, k),
                            ],
                        );
                    }
                }
            }
        }
        mesh
    }
}
