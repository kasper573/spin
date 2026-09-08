// Exclusive prefix sum of the per-cell particle counts, in one workgroup: every thread sums a
// contiguous run of cells, the run totals are scanned in shared memory, then each thread writes
// the starts of its run. The last entry holds the total.
#import fluid_common::params

@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cell_start: array<u32>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) local: vec3<u32>) {
    let t = local.x;
    let cells = u32(params.grid_dims.w);
    let run = (cells + THREADS - 1u) / THREADS;
    let first = t * run;
    let last = min(first + run, cells);
    var total = 0u;
    for (var c = first; c < last; c++) {
        total += atomicLoad(&cell_count[c]);
    }
    partial[t] = total;
    workgroupBarrier();
    for (var offset = 1u; offset < THREADS; offset <<= 1u) {
        var add = 0u;
        if (t >= offset) {
            add = partial[t - offset];
        }
        workgroupBarrier();
        partial[t] += add;
        workgroupBarrier();
    }
    var start = partial[t] - total;
    for (var c = first; c < last; c++) {
        cell_start[c] = start;
        start += atomicLoad(&cell_count[c]);
    }
    if (t == THREADS - 1u) {
        cell_start[cells] = partial[t];
    }
}
