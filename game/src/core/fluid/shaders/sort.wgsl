// Exclusive prefix sum of the per-cell particle counts, in three passes: every workgroup scans
// its own run of cells and notes the run's total, one workgroup scans those totals, and every
// workgroup adds its run's offset. The last entry of the starts holds the grand total.
#import fluid_common::params

@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(13) var<storage, read_write> run_total: array<u32>;

const THREADS: u32 = 256u;
// the runs a single workgroup can scan in the second pass
const MAX_RUNS: u32 = 1024u;

var<workgroup> partial: array<u32, 256>;

/// Exclusive scan of `partial` in place; returns the total.
fn scan_partial(t: u32) -> u32 {
    for (var offset = 1u; offset < THREADS; offset <<= 1u) {
        var add = 0u;
        if (t >= offset) {
            add = partial[t - offset];
        }
        workgroupBarrier();
        partial[t] += add;
        workgroupBarrier();
    }
    let total = partial[THREADS - 1u];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(256)
fn scan_runs(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let t = local.x;
    let c = group.x * THREADS + t;
    var count = 0u;
    if (c < params.cells) {
        count = atomicLoad(&cell_count[c]);
    }
    partial[t] = count;
    workgroupBarrier();
    let total = scan_partial(t);
    if (c < params.cells) {
        cell_start[c] = partial[t] - count;
    }
    if (t == 0u) {
        run_total[group.x] = total;
    }
}

@compute @workgroup_size(256)
fn scan_totals(@builtin(local_invocation_id) local: vec3<u32>) {
    let t = local.x;
    let runs = (params.cells + THREADS - 1u) / THREADS;
    let per_thread = MAX_RUNS / THREADS;
    var sum = 0u;
    for (var k = 0u; k < per_thread; k++) {
        let r = t * per_thread + k;
        if (r < runs) {
            sum += run_total[r];
        }
    }
    partial[t] = sum;
    workgroupBarrier();
    let total = scan_partial(t);
    var start = partial[t] - sum;
    for (var k = 0u; k < per_thread; k++) {
        let r = t * per_thread + k;
        if (r < runs) {
            let run = run_total[r];
            run_total[r] = start;
            start += run;
        }
    }
    if (t == 0u) {
        cell_start[params.cells] = total;
    }
}

@compute @workgroup_size(256)
fn add_offsets(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let c = group.x * THREADS + local.x;
    if (c < params.cells) {
        cell_start[c] += run_total[group.x];
    }
}
