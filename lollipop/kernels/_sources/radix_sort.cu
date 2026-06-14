/*
 *  Radix sort — multi-pass, multi-kernel GPU sorting.
 *
 *  Sorts unsigned 32-bit integers by processing BITS_PER_PASS bits at a
 *  time (default 4 bits = 16 buckets per pass, 8 passes for 32 bits).
 *
 *  Each pass has three phases, each a separate kernel launch:
 *
 *  1. radix_histogram — Count how many keys fall into each bucket.
 *     Each block processes a tile of keys and builds a local histogram
 *     in shared memory, then writes it to a per-block column in the
 *     global histogram table  [num_blocks x num_buckets].
 *
 *  2. (device-wide scan) — Exclusive prefix sum over the histogram table
 *     (flattened, column-major: all blocks' bucket-0 counts, then all
 *     blocks' bucket-1 counts, ...).  After this, each entry tells the
 *     global output offset for that block's bucket.  This phase is the
 *     hierarchical scan in prefix_sum.cu (scan_block_u32), driven from the
 *     Python wrapper — it replaced an in-file single-block Blelloch that
 *     capped n at 32768.
 *
 *  3. radix_scatter — Each block re-reads its tile, recomputes each
 *     key's bucket, looks up the global offset from the scanned
 *     histogram, and writes the key to its sorted position.  A local
 *     counter per bucket increments after each write.
 *
 *  After PASSES passes the keys are fully sorted.
 *
 *  This is a real multi-kernel pipeline: the host launches 3 kernels
 *  per pass, 8 passes = 24 kernel launches total.  Intermediate data
 *  flows through the histogram buffer on device memory.
 *
 *  Limitations (for clarity, not production):
 *      — n must be a multiple of BLOCK_SIZE (the wrapper pads if needed)
 *      — BLOCK_SIZE * num_buckets shared memory per block
 *
 *  Parameters (all three kernels share the same signature pattern):
 *      keys_in / keys_out — input / output uint32 arrays (n elements)
 *      histograms         — [num_blocks * num_buckets] uint32 work buffer
 *      n                  — number of keys
 *      shift              — bit position for this pass (0, 4, 8, ... 28)
 *
 *  Launch (each kernel): block=(256,), grid=((n+255)/256,)
 *      phase 2 scan:      see prefix_sum.cu / prefix_sum._scan_exclusive
 */

#define BITS_PER_PASS 4
#define NUM_BUCKETS   (1 << BITS_PER_PASS)   /* 16 */

/* ---- Phase 1: per-block histogram -------------------------------- */

extern "C" __global__
void radix_histogram(const unsigned int* keys_in,
                     unsigned int* histograms,
                     int n, int shift) {
    __shared__ unsigned int local_hist[NUM_BUCKETS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    /* Zero shared histogram */
    if (tid < NUM_BUCKETS) local_hist[tid] = 0;
    __syncthreads();

    /* Each thread contributes one key */
    if (gid < n) {
        unsigned int bucket = (keys_in[gid] >> shift) & (NUM_BUCKETS - 1);
        atomicAdd(&local_hist[bucket], 1);
    }
    __syncthreads();

    /* Write local histogram to global table (column-major layout):
       histograms[bucket * num_blocks + blockIdx.x] */
    int num_blocks = gridDim.x;
    if (tid < NUM_BUCKETS) {
        histograms[tid * num_blocks + blockIdx.x] = local_hist[tid];
    }
}

/* ---- Phase 2 lives in prefix_sum.cu (device-wide hierarchical scan) ---- */

/* ---- Phase 3: stable scatter keys to sorted positions ------------ */
/*
 *  Stability requirement: among keys with the same current digit, the one
 *  at a lower input index must go to a lower output index.  A bare
 *  atomicAdd on shared memory fails this because warp scheduling is
 *  non-deterministic, so threads race for slots in arbitrary order.
 *
 *  Fix: use __ballot_sync / __popc to assign positions in lane order
 *  (deterministic, low-lane → low position) within each warp, then
 *  combine across warps with a sequential prefix sum over warp counts.
 *
 *  Stable ordering guaranteed at three levels:
 *    1. across blocks  — from the pre-scanned histogram (block 0 < block 1 ...)
 *    2. across warps   — warp_prefix exclusive scan (warp 0 < warp 1 ...)
 *    3. within a warp  — __popc of lower-lane mask (lane 0 < lane 1 ...)
 */

#define NUM_WARPS (256 / 32)   /* must match BLOCK_SIZE in the Python wrapper */

extern "C" __global__
void radix_scatter(const unsigned int* keys_in,
                   unsigned int* keys_out,
                   const unsigned int* histograms,
                   int n, int shift) {
    __shared__ unsigned int block_offset[NUM_BUCKETS];
    __shared__ unsigned int warp_prefix[NUM_WARPS][NUM_BUCKETS];

    int tid       = threadIdx.x;
    int gid       = blockIdx.x * blockDim.x + tid;
    int num_blocks = gridDim.x;
    int warp_id   = tid >> 5;
    int lane      = tid & 31;

    /* Load this block's global starting offset for each bucket */
    if (tid < NUM_BUCKETS)
        block_offset[tid] = histograms[tid * num_blocks + blockIdx.x];

    /* Out-of-bounds threads use a sentinel bucket that is never written */
    int valid           = (gid < n);
    unsigned int key    = valid ? keys_in[gid] : 0u;
    unsigned int bucket = valid ? ((key >> shift) & (NUM_BUCKETS - 1))
                                : NUM_BUCKETS;   /* sentinel */

    /* For each bucket, ballot which lanes belong to it.
       popc of the mask for lanes below mine gives my intra-warp rank. */
    unsigned int intra_rank = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        unsigned int mask = __ballot_sync(0xFFFFFFFFu, bucket == (unsigned int)b);
        if (bucket == (unsigned int)b)
            intra_rank = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0)
            warp_prefix[warp_id][b] = __popc(mask);
    }
    __syncthreads();

    /* Convert per-warp counts to exclusive prefix sums across warps.
       Only NUM_BUCKETS threads needed; they all fit in a single warp. */
    if (tid < NUM_BUCKETS) {
        unsigned int sum = 0;
        for (int w = 0; w < NUM_WARPS; w++) {
            unsigned int cnt      = warp_prefix[w][tid];
            warp_prefix[w][tid]   = sum;
            sum                  += cnt;
        }
    }
    __syncthreads();

    if (valid) {
        unsigned int pos = block_offset[bucket]
                         + warp_prefix[warp_id][bucket]
                         + intra_rank;
        keys_out[pos] = key;
    }
}
