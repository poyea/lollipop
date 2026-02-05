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
 *  2. radix_prefix_sum — Exclusive prefix sum over the histogram table
 *     (flattened, column-major: all blocks' bucket-0 counts, then all
 *     blocks' bucket-1 counts, ...).  After this, each entry tells the
 *     global output offset for that block's bucket.
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
 *      radix_prefix_sum:  block=(512,), grid=(1,),
 *                         shared_mem = total_histogram_entries * sizeof(uint)
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

/* ---- Phase 2: exclusive prefix sum over histogram table ---------- */

extern "C" __global__
void radix_prefix_sum(unsigned int* histograms, int total) {
    extern __shared__ unsigned int temp[];

    int tid = threadIdx.x;

    /* Load into shared (two elements per thread, Blelloch style) */
    temp[2 * tid]     = (2 * tid < total)     ? histograms[2 * tid]     : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < total) ? histograms[2 * tid + 1] : 0;

    /* Up-sweep */
    int offset = 1;
    for (int d = total >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) temp[total - 1] = 0;

    /* Down-sweep */
    for (int d = 1; d < total; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    /* Write back */
    if (2 * tid < total)     histograms[2 * tid]     = temp[2 * tid];
    if (2 * tid + 1 < total) histograms[2 * tid + 1] = temp[2 * tid + 1];
}

/* ---- Phase 3: scatter keys to sorted positions ------------------- */

extern "C" __global__
void radix_scatter(const unsigned int* keys_in,
                   unsigned int* keys_out,
                   const unsigned int* histograms,
                   int n, int shift) {
    __shared__ unsigned int local_offset[NUM_BUCKETS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int num_blocks = gridDim.x;

    /* Load this block's starting offsets from the scanned histogram */
    if (tid < NUM_BUCKETS) {
        local_offset[tid] = histograms[tid * num_blocks + blockIdx.x];
    }
    __syncthreads();

    /* Scatter each key to its global position */
    if (gid < n) {
        unsigned int key = keys_in[gid];
        unsigned int bucket = (key >> shift) & (NUM_BUCKETS - 1);
        unsigned int pos = atomicAdd(&local_offset[bucket], 1);
        keys_out[pos] = key;
    }
}
