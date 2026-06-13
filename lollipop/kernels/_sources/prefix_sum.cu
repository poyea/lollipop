/*
 *  Device-wide exclusive prefix sum (scan) — hierarchical Blelloch.
 *
 *  The original single-block Blelloch scan topped out at n<=2048 (one block,
 *  n/2<=1024 threads).  This is the multi-block rebuild: a per-tile exclusive
 *  scan whose tile totals are themselves scanned (recursively, on the host
 *  side) and added back, so n is unbounded.
 *
 *  Three pieces, two dtype variants each (extern "C" can't template, so the
 *  DEFINE_* macros stamp out `_f32` and `_u32`):
 *
 *    1. scan_block_<T> — each block does a conflict-free exclusive Blelloch
 *       over its ELEMENTS_PER_BLOCK-wide tile, writes the scan to `out`, and
 *       writes the tile's total to `block_sums[blockIdx.x]`.  Out-of-range
 *       slots load 0, so n need not be a multiple of the tile (or a power of
 *       two — the per-tile Blelloch is always a fixed power-of-two width).
 *    2. (host recursion) — exclusive-scan `block_sums` into `block_offsets`
 *       by calling scan_block_<T> again; one level reaches ~4M elements, deeper
 *       n just recurses further.
 *    3. add_block_offsets_<T> — adds block_offsets[blockIdx.x] back onto every
 *       element of tile `blockIdx.x`, lifting the per-tile scans into one
 *       global exclusive scan.
 *
 *  Shared-memory bank conflicts: a naive Blelloch hits up to 32-way conflicts
 *  on the stride-2^d accesses.  CONFLICT_FREE_OFFSET spreads each logical index
 *  by index>>5 padding slots so consecutive powers-of-two land in distinct
 *  banks (GPU Gems 3, ch. 39).  Hence the +NUM_BANKS slack on the smem alloc.
 *
 *  Launch (per scan_block / add_block_offsets call):
 *      block = (ELEMENTS_PER_BLOCK/2,)            = (1024,)
 *      grid  = (ceil(n / ELEMENTS_PER_BLOCK),)
 *      shared_mem = (ELEMENTS_PER_BLOCK + ELEMENTS_PER_BLOCK/NUM_BANKS) * sizeof(T)
 */

#define NUM_BANKS     32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

#define ELEMENTS_PER_BLOCK 2048
#define SCAN_THREADS       (ELEMENTS_PER_BLOCK / 2)   /* 1024 */

/* ---- per-tile exclusive Blelloch + tile total ----------------------------- */

#define DEFINE_SCAN_BLOCK(SUFFIX, T)                                           \
extern "C" __global__                                                          \
void scan_block_##SUFFIX(const T* in, T* out, T* block_sums, int n) {          \
    extern __shared__ T temp_##SUFFIX[];                                       \
    T* temp = temp_##SUFFIX;                                                   \
                                                                              \
    int tid = threadIdx.x;                                                     \
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;                                \
                                                                              \
    int ai = tid;                                                             \
    int bi = tid + (ELEMENTS_PER_BLOCK / 2);                                   \
    int offA = CONFLICT_FREE_OFFSET(ai);                                       \
    int offB = CONFLICT_FREE_OFFSET(bi);                                       \
    int gai = base + ai;                                                       \
    int gbi = base + bi;                                                       \
                                                                              \
    temp[ai + offA] = (gai < n) ? in[gai] : (T)0;                              \
    temp[bi + offB] = (gbi < n) ? in[gbi] : (T)0;                              \
                                                                              \
    int offset = 1;                                                           \
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {                    \
        __syncthreads();                                                       \
        if (tid < d) {                                                         \
            int a = offset * (2 * tid + 1) - 1;                                \
            int b = offset * (2 * tid + 2) - 1;                                \
            a += CONFLICT_FREE_OFFSET(a);                                      \
            b += CONFLICT_FREE_OFFSET(b);                                      \
            temp[b] += temp[a];                                               \
        }                                                                     \
        offset *= 2;                                                          \
    }                                                                         \
                                                                              \
    if (tid == 0) {                                                            \
        int last = ELEMENTS_PER_BLOCK - 1;                                     \
        last += CONFLICT_FREE_OFFSET(last);                                    \
        if (block_sums != nullptr) block_sums[blockIdx.x] = temp[last];        \
        temp[last] = (T)0;                                                     \
    }                                                                         \
                                                                              \
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {                          \
        offset >>= 1;                                                          \
        __syncthreads();                                                       \
        if (tid < d) {                                                         \
            int a = offset * (2 * tid + 1) - 1;                                \
            int b = offset * (2 * tid + 2) - 1;                                \
            a += CONFLICT_FREE_OFFSET(a);                                      \
            b += CONFLICT_FREE_OFFSET(b);                                      \
            T t = temp[a];                                                     \
            temp[a] = temp[b];                                                 \
            temp[b] += t;                                                      \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    if (gai < n) out[gai] = temp[ai + offA];                                   \
    if (gbi < n) out[gbi] = temp[bi + offB];                                   \
}

/* ---- add scanned tile offsets back ---------------------------------------- */

#define DEFINE_ADD_OFFSETS(SUFFIX, T)                                         \
extern "C" __global__                                                         \
void add_block_offsets_##SUFFIX(T* out, const T* block_offsets, int n) {      \
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;                               \
    T add = block_offsets[blockIdx.x];                                        \
    int ai = base + threadIdx.x;                                             \
    int bi = base + threadIdx.x + (ELEMENTS_PER_BLOCK / 2);                   \
    if (ai < n) out[ai] += add;                                              \
    if (bi < n) out[bi] += add;                                              \
}

DEFINE_SCAN_BLOCK(f32, float)
DEFINE_SCAN_BLOCK(u32, unsigned int)
DEFINE_ADD_OFFSETS(f32, float)
DEFINE_ADD_OFFSETS(u32, unsigned int)
