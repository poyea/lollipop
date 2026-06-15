/*
 *  Parallel sum reduction with Cooperative Groups.
 *
 *  Same job as reduction.cu (sum a 1D float array), rewritten on the
 *  Cooperative Groups API instead of raw `__shfl_down_sync` + a manual smem
 *  tree.  This is the canonical, readable form of "warp-level primitives":
 *
 *    1. Grid-stride load — each thread folds its slice of the input into one
 *       register accumulator, so the launch geometry is decoupled from n.
 *    2. cg::reduce over a tiled_partition<32> — one library call replaces the
 *       five hand-rolled shuffle steps; on sm_80+ it lowers to the `redux.sync`
 *       hardware instruction, on Turing to the same shuffle tree by hand.
 *    3. One atomicAdd per warp into output[0].
 *
 *  No shared memory and no __syncthreads: the warp tile handles the
 *  intra-warp reduction, and warps cooperate purely through the final atomic.
 *  Numerically identical to reduction.cu up to fp32 summation order.
 *
 *  Two kernels here:
 *
 *    reduction_cg       — warp tile + one atomicAdd per warp. Any grid size;
 *                         the cross-block combine is the atomic.
 *    reduction_cg_grid  — single-launch, atomic-free. A grid-wide
 *                         `cg::this_grid().sync()` barrier lets block 0 fold
 *                         the per-block partials after every block has written
 *                         its own. Requires a *cooperative* launch (all blocks
 *                         co-resident), so the grid is capped at one block per
 *                         SM by the wrapper.
 *
 *  Parameters:
 *      input    — n float32 values to sum
 *      output   — single float32 accumulator
 *      partials — (grid kernel only) gridDim.x scratch slots for block sums
 *      n        — number of input elements
 *
 *  Launch (reduction_cg):      block=(256,), grid=(min(1024,ceil(n/256)),).
 *  Launch (reduction_cg_grid): block=(256,), grid=(#SMs,), cooperative.
 */
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

extern "C" __global__
void reduction_cg(const float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    /* Grid-stride fold into a per-thread accumulator. */
    float val = 0.0f;
    for (int i = gid; i < n; i += stride) val += input[i];

    /* Warp-wide sum: one cg::reduce replaces the manual shuffle ladder. */
    val = cg::reduce(warp, val, cg::plus<float>());

    if (warp.thread_rank() == 0) atomicAdd(output, val);
}

/* Block-wide sum of one per-thread value: warp-reduce, stash warp leaders in
 * smem, then have warp 0 reduce those. Returns the total on thread 0. */
__device__ static float block_sum(float val, float* ws,
                                   cg::thread_block_tile<32>& warp) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) / 32;

    val = cg::reduce(warp, val, cg::plus<float>());
    if (lane == 0) ws[wid] = val;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < nwarps) ? ws[lane] : 0.0f;
        v = cg::reduce(warp, v, cg::plus<float>());
        if (lane == 0) ws[0] = v;
    }
    __syncthreads();
    return ws[0];
}

extern "C" __global__
void reduction_cg_grid(const float* input, float* partials, float* output,
                       int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float ws[32];

    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    /* Pass 1: each block folds its grid-stride slice down to one partial. */
    float val = 0.0f;
    for (int i = gid; i < n; i += stride) val += input[i];
    float bsum = block_sum(val, ws, warp);
    if (threadIdx.x == 0) partials[blockIdx.x] = bsum;

    /* Grid-wide barrier: every block's partial is now visible. */
    grid.sync();

    /* Pass 2: block 0 reduces the gridDim.x partials. No atomics anywhere. */
    if (blockIdx.x == 0) {
        float v = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            v += partials[i];
        float total = block_sum(v, ws, warp);
        if (threadIdx.x == 0) output[0] = total;
    }
}
