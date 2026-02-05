/*
 *  Warp-level inclusive prefix sum using __shfl_up_sync only.
 *
 *  No shared memory, no __syncthreads() — entirely warp intrinsics.
 *
 *  Within a warp (32 threads), __shfl_up_sync lets thread i read
 *  the register value of thread (i - delta).  By doubling delta
 *  each step (1, 2, 4, 8, 16), we compute the inclusive scan in
 *  log2(32) = 5 steps:
 *
 *      step 1:  val += shfl_up(val, 1)   — each thread adds its left neighbor
 *      step 2:  val += shfl_up(val, 2)   — adds the pair two to the left
 *      step 4:  val += shfl_up(val, 4)   — adds the quad four to the left
 *      step 8:  val += shfl_up(val, 8)
 *      step 16: val += shfl_up(val, 16)
 *
 *  After 5 steps, thread i holds sum(input[0..i]) — an inclusive scan.
 *
 *  This kernel handles arrays up to 32 elements (one warp).
 *  For larger arrays, combine with a block-level or multi-block scan.
 *
 *  Contrast with prefix_sum_blelloch which uses shared memory.
 *
 *  Parameters:
 *      data — n float32 values (overwritten with inclusive prefix sums)
 *      n    — array length (max 32)
 *
 *  Launch: block=(32,), grid=(1,)
 */
extern "C" __global__
void warp_scan(float* data, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;

    float val = data[tid];

    /* 5-step inclusive scan across the warp */
    for (int delta = 1; delta < 32; delta <<= 1) {
        float received = __shfl_up_sync(0xFFFFFFFF, val, delta);
        if (tid >= delta) {
            val += received;
        }
    }

    data[tid] = val;
}
