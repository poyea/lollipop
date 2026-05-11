/*
 *  Sum reduction, float4-vectorized variant of `reduction`.
 *
 *  What changed vs. the scalar baseline (reduction.cu):
 *  ----------------------------------------------------
 *  Each thread now reads two `float4` values (8 floats) instead of two
 *  scalars (2 floats).  A single `float4` load compiles to one LDG.E.128
 *  (128-bit global load) instead of four LDG.E.32, which:
 *
 *    - issues 4x fewer load instructions (less LSU pipe pressure);
 *    - reduces address-generation work per byte moved;
 *    - keeps memory transactions naturally 16-byte aligned and coalesced
 *      across a warp (warp moves 32 * 16 = 512 B in one go).
 *
 *  Everything after the load is identical: per-thread partial sum,
 *  shared-mem tree reduction, warp-shuffle finish, atomicAdd into the
 *  global accumulator.
 *
 *  Block stride is now 2 * blockDim.x float4 = 8 * blockDim.x floats
 *  (2048 for blockDim.x = 256), vs 512 in the scalar version.  The caller
 *  must pass an array length that is a multiple of 4 (the Python wrapper
 *  handles the tail separately on the host side).
 *
 *  Parameters:
 *      input   — n_vec4 float4 values to sum  (= 4 * n_vec4 floats)
 *      output  — single float32 accumulator (caller zeroes it)
 *      n_vec4  — number of float4 elements
 *
 *  Launch: block=(256,), grid=((n_vec4 + 511) / 512,),
 *          shared_mem = 256 * sizeof(float)
 */
extern "C" __global__
void reduction_vec4(const float4* __restrict__ input, float* output, int n_vec4) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* Two float4 loads per thread -> 8 floats summed into a scalar partial. */
    float val = 0.0f;
    if (i < n_vec4) {
        float4 a = input[i];
        val += a.x + a.y + a.z + a.w;
    }
    if (i + blockDim.x < n_vec4) {
        float4 b = input[i + blockDim.x];
        val += b.x + b.y + b.z + b.w;
    }
    sdata[tid] = val;
    __syncthreads();

    /* Shared memory tree reduction down to one warp. */
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Warp shuffle finish. */
    if (tid < 32) {
        float wv = sdata[tid];
        wv += __shfl_down_sync(0xFFFFFFFF, wv, 16);
        wv += __shfl_down_sync(0xFFFFFFFF, wv, 8);
        wv += __shfl_down_sync(0xFFFFFFFF, wv, 4);
        wv += __shfl_down_sync(0xFFFFFFFF, wv, 2);
        wv += __shfl_down_sync(0xFFFFFFFF, wv, 1);
        if (tid == 0) {
            atomicAdd(output, wv);
        }
    }
}
