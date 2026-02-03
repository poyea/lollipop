/*
 *  Parallel sum reduction with warp shuffle.
 *
 *  Three-stage reduction:
 *    1. Each thread loads two elements and adds them (halves the work)
 *    2. Tree reduction in shared memory down to 32 threads
 *    3. Final warp uses __shfl_down_sync — no shared memory or
 *       __syncthreads needed within a warp (lockstep execution)
 *
 *  The final result is added to output[0] via atomicAdd, so multiple
 *  blocks can cooperate on large arrays.
 *
 *  Parameters:
 *      input  — n float32 values to sum
 *      output — single float32 accumulator (caller zeroes it)
 *      n      — number of input elements
 *
 *  Launch: block=(256,), grid=((n+511)/512,), shared_mem = 256 * sizeof(float)
 */
extern "C" __global__
void reduction(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* Load two elements per thread */
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    /* Shared memory tree reduction */
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Warp-level reduction using shuffle intrinsics (final 32 threads) */
    if (tid < 32) {
        float warp_val = sdata[tid];
        warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, 16);
        warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, 8);
        warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, 4);
        warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, 2);
        warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, 1);

        if (tid == 0) {
            atomicAdd(output, warp_val);
        }
    }
}
