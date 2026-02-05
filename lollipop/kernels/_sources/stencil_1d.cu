/*
 *  1D stencil with shared-memory halo cells.
 *
 *  Applies a symmetric averaging stencil of configurable radius:
 *      out[i] = (1 / (2*radius+1)) * sum(in[i-radius .. i+radius])
 *
 *  This is the simplest example of the *halo exchange* pattern:
 *
 *    1. Each block loads its interior elements into shared memory
 *    2. Threads at block edges also load halo cells — the extra
 *       elements needed from neighboring blocks
 *    3. After __syncthreads(), every thread can read its full stencil
 *       from shared memory (fast) instead of global memory (slow)
 *
 *  Shared memory layout:  [halo_left | block_interior | halo_right]
 *      total = blockDim.x + 2 * radius
 *
 *  Boundary: clamp to edge values (out-of-bounds reads use in[0] or in[n-1]).
 *
 *  Parameters:
 *      input  — n float32 values
 *      output — n float32 smoothed values
 *      n      — array length
 *      radius — stencil half-width
 *
 *  Launch: block=(256,), grid=((n+255)/256,)
 *          shared_mem = (256 + 2*radius) * sizeof(float)
 */
extern "C" __global__
void stencil_1d(const float* input, float* output, int n, int radius) {
    extern __shared__ float smem[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    /* Index into shared memory (offset by radius for left halo) */
    int s_idx = lid + radius;

    /* Load interior element */
    smem[s_idx] = (gid < n) ? input[gid] : 0.0f;

    /* Load left halo */
    if (lid < radius) {
        int left = gid - radius;
        smem[lid] = (left >= 0) ? input[left] : input[0];
    }

    /* Load right halo */
    if (lid >= blockDim.x - radius) {
        int right = gid + radius;
        smem[s_idx + radius] = (right < n) ? input[right] : input[n - 1];
    }

    __syncthreads();

    if (gid >= n) return;

    /* Compute stencil from shared memory */
    float sum = 0.0f;
    for (int offset = -radius; offset <= radius; offset++) {
        sum += smem[s_idx + offset];
    }
    output[gid] = sum / (float)(2 * radius + 1);
}
