/*
 *  Fused online softmax, float4-vectorized variant of `softmax`.
 *
 *  What changed vs. the scalar baseline (softmax.cu):
 *  --------------------------------------------------
 *  Both passes now move data in 128-bit chunks:
 *
 *    1. Streaming pass: each loop iteration issues one LDG.E.128 (a
 *       `float4`) instead of four LDG.E.32 scalar loads.  After the
 *       load, the four lanes are folded into the running (m, d) pair
 *       one at a time — the online-softmax recurrence is not
 *       associative-friendly enough to do four lanes "in parallel" any
 *       cheaper than serially, but the loads are now 4x fewer and the
 *       warp moves 32 * 16 = 512 B per iteration in one coalesced
 *       transaction.
 *
 *    2. Normalization pass: load one `float4`, compute
 *       `__expf(xj - M) * inv_D` for each lane, store one `float4`
 *       (STG.E.128).  Halves the number of issued L/S instructions
 *       relative to the scalar normalization loop.
 *
 *  Everything between the two passes (smem tree reduction of (m, d)
 *  pairs, warp-shuffle final merge) is byte-for-byte identical to the
 *  scalar kernel: the row reduction itself is scalar-bandwidth-trivial
 *  and there is nothing to gain by vectorizing it.
 *
 *  Block stride is `blockDim.x` float4 per iteration = 4 * blockDim.x
 *  floats (1024 with blockDim.x = 256), vs 256 floats in the scalar
 *  version.  The caller must guarantee `cols_vec4 = cols / 4` (cols
 *  divisible by 4) and that each row starts on a 16-byte boundary
 *  (which holds automatically for contiguous CuPy allocations whose
 *  row pitch in bytes is a multiple of 16).
 *
 *  Parameters:
 *      x         — input  [rows, cols], float32, row-major
 *      y         — output [rows, cols], float32, row-major
 *      cols_vec4 — row length in float4 units (= cols / 4)
 *
 *  Launch: block=(256,), grid=(rows,), shared_mem = 2 * 256 * sizeof(float)
 */

extern "C" __global__
void softmax_vec4(const float4* __restrict__ x, float4* __restrict__ y, int cols_vec4) {
    extern __shared__ float smem[];
    float* s_m = smem;
    float* s_d = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float4* x_row = x + row * cols_vec4;
    float4* y_row = y + row * cols_vec4;

    /* Streaming pass: one float4 per loop iteration; fold 4 lanes into (m, d). */
    float m = -3.4e38f;
    float d = 0.0f;
    for (int j = tid; j < cols_vec4; j += blockDim.x) {
        float4 v = x_row[j];
        float xs[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float m_new = fmaxf(m, xs[k]);
            d = d * __expf(m - m_new) + __expf(xs[k] - m_new);
            m = m_new;
        }
    }
    s_m[tid] = m;
    s_d[tid] = d;
    __syncthreads();

    /* Tree reduction of (m, d) pairs down to 32 threads. */
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            float ma = s_m[tid],       da = s_d[tid];
            float mb = s_m[tid + s],   db = s_d[tid + s];
            float mm = fmaxf(ma, mb);
            s_m[tid] = mm;
            s_d[tid] = da * __expf(ma - mm) + db * __expf(mb - mm);
        }
        __syncthreads();
    }

    /* Final warp-level merge using shuffles. */
    if (tid < 32) {
        float wm = s_m[tid];
        float wd = s_d[tid];
        for (int delta = 16; delta > 0; delta >>= 1) {
            float om = __shfl_down_sync(0xFFFFFFFF, wm, delta);
            float od = __shfl_down_sync(0xFFFFFFFF, wd, delta);
            float mm = fmaxf(wm, om);
            wd = wd * __expf(wm - mm) + od * __expf(om - mm);
            wm = mm;
        }
        if (tid == 0) {
            s_m[0] = wm;
            s_d[0] = wd;
        }
    }
    __syncthreads();

    float M = s_m[0];
    float inv_D = 1.0f / s_d[0];

    /* Normalization pass: one float4 in, one float4 out per iteration. */
    for (int j = tid; j < cols_vec4; j += blockDim.x) {
        float4 v = x_row[j];
        float4 r;
        r.x = __expf(v.x - M) * inv_D;
        r.y = __expf(v.y - M) * inv_D;
        r.z = __expf(v.z - M) * inv_D;
        r.w = __expf(v.w - M) * inv_D;
        y_row[j] = r;
    }
}
