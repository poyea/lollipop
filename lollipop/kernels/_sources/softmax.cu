/*
 *  Fused online (streaming) softmax over the last axis of a 2D tensor.
 *
 *  One block per row.  Each thread walks its row in a strided loop and
 *  maintains a local running (m, d) pair where:
 *      m = max over elements seen so far
 *      d = sum of exp(x - m) over elements seen so far
 *
 *  The online update rule (Milakov & Gimelshein 2018) combines a new
 *  element x into (m, d) without a separate max pass:
 *      m_new = max(m, x)
 *      d_new = d * exp(m - m_new) + exp(x - m_new)
 *
 *  After the per-thread streaming pass, the block reduces the 256 local
 *  (m, d) pairs into a single row-wide (M, D) using the associative
 *  merge:
 *      M = max(m_a, m_b)
 *      D = d_a * exp(m_a - M) + d_b * exp(m_b - M)
 *
 *  A second pass writes y[j] = exp(x[j] - M) / D.  No intermediate
 *  logits tensor is materialized and the kernel is numerically stable
 *  for arbitrary input magnitudes.
 *
 *  This is the same primitive that sits at the core of FlashAttention's
 *  tile-wise softmax.
 *
 *  Parameters:
 *      x — input  [rows, cols], float32, row-major
 *      y — output [rows, cols], float32, row-major
 *      cols — row length
 *
 *  Launch: block=(256,), grid=(rows,), shared_mem = 2 * 256 * sizeof(float)
 */

extern "C" __global__
void softmax(const float* __restrict__ x, float* __restrict__ y, int cols) {
    extern __shared__ float smem[];
    float* s_m = smem;                 /* [blockDim.x] running maxes */
    float* s_d = smem + blockDim.x;    /* [blockDim.x] running denoms */

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    /* Streaming pass: each thread folds its strided slice into (m, d). */
    float m = -3.4e38f;  /* sentinel below any finite float32 input */
    float d = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float xj = x_row[j];
        float m_new = fmaxf(m, xj);
        d = d * __expf(m - m_new) + __expf(xj - m_new);
        m = m_new;
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

    /* Normalization pass. */
    for (int j = tid; j < cols; j += blockDim.x) {
        y_row[j] = __expf(x_row[j] - M) * inv_D;
    }
}
