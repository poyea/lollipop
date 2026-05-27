/*
 *  RMSNorm -- forward + backward, one block per row of an [N, H] tensor.
 *
 *  Forward (per row):
 *      rrms = rsqrt(mean(x_i^2) + eps)
 *      y_i  = x_i * rrms * gamma_i
 *
 *  Backward.  Let n_i = x_i * rrms and dn_i = dy_i * gamma_i.  Differentiating
 *  y_i through both the scale and the row-wide rrms gives the standard
 *  fused form:
 *      dot   = (1/H) * sum_j(dn_j * n_j)
 *      dx_i  = rrms * (dn_i - n_i * dot)
 *      dgamma_j += dy_i * n_i        (reduced across rows)
 *
 *  The backward kernel does two reductions per row -- one to recompute
 *  `rrms`, one for the `dn . n` dot -- then one elementwise pass that
 *  writes `dx` and atomic-adds `dgamma` (in fp32 for safe cross-row
 *  accumulation; the wrapper casts back to the input dtype).
 *
 *  Block:    (256,)
 *  Grid:     (N,)
 *  Smem:     256 * sizeof(float)  (reduction scratch)
 */

#include <cuda_fp16.h>

#define BLOCK 256

__device__ __forceinline__ float block_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK / 2; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = smem[tid];
        for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, d);
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

/* ---------------- forward ---------------- */

extern "C" __global__
void rmsnorm_fwd_fp32(const float* __restrict__ x,
                     const float* __restrict__ gamma,
                     float*       __restrict__ y,
                     int H, float eps) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + (size_t)row * H;
    float*       y_row = y + (size_t)row * H;

    float ss = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj = x_row[j];
        ss += xj * xj;
    }
    ss = block_sum(ss, smem);
    const float rrms = rsqrtf(ss / (float)H + eps);

    for (int j = tid; j < H; j += BLOCK) {
        y_row[j] = x_row[j] * rrms * gamma[j];
    }
}

extern "C" __global__
void rmsnorm_fwd_fp16(const __half* __restrict__ x,
                     const __half* __restrict__ gamma,
                     __half*       __restrict__ y,
                     int H, float eps) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* x_row = x + (size_t)row * H;
    __half*       y_row = y + (size_t)row * H;

    float ss = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj = __half2float(x_row[j]);
        ss += xj * xj;
    }
    ss = block_sum(ss, smem);
    const float rrms = rsqrtf(ss / (float)H + eps);

    for (int j = tid; j < H; j += BLOCK) {
        float xj = __half2float(x_row[j]);
        float gj = __half2float(gamma[j]);
        y_row[j] = __float2half(xj * rrms * gj);
    }
}

/* ---------------- backward ----------------
 *  dgamma is accumulated in fp32 to keep cross-row sums clean.  The Python
 *  wrapper casts back to the input dtype to match torch's API. */

extern "C" __global__
void rmsnorm_bwd_fp32(const float* __restrict__ dy,
                     const float* __restrict__ x,
                     const float* __restrict__ gamma,
                     float*       __restrict__ dx,
                     float*       __restrict__ dgamma,    /* fp32 accum, [H] */
                     int H, float eps) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row  = x  + (size_t)row * H;
    const float* dy_row = dy + (size_t)row * H;
    float*       dx_row = dx + (size_t)row * H;

    /* Pass 1: rrms. */
    float ss = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj = x_row[j];
        ss += xj * xj;
    }
    ss = block_sum(ss, smem);
    const float rrms = rsqrtf(ss / (float)H + eps);

    /* Pass 2: row-wide dot of dn_j * n_j = rrms * sum_j(dy_j * gamma_j * x_j). */
    float dot = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        dot += dy_row[j] * gamma[j] * x_row[j];
    }
    dot = block_sum(dot, smem);
    const float dot_mean = (dot * rrms) / (float)H;   /* (1/H) * sum(dn*n) */

    /* Pass 3: dx + dgamma atomic accumulate. */
    for (int j = tid; j < H; j += BLOCK) {
        float xj  = x_row[j];
        float dyj = dy_row[j];
        float gj  = gamma[j];
        float n_j = xj * rrms;
        float dn_j = dyj * gj;
        dx_row[j] = rrms * (dn_j - n_j * dot_mean);
        atomicAdd(&dgamma[j], dyj * n_j);
    }
}

extern "C" __global__
void rmsnorm_bwd_fp16(const __half* __restrict__ dy,
                     const __half* __restrict__ x,
                     const __half* __restrict__ gamma,
                     __half*       __restrict__ dx,
                     float*        __restrict__ dgamma,   /* fp32 accum, [H] */
                     int H, float eps) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* x_row  = x  + (size_t)row * H;
    const __half* dy_row = dy + (size_t)row * H;
    __half*       dx_row = dx + (size_t)row * H;

    float ss = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj = __half2float(x_row[j]);
        ss += xj * xj;
    }
    ss = block_sum(ss, smem);
    const float rrms = rsqrtf(ss / (float)H + eps);

    float dot = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj  = __half2float(x_row[j]);
        float dyj = __half2float(dy_row[j]);
        float gj  = __half2float(gamma[j]);
        dot += dyj * gj * xj;
    }
    dot = block_sum(dot, smem);
    const float dot_mean = (dot * rrms) / (float)H;

    for (int j = tid; j < H; j += BLOCK) {
        float xj  = __half2float(x_row[j]);
        float dyj = __half2float(dy_row[j]);
        float gj  = __half2float(gamma[j]);
        float n_j = xj * rrms;
        float dn_j = dyj * gj;
        dx_row[j] = __float2half(rrms * (dn_j - n_j * dot_mean));
        atomicAdd(&dgamma[j], dyj * n_j);
    }
}
