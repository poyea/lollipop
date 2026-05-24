/*
 *  Fused FFN tail: RMSNorm -> scale by gamma -> + bias -> activation
 *                  -> + residual (optional).
 *
 *  One block per row of an `[M, H]` tensor.  All intermediates stay in
 *  registers; H is walked twice (one reduction pass for the inverse RMS,
 *  one elementwise pass for the rest), so global traffic is the
 *  irreducible minimum: read X, read gamma, optionally read bias and
 *  residual, write Y.
 *
 *  Math:
 *      rrms = rsqrt(mean(x_i^2) + eps)
 *      n_i  = x_i * rrms * gamma_i
 *      a_i  = n_i + bias_i                     (skipped if bias == nullptr)
 *      g_i  = act(a_i)                         (GELU-tanh or SiLU)
 *      y_i  = g_i + residual_i                 (skipped if residual == nullptr)
 *
 *  Activations:
 *      ACT_GELU_TANH = 0   gelu(a) = 0.5 * a * (1 + tanh(k * (a + 0.044715 * a^3)))
 *                                    with k = sqrt(2/pi).  Matches torch
 *                                    F.gelu(approximate='tanh').
 *      ACT_SILU      = 1   silu(a) = a * sigmoid(a) = a / (1 + exp(-a)).
 *                                    Llama-shaped FFN activation.
 *
 *  Two entry points: fp32 and fp16.  The fp16 variant does all compute in
 *  fp32 (the cost of one row's worth of converts is dwarfed by the
 *  reduction itself) and converts on store.
 *
 *  Block:    (256,)
 *  Grid:     (M,)
 *  Smem:     256 * sizeof(float)  (reduction scratch)
 *  Pre-cond: H > 0  (any H; vectorisation is left for v1.1)
 */

#include <cuda_fp16.h>

#define BLOCK 256
#define ACT_GELU_TANH 0
#define ACT_SILU      1
#define GELU_K 0.7978845608028654f      /* sqrt(2/pi) */
#define GELU_C 0.044715f

__device__ __forceinline__ float activation(float a, int act) {
    if (act == ACT_SILU) {
        return a / (1.0f + __expf(-a));
    }
    /* ACT_GELU_TANH (default) */
    float a3 = a * a * a;
    float t  = tanhf(GELU_K * (a + GELU_C * a3));
    return 0.5f * a * (1.0f + t);
}

/* Block-wide sum reduction. `val` is each thread's partial; result is broadcast
 * to every thread via `smem[0]`. `smem` must hold BLOCK floats. */
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

extern "C" __global__
void fused_ffn_tail_fp32(const float* __restrict__ x,
                         const float* __restrict__ gamma,
                         const float* __restrict__ bias,      /* may be nullptr */
                         const float* __restrict__ residual,  /* may be nullptr */
                         float*       __restrict__ y,
                         int H, float eps, int act) {
    __shared__ float smem[BLOCK];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + row * H;
    float*       y_row = y + row * H;
    const float* r_row = residual ? residual + row * H : nullptr;

    /* Pass 1: sum of squares. */
    float ss = 0.0f;
    for (int j = tid; j < H; j += BLOCK) {
        float xj = x_row[j];
        ss += xj * xj;
    }
    ss = block_sum(ss, smem);
    const float rrms = rsqrtf(ss / (float)H + eps);

    /* Pass 2: normalise, scale, bias, activation, residual. */
    for (int j = tid; j < H; j += BLOCK) {
        float n = x_row[j] * rrms * gamma[j];
        float a = bias ? n + bias[j] : n;
        float g = activation(a, act);
        y_row[j] = r_row ? g + r_row[j] : g;
    }
}

extern "C" __global__
void fused_ffn_tail_fp16(const __half* __restrict__ x,
                         const __half* __restrict__ gamma,
                         const __half* __restrict__ bias,      /* may be nullptr */
                         const __half* __restrict__ residual,  /* may be nullptr */
                         __half*       __restrict__ y,
                         int H, float eps, int act) {
    __shared__ float smem[BLOCK];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* x_row = x + row * H;
    __half*       y_row = y + row * H;
    const __half* r_row = residual ? residual + row * H : nullptr;

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
        float n = xj * rrms * gj;
        float a = bias ? n + __half2float(bias[j]) : n;
        float g = activation(a, act);
        if (r_row) g += __half2float(r_row[j]);
        y_row[j] = __float2half(g);
    }
}
