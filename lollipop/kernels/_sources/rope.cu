/*
 *  Rotary Positional Embedding (RoPE) -- forward, Llama half-rotation convention.
 *
 *  Reference: Su et al., "RoFormer", 2021.  Llama / HF convention applies
 *  `rotate_half`: pairs sit at offsets (i, i + D/2) within each head_dim,
 *  rotated by per-position (cos, sin).  Equivalent to:
 *
 *      x_lo = x[..., :D/2]                       (first half)
 *      x_hi = x[..., D/2:]                       (second half)
 *      y_lo = x_lo * cos - x_hi * sin
 *      y_hi = x_lo * sin + x_hi * cos
 *
 *  Where (cos, sin) come from a per-row table of shape [N, D/2].  We use
 *  the pre-gathered form -- the caller is responsible for shaping
 *  cos[n, i] = cos(m_n * theta_i) (and likewise sin) ahead of time.
 *
 *  ## v1 shortcut
 *  This API takes pre-gathered `cos`/`sin` tables of shape [N, D/2], one
 *  row per token.  The production layout (vLLM / HF / FlashAttention) is
 *  `(position_ids, cos_cache, sin_cache)` -- gather happens *inside* the
 *  kernel and saves B*S*H*D/2 of redundant traffic per layer.  Switching
 *  to that API is the v2 step; v1 is fine for benchmarking and parity.
 *
 *  ## Layout
 *      x      : [N, D]       fp16 or fp32, row-major.  N packs whatever
 *                            outer dims the caller wants (B*S*H typical).
 *      cos    : [N, D/2]     fp32 (HF default -- tighter numerics on fp16 x).
 *      sin    : [N, D/2]     fp32
 *      y      : [N, D]       output, same dtype as x.  In-place safe
 *                            (y == x); each thread reads both x_lo[i]
 *                            and x_hi[i] before writing either output.
 *      row_stride : int      stride between consecutive rows of x/y, in
 *                            elements.  D <= row_stride.  Lets callers
 *                            pass a non-contiguous view.
 *
 *  ## Launch
 *      Block : (256,)  threads stride D/2 pairs within a row.
 *      Grid  : (N,)    one block per row.  Tiny grids fine -- N is
 *                      already B*S*H, typically tens of thousands.
 *
 *  ## Vectorisation note
 *  Llama half-rotation pairs sit D/2 apart, so a `half2`-load of a pair
 *  is impossible.  Adjacent x_lo elements ARE contiguous, so a fp16
 *  variant could half2-load *two adjacent x_lo* and *two adjacent x_hi*
 *  in one instruction each -- deferred to v1.1.  Asserting D % 8 == 0
 *  in the wrapper keeps that path open.
 */

#include <cuda_fp16.h>

#define BLOCK 256

extern "C" __global__
void rope_fp32(const float* __restrict__ x,
               const float* __restrict__ cos_tab,
               const float* __restrict__ sin_tab,
               float*       __restrict__ y,
               int D, int x_stride, int y_stride) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int half = D >> 1;

    const float* x_row   = x       + (size_t)row * x_stride;
    float*       y_row   = y       + (size_t)row * y_stride;
    const float* cos_row = cos_tab + (size_t)row * half;
    const float* sin_row = sin_tab + (size_t)row * half;

    for (int i = tid; i < half; i += BLOCK) {
        const float xl = x_row[i];
        const float xh = x_row[i + half];
        const float c  = cos_row[i];
        const float s  = sin_row[i];
        /* Read both halves into registers before writing either -- makes
         * in-place (y == x) safe. */
        y_row[i]        = xl * c - xh * s;
        y_row[i + half] = xl * s + xh * c;
    }
}

extern "C" __global__
void rope_fp16(const __half* __restrict__ x,
               const float*  __restrict__ cos_tab,
               const float*  __restrict__ sin_tab,
               __half*       __restrict__ y,
               int D, int x_stride, int y_stride) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int half = D >> 1;

    const __half* x_row   = x       + (size_t)row * x_stride;
    __half*       y_row   = y       + (size_t)row * y_stride;
    const float*  cos_row = cos_tab + (size_t)row * half;
    const float*  sin_row = sin_tab + (size_t)row * half;

    for (int i = tid; i < half; i += BLOCK) {
        const float xl = __half2float(x_row[i]);
        const float xh = __half2float(x_row[i + half]);
        const float c  = cos_row[i];
        const float s  = sin_row[i];
        y_row[i]        = __float2half(xl * c - xh * s);
        y_row[i + half] = __float2half(xl * s + xh * c);
    }
}
