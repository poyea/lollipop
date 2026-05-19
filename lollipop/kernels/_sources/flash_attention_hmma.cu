/*
 * FlashAttention-2 forward, HMMA path: FP16 in, FP32 accumulate, FP16 out.
 *
 * Same online-softmax recurrence as `flash_attention.cu`, but the two
 * matmuls (QK^T and PV) go through `nvcuda::wmma` 16x16x16 fragments
 * (sm_75 HMMA with FP32 accumulator).  Designed for Turing+ where
 * tensor cores exist but FP8/BF16-HMMA do not.
 *
 * Tile geometry:
 *   BR = 64       Q rows per block
 *   BC = 32       KV cols per inner tile (kept small so 45 KB static smem fits sm_75's 48 KB cap)
 *   D  = 64       head_dim (fixed)
 *   Block = 128   threads = 4 warps
 *   Each warp owns 16 Q rows (warp_id * 16 .. +16) and computes its own
 *   slice of QK^T / PV via wmma; per-row softmax state is shared via smem.
 *
 * Smem layout (all unpadded; wmma load_matrix_sync's ld argument carries
 * the row stride explicitly):
 *   Qs[BR][D]   FP16   8192 B
 *   Ks[BC][D]   FP16   4096 B
 *   Vs[BC][D]   FP16   4096 B
 *   Ss[BR][BC]  FP32   8192 B   (scores; reused as Ps storage)
 *   Ps[BR][BC]  FP16   4096 B   (probs in FP16 for the PV matmul; separate buffer to dodge in-place aliasing)
 *   Os[BR][D]   FP32  16384 B   (running output accumulator)
 *   m_i, l_i, alpha [BR] FP32     768 B
 *   --------- total ~45.6 KB
 *
 * Numerics:
 *   Inputs FP16; m, l, alpha and the score+output accumulators stay
 *   FP32.  Online softmax recurrence unchanged from v1.  Parity bar is
 *   atol=rtol=1e-2 vs an FP32 materialised reference -- the FP16 input
 *   quantisation alone produces ~5e-3 relative error before the kernel
 *   adds anything on top.
 */

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BR 64
#define BC 32
#define D  64
#define NUM_WARPS  4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NEG_SENTINEL (-3.0e30f)

extern "C" __global__
__launch_bounds__(128, 1)
void flash_attention_hmma_fwd(
    const __half* __restrict__ Q,    // [BH, N, D]  FP16
    const __half* __restrict__ K,    // [BH, N, D]  FP16
    const __half* __restrict__ V,    // [BH, N, D]  FP16
    __half* __restrict__ O,          // [BH, N, D]  FP16
    int N,
    int causal)
{
    const int bh      = blockIdx.y;
    const int q_tile  = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;          // tid / 32
    const int lane    = tid & 31;
    const int warp_row_base = warp_id * WMMA_M;   // 0, 16, 32, 48

    __shared__ __half Qs[BR * D];
    __shared__ __half Ks[BC * D];
    __shared__ __half Vs[BC * D];
    __shared__ float  Ss[BR * BC];
    __shared__ __half Ps[BR * BC];
    __shared__ float  Os[BR * D];
    __shared__ float  m_i[BR];
    __shared__ float  l_i[BR];
    __shared__ float  alpha_arr[BR];

    const size_t head_stride = (size_t)N * D;
    const __half* Q_ptr = Q + bh * head_stride;
    const __half* K_ptr = K + bh * head_stride;
    const __half* V_ptr = V + bh * head_stride;
    __half*       O_ptr = O + bh * head_stride;

    const int q_base   = q_tile * BR;
    const float scale  = rsqrtf((float)D);
    const __half zero_h = __float2half(0.0f);

    // Cooperative load Q tile (BR*D = 4096 halves; 128 threads => 32 each).
    #pragma unroll
    for (int i = 0; i < (BR * D) / 128; ++i) {
        const int idx = tid + i * 128;
        const int r   = idx / D;
        const int d   = idx % D;
        const int gr  = q_base + r;
        Qs[idx] = (gr < N) ? Q_ptr[gr * D + d] : zero_h;
    }
    // Init Os = 0
    #pragma unroll
    for (int i = 0; i < (BR * D) / 128; ++i) {
        Os[tid + i * 128] = 0.0f;
    }
    // Init per-row state
    if (tid < BR) {
        m_i[tid] = NEG_SENTINEL;
        l_i[tid] = 0.0f;
    }
    __syncthreads();

    // 2C: whole-tile causal skip.
    const int n_tiles_full   = (N + BC - 1) / BC;
    const int row_block_end  = (q_tile + 1) * BR - 1;
    const int n_tiles_causal = (row_block_end / BC) + 1;
    const int n_tiles = causal
        ? (n_tiles_causal < n_tiles_full ? n_tiles_causal : n_tiles_full)
        : n_tiles_full;

    constexpr int BC_STEPS = BC / WMMA_N;   // 2
    constexpr int D_STEPS  = D  / WMMA_K;   // 4

    for (int j_tile = 0; j_tile < n_tiles; ++j_tile) {
        const int j_base = j_tile * BC;

        // Cooperative load K, V tiles (BC*D = 2048 halves => 16 each).
        #pragma unroll
        for (int i = 0; i < (BC * D) / 128; ++i) {
            const int idx = tid + i * 128;
            const int r   = idx / D;
            const int d   = idx % D;
            const int gr  = j_base + r;
            const bool ok = (gr < N);
            Ks[idx] = ok ? K_ptr[gr * D + d] : zero_h;
            Vs[idx] = ok ? V_ptr[gr * D + d] : zero_h;
        }
        __syncthreads();

        // -------- QK^T via wmma --------
        // Each warp computes Ss[warp_row_base : +16, 0 : BC] = Q_warp @ K^T tile.
        // Q is row_major [BR][D], K is row_major [BC][D]; B = K^T loaded with col_major
        // viewing the same [BC][D] row_major buffer (see derivation in the .md write-up).
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            #pragma unroll
            for (int n = 0; n < BC_STEPS; ++n) {
                wmma::fill_fragment(s_frag, 0.0f);
                #pragma unroll
                for (int k = 0; k < D_STEPS; ++k) {
                    // A = Qs[warp_row_base:+16, k*16:+16], row_major, ld=D
                    wmma::load_matrix_sync(a_frag,
                        Qs + warp_row_base * D + k * WMMA_K, D);
                    // B = K^T tile [k*16:+16][n*16:+16].
                    // K row-major has K(r,c) at offset r*D+c.  Loading K's
                    // sub-block K[n*16:+16, k*16:+16] as col_major treats
                    // adjacent rows as adjacent elements in a column ->
                    // that *is* the transpose.  ld=D.
                    wmma::load_matrix_sync(b_frag,
                        Ks + n * WMMA_N * D + k * WMMA_K, D);
                    wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
                }
                // Scale by 1/sqrt(D) inside the fragment.
                #pragma unroll
                for (int t = 0; t < s_frag.num_elements; ++t) {
                    s_frag.x[t] *= scale;
                }
                wmma::store_matrix_sync(
                    Ss + warp_row_base * BC + n * WMMA_N,
                    s_frag, BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // -------- Per-row softmax + Ps build --------
        // 128 threads, BR=64 rows: lower half does the work; one row per thread.
        if (tid < BR) {
            const int row = tid;
            const int gr  = q_base + row;
            const bool active_row = (gr < N);

            // First pass: mask + find row max.  Write masked values back into Ss so the
            // second pass can re-read them without recomputing the mask.
            float row_max = NEG_SENTINEL;
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                const int j = j_base + c;
                const bool valid = active_row && (j < N) && (!causal || j <= gr);
                float s = valid ? Ss[row * BC + c] : NEG_SENTINEL;
                Ss[row * BC + c] = s;
                if (s > row_max) row_max = s;
            }

            const float m_old = m_i[row];
            const float m_new = fmaxf(m_old, row_max);
            const float alpha = __expf(m_old - m_new);

            float sum_p = 0.0f;
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                const float p = __expf(Ss[row * BC + c] - m_new);
                sum_p += p;
                Ps[row * BC + c] = __float2half(p);
            }

            l_i[row] = l_i[row] * alpha + sum_p;
            m_i[row] = m_new;
            alpha_arr[row] = alpha;
        }
        __syncthreads();

        // -------- Scale Os by alpha[row] --------
        // BR*D = 4096 elements, 128 threads => 32 each.
        #pragma unroll
        for (int i = 0; i < (BR * D) / 128; ++i) {
            const int idx = tid + i * 128;
            const int r   = idx / D;
            Os[idx] *= alpha_arr[r];
        }
        __syncthreads();

        // -------- PV via wmma --------
        // Each warp computes Os[warp_row_base : +16, 0 : D] += P_warp @ V_tile.
        // P row_major [BR][BC], V row_major [BC][D].  Load accumulator from
        // Os, mma_sync into it, store back -- in-place += in the fragment.
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            constexpr int PV_N_STEPS = D  / WMMA_N;   // 4
            constexpr int PV_K_STEPS = BC / WMMA_K;   // 2

            #pragma unroll
            for (int n = 0; n < PV_N_STEPS; ++n) {
                wmma::load_matrix_sync(
                    o_frag,
                    Os + warp_row_base * D + n * WMMA_N,
                    D, wmma::mem_row_major);
                #pragma unroll
                for (int k = 0; k < PV_K_STEPS; ++k) {
                    // A = Ps[warp_row_base:+16, k*16:+16] row_major, ld=BC
                    wmma::load_matrix_sync(a_frag,
                        Ps + warp_row_base * BC + k * WMMA_K, BC);
                    // B = Vs[k*16:+16, n*16:+16] row_major, ld=D
                    wmma::load_matrix_sync(b_frag,
                        Vs + k * WMMA_K * D + n * WMMA_N, D);
                    wmma::mma_sync(o_frag, a_frag, b_frag, o_frag);
                }
                wmma::store_matrix_sync(
                    Os + warp_row_base * D + n * WMMA_N,
                    o_frag, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Final normalise + FP16 store.  BR*D = 4096 elements / 128 threads = 32 each.
    #pragma unroll
    for (int i = 0; i < (BR * D) / 128; ++i) {
        const int idx = tid + i * 128;
        const int r   = idx / D;
        const int d   = idx % D;
        const int gr  = q_base + r;
        if (gr < N) {
            const float inv_l = 1.0f / l_i[r];
            O_ptr[gr * D + d] = __float2half(Os[idx] * inv_l);
        }
    }
}
