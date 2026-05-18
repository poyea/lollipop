/*
 * FlashAttention-2 forward pass, FP32, single fixed head_dim (D=64).
 *
 * Implements the streaming-softmax recipe from Dao 2023 (FA-2): outer
 * loop tiles the query rows, inner loop streams over key/value tiles
 * while maintaining the per-row running max `m_i`, running denominator
 * `l_i`, and unscaled output accumulator `o_i` in registers.  Output
 * is normalised by `l_i` only once at the end (this is the FA-2
 * change vs FA-1, which rescaled the running `o` every iteration).
 *
 * Online softmax recurrence applied to each new score `s` for row i:
 *     m_new   = max(m_i, s)
 *     alpha   = exp(m_i - m_new)        // rescale factor for old state
 *     p       = exp(s   - m_new)        // contribution of this column
 *     l_i    <- l_i * alpha + p
 *     o_i    <- o_i * alpha + p * V[c]
 *     m_i    <- m_new
 * Final: O[i] = o_i / l_i.
 *
 * Layout:
 *   Q, K, V, O : [BH, N, D] row-major.  BH packs batch*heads; the
 *                caller flattens those into grid.y.  Single-head from
 *                the kernel's perspective.
 *   Grid       : (ceil(N / BR), BH)
 *   Block      : (BR,)   one thread per query row in the tile.
 *
 * Bank-conflict-free smem layout:
 *   Ks[D][BC+1], Vs[D][BC+1].  Stride BC+1=65 floats per row makes the
 *   32-thread cooperative store hit 32 distinct banks (row stride mod
 *   32 = 1, not 0).  Reads in the compute loop broadcast a single
 *   address across the warp -> no conflict either way.
 *
 * Numerics:
 *   `m_i` initialised to a large negative sentinel.  Masked entries
 *   (out-of-range j or causal j > i) get `s = -INF_SENTINEL`; the
 *   recurrence above naturally produces `p = 0` and `alpha = 1` so
 *   those columns contribute nothing.  Fully-masked rows would
 *   produce `l_i = 0` and divide-by-zero; we don't guard since
 *   causal+valid_row never fully masks.
 */

#define BR 64
#define BC 64
#define D  64
#define NEG_SENTINEL (-3.0e30f)

extern "C" __global__
__launch_bounds__(BR, 2)
void flash_attention_fwd(
    const float* __restrict__ Q,   // [BH, N, D]
    const float* __restrict__ K,   // [BH, N, D]
    const float* __restrict__ V,   // [BH, N, D]
    float* __restrict__ O,         // [BH, N, D]
    int N,
    int causal)                    // 0 or 1
{
    const int bh      = blockIdx.y;
    const int q_tile  = blockIdx.x;
    const int tid     = threadIdx.x;
    const int row     = q_tile * BR + tid;            // global query row
    const bool active = (row < N);
    const float scale = rsqrtf((float)D);             // 1 / sqrt(D)

    // [D][BC+1] layout: column index == key-row within tile, row index
    // == feature.  Stride 65 breaks the 32-way bank conflict on the
    // cooperative store path.
    __shared__ float Ks[D][BC + 1];
    __shared__ float Vs[D][BC + 1];

    const size_t head_stride = (size_t)N * D;
    const float* Q_ptr = Q + bh * head_stride;
    const float* K_ptr = K + bh * head_stride;
    const float* V_ptr = V + bh * head_stride;
    float*       O_ptr = O + bh * head_stride;

    // Load this thread's Q row into registers.  Unrolled so the
    // compiler keeps it in registers (constant indexing only).
    float q_reg[D];
    float o_reg[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q_reg[d] = active ? Q_ptr[row * D + d] : 0.0f;
        o_reg[d] = 0.0f;
    }

    float m_i = NEG_SENTINEL;
    float l_i = 0.0f;

    const int n_tiles = (N + BC - 1) / BC;
    for (int j_tile = 0; j_tile < n_tiles; ++j_tile) {
        const int j_base = j_tile * BC;

        // Cooperative coalesced load of K/V tiles.  BR threads load
        // BC*D floats in BC iterations: at iter k, lane `tid` reads
        // K_ptr[(j_base+k)*D + tid].  Across the warp `tid` is the
        // fast axis -> contiguous global addresses -> coalesced.
        // Stored transposed into Ks[d=tid][c=k] for bank-conflict-
        // free reads in the compute loop.
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
            const int j  = j_base + k;
            const bool ok = (j < N);
            Ks[tid][k] = ok ? K_ptr[j * D + tid] : 0.0f;
            Vs[tid][k] = ok ? V_ptr[j * D + tid] : 0.0f;
        }
        __syncthreads();

        if (active) {
            // Compute BC scores for this query row, fold each into the
            // online (m, l, o) state.
            #pragma unroll 8
            for (int c = 0; c < BC; ++c) {
                const int j = j_base + c;

                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    s += q_reg[d] * Ks[d][c];
                }
                s *= scale;

                const bool valid = (j < N) && (!causal || j <= row);
                if (!valid) s = NEG_SENTINEL;

                const float m_new = fmaxf(m_i, s);
                const float alpha = __expf(m_i - m_new);
                const float p     = __expf(s   - m_new);
                l_i = l_i * alpha + p;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    o_reg[d] = o_reg[d] * alpha + p * Vs[d][c];
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            O_ptr[row * D + d] = o_reg[d] * inv_l;
        }
    }
}
