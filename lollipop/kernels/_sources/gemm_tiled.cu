/*
 *  Shared-mem double-buffered tiled SGEMM (FP32, row-major).
 *
 *  Computes  C[M,N] = A[M,K] @ B[K,N]  with:
 *
 *    - 128x128 macro-tile per block (BM=BN=128, BK=8)
 *    - 8x8 register micro-tile per thread (256 threads/block, 16x16 grid)
 *    - manual smem double-buffer: while the math loop chews on buffer
 *      `buf`, the next K-tile is being LDG'd into registers, then STS'd
 *      into buffer `buf ^ 1`.  Mirrors the cp.async pipeline without
 *      needing sm_80+ instructions, so this runs on Turing/Ampere
 *      consumer cards (no `cp.async`, no `__pipeline_*`).
 *    - all global loads are float4 (LDG.E.128) — 4x fewer load issues
 *      than scalar loads, naturally 16-byte aligned and coalesced.
 *    - A is transposed into smem so the per-kk inner loop reads A in
 *      contiguous (M-major) order.  Each thread issues two LDS.128 for
 *      A and two for B per kk step.
 *
 *  Shape constraints (v1 — checked in the Python wrapper):
 *    - M and N must be multiples of 128
 *    - K must be a multiple of 8
 *    - A, B, C must be row-major contiguous and 16-byte aligned
 *
 *  Launch: block=(256,), grid=(N/128, M/128).  No dynamic smem.
 */

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__
__launch_bounds__(256, 2)
void gemm_tiled(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    /* Global-load mapping: each thread fetches one float4 from A and one
     * from B per K-tile (BM*BK/256 = 4 floats, BN*BK/256 = 4 floats). */
    const int a_row = tid >> 1;          /* 0..127 */
    const int a_col = (tid & 1) << 2;    /* 0 or 4 */
    const int b_row = tid >> 5;          /* 0..7   */
    const int b_col = (tid & 31) << 2;   /* 0,4,...,124 */

    /* Output mapping inside the macro-tile (16x16 thread grid). */
    const int ty = tid >> 4;
    const int tx = tid & 15;

    const int A_row_global = by * BM + a_row;
    const int B_col_global = bx * BN + b_col;

    /* ---------- Prologue: load tile 0 into buf 0 ------------------- */
    {
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[A_row_global * K + 0 * BK + a_col]);
        As[0][a_col + 0][a_row] = a4.x;
        As[0][a_col + 1][a_row] = a4.y;
        As[0][a_col + 2][a_row] = a4.z;
        As[0][a_col + 3][a_row] = a4.w;

        const float4 b4 = *reinterpret_cast<const float4*>(
            &B[(0 * BK + b_row) * N + B_col_global]);
        *reinterpret_cast<float4*>(&Bs[0][b_row][b_col]) = b4;
    }
    __syncthreads();

    float Creg[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            Creg[i][j] = 0.0f;

    const int num_tiles = K / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles; ++t) {
        const int next = buf ^ 1;
        const bool has_next = (t + 1) < num_tiles;

        /* Issue next tile's global loads BEFORE consuming current tile —
         * gives the LSU time to retire while the FFMAs run. */
        float4 a_pref, b_pref;
        if (has_next) {
            a_pref = *reinterpret_cast<const float4*>(
                &A[A_row_global * K + (t + 1) * BK + a_col]);
            b_pref = *reinterpret_cast<const float4*>(
                &B[((t + 1) * BK + b_row) * N + B_col_global]);
        }

        /* ---------- Compute on current buffer ---------------------- */
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float Areg[TM];
            float Breg[TN];
            *reinterpret_cast<float4*>(&Areg[0]) =
                *reinterpret_cast<float4*>(&As[buf][kk][ty * TM]);
            *reinterpret_cast<float4*>(&Areg[4]) =
                *reinterpret_cast<float4*>(&As[buf][kk][ty * TM + 4]);
            *reinterpret_cast<float4*>(&Breg[0]) =
                *reinterpret_cast<float4*>(&Bs[buf][kk][tx * TN]);
            *reinterpret_cast<float4*>(&Breg[4]) =
                *reinterpret_cast<float4*>(&Bs[buf][kk][tx * TN + 4]);
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }

        /* ---------- Stage prefetched tile into next buffer --------- */
        if (has_next) {
            As[next][a_col + 0][a_row] = a_pref.x;
            As[next][a_col + 1][a_row] = a_pref.y;
            As[next][a_col + 2][a_row] = a_pref.z;
            As[next][a_col + 3][a_row] = a_pref.w;
            *reinterpret_cast<float4*>(&Bs[next][b_row][b_col]) = b_pref;
        }
        __syncthreads();
        buf = next;
    }

    /* ---------- Epilogue: write Creg back to C --------------------- */
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        float* Crow = &C[(row0 + i) * N + col0];
        *reinterpret_cast<float4*>(&Crow[0]) =
            *reinterpret_cast<float4*>(&Creg[i][0]);
        *reinterpret_cast<float4*>(&Crow[4]) =
            *reinterpret_cast<float4*>(&Creg[i][4]);
    }
}
