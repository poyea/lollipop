/*
 *  INT8 W8A8 GEMM (DP4A's bigger cousin — wmma m16n16k16 INT8 on sm_75).
 *
 *  Computes  C[M,N] = a_scale[m] * b_scale[n] * (A_q @ B_q.T)[m, n]
 *
 *    A_q : [M, K]  int8  (per-row symmetric quantised activations)
 *    B_q : [N, K]  int8  (per-channel symmetric quantised weights, *already*
 *                         transposed into [N, K] — standard inference layout)
 *    C   : [M, N]  fp32
 *
 *  Tile geometry:
 *    - 64x64 macro tile, BK=32, 4 warps (128 threads) per block.
 *    - Each warp owns one 16-row m-tile × four 16-col n-tiles → covers the
 *      whole 64 wide macro-N with 4 wmma fragments.
 *    - K-loop: 2 wmma K-steps per macro-K tile (BK=32 / WK=16).
 *
 *  Smem:
 *    - As[64][32] int8 = 2 KB
 *    - Bs[64][32] int8 = 2 KB  (already N-major because B is pre-transposed)
 *    - Cs[64][64] int32 = 16 KB  (epilogue spill so the FP scale + store
 *      pass can be plain thread-streamed)
 *    Total = 20 KB, fits sm_75's 48 KB static cap with room for double
 *    occupancy.
 *
 *  Shape constraints:
 *    - M % 64 == 0, N % 64 == 0, K % 32 == 0
 *    - A_q, B_q must be 16-byte aligned
 *
 *  The fragment-load layout exploits the pre-transpose: a_frag is row_major
 *  over As (M-major, K fast) and b_frag is col_major over Bs (N-major, K
 *  fast). col_major reads memory at offset (k + n*ld) — matches Bs[n][k]
 *  flat-index ld=BK, so the math is B(k,n) under wmma's API even though
 *  Bs is physically n-major in smem.
 */

#include <mma.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 32
#define WM 16
#define WN 16
#define WK 16
#define NWARP 4

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_int8(const signed char* __restrict__ A,
               const signed char* __restrict__ B,
               float* __restrict__ C,
               const float* __restrict__ a_scale,
               const float* __restrict__ b_scale,
               int M, int N, int K) {
    __shared__ signed char As[BM][BK];
    __shared__ signed char Bs[BN][BK];
    __shared__ int Cs[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    wmma::fragment<wmma::matrix_a,    WM, WN, WK, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, int>                           c_frag[BN / WN];

    #pragma unroll
    for (int n = 0; n < BN / WN; ++n) wmma::fill_fragment(c_frag[n], 0);

    /* Cooperative load mapping: 128 threads * 16 B = 2048 B per tile.
     * Each thread moves one int4 (16 bytes / 16 int8s) along K. */
    const int t_row = tid >> 1;         /* 0..63 */
    const int t_col = (tid & 1) << 4;   /* 0 or 16 */

    const int num_tiles = K / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int gK0 = t * BK + t_col;

        /* ---- load A tile (M-major, K-fast) ---- */
        {
            const int gA_row = by * BM + t_row;
            const int4 a4 = *reinterpret_cast<const int4*>(&A[gA_row * K + gK0]);
            *reinterpret_cast<int4*>(&As[t_row][t_col]) = a4;
        }
        /* ---- load B tile (N-major, K-fast — B is pre-transposed) ---- */
        {
            const int gB_row = bx * BN + t_row;
            const int4 b4 = *reinterpret_cast<const int4*>(&B[gB_row * K + gK0]);
            *reinterpret_cast<int4*>(&Bs[t_row][t_col]) = b4;
        }
        __syncthreads();

        /* ---- wmma compute ---- */
        #pragma unroll
        for (int kk = 0; kk < BK / WK; ++kk) {
            wmma::load_matrix_sync(a_frag, &As[warp_id * WM][kk * WK], BK);
            #pragma unroll
            for (int n = 0; n < BN / WN; ++n) {
                wmma::load_matrix_sync(b_frag, &Bs[n * WN][kk * WK], BK);
                wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
            }
        }
        __syncthreads();
    }

    /* ---- spill c_frag -> Cs (row-major int32), then scale + store ---- */
    #pragma unroll
    for (int n = 0; n < BN / WN; ++n) {
        wmma::store_matrix_sync(&Cs[warp_id * WM][n * WN], c_frag[n], BN, wmma::mem_row_major);
    }
    __syncthreads();

    /* 128 threads, 4096 outputs → 32 outputs per thread, M-major. */
    const int e_row = tid >> 1;
    const int e_col_base = (tid & 1) << 5;
    const int gM = by * BM + e_row;
    const float as = a_scale[gM];

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int n_local = e_col_base + j;
        const int gN = bx * BN + n_local;
        const float bs = b_scale[gN];
        C[gM * N + gN] = static_cast<float>(Cs[e_row][n_local]) * as * bs;
    }
}
