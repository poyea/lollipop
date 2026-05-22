/*
 *  INT4 weight-only GEMM (W4A16) -- the AWQ/GPTQ inference recipe.
 *
 *  Computes  C[M, N] = A @ W.T   where W is reconstructed from packed
 *  INT4 quantised values + per-group FP16 scale + zero point:
 *
 *      W[k, n] = scale[g(k), n] * (W_q[n, k] - zero[g(k), n])
 *
 *  with group index `g(k) = k / G` and group size `G = 64`.
 *
 *  Layouts:
 *    A      : [M, K]    fp16     (activations -- no quantisation)
 *    Wq     : [N, K/2]  uint8    (packed: low nibble = k-even, high = k-odd)
 *    scales : [K/G, N]  fp16
 *    zeros  : [K/G, N]  fp16
 *    C      : [M, N]    fp16
 *
 *  Tile geometry:
 *    BM=BN=64, BK=64=G (one group per K-tile -> scales/zeros loaded once),
 *    4 warps (128 threads), wmma m16n16k16 fp16 × fp16 -> fp32.
 *    Each warp owns 16 m-rows x 64 n-cols (= 4 wmma n-tiles).
 *
 *  Dequant-fuse pattern: instead of materialising a dequantised W tensor
 *  in global memory, each K-iter does
 *      read packed Wq tile (2 KB) -> dequant in registers
 *      -> write fp16 Ws tile (8 KB) in smem -> wmma against As.
 *  The fp16 round-trip through smem is the cost of being able to use
 *  wmma at all (wmma requires its operands as fp16/int8/bf16 smem
 *  layouts; there's no direct INT4-input fragment on sm_75).
 *
 *  Smem footprint:
 *    As   : 64*64*2  =  8 KB  (fp16)
 *    Ws   : 64*64*2  =  8 KB  (fp16, dequantised)
 *    Cs32 : 64*64*4  = 16 KB  (fp32 accumulator spill)
 *                              ----
 *                              32 KB  -- fits sm_75's 48 KB static cap.
 *
 *  Shape constraints:
 *    - M % 64 == 0, N % 64 == 0, K % 64 == 0
 *    - K must be a multiple of the group size (64), enforced by the above
 *    - A, Wq must be 16-byte aligned
 */

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 64
#define G  64
#define WM 16
#define WN 16
#define WK 16

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_int4(const __half*       __restrict__ A,
               const unsigned char* __restrict__ Wq,
               const __half*       __restrict__ scales,
               const __half*       __restrict__ zeros,
               __half*             __restrict__ C,
               int M, int N, int K) {
    __shared__ __half As[BM][BK];
    __shared__ __half Ws[BN][BK];
    __shared__ float  Cs[BM][BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    wmma::fragment<wmma::matrix_a,    WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float>                    c_frag[BN / WN];

    #pragma unroll
    for (int n = 0; n < BN / WN; ++n) wmma::fill_fragment(c_frag[n], 0.0f);

    /* Load mapping (shared between A and W): 128 threads, 64 rows × 64 fp16
     * per tile. Each thread owns one row × 32 fp16 = 64 bytes.  Split into
     * 4 int4 (16 B = 8 fp16) loads per A-tile, or one int4 = 16 packed bytes
     * = 32 fp16 weights for the W dequant pass. */
    const int t_row      = tid >> 1;          /* 0..63 */
    const int t_col_base = (tid & 1) << 5;    /* 0 or 32 */

    const int num_tiles = K / BK;

    for (int t = 0; t < num_tiles; ++t) {

        /* ---- A tile (M-major, K-fast) -- coalesced int4 of fp16 ---- */
        {
            const int gA_row = by * BM + t_row;
            #pragma unroll
            for (int seg = 0; seg < 4; ++seg) {
                const int a_col  = t_col_base + seg * 8;
                const int gA_col = t * BK + a_col;
                const int4 a4 = *reinterpret_cast<const int4*>(&A[gA_row * K + gA_col]);
                *reinterpret_cast<int4*>(&As[t_row][a_col]) = a4;
            }
        }

        /* ---- scales / zeros for this group: each thread fetches its (n) pair ----
         *
         * Since BK == G, group index == K-tile index (t).  scales/zeros are
         * stored [K/G, N] row-major, so the per-thread fetch is one fp16 each
         * (we re-fetch per K-tile -- 4 bytes per thread per tile, trivial). */
        const int gN_row = bx * BN + t_row;
        const __half s_n = scales[t * N + gN_row];
        const __half z_n = zeros [t * N + gN_row];

        /* ---- Wq tile: load 16 packed bytes (= 32 fp16 weights), dequant inline,
         *      store fp16 into Ws[t_row][t_col_base + 0..31]. ---- */
        {
            const int gK_byte = (t * BK + t_col_base) >> 1;
            const int4 packed = *reinterpret_cast<const int4*>(&Wq[gN_row * (K / 2) + gK_byte]);
            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&packed);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const unsigned char b = bytes[i];
                const int q_lo = b & 0xF;
                const int q_hi = (b >> 4) & 0xF;
                const __half v_lo = __hmul(s_n, __hsub(__int2half_rn(q_lo), z_n));
                const __half v_hi = __hmul(s_n, __hsub(__int2half_rn(q_hi), z_n));
                Ws[t_row][t_col_base + i * 2 + 0] = v_lo;
                Ws[t_row][t_col_base + i * 2 + 1] = v_hi;
            }
        }

        __syncthreads();

        /* ---- wmma compute (4 K-steps × 4 n-tiles per warp) ---- */
        #pragma unroll
        for (int kk = 0; kk < BK / WK; ++kk) {
            wmma::load_matrix_sync(a_frag, &As[warp_id * WM][kk * WK], BK);
            #pragma unroll
            for (int n = 0; n < BN / WN; ++n) {
                wmma::load_matrix_sync(b_frag, &Ws[n * WN][kk * WK], BK);
                wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
            }
        }
        __syncthreads();
    }

    /* ---- epilogue: spill c_frag(fp32) into Cs, then convert + store fp16 ---- */
    #pragma unroll
    for (int n = 0; n < BN / WN; ++n) {
        wmma::store_matrix_sync(&Cs[warp_id * WM][n * WN], c_frag[n], BN, wmma::mem_row_major);
    }
    __syncthreads();

    const int e_row = tid >> 1;
    const int e_col_base = (tid & 1) << 5;
    const int gM = by * BM + e_row;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int n_local = e_col_base + j;
        const int gN = bx * BN + n_local;
        C[gM * N + gN] = __float2half(Cs[e_row][n_local]);
    }
}
