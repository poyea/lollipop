/*
 *  Naive matrix transpose — exists purely as a foil for
 *  `matrix_transpose` (the smem-tiled, +1-padded version).
 *
 *  What's bad about this kernel:
 *
 *    - The read  `input [y*width + x]`     is coalesced (consecutive
 *      threads in a warp hit consecutive x's, contiguous in memory).
 *    - The write `output[x*height + y]`   is *strided* by `height`:
 *      consecutive threads write 4*height bytes apart, so a 32-lane
 *      warp issues 32 separate STG.E.32 transactions instead of one
 *      coalesced 128 B transaction.  Effective write bandwidth drops
 *      by ~32x at large `height`.
 *
 *  Used in `tests/test_matrix_transpose.py` for parity and in the
 *  `docs/profiles/matrix_transpose.md` writeup to demonstrate the win
 *  from smem tiling + bank-conflict-free padding.
 *
 *  Launch: block=(32,8), grid=((w+31)/32, (h+31)/32)  (same as the
 *  tiled kernel, for an apples-to-apples comparison).
 */
#define TILE_DIM 32
#define BLOCK_ROWS 8

extern "C" __global__
void matrix_transpose_naive(const float* input, float* output,
                            int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            output[x * height + (y + j)] = input[(y + j) * width + x];
        }
    }
}
