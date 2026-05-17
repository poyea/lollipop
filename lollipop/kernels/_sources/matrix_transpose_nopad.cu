/*
 *  Tiled matrix transpose *without* the +1 smem padding.  Exists only
 *  to quantify the bank-conflict cost of the unpadded layout — the
 *  tiled+padded `matrix_transpose` is the version you actually ship.
 *
 *  Why this is slower than `matrix_transpose`:
 *
 *    Shared memory has 32 banks of 4 B each; `bank(addr) = (addr/4) %
 *    32`.  In the write-back pass each thread reads
 *
 *      tile[threadIdx.x][threadIdx.y + j]
 *
 *    where `threadIdx.x` varies across the warp.  With layout
 *    `tile[32][32]` the offset is `threadIdx.x * 32 + (ty+j)`, so all
 *    32 lanes hit the same bank `(ty+j) % 32`  →  32-way bank conflict
 *    per LDS, serialising the warp.  Padding the inner dim to 33
 *    breaks the alignment: offset becomes `tx * 33 + (ty+j)`, and the
 *    bank `(tx*33 + (ty+j)) % 32 = (tx + ty+j) % 32` is unique across
 *    the warp  →  no conflict.
 */
#define TILE_DIM 32
#define BLOCK_ROWS 8

extern "C" __global__
void matrix_transpose_nopad(const float* input, float* output,
                            int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  /* no +1 — induces conflicts */

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
