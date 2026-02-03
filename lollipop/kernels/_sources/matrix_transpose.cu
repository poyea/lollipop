/*
 *  Matrix transpose with shared-memory tiling.
 *
 *  Naive transpose suffers from uncoalesced global memory writes.
 *  This kernel fixes that by:
 *
 *    1. Coalesced read:  load a 32x32 tile from input into shared memory
 *    2. Coalesced write: write the transposed tile to output
 *
 *  The shared tile is declared as [32][33] — the +1 padding avoids
 *  bank conflicts when threads read down a column.
 *
 *  Each thread handles 4 rows (TILE_DIM / BLOCK_ROWS = 32/8 = 4).
 *
 *  Parameters:
 *      input  — source matrix (height x width, float32, row-major)
 *      output — transposed matrix (width x height, float32)
 *      width  — number of columns in input
 *      height — number of rows in input
 *
 *  Launch: block=(32,8), grid=((w+31)/32, (h+31)/32)
 */
#define TILE_DIM 32
#define BLOCK_ROWS 8

extern "C" __global__
void matrix_transpose(const float* input, float* output, int width, int height) {
    /* Shared memory tile with +1 padding to avoid bank conflicts */
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    /* Coalesced read: each thread loads 4 elements into shared tile */
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }

    __syncthreads();

    /* Transposed coordinates */
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    /* Coalesced write from transposed shared tile */
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
