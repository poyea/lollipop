/*
 *  Conway's Game of Life — one generation step.
 *
 *  Rules (applied simultaneously to every cell):
 *    - Live cell with 2 or 3 neighbors survives
 *    - Dead cell with exactly 3 neighbors becomes alive
 *    - All other cells die or stay dead
 *
 *  Boundary: toroidal wrap-around (left edge connects to right, etc.)
 *  Each thread processes one cell, counting its 8 neighbors.
 *
 *  Parameters:
 *      grid   — current state  (height x width, uint8, 0 or 1)
 *      next   — next state     (height x width, uint8)
 *      width, height — grid dimensions
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
extern "C" __global__
void game_of_life(const unsigned char* grid, unsigned char* next,
                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            neighbors += grid[ny * width + nx];
        }
    }

    unsigned char cell = grid[y * width + x];
    next[y * width + x] = (cell && (neighbors == 2 || neighbors == 3))
                           || (!cell && neighbors == 3);
}
