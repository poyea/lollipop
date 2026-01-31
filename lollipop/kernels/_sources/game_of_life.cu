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
