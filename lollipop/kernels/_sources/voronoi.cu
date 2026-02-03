/*
 *  Voronoi diagram via Jump Flooding Algorithm (JFA).
 *
 *  Two kernels:
 *
 *  voronoi_jfa — assigns each pixel to its nearest seed.
 *    Run multiple passes with step = max_dim/2, max_dim/4, ..., 1.
 *    Each pass, every pixel checks 9 neighbors at the current step
 *    distance and adopts the closest seed.  Converges in O(log n) passes.
 *
 *  voronoi_color — maps the nearest-seed index to an RGB color.
 *
 *  The nearest[] array stores flat pixel indices (y*width + x) of the
 *  assigned seed, or -1 if unassigned.
 *
 *  Parameters (jfa):
 *      nearest   — (width*height) int32, seed assignments
 *      seeds     — unused (kept for API compat)
 *      num_seeds — number of seed points
 *      width, height — grid size
 *      step      — current JFA step size
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
extern "C" __global__
void voronoi_jfa(int* nearest, const int* seeds, int num_seeds,
                 int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int best = nearest[idx];

    /* Compute distance to current best seed */
    int best_dist = 2147483647;
    if (best >= 0) {
        int bx = best % width;
        int by = best / width;
        int dx = x - bx;
        int dy = y - by;
        best_dist = dx * dx + dy * dy;
    }

    /* Check 9 neighbors at current step size */
    for (int dy = -step; dy <= step; dy += step) {
        for (int dx = -step; dx <= step; dx += step) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int neighbor = nearest[ny * width + nx];
            if (neighbor < 0) continue;

            int sx = neighbor % width;
            int sy = neighbor / width;
            int ddx = x - sx;
            int ddy = y - sy;
            int dist = ddx * ddx + ddy * ddy;

            if (dist < best_dist) {
                best_dist = dist;
                best = neighbor;
            }
        }
    }

    nearest[idx] = best;
}

extern "C" __global__
void voronoi_color(const int* nearest, unsigned char* output,
                   const unsigned char* colors, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int seed = nearest[idx];

    if (seed >= 0) {
        /* Map seed index to its color */
        int out_base = idx * 3;
        int color_base = seed * 3;
        output[out_base]     = colors[color_base];
        output[out_base + 1] = colors[color_base + 1];
        output[out_base + 2] = colors[color_base + 2];
    }
}
