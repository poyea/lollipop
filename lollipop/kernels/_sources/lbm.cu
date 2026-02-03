/*
 *  D2Q9 Lattice Boltzmann Method — BGK collision + streaming.
 *
 *  Nine velocity directions per cell on a 2-D grid:
 *
 *       6  2  5        (-1,+1) (0,+1) (+1,+1)
 *       3  0  1   =>   (-1, 0) (0, 0) (+1, 0)
 *       7  4  8        (-1,-1) (0,-1) (+1,-1)
 *
 *  Layout:  f_in / f_out are  [9 * width * height]  (direction-major).
 *           f_in [d * width * height + y * width + x]
 *
 *  Parameters:
 *      f_in   — current distribution functions   (read)
 *      f_out  — next distribution functions       (write)
 *      width, height — grid size
 *      omega  — relaxation parameter  (1 / tau),  e.g. 1.0 for tau = 1
 */
extern "C" __global__
void lbm(const float* f_in, float* f_out,
         int width, int height, float omega) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int N   = width * height;          /* stride between directions */

    /* --- D2Q9 lattice constants ---------------------------------- */

    /*  ex, ey for directions 0..8 */
    const int ex[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    const int ey[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    /*  weights */
    const float w[9] = {
        4.0f / 9.0f,                           /* centre        */
        1.0f / 9.0f, 1.0f / 9.0f,              /* axis          */
        1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 36.0f, 1.0f / 36.0f,            /* diagonals     */
        1.0f / 36.0f, 1.0f / 36.0f
    };

    /* --- 1.  Gather f_i for this cell (bounce-back at walls) ----- */

    float f[9];
    for (int d = 0; d < 9; d++) {
        int sx = x - ex[d];
        int sy = y - ey[d];

        /* solid / outside boundary → bounce back (reverse direction) */
        if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
            /* opposite direction index: 0→0, 1↔3, 2↔4, 5↔7, 6↔8 */
            const int opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
            f[d] = f_in[opp[d] * N + idx];
        } else {
            f[d] = f_in[d * N + sy * width + sx];
        }
    }

    /* --- 2.  Macroscopic quantities  (density, velocity) --------- */

    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int d = 0; d < 9; d++) {
        rho += f[d];
        ux  += (float)ex[d] * f[d];
        uy  += (float)ey[d] * f[d];
    }
    if (rho > 0.0f) { ux /= rho;  uy /= rho; }

    /* --- 3.  BGK collision: f_out = f - omega * (f - f_eq) ------- */

    float usq = ux * ux + uy * uy;          /* |u|^2 */

    for (int d = 0; d < 9; d++) {
        float eu  = (float)ex[d] * ux + (float)ey[d] * uy;   /* e_i . u */
        float feq = w[d] * rho * (1.0f + 3.0f * eu
                                       + 4.5f * eu * eu
                                       - 1.5f * usq);
        f_out[d * N + idx] = f[d] + omega * (feq - f[d]);
    }
}
