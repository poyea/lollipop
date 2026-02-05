/*
 *  2D heat equation — explicit finite-difference diffusion.
 *
 *  The heat equation describes how temperature spreads over time:
 *      du/dt = alpha * laplacian(u)
 *
 *  Discretised with forward Euler and the 4-neighbor Laplacian stencil:
 *      u_next[i,j] = u[i,j] + alpha * dt * (
 *          u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
 *      )
 *
 *  Simpler than the wave equation (only one time level, no u_prev).
 *  Energy always diffuses outward — the solution smooths over time
 *  and converges to a uniform temperature.
 *
 *  Stability requires:  alpha * dt <= 0.25  (for 2D, 4-neighbor stencil)
 *
 *  Boundary: Dirichlet (edges stay at zero).
 *
 *  Parameters:
 *      u      — current temperature field  (height x width, float32)
 *      u_next — next temperature field     (height x width, float32)
 *      width, height — grid dimensions
 *      alpha_dt — precomputed alpha * dt  (must be <= 0.25 for stability)
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
extern "C" __global__
void heat_equation(const float* u, float* u_next,
                   int width, int height, float alpha_dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;

    float lap = u[idx - 1] + u[idx + 1] + u[idx - width] + u[idx + width]
                - 4.0f * u[idx];

    u_next[idx] = u[idx] + alpha_dt * lap;
}
