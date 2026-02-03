/*
 *  2D wave equation — explicit finite differences.
 *
 *  Leapfrog (Verlet) time-stepping scheme:
 *      u_next = 2*u - u_prev + c^2 * dt^2 * laplacian(u)
 *
 *  The discrete Laplacian uses the 4-neighbor stencil:
 *      lap(u) = u[left] + u[right] + u[up] + u[down] - 4*u[center]
 *
 *  Boundary pixels are skipped (Dirichlet: zero displacement at edges).
 *  The caller ping-pongs three buffers: u_prev, u, u_next.
 *
 *  Parameters:
 *      u_prev        — displacement field at time t-1  (height x width)
 *      u             — displacement field at time t    (height x width)
 *      u_next        — displacement field at time t+1  (output)
 *      width, height — grid size
 *      c2dt2         — precomputed c^2 * dt^2
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
extern "C" __global__
void wave_equation(float* u_prev, float* u, float* u_next,
                   int width, int height, float c2dt2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;

    float lap = u[idx - 1] + u[idx + 1] + u[idx - width] + u[idx + width]
                - 4.0f * u[idx];

    u_next[idx] = 2.0f * u[idx] - u_prev[idx] + c2dt2 * lap;
}
