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
