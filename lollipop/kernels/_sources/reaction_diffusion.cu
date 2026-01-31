extern "C" __global__
void reaction_diffusion(float* u, float* v, float* u_next, float* v_next,
                        int width, int height,
                        float du, float dv, float f, float k, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;

    float lap_u = u[idx - 1] + u[idx + 1] + u[idx - width] + u[idx + width]
                  - 4.0f * u[idx];
    float lap_v = v[idx - 1] + v[idx + 1] + v[idx - width] + v[idx + width]
                  - 4.0f * v[idx];

    float uval = u[idx];
    float vval = v[idx];
    float uvv = uval * vval * vval;

    u_next[idx] = uval + dt * (du * lap_u - uvv + f * (1.0f - uval));
    v_next[idx] = vval + dt * (dv * lap_v + uvv - (f + k) * vval);
}
