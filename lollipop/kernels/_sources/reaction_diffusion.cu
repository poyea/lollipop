/*
 *  Gray-Scott reaction-diffusion.
 *
 *  Two chemicals (u and v) diffuse and react on a 2D grid:
 *      du/dt = Du * laplacian(u) - u*v^2 + f*(1-u)
 *      dv/dt = Dv * laplacian(v) + u*v^2 - (f+k)*v
 *
 *  The discrete Laplacian uses the 4-neighbor stencil.
 *  Different (f, k) values produce spots, stripes, or wave patterns.
 *  Typical: f=0.035, k=0.065 for mitosis-like spots.
 *
 *  Parameters:
 *      u, v          — current concentrations (height x width, float32)
 *      u_next,v_next — output concentrations  (height x width, float32)
 *      width, height — grid size
 *      du, dv        — diffusion rates
 *      f             — feed rate  (replenishes u, removes v)
 *      k             — kill rate  (removes v)
 *      dt            — time step
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
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
