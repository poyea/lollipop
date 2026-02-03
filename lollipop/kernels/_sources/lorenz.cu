extern "C" __global__
void lorenz(float* out_x, float* out_y, float* out_z,
            int num_trajectories, int num_steps, float dt,
            float sigma, float rho, float beta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_trajectories) return;

    /* Slightly perturbed initial conditions per trajectory */
    float x = 1.0f + 0.01f * (float)(tid - num_trajectories / 2);
    float y = 1.0f;
    float z = 1.0f;

    int stride = num_steps + 1;
    out_x[tid * stride] = x;
    out_y[tid * stride] = y;
    out_z[tid * stride] = z;

    for (int step = 1; step <= num_steps; step++) {
        /* RK4 integration */
        float k1x = sigma * (y - x);
        float k1y = x * (rho - z) - y;
        float k1z = x * y - beta * z;

        float mx = x + 0.5f * dt * k1x;
        float my = y + 0.5f * dt * k1y;
        float mz = z + 0.5f * dt * k1z;

        float k2x = sigma * (my - mx);
        float k2y = mx * (rho - mz) - my;
        float k2z = mx * my - beta * mz;

        mx = x + 0.5f * dt * k2x;
        my = y + 0.5f * dt * k2y;
        mz = z + 0.5f * dt * k2z;

        float k3x = sigma * (my - mx);
        float k3y = mx * (rho - mz) - my;
        float k3z = mx * my - beta * mz;

        mx = x + dt * k3x;
        my = y + dt * k3y;
        mz = z + dt * k3z;

        float k4x = sigma * (my - mx);
        float k4y = mx * (rho - mz) - my;
        float k4z = mx * my - beta * mz;

        x += dt / 6.0f * (k1x + 2.0f * k2x + 2.0f * k3x + k4x);
        y += dt / 6.0f * (k1y + 2.0f * k2y + 2.0f * k3y + k4y);
        z += dt / 6.0f * (k1z + 2.0f * k2z + 2.0f * k3z + k4z);

        out_x[tid * stride + step] = x;
        out_y[tid * stride + step] = y;
        out_z[tid * stride + step] = z;
    }
}
