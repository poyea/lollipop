extern "C" __global__
void nbody(float* px, float* py, float* pz,
           float* vx, float* vy, float* vz,
           float* mass, int n, float dt, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ax = 0, ay = 0, az = 0;

    for (int j = 0; j < n; j++) {
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float dz = pz[j] - pz[i];
        float dist_sqr = dx * dx + dy * dy + dz * dz + softening;
        float inv_dist = rsqrtf(dist_sqr);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float f = mass[j] * inv_dist3;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }

    vx[i] += ax * dt;
    vy[i] += ay * dt;
    vz[i] += az * dt;
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}
