/*
 *  N-body gravitational simulation.
 *
 *  Each thread computes the total gravitational acceleration on body i
 *  from all other bodies using Newton's law:
 *      a_i = sum_j  m_j * (r_j - r_i) / |r_j - r_i|^3
 *
 *  A softening factor prevents singularities when bodies get close.
 *  After computing the acceleration, velocity and position are updated
 *  using symplectic Euler integration (kick-drift).
 *
 *  Parameters:
 *      px,py,pz    — position arrays (n float32, updated in-place)
 *      vx,vy,vz    — velocity arrays (n float32, updated in-place)
 *      mass        — mass array (n float32)
 *      n           — number of bodies
 *      dt          — time step
 *      softening   — softening length squared added to distance
 *
 *  Launch: block=(256,), grid=((n+255)/256,)
 */
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
