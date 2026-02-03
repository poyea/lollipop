/*
 *  Geometric Brownian Motion (GBM) path generation.
 *
 *  Each thread simulates one independent price path using the exact
 *  log-normal discretisation:
 *      S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
 *  where Z ~ N(0,1) is generated via Box-Muller transform.
 *
 *  The xorshift32 PRNG provides fast per-thread random state.
 *  Box-Muller converts two uniform samples into a normal sample:
 *      Z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
 *
 *  Parameters:
 *      paths     — output array (num_paths x (num_steps+1), float32)
 *      S0        — initial asset price
 *      mu        — drift (expected annual return)
 *      sigma     — volatility (annual)
 *      dt        — time step (T / num_steps)
 *      num_steps — number of time steps per path
 *      num_paths — number of independent paths
 *      seed      — RNG seed
 *
 *  Launch: block=(256,), grid=((num_paths+255)/256,)
 */
extern "C" __global__
void gbm_paths(float* paths, float S0, float mu, float sigma,
               float dt, int num_steps, int num_paths, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    /* Per-thread xorshift32 RNG */
    unsigned int rng = (unsigned int)tid * 1099087573u + seed;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

    float drift = (mu - 0.5f * sigma * sigma) * dt;
    float diffusion = sigma * sqrtf(dt);

    int stride = num_steps + 1;
    float S = S0;
    paths[tid * stride] = S;

    for (int step = 1; step <= num_steps; step++) {
        /* Box-Muller: generate standard normal from two uniforms */
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u1 = (float)(rng & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
        u1 = fmaxf(u1, 1e-10f);
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u2 = (float)(rng & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;

        float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);

        S = S * expf(drift + diffusion * z);
        paths[tid * stride + step] = S;
    }
}
