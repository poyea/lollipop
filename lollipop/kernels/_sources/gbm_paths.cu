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
