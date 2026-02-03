/*
 *  Monte Carlo European option pricing.
 *
 *  Each thread simulates multiple GBM terminal stock prices and
 *  accumulates discounted payoffs:
 *      S_T = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)
 *      call payoff = max(S_T - K, 0)
 *      put  payoff = max(K - S_T, 0)
 *
 *  Thread-local payoff sums are added to global accumulators via
 *  atomicAdd.  The caller divides by total paths and discounts by
 *  e^(-rT) to get the option price.
 *
 *  Uses xorshift32 RNG + Box-Muller transform for normal samples.
 *
 *  Parameters:
 *      results          — float32[2]: [call_payoff_sum, put_payoff_sum]
 *      S0, K, r, sigma, T — option contract parameters
 *      paths_per_thread — how many paths each thread simulates
 *      num_threads      — total number of threads
 *      seed             — RNG seed
 *
 *  Launch: block=(256,), grid=((num_threads+255)/256,)
 */
extern "C" __global__
void monte_carlo_option(float* results, float S0, float K, float r,
                        float sigma, float T, int paths_per_thread,
                        int num_threads, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    /* Per-thread xorshift32 RNG */
    unsigned int rng = (unsigned int)tid * 1099087573u + seed;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

    float drift = (r - 0.5f * sigma * sigma) * T;
    float diffusion = sigma * sqrtf(T);

    float call_sum = 0.0f;
    float put_sum = 0.0f;

    for (int p = 0; p < paths_per_thread; p++) {
        /* Box-Muller: generate standard normal from two uniforms */
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u1 = (float)(rng & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
        u1 = fmaxf(u1, 1e-10f);
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u2 = (float)(rng & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;

        float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);

        float ST = S0 * expf(drift + diffusion * z);
        call_sum += fmaxf(ST - K, 0.0f);
        put_sum += fmaxf(K - ST, 0.0f);
    }

    /* results[0] = call payoff sum, results[1] = put payoff sum */
    atomicAdd(&results[0], call_sum);
    atomicAdd(&results[1], put_sum);
}
