/*
 *  Monte Carlo estimation of pi.
 *
 *  Each thread samples random (x,y) points in the square [-1,1]^2
 *  and counts how many fall inside the unit circle (x^2 + y^2 <= 1).
 *
 *      pi ~ 4 * (points inside circle) / (total points)
 *
 *  Uses a linear congruential generator (LCG) for fast per-thread
 *  random state.  Thread-local counts are added to a global counter
 *  via atomicAdd.
 *
 *  Parameters:
 *      counts            — uint32[1] accumulator (caller zeroes it)
 *      samples_per_thread — how many (x,y) pairs each thread tests
 *      seed              — RNG seed
 *
 *  Launch: block=(256,), grid=((num_threads+255)/256,)
 */
extern "C" __global__
void monte_carlo_pi(unsigned int* counts, int samples_per_thread, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long state = seed ^ ((unsigned long long)tid * 6364136223846793005ULL + 1);
    unsigned int inside = 0;

    for (int i = 0; i < samples_per_thread; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float x = (float)(state >> 33) / (float)(1ULL << 31) - 1.0f;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float y = (float)(state >> 33) / (float)(1ULL << 31) - 1.0f;

        if (x * x + y * y <= 1.0f)
            inside++;
    }

    atomicAdd(&counts[0], inside);
}
