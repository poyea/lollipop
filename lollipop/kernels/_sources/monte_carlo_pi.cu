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
