extern "C" __global__
void bitonic_sort(float* data, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    /* Load data into shared memory */
    if (tid < n) shared[tid] = data[tid];
    __syncthreads();

    /* Bitonic sort network */
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = tid ^ j;

            if (partner > tid && tid < n && partner < n) {
                /* Ascending if in first half of k-block, descending otherwise */
                int ascending = ((tid & k) == 0);

                float a = shared[tid];
                float b = shared[partner];

                if (ascending ? (a > b) : (a < b)) {
                    shared[tid] = b;
                    shared[partner] = a;
                }
            }
            __syncthreads();
        }
    }

    /* Write back to global memory */
    if (tid < n) data[tid] = shared[tid];
}
