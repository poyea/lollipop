/*
 *  Bitonic sort — data-parallel sorting network.
 *
 *  Works entirely in shared memory within a single thread block.
 *  The algorithm builds increasingly larger bitonic sequences and
 *  merges them:
 *
 *    for each k = 2, 4, 8, ... n:        (bitonic sequence length)
 *      for each j = k/2, k/4, ... 1:     (compare-and-swap distance)
 *        each thread compares element[tid] with element[tid ^ j]
 *        and swaps if they are out of order.
 *
 *  All threads sync between each compare-and-swap pass.
 *  O(n log^2 n) comparisons, fully parallel.
 *
 *  Parameters:
 *      data — n float32 values (sorted in-place, ascending)
 *      n    — array length (must be power of 2, max 1024)
 *
 *  Launch: block=(n,), grid=(1,), shared_mem = n * sizeof(float)
 */
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
