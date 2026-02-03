/*
 *  Exclusive prefix sum (scan) — Blelloch algorithm.
 *
 *  Two-phase work-efficient scan that runs inside a single block:
 *
 *    1. Up-sweep (reduce):   build partial sums bottom-up in shared memory
 *    2. Down-sweep:          propagate prefix sums top-down
 *
 *  Result: data[i] = sum of all elements before index i  (exclusive scan).
 *  Example: [1,2,3,4] -> [0,1,3,6]
 *
 *  Parameters:
 *      data — n float32 values (overwritten with exclusive prefix sums)
 *      n    — array length (must be a power of 2)
 *
 *  Launch: block=(n/2,), grid=(1,), shared_mem = n * sizeof(float)
 */
extern "C" __global__
void prefix_sum_blelloch(float* data, int n) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int offset = 1;

    /* Load input into shared memory */
    temp[2 * tid]     = (2 * tid < n)     ? data[2 * tid]     : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? data[2 * tid + 1] : 0;

    /* Up-sweep (reduce) */
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    /* Set last element to zero for exclusive scan */
    if (tid == 0) temp[n - 1] = 0;

    /* Down-sweep */
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    /* Write results back */
    if (2 * tid < n)     data[2 * tid]     = temp[2 * tid];
    if (2 * tid + 1 < n) data[2 * tid + 1] = temp[2 * tid + 1];
}
