/*
 *  Histogram of uint8 data via atomic operations.
 *
 *  Each thread reads one element, computes its bin index, and
 *  atomically increments the corresponding bin counter.  Simple but
 *  correct — atomicAdd serialises concurrent updates to the same bin.
 *
 *  Bin formula: bin = value * num_bins / 256
 *
 *  Parameters:
 *      input    — n uint8 values
 *      bins     — num_bins uint32 counters (caller zeroes them)
 *      n        — number of input elements
 *      num_bins — number of histogram bins
 *
 *  Launch: block=(256,), grid=((n+255)/256,)
 */
extern "C" __global__
void histogram(const unsigned char* input, unsigned int* bins,
               int n, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int bin = (int)input[i] * num_bins / 256;
    atomicAdd(&bins[bin], 1);
}
