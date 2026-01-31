extern "C" __global__
void histogram(const unsigned char* input, unsigned int* bins,
               int n, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int bin = (int)input[i] * num_bins / 256;
    atomicAdd(&bins[bin], 1);
}
