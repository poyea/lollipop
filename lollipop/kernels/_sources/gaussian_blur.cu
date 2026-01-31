extern "C" __global__
void gaussian_blur(const float* input, float* output,
                   int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sigma = (float)radius / 2.0f;
    float sum = 0, weight_sum = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            float w = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            sum += input[ny * width + nx] * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = sum / weight_sum;
}
