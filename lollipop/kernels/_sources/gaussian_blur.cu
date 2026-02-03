/*
 *  Gaussian blur — 2D image convolution.
 *
 *  Each thread computes one output pixel as a weighted average of its
 *  neighbors within a square window of size (2*radius+1).  Weights
 *  follow a Gaussian distribution:
 *      w(dx,dy) = exp(-(dx^2 + dy^2) / (2 * sigma^2))
 *  with sigma = radius / 2.
 *
 *  Boundary handling: clamp (edge pixels are repeated).
 *  The weights are normalised per pixel so the output stays in the
 *  same value range as the input.
 *
 *  Parameters:
 *      input  — source image  (height x width, float32)
 *      output — blurred image (height x width, float32)
 *      width, height — image dimensions
 *      radius — blur radius in pixels
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
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
