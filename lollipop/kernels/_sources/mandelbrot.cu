extern "C" __global__
void mandelbrot(unsigned char* output, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cx = (float)x / width * 3.5f - 2.5f;
    float cy = (float)y / height * 2.0f - 1.0f;
    float zx = 0, zy = 0;
    int iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < max_iter) {
        float tmp = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    output[y * width + x] = (unsigned char)(255 * iter / max_iter);
}
