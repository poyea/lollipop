extern "C" __global__
void julia(unsigned char* output, int width, int height, int max_iter,
           float c_re, float c_im) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float zx = (float)x / width * 3.5f - 1.75f;
    float zy = (float)y / height * 2.0f - 1.0f;
    int iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < max_iter) {
        float tmp = zx * zx - zy * zy + c_re;
        zy = 2.0f * zx * zy + c_im;
        zx = tmp;
        iter++;
    }

    output[y * width + x] = (unsigned char)(255 * iter / max_iter);
}
