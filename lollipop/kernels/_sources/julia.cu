/*
 *  Julia set fractal.
 *
 *  Like Mandelbrot, but the constant c is fixed across all pixels and
 *  each pixel provides the initial z(0):
 *      z(n+1) = z(n)^2 + c
 *
 *  Different (c_re, c_im) values produce wildly different patterns.
 *  Classic choice: c = -0.7 + 0.27015i
 *
 *  Parameters:
 *      output       — (width * height) uint8 image
 *      width, height — image dimensions
 *      max_iter     — escape-time iteration limit
 *      c_re, c_im   — real and imaginary parts of the Julia constant
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
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
