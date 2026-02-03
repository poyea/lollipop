/*
 *  Mandelbrot set fractal.
 *
 *  Each thread maps one pixel to the complex plane and iterates
 *      z(n+1) = z(n)^2 + c
 *  starting from z(0) = 0,  where c = pixel coordinate.
 *  The iteration count (scaled to 0-255) becomes the pixel brightness.
 *
 *  Parameters:
 *      output   — (width * height) uint8 image
 *      width, height — image dimensions
 *      max_iter — escape-time iteration limit
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
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
