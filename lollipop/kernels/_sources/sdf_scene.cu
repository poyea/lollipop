/*
 *  Ray-marched SDF (Signed Distance Function) scene.
 *
 *  Each thread casts one ray from a camera through a pixel and marches
 *  it through the scene using sphere tracing:
 *      repeat: t += SDF(ray_origin + t * ray_dir)
 *
 *  The scene contains three primitives:
 *    - Sphere  (radius 1.0 at origin)
 *    - Box     (0.8 half-size at x=2)
 *    - Torus   (R=1.5, r=0.4 at y=1.5)
 *
 *  The scene SDF is the minimum of all primitive SDFs (union).
 *  Shading is distance-based: closer surfaces appear brighter.
 *
 *  Parameters:
 *      output — (width * height) uint8 grayscale image
 *      width, height — image dimensions
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 */
extern "C" __global__
void sdf_scene(unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x / width * 2.0f - 1.0f) * (float)width / height;
    float v = (float)y / height * 2.0f - 1.0f;

    float ox = 0, oy = 0, oz = -3.0f;
    float dx = u, dy = v, dz = 1.5f;
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= len; dy /= len; dz /= len;

    float t = 0;
    for (int i = 0; i < 128; i++) {
        float px = ox + dx * t;
        float py = oy + dy * t;
        float pz = oz + dz * t;

        float sphere = sqrtf(px * px + py * py + pz * pz) - 1.0f;
        float box = fmaxf(fmaxf(fabsf(px - 2.0f) - 0.8f,
                                fabsf(py) - 0.8f),
                          fabsf(pz) - 0.8f);
        float torus_q = sqrtf(px * px + pz * pz) - 1.5f;
        float torus = sqrtf(torus_q * torus_q + (py - 1.5f) * (py - 1.5f)) - 0.4f;

        float d = fminf(fminf(sphere, box), torus);

        if (d < 0.001f) {
            float shade = 1.0f - t / 10.0f;
            output[y * width + x] = (unsigned char)(fmaxf(shade, 0.0f) * 255);
            return;
        }
        t += d;
        if (t > 10.0f) break;
    }

    output[y * width + x] = 0;
}
