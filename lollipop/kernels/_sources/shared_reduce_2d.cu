/*
 *  2D shared-memory tile reduction — sum a 2D grid of floats.
 *
 *  Demonstrates the most common shared-memory pattern in CUDA:
 *    1. Each thread loads one element from global into shared memory
 *    2. Tree reduction within the block using __syncthreads()
 *    3. Thread 0 of each block writes the block's partial sum via atomicAdd
 *
 *  Unlike a 1D reduction, the input is a 2D array (height x width).
 *  Each block handles a tile of up to (BLOCK_W x BLOCK_H) elements.
 *  The 2D thread index is flattened to a 1D index for the reduction tree:
 *      local_id = threadIdx.y * blockDim.x + threadIdx.x
 *
 *  This is simpler than the warp-shuffle reduction kernel — it shows
 *  the shared-memory fundamentals before introducing warp intrinsics.
 *
 *  Parameters:
 *      input  — (height x width) float32 matrix
 *      output — single float32 accumulator (caller zeroes it)
 *      width  — number of columns
 *      height — number of rows
 *
 *  Launch: block=(16,16), grid=((w+15)/16, (h+15)/16)
 *          shared_mem = 16*16 * sizeof(float)
 */
extern "C" __global__
void shared_reduce_2d(const float* input, float* output, int width, int height) {
    extern __shared__ float sdata[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    /* Load from global; out-of-bounds threads contribute zero */
    sdata[local_id] = (x < width && y < height) ? input[y * width + x] : 0.0f;
    __syncthreads();

    /* Tree reduction in shared memory */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] += sdata[local_id + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes block partial sum to global accumulator */
    if (local_id == 0) {
        atomicAdd(output, sdata[0]);
    }
}
