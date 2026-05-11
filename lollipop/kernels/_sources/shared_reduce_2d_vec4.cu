/*
 *  2D shared-memory tile reduction, float4-vectorized variant of
 *  `shared_reduce_2d`.
 *
 *  What changed vs. the scalar baseline (shared_reduce_2d.cu):
 *  ----------------------------------------------------------
 *  Each thread now reads one `float4` from the row (4 contiguous floats)
 *  instead of a single scalar.  A single `float4` load compiles to one
 *  LDG.E.128 (128-bit global load) instead of four LDG.E.32, which:
 *
 *    - issues 4x fewer load instructions across the block;
 *    - keeps memory transactions naturally 16-byte aligned and coalesced
 *      across a warp (the 32 threads in a warp move 32 * 16 = 512 B in
 *      one transaction along the row);
 *    - reduces address-generation work per byte moved.
 *
 *  Each thread sums its float4 (`a.x + a.y + a.z + a.w`) into a scalar
 *  partial *before* writing to shared memory, so the smem tree reduction
 *  (and its size, BLOCK_W * BLOCK_H floats) is unchanged.
 *
 *  The block now covers a tile of (4 * BLOCK_W) floats wide by BLOCK_H
 *  floats tall (64 x 16 with BLOCK_W=BLOCK_H=16), vs (16 x 16) in the
 *  scalar version.  The grid is therefore 4x smaller in x, which also
 *  means 4x fewer atomicAdds into the global accumulator.
 *
 *  The caller must pass a width that is a multiple of 4; the Python
 *  wrapper handles the (width % 4) tail columns on the host side via
 *  CuPy.  `width_vec4 = width / 4` is the number of float4 elements
 *  along each row, and `pitch_vec4` is the row stride in float4 units
 *  (== width_vec4 for contiguous input).
 *
 *  Parameters:
 *      input       — (height x width_vec4) float4 matrix
 *      output      — single float32 accumulator (caller zeroes it)
 *      width_vec4  — number of float4 elements per row (= width / 4)
 *      height      — number of rows
 *
 *  Launch: block=(16,16), grid=((width_vec4+15)/16, (height+15)/16)
 *          shared_mem = 16*16 * sizeof(float)
 */
extern "C" __global__
void shared_reduce_2d_vec4(const float4* __restrict__ input, float* output,
                           int width_vec4, int height) {
    extern __shared__ float sdata[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    /* One float4 load per thread -> 4 floats summed into a scalar partial. */
    float val = 0.0f;
    if (x < width_vec4 && y < height) {
        float4 a = input[y * width_vec4 + x];
        val = a.x + a.y + a.z + a.w;
    }
    sdata[local_id] = val;
    __syncthreads();

    /* Tree reduction in shared memory (identical to the scalar baseline). */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] += sdata[local_id + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes block partial sum to global accumulator. */
    if (local_id == 0) {
        atomicAdd(output, sdata[0]);
    }
}
