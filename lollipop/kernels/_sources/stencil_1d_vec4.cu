/*
 *  1D stencil, float4-vectorized variant of `stencil_1d`.
 *
 *  What changed vs. the scalar baseline (stencil_1d.cu):
 *  ----------------------------------------------------
 *  Each thread now owns 4 contiguous output lanes instead of 1.  The block
 *  tile covers 4 * blockDim.x floats (1024 for blockDim.x = 256), and the
 *  interior of the tile is loaded with one `float4` per thread (one
 *  LDG.E.128 instead of four LDG.E.32).  Each thread also writes a single
 *  `float4` to global memory at the end.
 *
 *  Why it should be faster:
 *    - 4x fewer global load instructions for the interior load;
 *    - 4x fewer global store instructions for the output;
 *    - 4x smaller grid, so 4x fewer block-launch / boundary-handling
 *      overhead per byte of work done.
 *
 *  Halo handling — the interesting bit:
 *  ------------------------------------
 *  The interior of the tile is a clean float4 load.  The 2*radius halo
 *  cells on each side of the tile do NOT line up with float4 boundaries
 *  in general, so we load them with *scalar* loads from the first/last
 *  `radius` threads of the block.  This means the per-block halo work is
 *  the same as in the scalar baseline (and there's nothing to vectorize
 *  there — it's `radius` scalars, not a multiple of 4 in general).
 *
 *  Therefore the expected speedup is bounded: the interior load+store is
 *  4x cheaper, but the halo loads are unchanged.  For radius=3 and
 *  blockDim.x=256 we save 1024 float loads (now 256 float4 loads) but
 *  still issue 6 scalar halo loads per block — negligible.  The bigger
 *  caveat is that the *compute* per thread is now 4x (each thread runs
 *  the (2*radius+1)-tap loop four times), and on a memory-bound kernel
 *  that doesn't hurt, but it also doesn't help.
 *
 *  Shared memory layout (same shape as baseline, just bigger tile):
 *      [halo_left | tile_interior (4 * blockDim.x) | halo_right]
 *      total = 4 * blockDim.x + 2 * radius floats
 *
 *  Boundary: clamp to edge values (out-of-bounds reads use in[0] or in[n-1]).
 *
 *  Parameters:
 *      input  — n float32 values  (must be 16-byte aligned, n % 4 == 0)
 *      output — n float32 smoothed values
 *      n      — array length (must be a multiple of 4)
 *      radius — stencil half-width
 *
 *  Launch: block=(256,), grid=((n/4 + 255) / 256,),
 *          shared_mem = (4 * 256 + 2 * radius) * sizeof(float)
 */
extern "C" __global__
void stencil_1d_vec4(const float4* __restrict__ input_v,
                     float4* __restrict__ output_v,
                     int n, int radius) {
    extern __shared__ float smem[];

    const int lid = threadIdx.x;
    const int tile = blockDim.x * 4;                 /* floats per block tile */
    const int block_start = blockIdx.x * tile;       /* float index */
    const int gid4 = blockIdx.x * blockDim.x + lid;  /* float4 index */
    const int n_vec4 = n / 4;

    /* Shared layout: [radius | tile | radius]; interior offset = radius. */
    const int s_interior = radius;

    /* Load interior as one float4 per thread. */
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gid4 < n_vec4) {
        v = input_v[gid4];
    }
    const int s_base = s_interior + lid * 4;
    smem[s_base + 0] = v.x;
    smem[s_base + 1] = v.y;
    smem[s_base + 2] = v.z;
    smem[s_base + 3] = v.w;

    /* Reinterpret input as scalar pointer for halo loads. */
    const float* input = reinterpret_cast<const float*>(input_v);

    /* Left halo: first `radius` threads load one scalar each. */
    if (lid < radius) {
        int left = block_start - radius + lid;
        smem[lid] = (left >= 0) ? input[left] : input[0];
    }

    /* Right halo: last `radius` threads load one scalar each. */
    if (lid >= blockDim.x - radius) {
        int offset_in_halo = lid - (blockDim.x - radius);  /* 0 .. radius-1 */
        int right = block_start + tile + offset_in_halo;
        int clamped = (right < n) ? right : (n - 1);
        smem[s_interior + tile + offset_in_halo] = input[clamped];
    }

    __syncthreads();

    if (gid4 >= n_vec4) return;

    /* Compute 4 output lanes from shared memory. */
    const float inv = 1.0f / (float)(2 * radius + 1);
    float4 out;
    float* out_arr = reinterpret_cast<float*>(&out);
    #pragma unroll
    for (int lane = 0; lane < 4; lane++) {
        float sum = 0.0f;
        int center = s_base + lane;
        for (int offset = -radius; offset <= radius; offset++) {
            sum += smem[center + offset];
        }
        out_arr[lane] = sum * inv;
    }
    output_v[gid4] = out;
}
