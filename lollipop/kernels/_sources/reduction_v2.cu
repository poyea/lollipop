/*
 *  Occupancy-tuned variants of `reduction`.
 *
 *  Four flavours, identical math (load N elements -> per-thread partial
 *  -> smem tree -> warp shuffle -> atomicAdd into the global accumulator),
 *  differing only in:
 *
 *    - threads per block
 *    - elements per thread (ITEMS)
 *    - `__launch_bounds__(threads, min_blocks_per_sm)` — the second arg
 *      tells the compiler to bound register usage so at least that many
 *      blocks can be resident on an SM.
 *
 *  ITEMS > 2 (the baseline) reduces grid size, which reduces atomicAdd
 *  contention and lets each thread amortise its smem-tree work over
 *  more bytes.  The trade-off is more registers per thread for the
 *  partial sum chain and (potentially) lower occupancy.  The benchmark
 *  in `bench_reduction_v2.py` shows ITEMS=8 wins on Turing.
 *
 *  Launch (per variant): block=(THREADS,), grid=((n + THREADS*ITEMS - 1)
 *  / (THREADS*ITEMS),), shared_mem = THREADS * sizeof(float).
 */

/* Per-block reduction body shared by all variants.
 *
 *  `partial`  — this thread's running partial sum (already includes
 *               however many global loads the variant chose to do).
 *  `tid`      — threadIdx.x
 *  `sdata`    — shared memory of size blockDim.x floats. */
template <int THREADS>
__device__ __forceinline__
void block_reduce_and_atomic(float partial, int tid, float* sdata,
                             float* output) {
    sdata[tid] = partial;
    __syncthreads();

    #pragma unroll
    for (int s = THREADS / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = sdata[tid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (tid == 0) atomicAdd(output, v);
    }
}

template <int THREADS, int ITEMS>
__device__ __forceinline__
void reduction_v2_body(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * (THREADS * ITEMS) + tid;

    float v = 0.0f;
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        int idx = base + k * THREADS;
        if (idx < n) v += input[idx];
    }
    block_reduce_and_atomic<THREADS>(v, tid, sdata, output);
}

extern "C" __global__
__launch_bounds__(128, 8)
void reduction_v2_t128_i8(const float* input, float* output, int n) {
    reduction_v2_body<128, 8>(input, output, n);
}

extern "C" __global__
__launch_bounds__(256, 4)
void reduction_v2_t256_i8(const float* input, float* output, int n) {
    reduction_v2_body<256, 8>(input, output, n);
}

extern "C" __global__
__launch_bounds__(512, 2)
void reduction_v2_t512_i8(const float* input, float* output, int n) {
    reduction_v2_body<512, 8>(input, output, n);
}

extern "C" __global__
__launch_bounds__(1024, 1)
void reduction_v2_t1024_i8(const float* input, float* output, int n) {
    reduction_v2_body<1024, 8>(input, output, n);
}
