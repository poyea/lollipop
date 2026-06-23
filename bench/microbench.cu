/*
 *  Native CUDA microbench -- the GPU-side counterpart to the Python bench
 *  scripts in this folder.
 *
 *  Why this exists: the Python benches time with `cp.cuda.Event` and lean on
 *  CuPy to reach the library baselines (cp.sum -> CUB, cp.sort -> Thrust,
 *  cp.matmul -> cuBLAS).  This harness does the same measurement with no
 *  Python in the loop at all: it `#include`s the kernel sources verbatim,
 *  times them with raw `cudaEvent_t`, and links the NVIDIA libraries
 *  directly -- Thrust (`thrust::reduce`) as the reduction baseline and
 *  cuBLAS (`cublasSgeam`) as the transpose baseline.  Same honesty (ratio
 *  vs library), measured the way you'd measure inside a profiler workflow.
 *
 *  Only `ncu`/`nsys` are absent on this rig; `cudaEvent` timing needs
 *  neither.  Covers the two kernels that are pure (no Python-side
 *  orchestration) so the native path measures exactly what ships:
 *  `reduction` and `matrix_transpose`.
 *
 *  Build (sm_75 = the test GPU's arch):
 *      nvcc -O3 -arch=sm_75 bench/microbench.cu -o bench/microbench -lcublas
 *  Run:
 *      ./bench/microbench            (Linux)
 *      .\bench\microbench.exe        (Windows)
 *
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

/* The kernels under test, included verbatim from the shipping sources so the
 * native bench measures the exact same code the CuPy wrappers nvrtc-compile. */
#include "../lollipop/kernels/_sources/reduction.cu"
#include "../lollipop/kernels/_sources/matrix_transpose.cu"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e_));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t s_ = (x);                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS %s:%d  status %d\n", __FILE__, __LINE__,   \
                    (int)s_);                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* Mean GPU-side ms per call, timed with CUDA events (warmup excluded). */
template <class Launch>
static float time_ms(Launch launch, int iters, int warmup) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / iters;
}

/* ----------------------------- reduction ------------------------------ */

static void bench_reduction() {
    const int n = 100'000'000;
    const double gb = (double)n * 4 / 1e9;

    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 7) - 3) * 0.5f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&d_out, 4));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block * 2 - 1) / (block * 2);
    const size_t smem = block * sizeof(float);

    auto mine = [&] {
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, 4));
        reduction<<<grid, block, smem>>>(d_in, d_out, n);
    };
    thrust::device_ptr<float> tp(d_in);
    auto lib = [&] { volatile float r = thrust::reduce(thrust::device, tp, tp + n, 0.0f); (void)r; };

    /* parity */
    mine();
    float s_mine = 0.0f;
    CUDA_CHECK(cudaMemcpy(&s_mine, d_out, 4, cudaMemcpyDeviceToHost));
    float s_lib = thrust::reduce(thrust::device, tp, tp + n, 0.0f);

    float t_mine = time_ms(mine, 50, 5);
    float t_lib = time_ms(lib, 50, 5);

    printf("\nreduction  (mine  vs  thrust::reduce)\n");
    printf("  n=%dM   mine %7.3f ms  %6.1f GB/s     lib %7.3f ms  %6.1f GB/s   mine/lib %4.2fx\n",
           n / 1'000'000, t_mine, gb / (t_mine * 1e-3), t_lib, gb / (t_lib * 1e-3),
           t_lib / t_mine);
    printf("  parity: mine=%.3f  lib=%.3f  (fp32 atomic vs thrust accum)\n", s_mine, s_lib);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

/* ----------------------------- transpose ------------------------------ */

static void bench_transpose(cublasHandle_t blas, int H, int W) {
    const double bytes = 2.0 * H * W * 4;  /* read + write */

    std::vector<float> h((size_t)H * W);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (float)(i % 101);

    float *d_in = nullptr, *d_mine = nullptr, *d_lib = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)H * W * 4));
    CUDA_CHECK(cudaMalloc(&d_mine, (size_t)H * W * 4));
    CUDA_CHECK(cudaMalloc(&d_lib, (size_t)H * W * 4));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), (size_t)H * W * 4, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, BLOCK_ROWS);  /* (32, 8) from matrix_transpose.cu */
    dim3 grid((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);
    auto mine = [&] { matrix_transpose<<<grid, block>>>(d_in, d_mine, W, H); };

    /* cuBLAS transpose of a row-major HxW matrix: the row-major buffer is a
     * column-major WxH matrix; its transpose is the row-major WxH we want. */
    const float one = 1.0f, zero = 0.0f;
    auto lib = [&] {
        CUBLAS_CHECK(cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, H, W, &one,
                                 d_in, W, &zero, d_lib, H, d_lib, H));
    };

    /* parity: compare a handful of elements */
    mine();
    lib();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> a((size_t)H * W), b((size_t)H * W);
    CUDA_CHECK(cudaMemcpy(a.data(), d_mine, (size_t)H * W * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_lib, (size_t)H * W * 4, cudaMemcpyDeviceToHost));
    int mism = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { ++mism; if (mism > 5) break; }

    float t_mine = time_ms(mine, 30, 3);
    float t_lib = time_ms(lib, 30, 3);

    printf("  %dx%d   mine %7.3f ms  %6.1f GB/s     cuBLAS %7.3f ms  %6.1f GB/s   mine/cuBLAS %4.2fx  (mismatch=%d)\n",
           H, W, t_mine, bytes / (t_mine * 1e-3) / 1e9, t_lib,
           bytes / (t_lib * 1e-3) / 1e9, t_lib / t_mine, mism);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_mine));
    CUDA_CHECK(cudaFree(d_lib));
}

int main() {
    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    bench_reduction();
    printf("\nmatrix_transpose  (mine  vs  cublasSgeam)\n");
    bench_transpose(blas, 4096, 4096);
    bench_transpose(blas, 8192, 8192);

    CUBLAS_CHECK(cublasDestroy(blas));
    return 0;
}
