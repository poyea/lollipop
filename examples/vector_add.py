import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time


def main():
    print("=== Vector Addition (CPU vs GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 20_000_000
    print(f"Adding two vectors of {n:,} elements\n")

    a_cpu = np.random.rand(n).astype(np.float32)
    b_cpu = np.random.rand(n).astype(np.float32)

    start = time.perf_counter()
    c_cpu = a_cpu + b_cpu
    cpu_time = time.perf_counter() - start
    print(f"  CPU: {cpu_time:.4f}s")

    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    c_gpu = a_gpu + b_gpu
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Speedup: {cpu_time / gpu_time:.1f}x")


if __name__ == "__main__":
    main()
