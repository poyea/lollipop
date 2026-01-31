import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time


def main():
    print("=== Matrix Multiplication (CPU vs GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    w = cp.zeros((2, 2), dtype=cp.float32)
    w @ w
    cp.cuda.Stream.null.synchronize()

    size = 4096
    print(f"Multiplying two {size}x{size} matrices\n")

    m_cpu = np.random.rand(size, size).astype(np.float32)

    start = time.perf_counter()
    r_cpu = m_cpu @ m_cpu
    cpu_time = time.perf_counter() - start
    print(f"  CPU: {cpu_time:.4f}s")

    m_gpu = cp.asarray(m_cpu)
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    r_gpu = m_gpu @ m_gpu
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Speedup: {cpu_time / gpu_time:.1f}x")


if __name__ == "__main__":
    main()
