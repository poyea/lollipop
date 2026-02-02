import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import bitonic_sort


def main():
    print("=== Bitonic Sort (Parallel Sorting Network) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 1024
    data = cp.random.default_rng(42).random(n, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = bitonic_sort(data)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {n} elements")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  First 8: {result[:8].get()}")
    print(f"  Last 8:  {result[-8:].get()}")

    expected = np.sort(data.get())
    match = np.allclose(result.get(), expected)
    print(f"  Correct: {match}")


if __name__ == "__main__":
    main()
