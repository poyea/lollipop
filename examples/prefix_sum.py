import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import prefix_sum


def main():
    print("=== Parallel Prefix Sum (Blelloch Scan) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 1024
    data = cp.ones(n, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = prefix_sum(data)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {n} elements (exclusive scan of all ones)")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  First 8: {result[:8].get()}")
    print(f"  Last 8:  {result[-8:].get()}")

    expected = np.arange(n, dtype=np.float32)
    match = np.allclose(result.get(), expected)
    print(f"  Correct: {match}")


if __name__ == "__main__":
    main()
