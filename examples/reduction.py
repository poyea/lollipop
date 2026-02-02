import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import reduction


def main():
    print("=== Parallel Reduction (Warp Shuffle Intrinsics) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 10_000_000
    data = cp.ones(n, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = reduction(data)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {n:,} elements (sum of all ones)")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Result: {result:,.0f}")
    print(f"  Expected: {n:,}")
    print(f"  Correct: {np.isclose(result, float(n))}")

    data2 = cp.arange(1, 1001, dtype=cp.float32)
    result2 = reduction(data2)
    expected2 = 1000 * 1001 / 2
    print(f"\n  Sum of 1..1000: {result2:,.0f} (expected {expected2:,.0f})")


if __name__ == "__main__":
    main()
