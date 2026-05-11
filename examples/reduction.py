import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import reduction, reduction_vec4


def _bench(fn, data, iters=5):
    # One warmup, then take the min over iters runs.
    fn(data)
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        result = fn(data)
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - start)
    return result, best


def main():
    print("=== Parallel Reduction (Warp Shuffle Intrinsics) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 100_000_000
    data = cp.ones(n, dtype=cp.float32)
    bytes_moved = n * 4

    r1, t1 = _bench(reduction, data)
    r2, t2 = _bench(reduction_vec4, data)

    print(f"  {n:,} elements (sum of all ones)")
    print(
        f"  scalar       : {t1 * 1e3:7.2f} ms   ({bytes_moved / t1 / 1e9:6.1f} GB/s)"
        f"   result={r1:,.0f}"
    )
    print(
        f"  vec4 (float4): {t2 * 1e3:7.2f} ms   ({bytes_moved / t2 / 1e9:6.1f} GB/s)"
        f"   result={r2:,.0f}"
    )
    print(f"  speedup      : {t1 / t2:.2f}x")
    print(f"  correct      : scalar={np.isclose(r1, n)}  vec4={np.isclose(r2, n)}")

    data2 = cp.arange(1, 1001, dtype=cp.float32)
    expected2 = 1000 * 1001 / 2
    print(
        f"\n  Sum of 1..1000   : scalar={reduction(data2):,.0f} "
        f"vec4={reduction_vec4(data2):,.0f}  (expected {expected2:,.0f})"
    )

    # Length not divisible by 4 — exercises the tail path.
    data3 = cp.arange(1, 1003, dtype=cp.float32)  # 1002 elements
    expected3 = 1002 * 1003 / 2
    print(
        f"  Sum of 1..1002   : scalar={reduction(data3):,.0f} "
        f"vec4={reduction_vec4(data3):,.0f}  (expected {expected3:,.0f})"
    )


if __name__ == "__main__":
    main()
