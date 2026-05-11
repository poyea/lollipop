import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop.kernels.shared_reduce_2d import shared_reduce_2d
from lollipop.kernels.shared_reduce_2d_vec4 import shared_reduce_2d_vec4


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
    print("=== 2D Shared-Memory Reduction (scalar vs float4) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    rows, cols = 4096, 8192
    n = rows * cols
    data = cp.ones((rows, cols), dtype=cp.float32)
    bytes_moved = n * 4

    r1, t1 = _bench(shared_reduce_2d, data)
    r2, t2 = _bench(shared_reduce_2d_vec4, data)
    ref = float(data.sum())

    print(f"  shape = ({rows}, {cols})  ({n:,} elements, {bytes_moved / 1e9:.2f} GB)")
    print(
        f"  scalar       : {t1 * 1e3:7.2f} ms   ({bytes_moved / t1 / 1e9:6.1f} GB/s)"
        f"   result={r1:,.0f}"
    )
    print(
        f"  vec4 (float4): {t2 * 1e3:7.2f} ms   ({bytes_moved / t2 / 1e9:6.1f} GB/s)"
        f"   result={r2:,.0f}"
    )
    print(f"  speedup      : {t1 / t2:.2f}x")
    print(
        f"  correct      : scalar={np.isclose(r1, ref, rtol=1e-4)}"
        f"  vec4={np.isclose(r2, ref, rtol=1e-4)}   (expected {ref:,.0f})"
    )

    # Width not divisible by 4 — exercises the tail path.
    data2 = cp.ones((128, 1023), dtype=cp.float32)
    expected2 = float(data2.sum())
    print(
        f"\n  Sum of ones (128 x 1023) : scalar={shared_reduce_2d(data2):,.0f} "
        f"vec4={shared_reduce_2d_vec4(data2):,.0f}  (expected {expected2:,.0f})"
    )


if __name__ == "__main__":
    main()
