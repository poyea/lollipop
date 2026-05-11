import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop.kernels.stencil_1d import stencil_1d
from lollipop.kernels.stencil_1d_vec4 import stencil_1d_vec4


def _bench(fn, data, iters=5):
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
    print("=== 1D Stencil (float4 vs scalar) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 32 * 1024 * 1024  # 32M floats = 128 MB input + 128 MB output
    radius = 3
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)
    # Bytes moved: one read + one write of the array (halo is amortised tiny).
    bytes_moved = 2 * n * 4

    r1, t1 = _bench(lambda d: stencil_1d(d, radius=radius), data)
    r2, t2 = _bench(lambda d: stencil_1d_vec4(d, radius=radius), data)

    max_err = float(cp.max(cp.abs(r1 - r2)))
    print(f"  n = {n:,} floats, radius = {radius}")
    print(f"  scalar       : {t1 * 1e3:7.2f} ms   ({bytes_moved / t1 / 1e9:6.1f} GB/s)")
    print(f"  vec4 (float4): {t2 * 1e3:7.2f} ms   ({bytes_moved / t2 / 1e9:6.1f} GB/s)")
    print(f"  speedup      : {t1 / t2:.2f}x")
    print(f"  max abs err  : {max_err:.3e}")

    # Sanity on a tiny non-divisible-by-4 length.
    data2 = cp.arange(1, 18, dtype=cp.float32)  # 17 elements
    s1 = stencil_1d(data2, radius=radius)
    s2 = stencil_1d_vec4(data2, radius=radius)
    print(f"\n  n=17 tail-path : max abs err = {float(cp.max(cp.abs(s1 - s2))):.3e}")


if __name__ == "__main__":
    main()
