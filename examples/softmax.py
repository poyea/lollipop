import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import softmax
from lollipop.kernels.softmax_vec4 import softmax_vec4


def _bench(fn, data, iters=5):
    # One warmup, then best-of-iters.
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
    print("=== Fused Online Softmax (FlashAttention-style streaming reduction) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    rows, cols = 2048, 8192
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((rows, cols), dtype=cp.float32) * 5.0

    def _cupy_ref(x):
        shifted = x - x.max(axis=1, keepdims=True)
        e = cp.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    y_ref, t_ref = _bench(_cupy_ref, x)
    y1, t1 = _bench(softmax, x)
    y2, t2 = _bench(softmax_vec4, x)

    bytes_moved = 2 * rows * cols * 4  # one read, one write
    err1 = float(cp.abs(y1 - y_ref).max())
    err2 = float(cp.abs(y2 - y_ref).max())

    print(f"  Shape: ({rows:,}, {cols:,})  ({rows * cols * 4 / 1e6:.1f} MB)")
    print(
        f"  scalar          : {t1 * 1e3:7.2f} ms   "
        f"({bytes_moved / t1 / 1e9:6.1f} GB/s)   max_err={err1:.2e}"
    )
    print(
        f"  vec4 (float4)   : {t2 * 1e3:7.2f} ms   "
        f"({bytes_moved / t2 / 1e9:6.1f} GB/s)   max_err={err2:.2e}"
    )
    print(f"  CuPy (3-kernel) : {t_ref * 1e3:7.2f} ms")
    print(f"  vec4 vs scalar  : {t1 / t2:.2f}x")
    print(f"  vec4 vs CuPy    : {t_ref / t2:.2f}x")

    print("\n  Small-input sanity check:")
    small = cp.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=cp.float32)
    print(f"    softmax_vec4([1,2,3,4]) = {softmax_vec4(small)[0]}")
    print(f"    softmax_vec4([0,0,0,0]) = {softmax_vec4(small)[1]}")


if __name__ == "__main__":
    main()
