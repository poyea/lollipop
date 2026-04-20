import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import softmax


def main():
    print("=== Fused Online Softmax (FlashAttention-style streaming reduction) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    rows, cols = 2048, 8192
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((rows, cols), dtype=cp.float32) * 5.0

    # Warmup.
    softmax(x)
    cp.exp(x - x.max(axis=1, keepdims=True))
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    y = softmax(x)
    cp.cuda.Stream.null.synchronize()
    fused = time.perf_counter() - start

    start = time.perf_counter()
    shifted = x - x.max(axis=1, keepdims=True)
    e = cp.exp(shifted)
    y_ref = e / e.sum(axis=1, keepdims=True)
    cp.cuda.Stream.null.synchronize()
    cupy_time = time.perf_counter() - start

    max_err = float(cp.abs(y - y_ref).max())
    row_sums = y.sum(axis=1)
    sum_err = float(cp.abs(row_sums - 1.0).max())

    bytes_moved = 2 * rows * cols * 4  # one read, one write
    bandwidth = bytes_moved / fused / 1e9

    print(f"  Shape: ({rows:,}, {cols:,})  ({rows * cols * 4 / 1e6:.1f} MB)")
    print(f"  Fused softmax: {fused * 1e3:.2f} ms   ({bandwidth:.1f} GB/s effective)")
    print(f"  CuPy (3-kernel) ref: {cupy_time * 1e3:.2f} ms")
    print(f"  Speedup vs CuPy: {cupy_time / fused:.2f}x")
    print(f"  Max |y - y_ref|: {max_err:.2e}")
    print(f"  Max |row_sum - 1|: {sum_err:.2e}")

    print("\n  Small-input sanity check:")
    small = cp.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=cp.float32)
    print(f"    softmax([1,2,3,4]) = {softmax(small)[0]}")
    print(f"    softmax([0,0,0,0]) = {softmax(small)[1]}")


if __name__ == "__main__":
    main()
