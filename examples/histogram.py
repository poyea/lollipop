import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import histogram


def main():
    print("=== GPU Histogram (Atomic Operations) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 50_000_000
    data = cp.random.randint(0, 256, size=n, dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    bins = histogram(data, num_bins=256)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {n:,} elements, 256 bins")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Sum of bins: {int(bins.sum()):,} (expected {n:,})")

    host_bins = bins.get()
    top_5 = np.argsort(host_bins)[-5:][::-1]
    print(f"  Top 5 bins: {', '.join(f'{b}={host_bins[b]}' for b in top_5)}")


if __name__ == "__main__":
    main()
