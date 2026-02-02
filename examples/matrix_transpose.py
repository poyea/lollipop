import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import matrix_transpose


def main():
    print("=== Matrix Transpose (Tiled Shared Memory) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    rows, cols = 4096, 2048
    matrix = cp.random.default_rng(42).random((rows, cols), dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = matrix_transpose(matrix)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {rows}x{cols} -> {result.shape[0]}x{result.shape[1]}")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Elements: {rows * cols:,}")

    expected = matrix.get().T
    match = np.allclose(result.get(), expected)
    print(f"  Correct: {match}")


if __name__ == "__main__":
    main()
