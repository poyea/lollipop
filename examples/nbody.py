import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import nbody


def main():
    print("=== N-Body Gravitational Simulation ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n, steps = 4096, 100

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    px, py, pz = nbody(n, steps)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {n:,} bodies, {steps} steps")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Interactions per step: {n * n:,}")
    print(f"  Total interactions: {n * n * steps:,}")


if __name__ == "__main__":
    main()
