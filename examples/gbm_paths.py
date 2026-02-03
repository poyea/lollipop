import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop import gbm_paths


def main():
    print("=== Geometric Brownian Motion (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    num_paths = 500_000
    num_steps = 252

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    paths = gbm_paths(
        s0=100.0, mu=0.05, sigma=0.2, t=1.0, num_paths=num_paths, num_steps=num_steps
    )
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    terminal = paths[:, -1].get()

    print(f"  Simulated {num_paths:,} paths x {num_steps} steps in {gpu_time:.4f}s")
    print(f"  Total data points: {num_paths * (num_steps + 1):,}\n")
    print(f"  Terminal price statistics (S0=100, mu=5%, sigma=20%, T=1yr):")
    print(
        f"    Mean:   {terminal.mean():10.2f}  (theoretical: {100 * np.exp(0.05):.2f})"
    )
    print(f"    Median: {np.median(terminal):10.2f}")
    print(f"    Std:    {terminal.std():10.2f}")
    print(f"    Min:    {terminal.min():10.2f}")
    print(f"    Max:    {terminal.max():10.2f}")


if __name__ == "__main__":
    main()
