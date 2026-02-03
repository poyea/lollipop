import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import monte_carlo_option


def main():
    print("=== Monte Carlo Option Pricing (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    num_paths = 10_000_000

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    mc_call, mc_put = monte_carlo_option(
        s0=100.0,
        k=100.0,
        r=0.05,
        sigma=0.2,
        t=1.0,
        num_paths=num_paths,
    )
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    # Analytical BS prices for comparison
    bs_call = 10.4506
    bs_put = 5.5735

    print(f"  S0=100, K=100, r=5%, sigma=20%, T=1yr")
    print(f"  {num_paths:,} Monte Carlo paths in {gpu_time:.4f}s\n")
    print(f"  {'':12} {'MC Price':>10} {'BS Exact':>10} {'Error':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Call':<12} {mc_call:10.4f} {bs_call:10.4f} {mc_call - bs_call:+10.4f}")
    print(f"  {'Put':<12} {mc_put:10.4f} {bs_put:10.4f} {mc_put - bs_put:+10.4f}")


if __name__ == "__main__":
    main()
