import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import black_scholes


def main():
    print("=== Black-Scholes Option Pricing (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 1_000_000

    spot = cp.full(n, 100.0, dtype=cp.float32)
    strike = cp.linspace(80, 120, n, dtype=cp.float32)
    ttm = cp.full(n, 1.0, dtype=cp.float32)
    rate = cp.full(n, 0.05, dtype=cp.float32)
    vol = cp.full(n, 0.2, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    call_prices, put_prices = black_scholes(spot, strike, ttm, rate, vol)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    c = call_prices.get()
    p = put_prices.get()

    print(f"  Priced {n:,} European options in {gpu_time:.4f}s")
    print(f"  Throughput: {n / gpu_time:,.0f} options/sec\n")

    print(f"  {'Strike':>8} {'Call':>10} {'Put':>10}")
    print(f"  {'-'*30}")
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        print(f"  {strike[i].get():8.2f} {c[i]:10.4f} {p[i]:10.4f}")


if __name__ == "__main__":
    main()
