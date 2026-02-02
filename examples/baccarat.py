import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import baccarat


def main():
    print("=== Baccarat Monte Carlo Simulation (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    num_hands = 10_000_000

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    player_wins, banker_wins, ties = baccarat(num_hands)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    total = player_wins + banker_wins + ties
    p_pct = player_wins / total * 100
    b_pct = banker_wins / total * 100
    t_pct = ties / total * 100

    print(f"  Simulated {num_hands:,} hands in {gpu_time:.4f}s\n")
    print(f"  {'Outcome':<12} {'Simulated':>10} {'Theoretical':>12} {'Diff':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Player':<12} {p_pct:>9.2f}% {44.62:>11.2f}% {p_pct - 44.62:>+7.2f}%")
    print(f"  {'Banker':<12} {b_pct:>9.2f}% {45.86:>11.2f}% {b_pct - 45.86:>+7.2f}%")
    print(f"  {'Tie':<12} {t_pct:>9.2f}% {9.52:>11.2f}% {t_pct - 9.52:>+7.2f}%")

    print(f"\n  House edge (theoretical):")
    print(f"    Player bet:  1.24%")
    print(f"    Banker bet:  1.06%")
    print(f"    Tie bet:    14.36%")


if __name__ == "__main__":
    main()
