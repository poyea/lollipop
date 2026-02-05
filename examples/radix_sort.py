import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import numpy as np
import time

from lollipop.kernels.radix_sort import (
    _get_kernels,
    _BLOCK_SIZE,
    _BITS_PER_PASS,
    _NUM_BUCKETS,
)


def _next_power_of_2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def main():
    print("=== Radix Sort (Multi-Kernel Pipeline) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    n = 8192
    rng = cp.random.default_rng(42)
    data = rng.integers(0, 1_000_000, size=n, dtype=cp.uint32)

    pad = (_BLOCK_SIZE - n % _BLOCK_SIZE) % _BLOCK_SIZE
    if pad > 0:
        padded = cp.concatenate([data, cp.full(pad, 0xFFFFFFFF, dtype=cp.uint32)])
    else:
        padded = data.copy()
    n_padded = padded.size

    num_blocks = n_padded // _BLOCK_SIZE
    total_hist = _NUM_BUCKETS * num_blocks
    scan_total = _next_power_of_2(total_hist)
    scan_threads = scan_total // 2
    num_passes = 32 // _BITS_PER_PASS

    print(f"  {n:,} keys  ({n_padded} padded, {num_blocks} blocks)")
    print(f"  {_BITS_PER_PASS} bits/pass, {_NUM_BUCKETS} buckets, {num_passes} passes")
    print(f"  {num_passes * 3} kernel launches total")
    print(f"  histogram table: {num_blocks} x {_NUM_BUCKETS} = {total_hist} entries")
    print()

    hist_kernel, scan_kernel, scatter_kernel = _get_kernels()

    keys_in = padded.copy()
    keys_out = cp.empty_like(keys_in)
    histograms = cp.zeros(scan_total, dtype=cp.uint32)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for pass_num in range(num_passes):
        shift = pass_num * _BITS_PER_PASS
        histograms[:] = 0

        hist_kernel(
            (num_blocks,), (_BLOCK_SIZE,),
            (keys_in, histograms, np.int32(n_padded), np.int32(shift)),
        )

        scan_kernel(
            (1,), (scan_threads,),
            (histograms, np.int32(scan_total)),
            shared_mem=scan_total * 4,
        )

        scatter_kernel(
            (num_blocks,), (_BLOCK_SIZE,),
            (keys_in, keys_out, histograms, np.int32(n_padded), np.int32(shift)),
        )

        keys_in, keys_out = keys_out, keys_in

        h = histograms[:total_hist].get()
        print(
            f"  pass {pass_num}: bits [{shift:>2}:{shift + _BITS_PER_PASS:>2}]  "
            f"histogram  scatter  "
            f"prefix_sum_range=[{h.min():>7,} .. {h.max():>7,}]"
        )

    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    result = keys_in[:n]

    print(f"\n  Sorted {n:,} keys in {gpu_time:.4f}s")
    print(f"  First 8: {result[:8].get()}")
    print(f"  Last 8:  {result[-8:].get()}")

    expected = np.sort(data.get())
    match = np.array_equal(result.get(), expected)
    print(f"  Correct: {match}")


if __name__ == "__main__":
    main()
