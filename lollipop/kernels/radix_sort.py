from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_hist_kernel = None
_scan_kernel = None
_scatter_kernel = None
_BLOCK_SIZE = 256
_BITS_PER_PASS = 4
_NUM_BUCKETS = 1 << _BITS_PER_PASS


def _get_kernels() -> tuple:
    global _hist_kernel, _scan_kernel, _scatter_kernel
    if _hist_kernel is None:
        source = (_SOURCES_DIR / "radix_sort.cu").read_text(encoding="utf-8")
        _hist_kernel = cp.RawKernel(source, "radix_histogram")
        _scan_kernel = cp.RawKernel(source, "radix_prefix_sum")
        _scatter_kernel = cp.RawKernel(source, "radix_scatter")
    return _hist_kernel, _scan_kernel, _scatter_kernel


def _next_power_of_2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def radix_sort(data: cp.ndarray) -> cp.ndarray:
    data = data.astype(cp.uint32).ravel()
    n_orig = data.size

    pad = (_BLOCK_SIZE - n_orig % _BLOCK_SIZE) % _BLOCK_SIZE
    if pad > 0:
        data = cp.concatenate([data, cp.full(pad, 0xFFFFFFFF, dtype=cp.uint32)])
    n = data.size

    num_blocks = n // _BLOCK_SIZE
    total_hist = _NUM_BUCKETS * num_blocks
    scan_threads = _next_power_of_2(total_hist) // 2
    scan_total = _next_power_of_2(total_hist)

    hist_kernel, scan_kernel, scatter_kernel = _get_kernels()

    keys_in = data.copy()
    keys_out = cp.empty_like(keys_in)
    histograms = cp.zeros(scan_total, dtype=cp.uint32)

    for shift in range(0, 32, _BITS_PER_PASS):
        histograms[:] = 0

        hist_kernel(
            (num_blocks,), (_BLOCK_SIZE,),
            (keys_in, histograms, np.int32(n), np.int32(shift)),
        )

        scan_kernel(
            (1,), (scan_threads,),
            (histograms, np.int32(scan_total)),
            shared_mem=scan_total * 4,
        )

        scatter_kernel(
            (num_blocks,), (_BLOCK_SIZE,),
            (keys_in, keys_out, histograms, np.int32(n), np.int32(shift)),
        )

        keys_in, keys_out = keys_out, keys_in

    return keys_in[:n_orig]
