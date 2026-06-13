import cupy as cp
import numpy as np

from lollipop.kernels._raw import load
from lollipop.kernels.prefix_sum import _scan_exclusive

_BLOCK_SIZE = 256
_BITS_PER_PASS = 4
_NUM_BUCKETS = 1 << _BITS_PER_PASS


def radix_sort(data: cp.ndarray) -> cp.ndarray:
    data = data.astype(cp.uint32).ravel()
    n_orig = data.size

    pad = (_BLOCK_SIZE - n_orig % _BLOCK_SIZE) % _BLOCK_SIZE
    if pad > 0:
        data = cp.concatenate([data, cp.full(pad, 0xFFFFFFFF, dtype=cp.uint32)])
    n = data.size

    num_blocks = n // _BLOCK_SIZE
    total_hist = _NUM_BUCKETS * num_blocks

    hist_kernel = load("radix_sort", "radix_histogram")
    scatter_kernel = load("radix_sort", "radix_scatter")

    keys_in = data.copy()
    keys_out = cp.empty_like(keys_in)
    histograms = cp.empty(total_hist, dtype=cp.uint32)
    hist_scanned = cp.empty(total_hist, dtype=cp.uint32)

    for shift in range(0, 32, _BITS_PER_PASS):
        hist_kernel(
            (num_blocks,),
            (_BLOCK_SIZE,),
            (keys_in, histograms, np.int32(n), np.int32(shift)),
        )

        # Phase 2: device-wide exclusive scan over the flattened histogram
        # table (column-major: all blocks' bucket-0 counts, then bucket-1, ...).
        # The old single-block scan capped this at n=32768; the hierarchical
        # scan lifts that to arbitrary n.
        _scan_exclusive(histograms, hist_scanned, "u32")

        scatter_kernel(
            (num_blocks,),
            (_BLOCK_SIZE,),
            (keys_in, keys_out, hist_scanned, np.int32(n), np.int32(shift)),
        )

        keys_in, keys_out = keys_out, keys_in

    return keys_in[:n_orig]
