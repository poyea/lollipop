import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

# Must match the #defines in _sources/prefix_sum.cu.
ELEMENTS_PER_BLOCK = 2048
_THREADS = ELEMENTS_PER_BLOCK // 2
# Conflict-free padding adds index>>5 slack; the worst-case index is the last
# slot, so reserve ELEMENTS_PER_BLOCK/NUM_BANKS extra floats.
_SMEM_SLOTS = ELEMENTS_PER_BLOCK + ELEMENTS_PER_BLOCK // 32


def _scan_exclusive(d_in: cp.ndarray, d_out: cp.ndarray, suffix: str) -> cp.ndarray:
    """Device-wide exclusive prefix sum of `d_in` into `d_out` (same dtype/size).

    Hierarchical Blelloch: scan each ELEMENTS_PER_BLOCK-wide tile, recursively
    scan the per-tile totals, then add the offsets back. Recurses until a single
    tile remains, so n is unbounded. `suffix` selects the dtype variant
    ("f32" or "u32").
    """
    n = d_in.size
    n_blocks = (n + ELEMENTS_PER_BLOCK - 1) // ELEMENTS_PER_BLOCK

    scan_block = load("prefix_sum", f"scan_block_{suffix}")
    block_sums = cp.empty(n_blocks, dtype=d_in.dtype)
    scan_block(
        (n_blocks,),
        (_THREADS,),
        (d_in, d_out, block_sums, np.int32(n)),
        shared_mem=_SMEM_SLOTS * d_in.itemsize,
    )

    if n_blocks > 1:
        block_offsets = cp.empty(n_blocks, dtype=d_in.dtype)
        _scan_exclusive(block_sums, block_offsets, suffix)
        add_offsets = load("prefix_sum", f"add_block_offsets_{suffix}")
        add_offsets(
            (n_blocks,),
            (_THREADS,),
            (d_out, block_offsets, np.int32(n)),
        )
    return d_out


def prefix_sum(data: cp.ndarray) -> cp.ndarray:
    """Exclusive prefix sum (scan) of a 1D array. Arbitrary length.

    Example: [1, 2, 3, 4] -> [0, 1, 3, 6].
    """
    src = data.astype(cp.float32).ravel()
    out = cp.empty_like(src)
    _scan_exclusive(src, out, "f32")
    return out
