from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "prefix_sum.cu").read_text()
        _kernel = cp.RawKernel(source, "prefix_sum_blelloch")
    return _kernel


def prefix_sum(data: cp.ndarray) -> cp.ndarray:
    """Compute exclusive prefix sum (scan) on GPU using Blelloch algorithm.

    Input must be a 1D float32 array with length that is a power of 2.
    Returns a new array with the exclusive prefix sum.
    """
    n = data.size
    if n & (n - 1) != 0:
        raise ValueError(f"Length must be a power of 2, got {n}")

    result = data.astype(cp.float32).copy()
    shared_mem = n * 4  # sizeof(float)
    _get_kernel()((1,), (n // 2,), (result, n), shared_mem=shared_mem)
    return result
