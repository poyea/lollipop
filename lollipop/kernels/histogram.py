from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "histogram.cu").read_text()
        _kernel = cp.RawKernel(source, "histogram")
    return _kernel


def histogram(data: cp.ndarray, num_bins: int = 256) -> cp.ndarray:
    n = data.size
    bins = cp.zeros(num_bins, dtype=cp.uint32)
    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _get_kernel()((grid,), (_BLOCK_SIZE,), (data.ravel(), bins, n, num_bins))
    return bins
