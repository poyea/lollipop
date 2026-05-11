import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def histogram(data: cp.ndarray, num_bins: int = 256) -> cp.ndarray:
    n = data.size
    bins = cp.zeros(num_bins, dtype=cp.uint32)
    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    load("histogram")((grid,), (_BLOCK_SIZE,), (data.ravel(), bins, n, num_bins))
    return bins
