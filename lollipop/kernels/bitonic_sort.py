import cupy as cp

from lollipop.kernels._raw import load


def bitonic_sort(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n & (n - 1) != 0:
        raise ValueError(f"Length must be a power of 2, got {n}")
    if n > 1024:
        raise ValueError(f"Max 1024 elements (single block), got {n}")

    result = data.astype(cp.float32).copy()
    shared_mem = n * 4  # sizeof(float)
    load("bitonic_sort")((1,), (n,), (result, n), shared_mem=shared_mem)
    return result
