import cupy as cp

from lollipop.kernels._raw import load


def prefix_sum(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n & (n - 1) != 0:
        raise ValueError(f"Length must be a power of 2, got {n}")

    result = data.astype(cp.float32).copy()
    shared_mem = n * 4  # sizeof(float)
    load("prefix_sum", "prefix_sum_blelloch")(
        (1,), (n // 2,), (result, n), shared_mem=shared_mem
    )
    return result
