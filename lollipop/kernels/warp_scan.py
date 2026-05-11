import cupy as cp

from lollipop.kernels._raw import load


def warp_scan(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n > 32:
        raise ValueError(f"Max 32 elements (single warp), got {n}")

    result = data.astype(cp.float32).ravel().copy()
    load("warp_scan")((1,), (32,), (result, n))
    return result
