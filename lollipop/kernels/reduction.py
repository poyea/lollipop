import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def reduction(data: cp.ndarray) -> float:
    data = data.astype(cp.float32).ravel()
    n = data.size

    output = cp.zeros(1, dtype=cp.float32)
    grid = (n + _BLOCK_SIZE * 2 - 1) // (_BLOCK_SIZE * 2)
    shared_mem = _BLOCK_SIZE * 4  # sizeof(float)

    load("reduction")((grid,), (_BLOCK_SIZE,), (data, output, n), shared_mem=shared_mem)
    return float(output[0])
