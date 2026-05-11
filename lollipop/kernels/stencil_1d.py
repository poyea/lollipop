import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def stencil_1d(data: cp.ndarray, radius: int = 3) -> cp.ndarray:
    data = data.astype(cp.float32).ravel()
    n = data.size
    output = cp.empty(n, dtype=cp.float32)

    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    shared_mem = (_BLOCK_SIZE + 2 * radius) * 4

    load("stencil_1d")(
        (grid,),
        (_BLOCK_SIZE,),
        (data, output, np.int32(n), np.int32(radius)),
        shared_mem=shared_mem,
    )
    return output
