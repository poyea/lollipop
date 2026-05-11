import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def baccarat(
    num_shoes: int = 1_000_000, seed: int = 42
) -> tuple[int, int, int, int, int, int]:
    results = cp.zeros(6, dtype=cp.uint32)
    grid = (num_shoes + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    load("baccarat")(
        (grid,),
        (_BLOCK_SIZE,),
        (results, np.int32(num_shoes), np.uint32(seed)),
    )

    r = results.get()
    return int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])
