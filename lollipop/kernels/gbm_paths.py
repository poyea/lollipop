import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def gbm_paths(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    t: float = 1.0,
    num_paths: int = 100_000,
    num_steps: int = 252,
    seed: int = 42,
) -> cp.ndarray:
    dt = t / num_steps
    paths = cp.empty((num_paths, num_steps + 1), dtype=cp.float32)

    grid = (num_paths + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    load("gbm_paths")(
        (grid,),
        (_BLOCK_SIZE,),
        (
            paths,
            cp.float32(s0),
            cp.float32(mu),
            cp.float32(sigma),
            cp.float32(dt),
            np.int32(num_steps),
            np.int32(num_paths),
            np.uint32(seed),
        ),
    )

    return paths
