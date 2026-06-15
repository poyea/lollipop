import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256
_MAX_BLOCKS = 1024  # grid-stride loop caps the grid; one atomicAdd per warp


def reduction_cg(data: cp.ndarray) -> float:
    """Sum a 1D array via the Cooperative Groups reduction kernel.

    Numerically equivalent to `reduction`, but built on `cg::reduce` over a
    32-lane tile instead of hand-rolled warp shuffles, with a grid-stride load.
    The cross-block combine is one atomicAdd per warp.
    """
    data = data.astype(cp.float32).ravel()
    n = data.size

    output = cp.zeros(1, dtype=cp.float32)
    grid = min(_MAX_BLOCKS, (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
    grid = max(grid, 1)

    load("reduction_cg")((grid,), (_BLOCK_SIZE,), (data, output, np.int32(n)))
    return float(output[0])


def reduction_cg_grid(data: cp.ndarray) -> float:
    """Sum a 1D array with a single atomic-free cooperative launch.

    Uses a grid-wide `cg::this_grid().sync()` barrier so block 0 can fold the
    per-block partials in the same launch. Cooperative launch requires every
    block to be co-resident, so the grid is capped at one block per SM.
    """
    data = data.astype(cp.float32).ravel()
    n = data.size

    num_sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    grid = max(1, num_sms)

    partials = cp.empty(grid, dtype=cp.float32)
    output = cp.zeros(1, dtype=cp.float32)
    load("reduction_cg", "reduction_cg_grid", cooperative=True)(
        (grid,), (_BLOCK_SIZE,), (data, partials, output, np.int32(n))
    )
    return float(output[0])
