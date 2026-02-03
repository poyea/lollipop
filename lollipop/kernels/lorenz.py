from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "lorenz.cu").read_text()
        _kernel = cp.RawKernel(source, "lorenz")
    return _kernel


def lorenz(
    num_trajectories: int = 128,
    num_steps: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    stride = num_steps + 1
    out_x = cp.empty((num_trajectories, stride), dtype=cp.float32)
    out_y = cp.empty((num_trajectories, stride), dtype=cp.float32)
    out_z = cp.empty((num_trajectories, stride), dtype=cp.float32)

    grid = (num_trajectories + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (
            out_x,
            out_y,
            out_z,
            np.int32(num_trajectories),
            np.int32(num_steps),
            cp.float32(dt),
            cp.float32(sigma),
            cp.float32(rho),
            cp.float32(beta),
        ),
    )

    return out_x, out_y, out_z
