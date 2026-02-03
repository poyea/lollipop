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
    """Integrate the Lorenz system for many trajectories in parallel on GPU.

    Each CUDA thread integrates one trajectory using fourth-order Runge-Kutta
    with a slightly different initial condition, demonstrating sensitive
    dependence on initial conditions (chaos).

    The Lorenz equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Returns (x, y, z) arrays of shape (num_trajectories, num_steps + 1)
    as float32.
    """
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
