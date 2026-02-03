from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "gbm_paths.cu").read_text()
        _kernel = cp.RawKernel(source, "gbm_paths")
    return _kernel


def gbm_paths(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    t: float = 1.0,
    num_paths: int = 100_000,
    num_steps: int = 252,
    seed: int = 42,
) -> cp.ndarray:
    """Simulate Geometric Brownian Motion paths on GPU.

    Each CUDA thread generates one independent price path using the exact
    GBM discretisation:
        S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    s0        : float – Initial asset price.
    mu        : float – Drift (expected annual return).
    sigma     : float – Volatility (annual).
    t         : float – Time horizon in years.
    num_paths : int   – Number of independent paths to simulate.
    num_steps : int   – Number of time steps per path (e.g. 252 trading days).
    seed      : int   – RNG seed.

    Returns an array of shape (num_paths, num_steps + 1) as float32.
    """
    dt = t / num_steps
    paths = cp.empty((num_paths, num_steps + 1), dtype=cp.float32)

    grid = (num_paths + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
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
