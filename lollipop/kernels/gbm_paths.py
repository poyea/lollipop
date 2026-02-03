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
