from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "stencil_1d.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "stencil_1d")
    return _kernel


def stencil_1d(data: cp.ndarray, radius: int = 3) -> cp.ndarray:
    data = data.astype(cp.float32).ravel()
    n = data.size
    output = cp.empty(n, dtype=cp.float32)

    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    shared_mem = (_BLOCK_SIZE + 2 * radius) * 4

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (data, output, np.int32(n), np.int32(radius)),
        shared_mem=shared_mem,
    )
    return output
