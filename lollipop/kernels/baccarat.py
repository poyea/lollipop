from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "baccarat.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "baccarat")
    return _kernel


def baccarat(num_shoes: int = 1_000_000, seed: int = 42) -> tuple[int, int, int, int]:
    results = cp.zeros(4, dtype=cp.uint32)
    grid = (num_shoes + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (results, np.int32(num_shoes), np.uint32(seed)),
    )

    r = results.get()
    return int(r[0]), int(r[1]), int(r[2]), int(r[3])
