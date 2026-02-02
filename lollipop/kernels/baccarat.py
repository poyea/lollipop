from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "baccarat.cu").read_text()
        _kernel = cp.RawKernel(source, "baccarat")
    return _kernel


def baccarat(num_hands: int = 10_000_000, seed: int = 42) -> tuple[int, int, int]:
    """Run a Monte Carlo Baccarat simulation on the GPU.

    Each CUDA thread simulates one complete hand using per-thread xorshift32
    RNG and the full third-card drawing rules. Results are accumulated via
    atomic operations.

    Returns (player_wins, banker_wins, ties) as Python ints.
    """
    results = cp.zeros(3, dtype=cp.uint32)
    grid = (num_hands + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (results, np.int32(num_hands), np.uint32(seed)),
    )

    r = results.get()
    return int(r[0]), int(r[1]), int(r[2])
