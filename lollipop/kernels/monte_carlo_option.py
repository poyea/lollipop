from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "monte_carlo_option.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "monte_carlo_option")
    return _kernel


def monte_carlo_option(
    s0: float = 100.0,
    k: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    t: float = 1.0,
    num_paths: int = 1_000_000,
    seed: int = 42,
) -> tuple[float, float]:
    num_threads = min(num_paths, 65536)
    paths_per_thread = num_paths // num_threads

    results = cp.zeros(2, dtype=cp.float32)
    grid = (num_threads + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (
            results,
            cp.float32(s0),
            cp.float32(k),
            cp.float32(r),
            cp.float32(sigma),
            cp.float32(t),
            np.int32(paths_per_thread),
            np.int32(num_threads),
            np.uint32(seed),
        ),
    )

    total_paths = num_threads * paths_per_thread
    discount = float(np.exp(-r * t))
    res = results.get()
    call_price = discount * float(res[0]) / total_paths
    put_price = discount * float(res[1]) / total_paths

    return call_price, put_price
