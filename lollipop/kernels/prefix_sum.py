from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "prefix_sum.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "prefix_sum_blelloch")
    return _kernel


def prefix_sum(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n & (n - 1) != 0:
        raise ValueError(f"Length must be a power of 2, got {n}")

    result = data.astype(cp.float32).copy()
    shared_mem = n * 4  # sizeof(float)
    _get_kernel()((1,), (n // 2,), (result, n), shared_mem=shared_mem)
    return result
