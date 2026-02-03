from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "bitonic_sort.cu").read_text()
        _kernel = cp.RawKernel(source, "bitonic_sort")
    return _kernel


def bitonic_sort(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n & (n - 1) != 0:
        raise ValueError(f"Length must be a power of 2, got {n}")
    if n > 1024:
        raise ValueError(f"Max 1024 elements (single block), got {n}")

    result = data.astype(cp.float32).copy()
    shared_mem = n * 4  # sizeof(float)
    _get_kernel()((1,), (n,), (result, n), shared_mem=shared_mem)
    return result
