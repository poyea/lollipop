from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "reduction.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "reduction")
    return _kernel


def reduction(data: cp.ndarray) -> float:
    data = data.astype(cp.float32).ravel()
    n = data.size

    output = cp.zeros(1, dtype=cp.float32)
    grid = (n + _BLOCK_SIZE * 2 - 1) // (_BLOCK_SIZE * 2)
    shared_mem = _BLOCK_SIZE * 4  # sizeof(float)

    _get_kernel()((grid,), (_BLOCK_SIZE,), (data, output, n), shared_mem=shared_mem)
    return float(output[0])
