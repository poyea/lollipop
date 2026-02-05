from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "shared_reduce_2d.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "shared_reduce_2d")
    return _kernel


def shared_reduce_2d(data: cp.ndarray) -> float:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    data = data.astype(cp.float32)
    height, width = data.shape
    output = cp.zeros(1, dtype=cp.float32)

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    shared_mem = _BLOCK_SIZE[0] * _BLOCK_SIZE[1] * 4

    _get_kernel()(grid, _BLOCK_SIZE, (data, output, width, height), shared_mem=shared_mem)
    return float(output[0])
