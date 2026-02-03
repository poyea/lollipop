from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_TILE_DIM = 32
_BLOCK_ROWS = 8


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "matrix_transpose.cu").read_text()
        _kernel = cp.RawKernel(source, "matrix_transpose")
    return _kernel


def matrix_transpose(matrix: cp.ndarray) -> cp.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got {matrix.ndim}D")

    matrix = matrix.astype(cp.float32)
    height, width = matrix.shape
    output = cp.empty((width, height), dtype=cp.float32)

    grid = (
        (width + _TILE_DIM - 1) // _TILE_DIM,
        (height + _TILE_DIM - 1) // _TILE_DIM,
    )
    block = (_TILE_DIM, _BLOCK_ROWS)

    _get_kernel()(grid, block, (matrix, output, width, height))
    return output
