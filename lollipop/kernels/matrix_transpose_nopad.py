import cupy as cp

from lollipop.kernels._raw import load

_TILE_DIM = 32
_BLOCK_ROWS = 8


def matrix_transpose_nopad(matrix: cp.ndarray) -> cp.ndarray:
    """Tiled transpose with unpadded smem (32-way bank conflict on writeback).

    Foil for the +1-padded `matrix_transpose`; quantifies the bank-conflict
    cost specifically (everything else — tile size, launch geo, global access
    pattern — is byte-for-byte identical).
    """
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
    load("matrix_transpose_nopad")(grid, block, (matrix, output, width, height))
    return output
