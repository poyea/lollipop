import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def sparse_matvec(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values: cp.ndarray,
    x: cp.ndarray,
) -> cp.ndarray:
    num_rows = row_ptr.size - 1
    y = cp.zeros(num_rows, dtype=cp.float32)

    grid = (num_rows + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    load("sparse_matvec")(
        (grid,),
        (_BLOCK_SIZE,),
        (
            row_ptr.astype(cp.int32),
            col_idx.astype(cp.int32),
            values.astype(cp.float32),
            x.astype(cp.float32),
            y,
            num_rows,
        ),
    )
    return y
