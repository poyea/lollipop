from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "sparse_matvec.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "sparse_matvec")
    return _kernel


def sparse_matvec(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values: cp.ndarray,
    x: cp.ndarray,
) -> cp.ndarray:
    num_rows = row_ptr.size - 1
    y = cp.zeros(num_rows, dtype=cp.float32)

    grid = (num_rows + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,), (_BLOCK_SIZE,),
        (row_ptr.astype(cp.int32), col_idx.astype(cp.int32),
         values.astype(cp.float32), x.astype(cp.float32), y, num_rows),
    )
    return y
