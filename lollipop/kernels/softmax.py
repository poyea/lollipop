from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "softmax.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "softmax")
    return _kernel


def softmax(data: cp.ndarray) -> cp.ndarray:
    if data.ndim == 1:
        data = data[None, :]
        squeeze = True
    elif data.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected 1D or 2D input, got {data.ndim}D")

    x = cp.ascontiguousarray(data, dtype=cp.float32)
    rows, cols = x.shape
    y = cp.empty_like(x)

    shared_mem = 2 * _BLOCK_SIZE * 4  # two float32 buffers
    _get_kernel()((rows,), (_BLOCK_SIZE,), (x, y, cols), shared_mem=shared_mem)

    return y[0] if squeeze else y
