from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "softmax_vec4.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "softmax_vec4")
    return _kernel


def softmax_vec4(data: cp.ndarray) -> cp.ndarray:
    """Float4-vectorized fused online softmax over the last axis.

    Requires 16-byte aligned input rows.  Each row must start on a
    16-byte boundary, which CuPy's default allocator guarantees for
    contiguous arrays whose row-stride-in-bytes is a multiple of 16
    (true when `cols % 4 == 0`).

    Tail handling: when `cols % 4 != 0` we fall back to the scalar
    `softmax` kernel for the whole input.  This is the simplest correct
    choice — a per-row tail pass would need to participate in the
    row-wide (m, d) reduction, which is structurally awkward and
    defeats the point of a clean "vectorized variant" demo.
    """
    if data.ndim == 1:
        data = data[None, :]
        squeeze = True
    elif data.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected 1D or 2D input, got {data.ndim}D")

    x = cp.ascontiguousarray(data, dtype=cp.float32)
    rows, cols = x.shape

    if cols % 4 != 0:
        # Fall back to the scalar kernel for awkward widths.
        from lollipop.kernels.softmax import softmax as _softmax_scalar

        y = _softmax_scalar(x)
        return y[0] if squeeze else y

    if x.data.ptr % 16 != 0:
        raise ValueError(
            "softmax_vec4 requires 16-byte aligned input; "
            f"got ptr % 16 = {x.data.ptr % 16}.  "
            "Make a fresh copy with cp.ascontiguousarray."
        )

    y = cp.empty_like(x)
    cols_vec4 = cols // 4
    shared_mem = 2 * _BLOCK_SIZE * 4
    _get_kernel()((rows,), (_BLOCK_SIZE,), (x, y, cols_vec4), shared_mem=shared_mem)

    return y[0] if squeeze else y
