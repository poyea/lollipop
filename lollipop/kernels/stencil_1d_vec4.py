from pathlib import Path

import cupy as cp
import numpy as np

from lollipop.kernels.stencil_1d import stencil_1d

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "stencil_1d_vec4.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "stencil_1d_vec4")
    return _kernel


def stencil_1d_vec4(data: cp.ndarray, radius: int = 3) -> cp.ndarray:
    """Float4-vectorized 1D stencil.

    Requires the input pointer to be 16-byte aligned (CuPy's default
    allocator satisfies this for fresh allocations).  Raises on misaligned
    input.

    Length handling: the kernel only operates on the largest n // 4 * 4
    prefix.  If ``n % 4 != 0``, the tail elements are computed by
    delegating the *entire* input to the scalar ``stencil_1d`` kernel and
    overwriting only the tail of the vec4 result with the scalar result.
    This keeps the boundary math identical to the baseline without
    needing a second specialised tail kernel.
    """
    data = data.astype(cp.float32).ravel()
    n = data.size

    if data.data.ptr % 16 != 0:
        raise ValueError(
            "stencil_1d_vec4 requires 16-byte aligned input; "
            "got ptr % 16 = "
            f"{data.data.ptr % 16}.  Make a fresh copy with cp.ascontiguousarray."
        )

    output = cp.empty(n, dtype=cp.float32)
    n_aligned = (n // 4) * 4

    if n_aligned > 0:
        n_vec4 = n_aligned // 4
        grid = (n_vec4 + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        shared_mem = (4 * _BLOCK_SIZE + 2 * radius) * 4  # sizeof(float)
        _get_kernel()(
            (grid,),
            (_BLOCK_SIZE,),
            (data, output, np.int32(n_aligned), np.int32(radius)),
            shared_mem=shared_mem,
        )

    if n_aligned < n:
        # Tail — recompute via the scalar kernel and copy the tail portion.
        # We also touch the last `radius` aligned elements because their
        # right-halo extended past n_aligned (clamped) in the vec4 kernel,
        # whereas the scalar kernel sees the true tail.
        scalar_out = stencil_1d(data, radius=radius)
        overlap = min(n, max(n_aligned - radius, 0))
        output[overlap:] = scalar_out[overlap:]

    return output
