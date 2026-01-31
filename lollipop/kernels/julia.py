from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "julia.cu").read_text()
        _kernel = cp.RawKernel(source, "julia")
    return _kernel


def julia(
    width: int = 2048,
    height: int = 2048,
    max_iter: int = 500,
    c_re: float = -0.7,
    c_im: float = 0.27015,
) -> cp.ndarray:
    """Compute a Julia set on GPU.

    Returns a (height, width) uint8 array of iteration counts scaled to 0-255.
    """
    output = cp.zeros(width * height, dtype=cp.uint8)
    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    _get_kernel()(
        grid,
        _BLOCK_SIZE,
        (output, width, height, max_iter, cp.float32(c_re), cp.float32(c_im)),
    )
    return output.reshape(height, width)
