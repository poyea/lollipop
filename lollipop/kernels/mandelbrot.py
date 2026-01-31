from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "mandelbrot.cu").read_text()
        _kernel = cp.RawKernel(source, "mandelbrot")
    return _kernel


def mandelbrot(
    width: int = 2048,
    height: int = 2048,
    max_iter: int = 500,
) -> cp.ndarray:
    """Compute the Mandelbrot set on GPU.

    Returns a (height, width) uint8 array of iteration counts scaled to 0-255.
    """
    output = cp.zeros(width * height, dtype=cp.uint8)
    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    _get_kernel()(grid, _BLOCK_SIZE, (output, width, height, max_iter))
    return output.reshape(height, width)
