import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = (16, 16)


def julia(
    width: int = 2048,
    height: int = 2048,
    max_iter: int = 500,
    c_re: float = -0.7,
    c_im: float = 0.27015,
) -> cp.ndarray:
    output = cp.zeros(width * height, dtype=cp.uint8)
    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    load("julia")(
        grid,
        _BLOCK_SIZE,
        (output, width, height, max_iter, cp.float32(c_re), cp.float32(c_im)),
    )
    return output.reshape(height, width)
