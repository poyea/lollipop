import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK_SIZE = (16, 16)


def heat_equation(
    width: int = 256,
    height: int = 256,
    steps: int = 1000,
    alpha: float = 1.0,
    dt: float = 0.2,
) -> cp.ndarray:
    u = cp.zeros((height, width), dtype=cp.float32)

    cy, cx = height // 2, width // 2
    r = min(width, height) // 10
    u[cy - r : cy + r, cx - r : cx + r] = 1.0

    u_next = u.copy()
    alpha_dt = np.float32(alpha * dt)

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    kernel = load("heat_equation")

    for _ in range(steps):
        kernel(grid, _BLOCK_SIZE, (u, u_next, width, height, alpha_dt))
        u, u_next = u_next, u

    return u
