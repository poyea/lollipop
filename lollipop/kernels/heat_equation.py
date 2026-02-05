from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "heat_equation.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "heat_equation")
    return _kernel


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
    kernel = _get_kernel()

    for _ in range(steps):
        kernel(grid, _BLOCK_SIZE, (u, u_next, width, height, alpha_dt))
        u, u_next = u_next, u

    return u
