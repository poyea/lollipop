from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "lbm.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "lbm")
    return _kernel


def lbm(
    width: int = 256,
    height: int = 256,
    steps: int = 200,
    omega: float = 1.0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    N = width * height

    w = np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        dtype=np.float32,
    )
    f_in = cp.zeros(9 * N, dtype=cp.float32)
    for d in range(9):
        f_in[d * N : (d + 1) * N] = w[d]

    cy = height // 2
    for y in range(cy - 2, cy + 3):
        for x in range(width // 4, 3 * width // 4):
            idx = y * width + x
            f_in[1 * N + idx] += 0.01
            f_in[3 * N + idx] -= 0.01

    f_out = cp.zeros_like(f_in)

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    kernel = _get_kernel()

    for _ in range(steps):
        kernel(
            grid,
            _BLOCK_SIZE,
            (f_in, f_out, np.int32(width), np.int32(height), np.float32(omega)),
        )
        f_in, f_out = f_out, f_in

    ex = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=cp.float32)
    ey = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=cp.float32)

    f = f_in.reshape(9, N)
    rho = f.sum(axis=0).reshape(height, width)
    ux = (ex[:, None] * f).sum(axis=0).reshape(height, width) / rho
    uy = (ey[:, None] * f).sum(axis=0).reshape(height, width) / rho

    return rho, ux, uy
