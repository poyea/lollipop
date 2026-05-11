import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = (16, 16)


def wave_equation(
    width: int = 256,
    height: int = 256,
    steps: int = 500,
    c: float = 1.0,
    dt: float = 0.1,
) -> cp.ndarray:
    u_prev = cp.zeros((height, width), dtype=cp.float32)
    u = cp.zeros((height, width), dtype=cp.float32)

    cy, cx = height // 2, width // 2
    ys = cp.arange(height, dtype=cp.float32)[:, None]
    xs = cp.arange(width, dtype=cp.float32)[None, :]
    sigma = min(width, height) / 20.0
    u[:] = cp.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    u_prev[:] = u

    u_next = cp.zeros_like(u)
    c2dt2 = cp.float32(c * c * dt * dt)

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    kernel = load("wave_equation")

    for _ in range(steps):
        kernel(
            grid,
            _BLOCK_SIZE,
            (u_prev, u, u_next, width, height, c2dt2),
        )
        u_prev, u, u_next = u, u_next, u_prev

    return u
