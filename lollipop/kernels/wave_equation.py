from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "wave_equation.cu").read_text()
        _kernel = cp.RawKernel(source, "wave_equation")
    return _kernel


def wave_equation(
    width: int = 256,
    height: int = 256,
    steps: int = 500,
    c: float = 1.0,
    dt: float = 0.1,
) -> cp.ndarray:
    """Solve the 2D wave equation on GPU using explicit finite differences.

    Uses the leapfrog (Verlet) scheme:
        u_next = 2*u - u_prev + c^2 * dt^2 * laplacian(u)

    Initialised with a Gaussian pulse at the centre.  Dirichlet boundary
    conditions (zero displacement at edges).

    Returns the displacement field u of shape (height, width) as float32.
    """
    u_prev = cp.zeros((height, width), dtype=cp.float32)
    u = cp.zeros((height, width), dtype=cp.float32)

    # Gaussian pulse initial condition
    cy, cx = height // 2, width // 2
    ys = cp.arange(height, dtype=cp.float32)[:, None]
    xs = cp.arange(width, dtype=cp.float32)[None, :]
    sigma = min(width, height) / 20.0
    u[:] = cp.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    u_prev[:] = u  # zero initial velocity

    u_next = cp.zeros_like(u)
    c2dt2 = cp.float32(c * c * dt * dt)

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    kernel = _get_kernel()

    for _ in range(steps):
        kernel(
            grid,
            _BLOCK_SIZE,
            (u_prev, u, u_next, width, height, c2dt2),
        )
        u_prev, u, u_next = u, u_next, u_prev

    return u
