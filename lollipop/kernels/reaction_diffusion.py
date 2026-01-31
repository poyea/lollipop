from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "reaction_diffusion.cu").read_text()
        _kernel = cp.RawKernel(source, "reaction_diffusion")
    return _kernel


def reaction_diffusion(
    width: int = 256,
    height: int = 256,
    steps: int = 5000,
    du: float = 0.16,
    dv: float = 0.08,
    f: float = 0.035,
    k: float = 0.065,
    dt: float = 1.0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Run Gray-Scott reaction-diffusion simulation on GPU.

    Returns (u, v) arrays of shape (height, width) as float32.
    """
    u = cp.ones((height, width), dtype=cp.float32)
    v = cp.zeros((height, width), dtype=cp.float32)

    cx, cy = width // 2, height // 2
    r = min(width, height) // 10
    u[cy - r : cy + r, cx - r : cx + r] = 0.5
    v[cy - r : cy + r, cx - r : cx + r] = 0.25

    u_next = u.copy()
    v_next = v.copy()

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    kernel = _get_kernel()

    for _ in range(steps):
        kernel(
            grid,
            _BLOCK_SIZE,
            (
                u,
                v,
                u_next,
                v_next,
                width,
                height,
                cp.float32(du),
                cp.float32(dv),
                cp.float32(f),
                cp.float32(k),
                cp.float32(dt),
            ),
        )
        u, u_next = u_next, u
        v, v_next = v_next, v

    return u, v
