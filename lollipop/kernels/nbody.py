import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


def nbody(
    n: int = 4096,
    steps: int = 100,
    dt: float = 0.001,
    softening: float = 0.01,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    rng = cp.random.default_rng(42)
    px = rng.standard_normal(n, dtype=cp.float32)
    py = rng.standard_normal(n, dtype=cp.float32)
    pz = rng.standard_normal(n, dtype=cp.float32)
    vx = cp.zeros(n, dtype=cp.float32)
    vy = cp.zeros(n, dtype=cp.float32)
    vz = cp.zeros(n, dtype=cp.float32)
    mass = cp.ones(n, dtype=cp.float32)

    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    kernel = load("nbody")

    for _ in range(steps):
        kernel(
            (grid,),
            (_BLOCK_SIZE,),
            (px, py, pz, vx, vy, vz, mass, n, cp.float32(dt), cp.float32(softening)),
        )

    return px, py, pz
