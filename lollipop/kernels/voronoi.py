from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_jfa_kernel = None
_color_kernel = None
_BLOCK_SIZE = (16, 16)


def _get_kernels() -> tuple:
    global _jfa_kernel, _color_kernel
    if _jfa_kernel is None:
        source = (_SOURCES_DIR / "voronoi.cu").read_text()
        _jfa_kernel = cp.RawKernel(source, "voronoi_jfa")
        _color_kernel = cp.RawKernel(source, "voronoi_color")
    return _jfa_kernel, _color_kernel


def voronoi(
    width: int = 512,
    height: int = 512,
    num_seeds: int = 64,
) -> cp.ndarray:
    jfa_kernel, color_kernel = _get_kernels()

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )

    nearest = cp.full(width * height, -1, dtype=cp.int32)

    rng = cp.random.default_rng(42)
    seed_x = rng.integers(0, width, size=num_seeds)
    seed_y = rng.integers(0, height, size=num_seeds)
    seed_indices = seed_y * width + seed_x
    nearest[seed_indices] = seed_indices

    colors = cp.zeros(width * height * 3, dtype=cp.uint8)
    seed_colors = rng.integers(0, 256, size=(num_seeds, 3), dtype=cp.uint8)
    for i in range(num_seeds):
        base = int(seed_indices[i]) * 3
        colors[base] = seed_colors[i, 0]
        colors[base + 1] = seed_colors[i, 1]
        colors[base + 2] = seed_colors[i, 2]

    max_dim = max(width, height)
    step = max_dim // 2
    while step >= 1:
        jfa_kernel(
            grid,
            _BLOCK_SIZE,
            (nearest, nearest, num_seeds, width, height, step),
        )
        step //= 2

    output = cp.zeros(width * height * 3, dtype=cp.uint8)
    color_kernel(grid, _BLOCK_SIZE, (nearest, output, colors, width, height))

    return output.reshape(height, width, 3)
