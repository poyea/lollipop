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
    """Generate a Voronoi diagram on GPU using the Jump Flood Algorithm.

    Seeds are placed randomly. Multiple JFA passes at geometrically decreasing
    step sizes converge to nearest-seed assignment in O(log n) passes.

    Returns an RGB image as a (height, width, 3) uint8 array.
    """
    jfa_kernel, color_kernel = _get_kernels()

    grid = (
        (width + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )

    # Initialize nearest-seed map to -1 (unassigned)
    nearest = cp.full(width * height, -1, dtype=cp.int32)

    # Place seeds randomly and assign them to themselves
    rng = cp.random.default_rng(42)
    seed_x = rng.integers(0, width, size=num_seeds)
    seed_y = rng.integers(0, height, size=num_seeds)
    seed_indices = seed_y * width + seed_x
    nearest[seed_indices] = seed_indices

    # Pre-compute a color per pixel position used as seed
    # (store color at the flat-index position so the color kernel can look it up)
    colors = cp.zeros(width * height * 3, dtype=cp.uint8)
    seed_colors = rng.integers(0, 256, size=(num_seeds, 3), dtype=cp.uint8)
    for i in range(num_seeds):
        base = int(seed_indices[i]) * 3
        colors[base] = seed_colors[i, 0]
        colors[base + 1] = seed_colors[i, 1]
        colors[base + 2] = seed_colors[i, 2]

    # Jump Flood passes: step = max_dim/2, max_dim/4, ..., 1
    max_dim = max(width, height)
    step = max_dim // 2
    while step >= 1:
        jfa_kernel(
            grid,
            _BLOCK_SIZE,
            (nearest, nearest, num_seeds, width, height, step),
        )
        step //= 2

    # Colorize the result
    output = cp.zeros(width * height * 3, dtype=cp.uint8)
    color_kernel(grid, _BLOCK_SIZE, (nearest, output, colors, width, height))

    return output.reshape(height, width, 3)
