from functools import lru_cache
from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"


@lru_cache(maxsize=None)
def load(
    source: str, kernel: str | None = None, cooperative: bool = False
) -> cp.RawKernel:
    """Lazy-load and cache a CUDA kernel from `_sources/<source>.cu`.

    If `kernel` is omitted, the kernel function name is assumed equal to
    `source`. Multi-kernel sources (e.g. radix_sort, voronoi) pass both.
    Set `cooperative=True` for kernels that use a grid-wide `cg::this_grid()`
    barrier. They must be launched cooperatively (cudaLaunchCooperativeKernel).
    """
    code = (_SOURCES_DIR / f"{source}.cu").read_text(encoding="utf-8")
    return cp.RawKernel(code, kernel or source, enable_cooperative_groups=cooperative)
