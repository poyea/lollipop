from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "warp_scan.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "warp_scan")
    return _kernel


def warp_scan(data: cp.ndarray) -> cp.ndarray:
    n = data.size
    if n > 32:
        raise ValueError(f"Max 32 elements (single warp), got {n}")

    result = data.astype(cp.float32).ravel().copy()
    _get_kernel()((1,), (32,), (result, n))
    return result
