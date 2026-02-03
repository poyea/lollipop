from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "black_scholes.cu").read_text()
        _kernel = cp.RawKernel(source, "black_scholes")
    return _kernel


def black_scholes(
    spot: cp.ndarray,
    strike: cp.ndarray,
    ttm: cp.ndarray,
    rate: cp.ndarray,
    vol: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    n = spot.shape[0]
    call_prices = cp.empty(n, dtype=cp.float32)
    put_prices = cp.empty(n, dtype=cp.float32)

    grid = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _get_kernel()(
        (grid,),
        (_BLOCK_SIZE,),
        (spot, strike, ttm, rate, vol, call_prices, put_prices, n),
    )

    return call_prices, put_prices
