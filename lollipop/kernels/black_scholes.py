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
    """Price European options using the Black-Scholes analytical formula.

    Each element across the input arrays defines one option contract.
    Computes call and put prices in parallel on the GPU using the closed-form
    solution with the cumulative normal distribution.

    Parameters
    ----------
    spot   : cp.ndarray (float32) – Current underlying prices.
    strike : cp.ndarray (float32) – Strike prices.
    ttm    : cp.ndarray (float32) – Time to maturity in years.
    rate   : cp.ndarray (float32) – Risk-free interest rates.
    vol    : cp.ndarray (float32) – Implied volatilities.

    Returns (call_prices, put_prices) as float32 GPU arrays.
    """
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
