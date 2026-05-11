import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = 256


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

    load("black_scholes")(
        (grid,),
        (_BLOCK_SIZE,),
        (spot, strike, ttm, rate, vol, call_prices, put_prices, n),
    )

    return call_prices, put_prices
