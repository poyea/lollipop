import cupy as cp

from lollipop.kernels._raw import load

_ITEMS = 8
_CONFIGS = {
    128: "reduction_v2_t128_i8",
    256: "reduction_v2_t256_i8",
    512: "reduction_v2_t512_i8",
    1024: "reduction_v2_t1024_i8",
}
_DEFAULT_THREADS = 128  # Best on Turing — see docs/profiles/reduction_v2.md.


def reduction_v2(data: cp.ndarray, threads: int = _DEFAULT_THREADS) -> float:
    """Occupancy-tuned sum reduction.

    ``threads`` selects the launch config (128 / 256 / 512 / 1024); each
    is a separate `__launch_bounds__`-annotated kernel in
    ``_sources/reduction_v2.cu``.  Default 128 is the empirical winner
    on Turing — see ``docs/profiles/reduction_v2.md``.
    """
    if threads not in _CONFIGS:
        raise ValueError(f"threads must be one of {sorted(_CONFIGS)}; got {threads}")

    data = data.astype(cp.float32).ravel()
    n = data.size

    output = cp.zeros(1, dtype=cp.float32)
    work_per_block = threads * _ITEMS
    grid = (n + work_per_block - 1) // work_per_block
    shared_mem = threads * 4

    load("reduction_v2", _CONFIGS[threads])(
        (grid,), (threads,), (data, output, n), shared_mem=shared_mem
    )
    return float(output[0])
