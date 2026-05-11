from pathlib import Path

import cupy as cp

_SOURCES_DIR = Path(__file__).parent / "_sources"
_kernel = None
_BLOCK_SIZE = 256


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        source = (_SOURCES_DIR / "reduction_vec4.cu").read_text(encoding="utf-8")
        _kernel = cp.RawKernel(source, "reduction_vec4")
    return _kernel


def reduction_vec4(data: cp.ndarray) -> float:
    """Float4-vectorized sum reduction.

    Requires the input pointer to be 16-byte aligned (CuPy's default
    allocator satisfies this for fresh allocations and most views).
    Handles arbitrary length: the bulk is processed with float4 loads,
    the n % 4 tail elements are summed on-device via CuPy.
    """
    data = data.astype(cp.float32).ravel()
    n = data.size

    if data.data.ptr % 16 != 0:
        raise ValueError(
            "reduction_vec4 requires 16-byte aligned input; "
            "got ptr % 16 = "
            f"{data.data.ptr % 16}.  Make a fresh copy with cp.ascontiguousarray."
        )

    n_vec4 = n // 4
    output = cp.zeros(1, dtype=cp.float32)

    if n_vec4 > 0:
        grid = (n_vec4 + _BLOCK_SIZE * 2 - 1) // (_BLOCK_SIZE * 2)
        shared_mem = _BLOCK_SIZE * 4  # sizeof(float)
        # Reinterpret the float32 view as float4 for the kernel signature.
        _get_kernel()(
            (grid,), (_BLOCK_SIZE,), (data, output, n_vec4), shared_mem=shared_mem
        )

    tail_start = n_vec4 * 4
    if tail_start < n:
        output += data[tail_start:].sum()

    return float(output[0])
