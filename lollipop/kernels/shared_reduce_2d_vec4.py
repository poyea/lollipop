import cupy as cp

from lollipop.kernels._raw import load

_BLOCK_SIZE = (16, 16)


def shared_reduce_2d_vec4(data: cp.ndarray) -> float:
    """Float4-vectorized 2D sum reduction.

    Requires the input pointer to be 16-byte aligned (CuPy's default
    allocator satisfies this for fresh allocations).  Handles arbitrary
    width: the largest (width // 4) * 4 column prefix is processed with
    float4 loads, the (width % 4) tail columns are summed on-device via
    CuPy.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    data = cp.ascontiguousarray(data.astype(cp.float32))
    height, width = data.shape

    if data.data.ptr % 16 != 0:
        raise ValueError(
            "shared_reduce_2d_vec4 requires 16-byte aligned input; "
            "got ptr % 16 = "
            f"{data.data.ptr % 16}.  Make a fresh copy with cp.ascontiguousarray."
        )

    output = cp.zeros(1, dtype=cp.float32)
    width_vec4 = width // 4

    if width_vec4 > 0 and height > 0:
        if width % 4 == 0:
            head = data
        else:
            head = cp.ascontiguousarray(data[:, : width_vec4 * 4])

        grid = (
            (width_vec4 + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
            (height + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
        )
        shared_mem = _BLOCK_SIZE[0] * _BLOCK_SIZE[1] * 4

        load("shared_reduce_2d_vec4")(
            grid,
            _BLOCK_SIZE,
            (head, output, width_vec4, height),
            shared_mem=shared_mem,
        )

    tail_start = width_vec4 * 4
    if tail_start < width:
        output += data[:, tail_start:].sum()

    return float(output[0])
