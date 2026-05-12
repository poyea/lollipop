import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BM = 128
_BN = 128
_BK = 8
_BLOCK_SIZE = 256


def gemm_tiled(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    """Shared-mem double-buffered tiled SGEMM.

    Shape constraints (v1): ``M, N`` multiples of 128 and ``K`` a multiple
    of 8. Inputs must be FP32, row-major contiguous and 16-byte aligned
    (CuPy's default allocator satisfies alignment for fresh allocations).
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got A.ndim={A.ndim} B.ndim={B.ndim}")
    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"Inner dim mismatch: A is {A.shape}, B is {B.shape}")
    if M % _BM or N % _BN or K % _BK:
        raise ValueError(
            f"gemm_tiled v1 requires M%{_BM}==0, N%{_BN}==0, K%{_BK}==0; "
            f"got M={M}, N={N}, K={K}"
        )

    A = cp.ascontiguousarray(A, dtype=cp.float32)
    B = cp.ascontiguousarray(B, dtype=cp.float32)

    if A.data.ptr % 16 or B.data.ptr % 16:
        raise ValueError("gemm_tiled requires 16-byte aligned A and B")

    C = cp.empty((M, N), dtype=cp.float32)

    grid = (N // _BN, M // _BM)
    load("gemm_tiled")(
        grid,
        (_BLOCK_SIZE,),
        (A, B, C, np.int32(M), np.int32(N), np.int32(K)),
    )
    return C
