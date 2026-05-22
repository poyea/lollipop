import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BM = 64
_BN = 64
_BK = 32
_BLOCK = 128


def quantise_per_row_symmetric(X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Symmetric per-row INT8 quantisation.

    Returns ``(X_q, scale)`` with ``X_q`` ``int8`` and ``scale`` FP32 of
    shape ``X.shape[:-1]``. Reconstruction is ``X ≈ X_q * scale[..., None]``.
    """
    amax = cp.max(cp.abs(X), axis=-1, keepdims=True)
    amax = cp.maximum(amax, 1e-8)
    scale = (amax / 127.0).astype(cp.float32)
    Xq = cp.round(X.astype(cp.float32) / scale).clip(-127, 127).astype(cp.int8)
    return Xq, scale.squeeze(-1).astype(cp.float32)


def gemm_int8(
    A_q: cp.ndarray,
    B_q: cp.ndarray,
    a_scale: cp.ndarray,
    b_scale: cp.ndarray,
) -> cp.ndarray:
    """INT8 W8A8 GEMM with per-row + per-channel symmetric scales.

    Parameters
    ----------
    A_q : ``[M, K]`` int8 -- quantised activations (per-row scale).
    B_q : ``[N, K]`` int8 -- quantised weights, *pre-transposed* into
        ``[N, K]`` row-major (standard inference layout).
    a_scale : ``[M]`` float32 -- per-row activation scale.
    b_scale : ``[N]`` float32 -- per-channel weight scale.

    Returns
    -------
    ``[M, N]`` float32  ``C[m,n] = a_scale[m] * b_scale[n] * sum_k A_q[m,k] * B_q[n,k]``.

    Shape constraints: ``M % 64 == 0``, ``N % 64 == 0``, ``K % 32 == 0``.
    """
    if A_q.dtype != cp.int8 or B_q.dtype != cp.int8:
        raise ValueError(f"A_q and B_q must be int8; got {A_q.dtype}, {B_q.dtype}")
    if A_q.ndim != 2 or B_q.ndim != 2:
        raise ValueError(
            f"Expected 2D matrices, got A.ndim={A_q.ndim} B.ndim={B_q.ndim}"
        )
    M, K = A_q.shape
    N, Kb = B_q.shape
    if K != Kb:
        raise ValueError(
            f"Inner dim mismatch: A_q is [M={M}, K={K}], B_q is [N={N}, K={Kb}] "
            "(B_q must be pre-transposed to [N, K])"
        )
    if M % _BM or N % _BN or K % _BK:
        raise ValueError(
            f"gemm_int8 requires M%{_BM}==0, N%{_BN}==0, K%{_BK}==0; "
            f"got M={M}, N={N}, K={K}"
        )
    if a_scale.shape != (M,) or b_scale.shape != (N,):
        raise ValueError(
            f"a_scale must be [M={M}], b_scale must be [N={N}]; "
            f"got {a_scale.shape}, {b_scale.shape}"
        )

    A_q = cp.ascontiguousarray(A_q)
    B_q = cp.ascontiguousarray(B_q)
    a_scale = cp.ascontiguousarray(a_scale, dtype=cp.float32)
    b_scale = cp.ascontiguousarray(b_scale, dtype=cp.float32)

    if A_q.data.ptr % 16 or B_q.data.ptr % 16:
        raise ValueError("gemm_int8 requires 16-byte aligned A_q and B_q")

    C = cp.empty((M, N), dtype=cp.float32)
    grid = (N // _BN, M // _BM)
    load("gemm_int8")(
        grid,
        (_BLOCK,),
        (A_q, B_q, C, a_scale, b_scale, np.int32(M), np.int32(N), np.int32(K)),
    )
    return C
