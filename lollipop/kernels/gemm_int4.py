import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BM = 64
_BN = 64
_BK = 64
_GROUP = 64
_BLOCK = 128


def pack_int4_w4a16(
    W: cp.ndarray, group: int = _GROUP
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Asymmetric per-group INT4 weight quantisation (AWQ/GPTQ convention).

    Parameters
    ----------
    W : ``[N, K]`` float -- weight tensor, output-channel-major (i.e. already
        in the "linear's stored weight" layout).
    group : group size along K. Default 64; must divide ``K``.

    Returns
    -------
    Wq     : ``[N, K // 2]`` uint8 -- 2 nibbles per byte, low nibble = even k,
        high nibble = odd k.
    scales : ``[K // group, N]`` float16
    zeros  : ``[K // group, N]`` float16 -- stored as float so the kernel can
        load via the same path; values are integers in ``[0, 15]``.

    Reconstruction: ``W[k, n] = scales[k//group, n] * (Wq[n, k] - zeros[k//group, n])``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D [N, K]; got {W.shape}")
    N, K = W.shape
    if K % group:
        raise ValueError(f"K={K} must be a multiple of group={group}")
    G = K // group

    W = W.astype(cp.float32)
    W_grouped = W.reshape(N, G, group)
    wmin = W_grouped.min(axis=-1, keepdims=True)
    wmax = W_grouped.max(axis=-1, keepdims=True)
    scale = (wmax - wmin) / 15.0
    scale = cp.maximum(scale, 1e-8)
    zero = cp.round(-wmin / scale).clip(0, 15)

    q = cp.round(W_grouped / scale + zero).clip(0, 15).astype(cp.uint8)
    q = q.reshape(N, K)

    q_low = q[:, 0::2]  # [N, K/2]  k-even nibbles
    q_high = q[:, 1::2]  # [N, K/2]  k-odd nibbles
    packed = (q_low | (q_high << 4)).astype(cp.uint8)

    scales_out = scale.squeeze(-1).T.astype(cp.float16)  # [G, N]
    zeros_out = zero.squeeze(-1).T.astype(cp.float16)  # [G, N]
    return packed, scales_out, zeros_out


def dequantise_int4(
    Wq: cp.ndarray, scales: cp.ndarray, zeros: cp.ndarray, K: int, group: int = _GROUP
) -> cp.ndarray:
    """Inverse of :func:`pack_int4_w4a16`, for testing.  Returns ``[N, K] fp32``."""
    N, half = Wq.shape
    assert 2 * half == K
    G = K // group
    q = cp.empty((N, K), dtype=cp.uint8)
    q[:, 0::2] = Wq & 0x0F
    q[:, 1::2] = (Wq >> 4) & 0x0F
    q = q.astype(cp.float32).reshape(N, G, group)
    s = scales.T.astype(cp.float32).reshape(N, G, 1)
    z = zeros.T.astype(cp.float32).reshape(N, G, 1)
    return (s * (q - z)).reshape(N, K)


def gemm_int4(
    A: cp.ndarray,
    Wq: cp.ndarray,
    scales: cp.ndarray,
    zeros: cp.ndarray,
) -> cp.ndarray:
    """INT4 weight-only GEMM (W4A16).

    Parameters
    ----------
    A      : ``[M, K]`` float16 -- activations.
    Wq     : ``[N, K // 2]`` uint8 -- packed weights from :func:`pack_int4_w4a16`.
    scales : ``[K // 64, N]`` float16
    zeros  : ``[K // 64, N]`` float16

    Returns ``[M, N] float16`` :math:`C = A \\cdot W^{\\!\\top}` with `W`
    dequantised inside the kernel.
    """
    if A.dtype != cp.float16:
        raise ValueError(f"A must be float16; got {A.dtype}")
    if Wq.dtype != cp.uint8:
        raise ValueError(f"Wq must be uint8 (packed int4); got {Wq.dtype}")
    if scales.dtype != cp.float16 or zeros.dtype != cp.float16:
        raise ValueError(
            f"scales/zeros must be float16; got {scales.dtype}, {zeros.dtype}"
        )
    if A.ndim != 2 or Wq.ndim != 2:
        raise ValueError(f"A and Wq must be 2D; got {A.shape}, {Wq.shape}")

    M, K = A.shape
    N, K_half = Wq.shape
    if K_half * 2 != K:
        raise ValueError(
            f"Wq's second dim must be K/2; got Wq.shape={Wq.shape}, A.K={K}"
        )
    if M % _BM or N % _BN or K % _BK:
        raise ValueError(
            f"gemm_int4 requires M%{_BM}==0, N%{_BN}==0, K%{_BK}==0; "
            f"got M={M}, N={N}, K={K}"
        )
    G = K // _GROUP
    if scales.shape != (G, N) or zeros.shape != (G, N):
        raise ValueError(
            f"scales and zeros must be [G={G}, N={N}]; "
            f"got {scales.shape}, {zeros.shape}"
        )

    A = cp.ascontiguousarray(A)
    Wq = cp.ascontiguousarray(Wq)
    scales = cp.ascontiguousarray(scales)
    zeros = cp.ascontiguousarray(zeros)

    if A.data.ptr % 16 or Wq.data.ptr % 16:
        raise ValueError("gemm_int4 requires 16-byte aligned A and Wq")

    C = cp.empty((M, N), dtype=cp.float16)
    grid = (N // _BN, M // _BM)
    load("gemm_int4")(
        grid,
        (_BLOCK,),
        (A, Wq, scales, zeros, C, np.int32(M), np.int32(N), np.int32(K)),
    )
    return C
