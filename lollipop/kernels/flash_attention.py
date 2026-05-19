import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BR_FP32 = 64
_BR_HMMA = 64
_HEAD_DIM = 64


def flash_attention(
    Q: cp.ndarray,
    K: cp.ndarray,
    V: cp.ndarray,
    causal: bool = False,
) -> cp.ndarray:
    """FlashAttention-2 forward (head_dim=64, single-head per BH).

    Inputs are 3D ``[BH, N, D]`` with ``D == 64`` and ``BH`` packing any
    leading batch / head dims.  4D ``[B, H, N, D]`` is accepted and
    flattened.  Returns the same shape and dtype as ``Q``.

    Dispatch:
      * ``float32`` -> scalar-FFMA kernel (``flash_attention.cu``).
      * ``float16`` -> wmma HMMA kernel with FP32 accumulator
        (``flash_attention_hmma.cu``).  Inputs/output FP16; ``m, l, O``
        stay FP32 internally.
    """
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q/K/V shape mismatch: {Q.shape}, {K.shape}, {V.shape}")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(f"Q/K/V dtype mismatch: {Q.dtype}, {K.dtype}, {V.dtype}")
    if Q.ndim == 4:
        B, H, N, D = Q.shape
        out_shape = (B, H, N, D)
        Q = Q.reshape(B * H, N, D)
        K = K.reshape(B * H, N, D)
        V = V.reshape(B * H, N, D)
    elif Q.ndim == 3:
        out_shape = Q.shape
    else:
        raise ValueError(f"Expected 3D [BH,N,D] or 4D [B,H,N,D], got {Q.ndim}D")

    BH, N, D = Q.shape
    if D != _HEAD_DIM:
        raise ValueError(f"flash_attention requires head_dim={_HEAD_DIM}, got D={D}")

    if Q.dtype == cp.float16:
        Q = cp.ascontiguousarray(Q)
        K = cp.ascontiguousarray(K)
        V = cp.ascontiguousarray(V)
        O = cp.empty_like(Q)
        grid = ((N + _BR_HMMA - 1) // _BR_HMMA, BH)
        load("flash_attention_hmma", "flash_attention_hmma_fwd")(
            grid,
            (128,),
            (Q, K, V, O, np.int32(N), np.int32(1 if causal else 0)),
        )
        return O.reshape(out_shape)

    Q = cp.ascontiguousarray(Q, dtype=cp.float32)
    K = cp.ascontiguousarray(K, dtype=cp.float32)
    V = cp.ascontiguousarray(V, dtype=cp.float32)
    O = cp.empty_like(Q)
    grid = ((N + _BR_FP32 - 1) // _BR_FP32, BH)
    load("flash_attention", "flash_attention_fwd")(
        grid,
        (_BR_FP32,),
        (Q, K, V, O, np.int32(N), np.int32(1 if causal else 0)),
    )
    return O.reshape(out_shape)
