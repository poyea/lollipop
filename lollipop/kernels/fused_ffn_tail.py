import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK = 256

_ACTIVATIONS = {"gelu": 0, "gelu_tanh": 0, "silu": 1, "swish": 1}


def fused_ffn_tail(
    x: cp.ndarray,
    gamma: cp.ndarray,
    bias: cp.ndarray | None = None,
    residual: cp.ndarray | None = None,
    eps: float = 1e-5,
    activation: str = "gelu",
) -> cp.ndarray:
    """Fused RMSNorm + scale + (optional) bias + activation + (optional) residual.

    Computes, row-wise over the last axis::

        n = x / sqrt(mean(x**2) + eps) * gamma
        a = n + bias                    # if bias is not None
        y = act(a)                      # gelu_tanh or silu
        y = y + residual                # if residual is not None

    where ``act`` is the tanh-approximation GELU (matches
    ``F.gelu(approximate='tanh')``) or SiLU (``a * sigmoid(a)``,
    a.k.a. Swish, the Llama FFN activation).

    Parameters
    ----------
    x          : ``[M, H]`` float16 or float32.
    gamma      : ``[H]`` -- RMSNorm scale, same dtype as ``x``.
    bias       : ``[H]`` or ``None`` -- optional per-channel bias.
    residual   : ``[M, H]`` or ``None`` -- optional skip connection.
    eps        : numerical stabiliser inside the RMS.
    activation : ``"gelu"`` (default) or ``"silu"``.

    Returns ``[M, H]`` array of the same dtype as ``x``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [M, H]; got {x.shape}")
    M, H = x.shape
    if x.dtype not in (cp.float16, cp.float32):
        raise ValueError(f"x must be float16 or float32; got {x.dtype}")
    if gamma.shape != (H,) or gamma.dtype != x.dtype:
        raise ValueError(
            f"gamma must be [{H}] with dtype {x.dtype}; got {gamma.shape}, {gamma.dtype}"
        )
    if bias is not None and (bias.shape != (H,) or bias.dtype != x.dtype):
        raise ValueError(
            f"bias must be [{H}] with dtype {x.dtype}; got {bias.shape}, {bias.dtype}"
        )
    if residual is not None and (
        residual.shape != (M, H) or residual.dtype != x.dtype
    ):
        raise ValueError(
            f"residual must be [{M}, {H}] with dtype {x.dtype}; "
            f"got {residual.shape}, {residual.dtype}"
        )
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {sorted(set(_ACTIVATIONS))}; got {activation!r}"
        )
    act_id = _ACTIVATIONS[activation]

    x = cp.ascontiguousarray(x)
    gamma = cp.ascontiguousarray(gamma)
    bias = cp.ascontiguousarray(bias) if bias is not None else None
    residual = cp.ascontiguousarray(residual) if residual is not None else None
    y = cp.empty_like(x)

    kernel_name = (
        "fused_ffn_tail_fp16" if x.dtype == cp.float16 else "fused_ffn_tail_fp32"
    )
    # CuPy passes Python int -> CUDA pointer via the integer-as-pointer path; 0 == nullptr.
    bias_arg = bias if bias is not None else np.uint64(0)
    res_arg = residual if residual is not None else np.uint64(0)

    load("fused_ffn_tail", kernel_name)(
        (M,),
        (_BLOCK,),
        (x, gamma, bias_arg, res_arg, y, np.int32(H), np.float32(eps), np.int32(act_id)),
    )
    return y
