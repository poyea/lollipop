import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK = 256


def rmsnorm(
    x: cp.ndarray,
    gamma: cp.ndarray,
    eps: float = 1e-5,
) -> cp.ndarray:
    """Root-Mean-Square LayerNorm forward (Llama / T5 style, no bias, no mean-subtract).

    Computes, row-wise over the last axis::

        y = x / sqrt(mean(x**2) + eps) * gamma

    Parameters
    ----------
    x     : ``[N, H]`` float16 or float32.
    gamma : ``[H]`` per-channel scale, same dtype as ``x``.
    eps   : numerical stabiliser inside the RMS.

    Returns ``[N, H]`` array of the same dtype as ``x``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, H]; got {x.shape}")
    N, H = x.shape
    if x.dtype not in (cp.float16, cp.float32):
        raise ValueError(f"x must be float16 or float32; got {x.dtype}")
    if gamma.shape != (H,) or gamma.dtype != x.dtype:
        raise ValueError(
            f"gamma must be [{H}] with dtype {x.dtype}; got {gamma.shape}, {gamma.dtype}"
        )

    x = cp.ascontiguousarray(x)
    gamma = cp.ascontiguousarray(gamma)
    y = cp.empty_like(x)

    kernel_name = "rmsnorm_fwd_fp16" if x.dtype == cp.float16 else "rmsnorm_fwd_fp32"
    load("rmsnorm", kernel_name)(
        (N,),
        (_BLOCK,),
        (x, gamma, y, np.int32(H), np.float32(eps)),
    )
    return y


def rmsnorm_backward(
    dy: cp.ndarray,
    x: cp.ndarray,
    gamma: cp.ndarray,
    eps: float = 1e-5,
) -> tuple[cp.ndarray, cp.ndarray]:
    """RMSNorm backward.

    Given upstream gradient ``dy`` of shape ``[N, H]`` and the same ``x, gamma,
    eps`` that produced the forward, returns ``(dx, dgamma)`` with::

        n      = x * rrms                  (rrms recomputed)
        dot    = (1/H) * sum_j(dy_j * gamma_j * n_j)
        dx     = rrms * (dy * gamma - n * dot)
        dgamma = sum over rows of (dy * n)

    ``dgamma`` is accumulated in fp32 inside the kernel and cast back to
    ``x.dtype`` on return, matching ``torch.nn.functional.rms_norm`` conventions.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, H]; got {x.shape}")
    N, H = x.shape
    if x.dtype not in (cp.float16, cp.float32):
        raise ValueError(f"x must be float16 or float32; got {x.dtype}")
    if dy.shape != x.shape or dy.dtype != x.dtype:
        raise ValueError(
            f"dy must be {x.shape} with dtype {x.dtype}; got {dy.shape}, {dy.dtype}"
        )
    if gamma.shape != (H,) or gamma.dtype != x.dtype:
        raise ValueError(
            f"gamma must be [{H}] with dtype {x.dtype}; got {gamma.shape}, {gamma.dtype}"
        )

    x = cp.ascontiguousarray(x)
    dy = cp.ascontiguousarray(dy)
    gamma = cp.ascontiguousarray(gamma)
    dx = cp.empty_like(x)
    dgamma_fp32 = cp.zeros((H,), dtype=cp.float32)

    kernel_name = "rmsnorm_bwd_fp16" if x.dtype == cp.float16 else "rmsnorm_bwd_fp32"
    load("rmsnorm", kernel_name)(
        (N,),
        (_BLOCK,),
        (dy, x, gamma, dx, dgamma_fp32, np.int32(H), np.float32(eps)),
    )
    dgamma = dgamma_fp32.astype(x.dtype, copy=False)
    return dx, dgamma
