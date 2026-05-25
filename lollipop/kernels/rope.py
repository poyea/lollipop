import cupy as cp
import numpy as np

from lollipop.kernels._raw import load

_BLOCK = 256


def rope(
    x: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    out: cp.ndarray | None = None,
) -> cp.ndarray:
    """Apply Rotary Positional Embedding (Llama / HF half-rotation convention).

    Computes, per row::

        y[..., :D/2] = x[..., :D/2] * cos - x[..., D/2:] * sin
        y[..., D/2:] = x[..., :D/2] * sin + x[..., D/2:] * cos

    Parameters
    ----------
    x   : ``[N, D]``  fp16 or fp32, row-major.  ``N`` packs whatever outer
          dims the caller wants (``B * S * H`` typical).  ``D`` must be a
          multiple of 8 (keeps the v1.1 vectorised path open).
    cos : ``[N, D/2]`` fp32 -- pre-gathered cosine table.
    sin : ``[N, D/2]`` fp32 -- pre-gathered sine table.
    out : optional output buffer.  Can alias ``x`` for in-place; the
          kernel reads both halves before writing either.

    Returns ``[N, D]`` with the same dtype as ``x``.

    Notes
    -----
    The ``(cos, sin)`` pre-gather is a v1 shortcut.  Production callers
    (vLLM / HF / FlashAttention) pass ``(position_ids, cos_cache,
    sin_cache)`` and gather inside the kernel, saving ``B*S*H*D/2`` of
    redundant traffic per layer.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, D]; got {x.shape}")
    N, D = x.shape
    if D % 8 != 0:
        raise ValueError(f"D must be a multiple of 8; got {D}")
    if x.dtype not in (cp.float16, cp.float32):
        raise ValueError(f"x must be float16 or float32; got {x.dtype}")
    half = D // 2
    if cos.shape != (N, half) or sin.shape != (N, half):
        raise ValueError(
            f"cos and sin must be [{N}, {half}]; got {cos.shape}, {sin.shape}"
        )
    if cos.dtype != cp.float32 or sin.dtype != cp.float32:
        raise ValueError(
            f"cos/sin must be fp32 (HF convention); got {cos.dtype}, {sin.dtype}"
        )

    cos = cp.ascontiguousarray(cos)
    sin = cp.ascontiguousarray(sin)

    # Require x C-contiguous along D (the fast axis). row_stride may
    # exceed D for callers passing a padded view.
    elem = x.itemsize
    if x.strides[1] != elem:
        if out is x:
            raise ValueError(
                "in-place rope requires x contiguous along D; "
                "got strides[1] != itemsize"
            )
        x = cp.ascontiguousarray(x)
    x_stride = x.strides[0] // elem
    if x_stride < D:
        raise ValueError(f"x_stride ({x_stride}) must be >= D ({D})")

    if out is None:
        # Default: fresh contiguous output ([N, D] tight).
        out = cp.empty((N, D), dtype=x.dtype)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(
            f"out must be {x.shape} with dtype {x.dtype}; got {out.shape}, {out.dtype}"
        )
    elif out.strides[1] != elem:
        raise ValueError(
            f"out must be contiguous along D; got strides={out.strides}"
        )
    y_stride = out.strides[0] // elem
    if y_stride < D:
        raise ValueError(f"y_stride ({y_stride}) must be >= D ({D})")

    kernel_name = "rope_fp16" if x.dtype == cp.float16 else "rope_fp32"
    load("rope", kernel_name)(
        (N,),
        (_BLOCK,),
        (x, cos, sin, out, np.int32(D), np.int32(x_stride), np.int32(y_stride)),
    )
    return out
