import cupy as cp
import pytest

from lollipop import fused_ffn_tail


def _rmsnorm_ref(x: cp.ndarray, gamma: cp.ndarray, eps: float) -> cp.ndarray:
    x32 = x.astype(cp.float32)
    rrms = cp.reciprocal(cp.sqrt((x32 * x32).mean(axis=-1, keepdims=True) + eps))
    return (x32 * rrms * gamma.astype(cp.float32)).astype(x.dtype)


def _gelu_tanh(a: cp.ndarray) -> cp.ndarray:
    a32 = a.astype(cp.float32)
    k = 0.7978845608028654
    return (0.5 * a32 * (1.0 + cp.tanh(k * (a32 + 0.044715 * a32 * a32 * a32)))).astype(
        a.dtype
    )


def _silu(a: cp.ndarray) -> cp.ndarray:
    a32 = a.astype(cp.float32)
    return (a32 / (1.0 + cp.exp(-a32))).astype(a.dtype)


_ACTS = {"gelu": _gelu_tanh, "silu": _silu}


def _reference(
    x: cp.ndarray,
    gamma: cp.ndarray,
    bias: cp.ndarray | None,
    residual: cp.ndarray | None,
    eps: float,
    activation: str,
) -> cp.ndarray:
    n = _rmsnorm_ref(x, gamma, eps)
    a = n + bias if bias is not None else n
    g = _ACTS[activation](a)
    return g + residual if residual is not None else g


@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
@pytest.mark.parametrize("M,H", [(2, 128), (8, 512), (64, 1024), (3, 257)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_fused_ffn_tail_matches_reference(
    dtype, M, H, with_bias, with_residual, activation
):
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((M, H), dtype=cp.float32).astype(dtype)
    gamma = (1.0 + 0.1 * rng.standard_normal((H,), dtype=cp.float32)).astype(dtype)
    bias = (
        (0.1 * rng.standard_normal((H,), dtype=cp.float32)).astype(dtype)
        if with_bias
        else None
    )
    residual = (
        (0.1 * rng.standard_normal((M, H), dtype=cp.float32)).astype(dtype)
        if with_residual
        else None
    )
    eps = 1e-5

    actual = fused_ffn_tail(x, gamma, bias, residual, eps=eps, activation=activation)
    expected = _reference(x, gamma, bias, residual, eps, activation)

    assert actual.dtype == dtype
    if dtype == cp.float32:
        tol = dict(atol=1e-5, rtol=1e-5)
    else:
        tol = dict(atol=3e-3, rtol=3e-3)
    assert cp.allclose(
        actual, expected, **tol
    ), f"max abs err = {float(cp.abs(actual.astype(cp.float32) - expected.astype(cp.float32)).max())}"


def test_fused_ffn_tail_rejects_bad_shape():
    x = cp.zeros((4, 64), dtype=cp.float32)
    gamma = cp.ones((63,), dtype=cp.float32)
    with pytest.raises(ValueError):
        fused_ffn_tail(x, gamma)


def test_fused_ffn_tail_rejects_bad_dtype():
    x = cp.zeros((4, 64), dtype=cp.float64)
    gamma = cp.ones((64,), dtype=cp.float64)
    with pytest.raises(ValueError):
        fused_ffn_tail(x, gamma)


def test_fused_ffn_tail_rejects_unknown_activation():
    x = cp.zeros((4, 64), dtype=cp.float32)
    gamma = cp.ones((64,), dtype=cp.float32)
    with pytest.raises(ValueError):
        fused_ffn_tail(x, gamma, activation="relu")
