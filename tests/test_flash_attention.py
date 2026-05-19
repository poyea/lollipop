import cupy as cp
import pytest

from lollipop import flash_attention


def _reference(Q: cp.ndarray, K: cp.ndarray, V: cp.ndarray, causal: bool) -> cp.ndarray:
    """Numerically-stable reference attention via CuPy: softmax(QK^T/sqrt(D))V."""
    D = Q.shape[-1]
    scale = 1.0 / float(D) ** 0.5
    scores = cp.einsum("bnd,bmd->bnm", Q, K) * scale
    if causal:
        N = Q.shape[-2]
        mask = cp.triu(cp.ones((N, N), dtype=cp.bool_), k=1)
        scores = cp.where(mask[None, :, :], cp.float32(-1e30), scores)
    m = scores.max(axis=-1, keepdims=True)
    p = cp.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    return cp.einsum("bnm,bmd->bnd", p, V)


@pytest.mark.parametrize("BH,N", [(1, 64), (2, 128), (4, 130), (1, 512)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_matches_reference(BH: int, N: int, causal: bool) -> None:
    D = 64
    rng = cp.random.default_rng(0)
    Q = rng.standard_normal((BH, N, D), dtype=cp.float32)
    K = rng.standard_normal((BH, N, D), dtype=cp.float32)
    V = rng.standard_normal((BH, N, D), dtype=cp.float32)

    expected = _reference(Q, K, V, causal)
    actual = flash_attention(Q, K, V, causal=causal)

    assert actual.shape == expected.shape
    assert cp.allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BH,N", [(1, 64), (2, 128), (4, 130), (1, 512)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_hmma_matches_reference(BH: int, N: int, causal: bool) -> None:
    D = 64
    rng = cp.random.default_rng(0)
    Q32 = rng.standard_normal((BH, N, D), dtype=cp.float32)
    K32 = rng.standard_normal((BH, N, D), dtype=cp.float32)
    V32 = rng.standard_normal((BH, N, D), dtype=cp.float32)

    expected = _reference(Q32, K32, V32, causal)

    Q = Q32.astype(cp.float16)
    K = K32.astype(cp.float16)
    V = V32.astype(cp.float16)
    actual = flash_attention(Q, K, V, causal=causal)

    assert actual.dtype == cp.float16
    assert actual.shape == expected.shape
    assert cp.allclose(actual.astype(cp.float32), expected, atol=1e-2, rtol=1e-2)


def test_flash_attention_accepts_4d() -> None:
    rng = cp.random.default_rng(1)
    Q = rng.standard_normal((2, 3, 128, 64), dtype=cp.float32)
    K = rng.standard_normal((2, 3, 128, 64), dtype=cp.float32)
    V = rng.standard_normal((2, 3, 128, 64), dtype=cp.float32)
    O = flash_attention(Q, K, V, causal=True)
    assert O.shape == Q.shape


def test_flash_attention_rejects_bad_head_dim() -> None:
    Q = cp.zeros((1, 64, 32), dtype=cp.float32)
    with pytest.raises(ValueError):
        flash_attention(Q, Q, Q)
