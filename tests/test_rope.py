import cupy as cp
import pytest

from lollipop import rope


def _make_cos_sin(N: int, D: int, base: float = 10000.0):
    """HF-style cos/sin table: theta_i = base^(-2i/D), m_n = n."""
    half = D // 2
    inv_freq = 1.0 / (base ** (cp.arange(half, dtype=cp.float32) * 2.0 / D))
    positions = cp.arange(N, dtype=cp.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    return cp.cos(freqs), cp.sin(freqs)


def _rope_ref(x: cp.ndarray, cos: cp.ndarray, sin: cp.ndarray) -> cp.ndarray:
    """HF rotate_half reference: y = x*cos + rotate_half(x)*sin
    where rotate_half(x) = cat(-x_hi, x_lo)."""
    x32 = x.astype(cp.float32)
    half = x32.shape[-1] // 2
    x_lo, x_hi = x32[..., :half], x32[..., half:]
    rot = cp.concatenate([-x_hi, x_lo], axis=-1)
    cos_full = cp.concatenate([cos, cos], axis=-1)
    sin_full = cp.concatenate([sin, sin], axis=-1)
    y = x32 * cos_full + rot * sin_full
    return y.astype(x.dtype)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
@pytest.mark.parametrize("N,D", [(4, 64), (32, 128), (1, 256), (17, 64), (257, 128)])
def test_rope_matches_hf_rotate_half(dtype, N, D):
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((N, D), dtype=cp.float32).astype(dtype)
    cos, sin = _make_cos_sin(N, D)

    actual = rope(x, cos, sin)
    expected = _rope_ref(x, cos, sin)

    if dtype == cp.float32:
        tol = dict(atol=1e-5, rtol=1e-5)
    else:
        tol = dict(atol=3e-3, rtol=3e-3)
    assert actual.dtype == dtype
    assert cp.allclose(actual, expected, **tol), (
        f"max abs err = "
        f"{float(cp.abs(actual.astype(cp.float32) - expected.astype(cp.float32)).max())}"
    )


@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
def test_rope_inplace_safe(dtype):
    N, D = 16, 64
    rng = cp.random.default_rng(1)
    x = rng.standard_normal((N, D), dtype=cp.float32).astype(dtype)
    cos, sin = _make_cos_sin(N, D)
    expected = _rope_ref(x, cos, sin)

    y = rope(x, cos, sin, out=x)
    assert y is x
    tol = dict(atol=1e-5, rtol=1e-5) if dtype == cp.float32 else dict(atol=3e-3, rtol=3e-3)
    assert cp.allclose(y, expected, **tol)


def test_rope_handles_padded_rows():
    """row_stride > D: x is a view into a [N, D+pad] buffer."""
    N, D, pad = 8, 64, 16
    rng = cp.random.default_rng(2)
    buf = rng.standard_normal((N, D + pad), dtype=cp.float32)
    x_view = buf[:, :D]  # contiguous along D, row stride is D+pad
    assert x_view.strides[1] == buf.itemsize
    assert x_view.strides[0] == (D + pad) * buf.itemsize

    cos, sin = _make_cos_sin(N, D)
    actual = rope(x_view, cos, sin)
    expected = _rope_ref(cp.ascontiguousarray(x_view), cos, sin)
    assert cp.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rope_rejects_odd_D():
    x = cp.zeros((4, 65), dtype=cp.float32)
    cos = cp.zeros((4, 32), dtype=cp.float32)
    sin = cp.zeros((4, 32), dtype=cp.float32)
    with pytest.raises(ValueError, match="multiple of 8"):
        rope(x, cos, sin)


def test_rope_rejects_non_multiple_of_8():
    x = cp.zeros((4, 4), dtype=cp.float32)
    cos = cp.zeros((4, 2), dtype=cp.float32)
    sin = cp.zeros((4, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="multiple of 8"):
        rope(x, cos, sin)


def test_rope_rejects_bad_cos_dtype():
    x = cp.zeros((4, 64), dtype=cp.float32)
    cos = cp.zeros((4, 32), dtype=cp.float16)
    sin = cp.zeros((4, 32), dtype=cp.float16)
    with pytest.raises(ValueError, match="fp32"):
        rope(x, cos, sin)


def test_rope_rejects_shape_mismatch():
    x = cp.zeros((4, 64), dtype=cp.float32)
    cos = cp.zeros((4, 31), dtype=cp.float32)
    sin = cp.zeros((4, 32), dtype=cp.float32)
    with pytest.raises(ValueError):
        rope(x, cos, sin)
