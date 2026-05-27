import cupy as cp
import pytest

from lollipop import rmsnorm, rmsnorm_backward

# ---------------- numpy/cupy references ----------------


def _rmsnorm_ref(x: cp.ndarray, gamma: cp.ndarray, eps: float) -> cp.ndarray:
    x32 = x.astype(cp.float32)
    rrms = cp.reciprocal(cp.sqrt(cp.mean(x32 * x32, axis=-1, keepdims=True) + eps))
    y = x32 * rrms * gamma.astype(cp.float32)
    return y.astype(x.dtype)


def _rmsnorm_bwd_ref(dy, x, gamma, eps):
    """Reference backward in fp32."""
    x32 = x.astype(cp.float32)
    dy32 = dy.astype(cp.float32)
    g32 = gamma.astype(cp.float32)
    H = x32.shape[-1]
    rrms = cp.reciprocal(cp.sqrt(cp.mean(x32 * x32, axis=-1, keepdims=True) + eps))
    n = x32 * rrms
    dn = dy32 * g32
    dot = cp.sum(dn * n, axis=-1, keepdims=True) / H
    dx = rrms * (dn - n * dot)
    dgamma = cp.sum(dy32 * n, axis=0)
    return dx.astype(x.dtype), dgamma.astype(x.dtype)


# ---------------- forward parity ----------------


@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
@pytest.mark.parametrize("N,H", [(4, 64), (32, 128), (1, 256), (17, 65), (257, 4096)])
def test_rmsnorm_forward_parity(dtype, N, H):
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((N, H), dtype=cp.float32).astype(dtype)
    gamma = rng.standard_normal((H,), dtype=cp.float32).astype(dtype)

    actual = rmsnorm(x, gamma, eps=1e-5)
    expected = _rmsnorm_ref(x, gamma, eps=1e-5)

    tol = (
        dict(atol=1e-5, rtol=1e-5)
        if dtype == cp.float32
        else dict(atol=3e-3, rtol=3e-3)
    )
    assert actual.dtype == dtype
    assert actual.shape == (N, H)
    assert cp.allclose(actual, expected, **tol), (
        f"max abs err = "
        f"{float(cp.abs(actual.astype(cp.float32) - expected.astype(cp.float32)).max())}"
    )


# ---------------- backward parity ----------------


@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
@pytest.mark.parametrize("N,H", [(4, 64), (32, 128), (17, 65), (257, 1024)])
def test_rmsnorm_backward_parity(dtype, N, H):
    rng = cp.random.default_rng(1)
    x = rng.standard_normal((N, H), dtype=cp.float32).astype(dtype)
    gamma = rng.standard_normal((H,), dtype=cp.float32).astype(dtype)
    dy = rng.standard_normal((N, H), dtype=cp.float32).astype(dtype)

    dx, dgamma = rmsnorm_backward(dy, x, gamma, eps=1e-5)
    dx_ref, dgamma_ref = _rmsnorm_bwd_ref(dy, x, gamma, eps=1e-5)

    if dtype == cp.float32:
        tol = dict(atol=2e-5, rtol=2e-5)
        # dgamma sums N rows; loosen a touch.
        dg_tol = dict(atol=1e-4, rtol=1e-4)
    else:
        tol = dict(atol=5e-3, rtol=5e-3)
        dg_tol = dict(atol=2e-2, rtol=2e-2)

    assert dx.dtype == dtype and dx.shape == (N, H)
    assert dgamma.dtype == dtype and dgamma.shape == (H,)
    assert cp.allclose(dx, dx_ref, **tol), (
        f"dx max abs err = "
        f"{float(cp.abs(dx.astype(cp.float32) - dx_ref.astype(cp.float32)).max())}"
    )
    assert cp.allclose(dgamma, dgamma_ref, **dg_tol), (
        f"dgamma max abs err = "
        f"{float(cp.abs(dgamma.astype(cp.float32) - dgamma_ref.astype(cp.float32)).max())}"
    )


# ---------------- rejections ----------------


def test_rmsnorm_rejects_non_2d():
    x = cp.zeros((4, 8, 16), dtype=cp.float32)
    gamma = cp.zeros((16,), dtype=cp.float32)
    with pytest.raises(ValueError, match="2D"):
        rmsnorm(x, gamma)


def test_rmsnorm_rejects_bad_gamma_shape():
    x = cp.zeros((4, 64), dtype=cp.float32)
    gamma = cp.zeros((32,), dtype=cp.float32)
    with pytest.raises(ValueError, match="gamma"):
        rmsnorm(x, gamma)


def test_rmsnorm_rejects_dtype_mismatch():
    x = cp.zeros((4, 64), dtype=cp.float32)
    gamma = cp.zeros((64,), dtype=cp.float16)
    with pytest.raises(ValueError, match="dtype"):
        rmsnorm(x, gamma)


def test_rmsnorm_rejects_bad_dtype():
    x = cp.zeros((4, 64), dtype=cp.float64)
    gamma = cp.zeros((64,), dtype=cp.float64)
    with pytest.raises(ValueError, match="float16 or float32"):
        rmsnorm(x, gamma)


def test_rmsnorm_backward_rejects_dy_shape_mismatch():
    x = cp.zeros((4, 64), dtype=cp.float32)
    gamma = cp.zeros((64,), dtype=cp.float32)
    dy = cp.zeros((4, 32), dtype=cp.float32)
    with pytest.raises(ValueError, match="dy"):
        rmsnorm_backward(dy, x, gamma)
