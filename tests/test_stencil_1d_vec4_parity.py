import cupy as cp
import pytest

from lollipop.kernels.stencil_1d import stencil_1d
from lollipop.kernels.stencil_1d_vec4 import stencil_1d_vec4


@pytest.mark.parametrize("n", [16, 17, 1023, 1024, 1_000_003])
@pytest.mark.parametrize("radius", [1, 3, 5])
def test_stencil_1d_vec4_matches_scalar(n: int, radius: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)
    expected = stencil_1d(data, radius=radius)
    actual = stencil_1d_vec4(data, radius=radius)
    assert cp.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n", [64, 1024, 1_000_003])
@pytest.mark.parametrize("radius", [1, 3, 5])
def test_stencil_1d_vec4_boundary_exact(n: int, radius: int) -> None:
    """Boundary lanes (first/last `radius` elements) are the lanes most
    likely to break under vectorisation — check them exactly."""
    rng = cp.random.default_rng(1)
    data = rng.standard_normal(n, dtype=cp.float32)
    expected = stencil_1d(data, radius=radius)
    actual = stencil_1d_vec4(data, radius=radius)
    k = radius + 4
    assert cp.allclose(actual[:k], expected[:k], atol=1e-5, rtol=1e-5)
    assert cp.allclose(actual[-k:], expected[-k:], atol=1e-5, rtol=1e-5)
