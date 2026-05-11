import cupy as cp
import pytest

from lollipop import softmax
from lollipop.kernels.softmax_vec4 import softmax_vec4


@pytest.mark.parametrize("cols", [4, 5, 17, 64, 1023, 1024, 1025, 8192])
@pytest.mark.parametrize("rows", [1, 64])
def test_softmax_vec4_matches_scalar_2d(rows: int, cols: int) -> None:
    rng = cp.random.default_rng(0)
    x = rng.standard_normal((rows, cols), dtype=cp.float32) * 3.0
    expected = softmax(x)
    actual = softmax_vec4(x)
    assert actual.shape == expected.shape
    assert cp.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cols", [4, 5, 17, 64, 1023, 1024, 1025, 8192])
def test_softmax_vec4_matches_scalar_1d(cols: int) -> None:
    rng = cp.random.default_rng(1)
    x = rng.standard_normal(cols, dtype=cp.float32) * 3.0
    expected = softmax(x)
    actual = softmax_vec4(x)
    assert actual.shape == expected.shape
    assert cp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
