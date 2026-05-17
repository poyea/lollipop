import cupy as cp
import pytest

from lollipop import matrix_transpose, matrix_transpose_naive


@pytest.mark.parametrize(
    "h,w",
    [(64, 64), (256, 512), (1024, 1024), (1024, 4096)],
)
def test_matrix_transpose_naive_matches_tiled(h: int, w: int) -> None:
    rng = cp.random.default_rng(0)
    m = rng.standard_normal((h, w), dtype=cp.float32)
    expected = matrix_transpose(m)
    actual = matrix_transpose_naive(m)
    assert cp.array_equal(actual, expected)
