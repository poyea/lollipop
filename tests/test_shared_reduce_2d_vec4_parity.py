import cupy as cp
import pytest

from lollipop.kernels.shared_reduce_2d import shared_reduce_2d
from lollipop.kernels.shared_reduce_2d_vec4 import shared_reduce_2d_vec4


@pytest.mark.parametrize("rows", [1, 16, 256])
@pytest.mark.parametrize("cols", [4, 5, 1023, 1024, 8192])
def test_shared_reduce_2d_vec4_matches_scalar(rows: int, cols: int) -> None:
    rng = cp.random.default_rng(rows * 100003 + cols)
    data = rng.standard_normal((rows, cols), dtype=cp.float32)
    expected = shared_reduce_2d(data)
    actual = shared_reduce_2d_vec4(data)
    assert cp.allclose(
        cp.asarray(actual), cp.asarray(expected), atol=1e-3, rtol=1e-5
    ), f"vec4={actual} scalar={expected} (rows={rows}, cols={cols})"
