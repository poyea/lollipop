import cupy as cp
import pytest

from lollipop import reduction, reduction_vec4


@pytest.mark.parametrize(
    "n",
    [1, 3, 4, 5, 1023, 1024, 1025, 100_003, 1_000_000, 1_048_577],
)
def test_reduction_vec4_matches_scalar(n: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)
    expected = reduction(data)
    actual = reduction_vec4(data)
    # Same algorithm modulo summation order; tolerance reflects fp32 reassoc.
    assert abs(actual - expected) <= 1e-3 + 1e-5 * abs(expected)
