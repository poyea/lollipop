import cupy as cp
import pytest

from lollipop import prefix_sum


@pytest.mark.parametrize(
    "n",
    [
        1,
        7,
        2048,  # exactly one tile
        2049,  # one element into a second tile
        100_003,  # non-power-of-two, multi-tile
        1_000_000,  # forces the recursive block-sums scan
        4_200_000,  # past one recursion level (> 2048 tiles)
    ],
)
def test_prefix_sum_matches_cumsum(n: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)

    actual = prefix_sum(data)
    # exclusive scan == inclusive cumsum shifted by one == cumsum - x
    expected = cp.cumsum(data) - data

    # fp32 accumulation order differs (tree vs sequential), so allow a
    # magnitude-scaled tolerance.
    scale = float(cp.abs(expected).max()) + 1.0
    cp.testing.assert_allclose(actual, expected, atol=1e-3 * scale, rtol=1e-4)


def test_prefix_sum_small_exact() -> None:
    data = cp.array([1, 2, 3, 4], dtype=cp.float32)
    actual = prefix_sum(data)
    cp.testing.assert_array_equal(actual, cp.array([0, 1, 3, 6], dtype=cp.float32))


def test_prefix_sum_flattens_input() -> None:
    data = cp.arange(12, dtype=cp.float32).reshape(3, 4)
    actual = prefix_sum(data)
    expected = cp.cumsum(data.ravel()) - data.ravel()
    cp.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
