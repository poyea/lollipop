import cupy as cp
import pytest

from lollipop import reduction, reduction_cg, reduction_cg_grid


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1024, 100_003, 1_000_000, 50_000_000])
def test_reduction_cg_matches_baseline(n: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)

    expected = reduction(data)
    actual = reduction_cg(data)

    assert abs(actual - expected) <= 1e-2 + 1e-4 * abs(expected)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1024, 100_003, 1_000_000, 50_000_000])
def test_reduction_cg_grid_matches_baseline(n: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)

    expected = reduction(data)
    actual = reduction_cg_grid(data)

    # atomic-free cooperative reduction: same tolerance as the warp variant
    assert abs(actual - expected) <= 1e-2 + 1e-4 * abs(expected)


def test_reduction_cg_flattens_input() -> None:
    rng = cp.random.default_rng(1)
    data = rng.standard_normal((512, 384), dtype=cp.float32)
    expected = float(data.sum())
    actual = reduction_cg(data)
    assert abs(actual - expected) <= 1e-2 + 1e-4 * abs(expected)
