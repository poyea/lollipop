import cupy as cp
import pytest

from lollipop import reduction, reduction_v2


@pytest.mark.parametrize("threads", [128, 256, 512, 1024])
@pytest.mark.parametrize("n", [1024, 100_003, 1_000_000])
def test_reduction_v2_matches_baseline(n: int, threads: int) -> None:
    rng = cp.random.default_rng(0)
    data = rng.standard_normal(n, dtype=cp.float32)
    expected = reduction(data)
    actual = reduction_v2(data, threads=threads)
    assert abs(actual - expected) <= 1e-3 + 1e-5 * abs(expected)
