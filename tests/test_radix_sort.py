import cupy as cp
import pytest

from lollipop import radix_sort


def _rand_u32(rng, size):
    # 2**32 overflows CuPy's C-long arg on Windows; build from 16-bit halves.
    hi = rng.integers(0, 1 << 16, size=size, dtype=cp.uint32)
    lo = rng.integers(0, 1 << 16, size=size, dtype=cp.uint32)
    return (hi << 16) | lo


@pytest.mark.parametrize(
    "n",
    [
        1,
        255,  # smaller than one block
        256,  # exactly one block
        32768,  # the old single-block-scan cap
        32769,  # one past the old cap
        1_000_000,  # device-wide scan territory
    ],
)
def test_radix_sort_matches_cupy(n: int) -> None:
    rng = cp.random.default_rng(0)
    data = _rand_u32(rng, n)

    actual = radix_sort(data)
    expected = cp.sort(data)

    assert actual.shape == (n,)
    cp.testing.assert_array_equal(actual, expected)


def test_radix_sort_handles_duplicates() -> None:
    rng = cp.random.default_rng(1)
    # heavy duplication stresses the stable scatter and per-bucket offsets
    data = rng.integers(0, 16, size=50_000, dtype=cp.uint32)
    cp.testing.assert_array_equal(radix_sort(data), cp.sort(data))


def test_radix_sort_already_sorted() -> None:
    data = cp.arange(100_000, dtype=cp.uint32)
    cp.testing.assert_array_equal(radix_sort(data), data)
