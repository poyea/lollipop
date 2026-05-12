import cupy as cp
import pytest

from lollipop import gemm_tiled


@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 8),
        (128, 128, 128),
        (256, 512, 128),
        (1024, 1024, 1024),
    ],
)
def test_gemm_tiled_matches_cupy(M: int, N: int, K: int) -> None:
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=cp.float32)
    B = rng.standard_normal((K, N), dtype=cp.float32)
    expected = A @ B
    actual = gemm_tiled(A, B)
    # FP32 GEMM accumulation order differs from cuBLAS; relax tolerance with K.
    tol = 1e-4 * K
    assert cp.allclose(actual, expected, atol=tol, rtol=1e-4)


def test_gemm_tiled_rejects_bad_shapes() -> None:
    A = cp.zeros((127, 8), dtype=cp.float32)
    B = cp.zeros((8, 128), dtype=cp.float32)
    with pytest.raises(ValueError):
        gemm_tiled(A, B)
