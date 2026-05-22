import cupy as cp
import pytest

from lollipop import gemm_int8
from lollipop.kernels.gemm_int8 import quantise_per_row_symmetric


@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 64, 32),
        (64, 64, 256),
        (128, 256, 512),
        (1024, 1024, 1024),
    ],
)
def test_gemm_int8_matches_dequantised_reference(M: int, N: int, K: int) -> None:
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=cp.float32)
    B = rng.standard_normal((N, K), dtype=cp.float32)  # [N, K] convention

    A_q, a_scale = quantise_per_row_symmetric(A)
    B_q, b_scale = quantise_per_row_symmetric(B)  # per-row of B == per-channel of W

    # Reference: dequantise and run FP32 matmul. This tracks rounding error
    # exactly, isolating any kernel-side bug from quantisation noise.
    A_dq = A_q.astype(cp.float32) * a_scale[:, None]
    B_dq = B_q.astype(cp.float32) * b_scale[:, None]
    expected = A_dq @ B_dq.T  # [M, N]

    actual = gemm_int8(A_q, B_q, a_scale, b_scale)

    # INT32 accumulation is exact; only the final FP32 multiply-by-scales
    # introduces error. Tolerance scales with K (sum of K terms).
    tol = max(1e-4 * K * (a_scale.max() * b_scale.max()).item(), 1e-3)
    assert actual.dtype == cp.float32
    assert cp.allclose(actual, expected, atol=tol, rtol=1e-3)


def test_gemm_int8_accuracy_vs_fp32_baseline() -> None:
    """End-to-end accuracy: quantise random inputs, run INT8 GEMM, compare to
    FP32 GEMM on the *original* (unquantised) inputs. This is the realistic
    "what does W8A8 cost me?" question."""
    rng = cp.random.default_rng(1)
    M, N, K = 256, 256, 512
    A = rng.standard_normal((M, K), dtype=cp.float32)
    B = rng.standard_normal((N, K), dtype=cp.float32)
    expected_fp32 = A @ B.T

    A_q, a_scale = quantise_per_row_symmetric(A)
    B_q, b_scale = quantise_per_row_symmetric(B)
    actual_int8 = gemm_int8(A_q, B_q, a_scale, b_scale)

    # Symmetric int8 ε ≈ scale/2 per element; over K terms the error stddev
    # grows as sqrt(K) * scale_a * scale_b. With unit-stddev inputs, scales
    # ≈ 3/127, so error stddev ≈ sqrt(K) * (3/127)^2 ≈ 0.013 for K=512.
    # Relative error against output stddev (~sqrt(K)) is ~6e-4 — well under 1%.
    rel_err = cp.linalg.norm(actual_int8 - expected_fp32) / cp.linalg.norm(
        expected_fp32
    )
    # ~1% relative error is the published headline for W8A8 on random Gaussian
    # inputs without calibration; 2% leaves headroom for accumulator quirks.
    assert float(rel_err) < 0.02


def test_gemm_int8_rejects_bad_shapes() -> None:
    A = cp.zeros((63, 32), dtype=cp.int8)
    B = cp.zeros((64, 32), dtype=cp.int8)
    sa = cp.ones((63,), dtype=cp.float32)
    sb = cp.ones((64,), dtype=cp.float32)
    with pytest.raises(ValueError):
        gemm_int8(A, B, sa, sb)


def test_gemm_int8_rejects_bad_dtype() -> None:
    A = cp.zeros((64, 32), dtype=cp.float32)
    B = cp.zeros((64, 32), dtype=cp.int8)
    sa = cp.ones((64,), dtype=cp.float32)
    sb = cp.ones((64,), dtype=cp.float32)
    with pytest.raises(ValueError):
        gemm_int8(A, B, sa, sb)
