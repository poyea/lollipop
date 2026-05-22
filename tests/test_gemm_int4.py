import cupy as cp
import pytest

from lollipop import gemm_int4
from lollipop.kernels.gemm_int4 import pack_int4_w4a16, dequantise_int4


@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 64, 64),
        (64, 64, 256),
        (128, 256, 512),
        (256, 256, 1024),
    ],
)
def test_gemm_int4_matches_dequantised_reference(M: int, N: int, K: int) -> None:
    """Kernel output should match A @ dequant(Wq).T exactly up to fp16 rounding,
    isolating any kernel bug from the (separate) quantisation error question."""
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=cp.float32)
    W = rng.standard_normal((N, K), dtype=cp.float32)

    Wq, scales, zeros = pack_int4_w4a16(W, group=64)
    W_dq = dequantise_int4(Wq, scales, zeros, K, group=64)  # [N, K] fp32
    expected = (A @ W_dq.T).astype(cp.float32)  # [M, N]

    A_h = A.astype(cp.float16)
    actual = gemm_int4(A_h, Wq, scales, zeros)
    assert actual.dtype == cp.float16

    # fp16 accumulation in the (final) cast + fp16 input rounding give roughly
    # ~5e-3 relative + K-scaled absolute error vs the fp32 dequant reference.
    tol = max(0.02 * cp.linalg.norm(expected).item() / (M * N) ** 0.5, 5e-2)
    assert cp.allclose(actual.astype(cp.float32), expected, atol=tol, rtol=5e-2)


def test_gemm_int4_accuracy_vs_fp32_baseline() -> None:
    """End-to-end W4A16 accuracy: quantise W to INT4, run kernel, compare to
    FP32 matmul on the *original* W. This is the realistic
    "what does W4A16 cost me?" question."""
    rng = cp.random.default_rng(1)
    M, N, K = 256, 256, 512
    A = rng.standard_normal((M, K), dtype=cp.float32)
    W = rng.standard_normal((N, K), dtype=cp.float32)
    expected_fp32 = A @ W.T

    Wq, scales, zeros = pack_int4_w4a16(W, group=64)
    A_h = A.astype(cp.float16)
    actual_int4 = gemm_int4(A_h, Wq, scales, zeros).astype(cp.float32)

    rel_err = cp.linalg.norm(actual_int4 - expected_fp32) / cp.linalg.norm(
        expected_fp32
    )
    # Uncalibrated group-64 INT4 on random Gaussians: per-element ε ≈ scale/√12
    # with scale ≈ (max-min)/15 ≈ 0.32 for a 64-sample group → ε ≈ 0.09. Over a
    # K=512 inner product the relative error reaches ~9% — that's the
    # *expected* uncalibrated W4A16 cost. Published 1-3% figures need calibration
    # data (AWQ/GPTQ), which is a quantiser-side concern not a kernel concern.
    assert float(rel_err) < 0.12


def test_gemm_int4_rejects_bad_shapes() -> None:
    A = cp.zeros((64, 64), dtype=cp.float16)
    Wq = cp.zeros((63, 32), dtype=cp.uint8)
    scales = cp.zeros((1, 63), dtype=cp.float16)
    zeros = cp.zeros((1, 63), dtype=cp.float16)
    with pytest.raises(ValueError):
        gemm_int4(A, Wq, scales, zeros)


def test_gemm_int4_rejects_bad_dtype() -> None:
    A = cp.zeros((64, 64), dtype=cp.float32)  # should be fp16
    Wq = cp.zeros((64, 32), dtype=cp.uint8)
    scales = cp.zeros((1, 64), dtype=cp.float16)
    zeros = cp.zeros((1, 64), dtype=cp.float16)
    with pytest.raises(ValueError):
        gemm_int4(A, Wq, scales, zeros)
