"""Exercise the scan / sort / reduce kernels for a compute-sanitizer sweep.

compute-sanitizer catches what `assert_allclose` cannot: a racing smem write or
a missing `__syncthreads()` can still produce the right answer on a lucky run.
This driver runs each primitive once at a small-but-multi-block size (so the
cross-block paths and the grid barrier are actually exercised) and checks
parity, so the sanitizer has real work to instrument.

Run under each tool (from the repo root, using the venv interpreter):

    compute-sanitizer --tool memcheck   <python> tools/sanitizer_check.py
    compute-sanitizer --tool racecheck  <python> tools/sanitizer_check.py
    compute-sanitizer --tool synccheck  <python> tools/sanitizer_check.py
    compute-sanitizer --tool initcheck  <python> tools/sanitizer_check.py

A clean run prints the parity line and exits 0 with no sanitizer errors.
"""

import lollipop._cuda_setup  # noqa: F401
import cupy as cp

from lollipop import (
    reduction,
    reduction_cg,
    reduction_cg_grid,
    prefix_sum,
    radix_sort,
)


def _check(name, ok):
    print(f"  {name:18s} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"parity failed: {name}")


def main():
    rng = cp.random.default_rng(0)

    # Multi-block but small, so racecheck/synccheck stay fast.
    x = rng.standard_normal(70_000, dtype=cp.float32)
    ref = float(x.sum())
    tol = 1e-2 + 1e-4 * abs(ref)
    _check("reduction", abs(reduction(x) - ref) <= tol)
    _check("reduction_cg", abs(reduction_cg(x) - ref) <= tol)
    _check("reduction_cg_grid", abs(reduction_cg_grid(x) - ref) <= tol)

    # Spans several scan tiles (ELEMENTS_PER_BLOCK = 2048) -> recursive path.
    s = rng.standard_normal(10_000, dtype=cp.float32)
    excl = prefix_sum(s)
    excl_ref = cp.cumsum(s) - s
    scale = float(cp.abs(excl_ref).max()) + 1.0
    _check("prefix_sum", bool(cp.abs(excl - excl_ref).max() <= 1e-3 * scale))

    hi = rng.integers(0, 1 << 16, size=40_000, dtype=cp.uint32)
    lo = rng.integers(0, 1 << 16, size=40_000, dtype=cp.uint32)
    keys = (hi << 16) | lo
    _check("radix_sort", bool((radix_sort(keys) == cp.sort(keys)).all()))

    print("all primitives parity-clean")


if __name__ == "__main__":
    main()
