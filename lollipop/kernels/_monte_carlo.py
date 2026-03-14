"""
Reusable base class for GPU Monte Carlo simulations.

Subclasses set CU_FILE, KERNEL_NAME, and (optionally) NUM_OUTPUTS,
then call ``_run`` with any simulation-specific kernel arguments::

    class PiEstimator(MonteCarloKernel):
        CU_FILE = "monte_carlo_pi.cu"
        KERNEL_NAME = "monte_carlo_pi"

        def estimate(self, num_paths: int = 1_000_000) -> float:
            out, total = self._run(num_paths=num_paths)
            return float(out[0]) / total

The CUDA kernel's last three parameters are always
``float* out, int paths_per_thread, unsigned int seed``;
everything before that is the simulation-specific payload passed
via ``*kernel_args`` to ``_run``.
"""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

_SOURCES_DIR = Path(__file__).parent / "_sources"


class MonteCarloKernel:
    BLOCK_SIZE: int = 256
    CU_FILE: str = ""
    KERNEL_NAME: str = ""
    NUM_OUTPUTS: int = 1

    def __init__(self) -> None:
        self._kernel: cp.RawKernel | None = None

    def _get_kernel(self) -> cp.RawKernel:
        if self._kernel is None:
            source = (_SOURCES_DIR / self.CU_FILE).read_text(encoding="utf-8")
            self._kernel = cp.RawKernel(
                source,
                self.KERNEL_NAME,
                options=(f"-I{_SOURCES_DIR}",),
            )
        return self._kernel

    def _run(
        self,
        *kernel_args: object,
        num_paths: int = 1_000_000,
        seed: int = 42,
    ) -> tuple[cp.ndarray, int]:
        num_threads = min(num_paths, 65_536)
        paths_per_thread = max(1, num_paths // num_threads)
        total_paths = num_threads * paths_per_thread

        grid = (num_threads + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        out = cp.zeros(self.NUM_OUTPUTS, dtype=cp.float32)

        self._get_kernel()(
            (grid,),
            (self.BLOCK_SIZE,),
            (
                *kernel_args,
                out,
                np.int32(paths_per_thread),
                np.uint32(seed),
            ),
        )

        return out, total_paths
