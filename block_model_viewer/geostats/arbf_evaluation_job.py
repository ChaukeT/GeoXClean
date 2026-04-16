"""
ARBF Evaluation Job — runs one estimation from definition objects.

Inputs:
  - BlockModelDefinition (target geometry)
  - ActiveCellMask (which blocks to estimate)
  - ARBFEstimatorDefinition (sample data + engine params)

Outputs:
  - Full-model-size result arrays (NaN for inactive)
  - Profiling metrics
  - Job summary
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..models.block_model_definition import BlockModelDefinition
from ..models.active_cell_mask import ActiveCellMask
from .arbf_estimator_definition import ARBFEstimatorDefinition
from .arbf_target_preparer import prepare_targets
from .arbf_output_writer import expand_to_full_model
from .arbf_profiler import EstimationProfiler
from .fastrbf_engine_v2 import FastRBFEstimator

logger = logging.getLogger(__name__)


@dataclass
class ARBFJobResult:
    """Complete result of one ARBF evaluation job."""
    grades: np.ndarray               # (n_total,) — NaN for inactive
    variances: np.ndarray            # (n_total,) — NaN for inactive
    all_outputs: Dict[str, np.ndarray]  # Full output dict
    n_total: int
    n_active: int
    n_estimated: int
    elapsed_seconds: float
    mode_used: str
    downgraded: bool
    warnings: list
    profile: Dict[str, Any]
    status: str                      # "complete", "failed", "downgraded"


class ARBFEvaluationJob:
    """Runs one ARBF estimation from separated objects."""

    def __init__(
        self,
        block_def: BlockModelDefinition,
        mask: ActiveCellMask,
        estimator_def: ARBFEstimatorDefinition,
    ) -> None:
        self.block_def = block_def
        self.mask = mask
        self.estimator_def = estimator_def

    def run(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> ARBFJobResult:
        """Execute the estimation job.

        Returns ARBFJobResult with full-model-size arrays.
        """
        prof = EstimationProfiler()

        def _report(pct: int, msg: str) -> None:
            if progress_callback:
                try:
                    progress_callback(pct, msg)
                except Exception:
                    pass

        t0 = time.perf_counter()
        ed = self.estimator_def

        # ── Stage 1: Prepare targets ─────────────────────────────────
        _report(2, "Preparing active targets...")
        with prof.time("prepare"):
            plan = prepare_targets(self.mask.active_centres, ed.mode)

        if plan.warnings:
            for w in plan.warnings:
                _report(3, f"Warning: {w}")

        n_active = plan.n_blocks
        _report(5, f"Active blocks: {n_active:,} of {self.block_def.n_blocks:,} "
                    f"({n_active / max(self.block_def.n_blocks, 1) * 100:.1f}%)")

        # ── Stage 2: Build estimator engine ──────────────────────────
        _report(8, "Building FastRBF estimator...")
        with prof.time("engine_build"):
            estimator = FastRBFEstimator(
                coords=ed.sample_coords,
                values=ed.sample_values,
                anisotropy=ed.anisotropy,
                variogram=ed.variogram,
                rbf_settings=ed.rbf_settings,
                neighbourhood=ed.neighbourhood,
                block_settings=ed.block_settings,
                partition_settings=ed.partition_settings,
            )

        # ── Stage 3: Estimate on active cells only ───────────────────
        _report(12, f"Estimating {n_active:,} blocks ({plan.mode.name} mode)...")
        with prof.time("estimation"):
            raw = self._estimate_with_progress(
                estimator, plan.centres, _report, 12, 85,
            )

        # ── Stage 4: Expand to full model ────────────────────────────
        _report(88, "Expanding results to full block model...")
        with prof.time("expand"):
            full = expand_to_full_model(
                n_total=self.block_def.n_blocks,
                active_mask=self.mask.active,
                active_results=raw,
            )

        elapsed = time.perf_counter() - t0
        prof.set("n_total", self.block_def.n_blocks)
        prof.set("n_active", n_active)
        prof.set("mode", plan.mode.name)
        prof.set("elapsed_seconds", round(elapsed, 3))
        prof.set("ms_per_block", round(elapsed / max(n_active, 1) * 1000, 2))

        _report(95, f"Complete: {n_active:,} blocks in {elapsed:.1f}s "
                     f"({elapsed / max(n_active, 1) * 1000:.1f} ms/block)")

        logger.info(
            "ARBF job complete: %d/%d blocks, %.1fs, mode=%s",
            n_active, self.block_def.n_blocks, elapsed, plan.mode.name,
        )

        return ARBFJobResult(
            grades=full.get("ARBF_GRADE", np.full(self.block_def.n_blocks, np.nan)),
            variances=full.get("ARBF_VAR_BLOCK", np.full(self.block_def.n_blocks, np.nan)),
            all_outputs=full,
            n_total=self.block_def.n_blocks,
            n_active=n_active,
            n_estimated=n_active,
            elapsed_seconds=elapsed,
            mode_used=plan.mode.name,
            downgraded=plan.downgraded,
            warnings=plan.warnings,
            profile=prof.summary(),
            status="complete",
        )

    def _estimate_with_progress(
        self,
        estimator: FastRBFEstimator,
        centres: np.ndarray,
        report: Callable,
        pct_start: int,
        pct_end: int,
    ) -> Dict[str, np.ndarray]:
        """Estimate blocks with progress updates per chunk."""
        n = centres.shape[0]
        chunk_size = max(n // 20, 1)
        out = {
            "ARBF_GRADE": np.zeros(n, dtype=float),
            "ARBF_VAR_BLOCK": np.zeros(n, dtype=float),
            "ARBF_STD_BLOCK": np.zeros(n, dtype=float),
            "ARBF_NEFF": np.full(n, np.nan, dtype=float),
            "ARBF_CONDNUM": np.full(n, np.nan, dtype=float),
            "ARBF_PUM_COUNT": np.zeros(n, dtype=float),
            "ARBF_STITCH_VAR": np.zeros(n, dtype=float),
            "ARBF_HIGHVAR_FLAG": np.zeros(n, dtype=int),
            "ARBF_FAIL_FLAG": np.zeros(n, dtype=int),
        }

        for i in range(n):
            est = estimator.estimate_block(centres[i])
            out["ARBF_GRADE"][i] = est.mean
            out["ARBF_VAR_BLOCK"][i] = est.variance
            out["ARBF_STD_BLOCK"][i] = est.std
            out["ARBF_NEFF"][i] = est.neff
            out["ARBF_CONDNUM"][i] = est.condnum
            out["ARBF_PUM_COUNT"][i] = est.coverage_count
            out["ARBF_STITCH_VAR"][i] = est.stitch_variance
            out["ARBF_HIGHVAR_FLAG"][i] = int(est.highvar_flag)
            out["ARBF_FAIL_FLAG"][i] = int(est.fail_flag)

            if (i + 1) % chunk_size == 0 or i == n - 1:
                pct = pct_start + int((pct_end - pct_start) * (i + 1) / n)
                report(pct, f"Block {i + 1:,}/{n:,}")

        return out
