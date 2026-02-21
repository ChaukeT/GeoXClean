"""
UI-side controller that provides structured access to IRR analysis results.

This module contains no PyQt widgets. It converts raw engine output into useful
views that the IRR panel can bind to tables, charts, and exports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..irr_engine.results_model import IRRResult, build_irr_result

logger = logging.getLogger(__name__)


class IRRResultsController:
    """Logic-only controller for IRR analysis results."""

    def __init__(self) -> None:
        self.raw_results: Dict[str, Any] = {}
        self.result: Optional[IRRResult] = None

    # ------------------------------------------------------------------
    # Update / access helpers
    # ------------------------------------------------------------------
    def update_results(
        self,
        raw_results: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> IRRResult:
        """Store raw results and build a normalized IRRResult."""
        if raw_results is None:
            raise ValueError("IRRResultsController.update_results() requires data.")

        # Preserve the raw dictionary (shallow copy for audit trail)
        self.raw_results = dict(raw_results)
        self.result = build_irr_result(self.raw_results, metadata=metadata)
        return self.result

    def has_results(self) -> bool:
        """Return True when a structured result is available."""
        return self.result is not None

    def get_result(self) -> Optional[IRRResult]:
        """Return the current IRRResult, if any."""
        return self.result

    def clear(self) -> None:
        """Reset stored results."""
        self.raw_results = {}
        self.result = None

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def summary_rows(self) -> List[Tuple[str, str]]:
        """Return metric/value rows suitable for a summary table."""
        result = self._require_result()

        npvs = self._safe_distribution()
        quantiles = self.npv_quantiles([0.05, 0.1, 0.5, 0.9])

        if npvs.size:
            var_5 = float(np.percentile(npvs, 5))
            tail_mask = npvs <= var_5
            if np.any(tail_mask):
                cvar_5 = float(np.mean(npvs[tail_mask]))
            else:
                cvar_5 = var_5
            positive = int(np.sum(npvs >= 0))
        else:
            var_5 = float("nan")
            cvar_5 = float("nan")
            positive = 0

        num_scenarios = result.num_scenarios

        return [
            ("IRR_α (Risk-Adjusted)", f"{result.irr_alpha:.2%}"),
            ("Confidence Level (α)", f"{result.alpha_target:.2%}"),
            ("Satisfaction Rate", f"{result.satisfaction_rate:.2%}"),
            ("Mean NPV", self._format_currency(result.mean_npv)),
            ("Std Dev NPV", self._format_currency(result.std_npv)),
            ("Min NPV", self._format_currency(result.min_npv)),
            ("Max NPV", self._format_currency(result.max_npv)),
            ("P5 NPV (VaR 5%)", self._format_currency(var_5)),
            ("Downside CVaR (5%)", self._format_currency(cvar_5)),
            ("P10 NPV", self._format_currency(quantiles.get(0.1))),
            ("P50 NPV", self._format_currency(quantiles.get(0.5))),
            ("P90 NPV", self._format_currency(quantiles.get(0.9))),
            ("Scenarios ≥ 0 NPV", f"{positive:,} / {num_scenarios:,}"),
            ("Scenarios", f"{result.num_scenarios:,}"),
            ("Iterations", f"{result.iterations:,}"),
        ]

    def narrative_summary(self) -> str:
        """Return a concise narrative summary suitable for reports."""
        result = self._require_result()
        npvs = self._safe_distribution()

        if not npvs.size:
            return (
                "IRR analysis completed, but no NPV distribution was returned. "
                "Verify scenario generation and optimization outputs."
            )

        quantiles = self.npv_quantiles([0.1, 0.5, 0.9])
        prob_negative = float(np.sum(npvs < 0)) / npvs.size

        return (
            f"Risk-adjusted IRR_α at α = {result.alpha_target:.0%} is "
            f"{result.irr_alpha:.1%}. "
            f"{result.satisfaction_rate:.1%} of scenarios deliver NPV ≥ 0 at this rate. "
            f"Mean NPV is {self._format_currency(result.mean_npv)} "
            f"(P10: {self._format_currency(quantiles.get(0.1))}, "
            f"P50: {self._format_currency(quantiles.get(0.5))}, "
            f"P90: {self._format_currency(quantiles.get(0.9))}). "
            f"Probability of negative NPV is {prob_negative:.1%}."
        )

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def npv_histogram(self, bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """Return histogram counts and bin edges for NPV distribution."""
        npvs = self._safe_distribution()
        if not npvs.size:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        counts, edges = np.histogram(npvs, bins=bins)
        return counts.astype(float), edges.astype(float)

    def npv_quantiles(
        self, quantiles: Sequence[float]
    ) -> Dict[float, float]:
        """Return specified quantiles (e.g., [0.1, 0.5, 0.9]) of the NPV distribution."""
        npvs = self._safe_distribution()
        if not npvs.size:
            return {float(q): float("nan") for q in quantiles}

        percentiles = [q * 100 for q in quantiles]
        values = np.percentile(npvs, percentiles)
        return {float(q): float(v) for q, v in zip(quantiles, values)}

    def npv_var_es(self, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Compute Value-at-Risk and Expected Shortfall for the NPV distribution.

        Args:
            alpha: Confidence level (e.g., 0.95 for 95%).

        Returns:
            (VaR, ES) pair in currency units.
            VaR is the quantile at (1 - alpha), ES is the mean of losses beyond that point.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be within (0, 1].")

        npvs = self._safe_distribution()
        if not npvs.size:
            return float("nan"), float("nan")

        tail_probability = 1.0 - alpha
        var_threshold = np.percentile(npvs, tail_probability * 100.0)
        tail_values = npvs[npvs <= var_threshold]

        if not tail_values.size:
            es_value = var_threshold
        else:
            es_value = float(np.mean(tail_values))

        return float(var_threshold), float(es_value)

    # ------------------------------------------------------------------
    # Economic helpers
    # ------------------------------------------------------------------
    def cashflow_series(self) -> pd.Series:
        """Return a pandas Series of cashflows per period (best scenario)."""
        result = self._require_result()
        cashflows = result.best_cashflows

        if not cashflows.size:
            return pd.Series([], dtype=float)

        if cashflows.ndim == 1:
            index = self._cashflow_index(len(cashflows), result.best_npv_details)
            return pd.Series(cashflows, index=index, dtype=float)

        # If 2D (e.g., period x components), sum across columns
        collapsed = cashflows.sum(axis=1)
        index = self._cashflow_index(len(collapsed), result.best_npv_details)
        return pd.Series(collapsed, index=index, dtype=float)

    def economic_breakdown(self) -> Dict[str, Any]:
        """Return revenue/cost breakdown suitable for tables or charts."""
        result = self._require_result()
        if result.economic_breakdown:
            return dict(result.economic_breakdown)

        details = result.best_npv_details or {}
        return dict(details.get("economic_breakdown", {}))

    # ------------------------------------------------------------------
    # Schedule / convergence helpers
    # ------------------------------------------------------------------
    def best_schedule(self) -> Optional[pd.DataFrame]:
        """Return the best schedule DataFrame, if available."""
        result = self._require_result()
        if result.best_schedule is None:
            return None
        return result.best_schedule.copy()

    def convergence_curves(self) -> Dict[str, np.ndarray]:
        """Return convergence history arrays for plotting diagnostics."""
        result = self._require_result()
        return {
            key: value.copy()
            for key, value in (result.convergence_history or {}).items()
        }

    def optimal_pit_blocks(self) -> Optional[pd.DataFrame]:
        """Return DataFrame of blocks belonging to the IRR-optimal pit shell."""
        blocks = self.raw_results.get("optimal_pit_blocks")
        if blocks is None:
            return None
        if isinstance(blocks, pd.DataFrame):
            return blocks.copy()
        try:
            return pd.DataFrame(blocks)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _require_result(self) -> IRRResult:
        if self.result is None:
            raise RuntimeError("No IRR results are available.")
        return self.result

    def _safe_distribution(self) -> np.ndarray:
        result = self._require_result()
        npvs = result.npv_distribution
        if npvs is None:
            return np.zeros(0, dtype=float)
        return np.asarray(npvs, dtype=float)

    @staticmethod
    def _format_currency(value: Any) -> str:
        if value is None or isinstance(value, float) and np.isnan(value):
            return "N/A"
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _cashflow_index(length: int, details: Optional[Dict[str, Any]]) -> pd.Index:
        """Derive sensible index labels for cashflow series."""
        if details and isinstance(details.get("periods"), Iterable):
            periods = list(details.get("periods"))
            if len(periods) == length:
                return pd.Index(periods, name="Period")
        return pd.Index(range(1, length + 1), name="Period")


