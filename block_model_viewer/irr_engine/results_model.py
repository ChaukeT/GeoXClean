"""
IRR results data structures and builders.

Provides a normalized `IRRResult` dataclass and a factory for transforming
raw dictionaries returned by `find_irr_alpha` into audit-ready objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class IRRResult:
    """
    Structured representation of an IRR analysis result.
    
    Updated 2025-12 with audit compliance fields:
    - IRR distribution for full analysis
    - Provenance record for reproducibility
    - Multiple IRR warning tracking
    - Classification filter metadata
    """

    irr_alpha: float
    alpha_target: float
    satisfaction_rate: float
    num_scenarios: int
    iterations: int

    npv_distribution: np.ndarray = field(repr=False)
    mean_npv: float = 0.0
    std_npv: float = 0.0
    min_npv: float = 0.0
    max_npv: float = 0.0
    
    # NEW: IRR distribution statistics
    irr_distribution: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    mean_irr: float = 0.0
    std_irr: float = 0.0

    best_schedule: Optional[pd.DataFrame] = field(default=None, repr=False)
    best_cashflows: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    economic_breakdown: Dict[str, Any] = field(default_factory=dict)
    convergence_history: Dict[str, np.ndarray] = field(default_factory=dict)
    best_npv_details: Dict[str, Any] = field(default_factory=dict, repr=False)

    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # === AUDIT COMPLIANCE FIELDS (2025-12) ===
    provenance: Dict[str, Any] = field(default_factory=dict, repr=False)
    scenario_cashflows: Optional[list] = field(default=None, repr=False)
    multiple_irr_warnings: int = 0
    classification_filter_applied: bool = False
    blocks_before_filter: int = 0
    blocks_after_filter: int = 0
    validation_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)


def build_irr_result(
    raw: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None
) -> IRRResult:
    """
    Build a normalized `IRRResult` from the raw dictionary returned by
    `find_irr_alpha`.

    Args:
        raw: Raw dictionary returned by the engine.
        metadata: Optional metadata to attach (block model hash, config snapshot, etc.).

    Returns:
        IRRResult: Normalized result object.
    """
    if raw is None:
        raise ValueError("build_irr_result() requires raw result data.")

    # Coerce core numerical fields
    irr_alpha = _coerce_float(raw.get("irr_alpha"))
    alpha_target = _coerce_float(raw.get("alpha_target"))
    satisfaction_rate = _coerce_float(raw.get("satisfaction_rate"))
    iterations = _coerce_int(raw.get("iterations"))

    npv_distribution = _coerce_np_array(raw.get("npv_distribution"))

    if alpha_target is None:
        alpha_target = 0.0
    if satisfaction_rate is None:
        satisfaction_rate = 0.0
    if iterations is None:
        iterations = 0

    num_scenarios = _coerce_int(raw.get("num_scenarios"))
    if num_scenarios is None:
        num_scenarios = int(npv_distribution.size)

    # Derive NPV stats with safe defaults.
    if npv_distribution.size:
        mean_fallback = float(np.mean(npv_distribution))
        std_fallback = float(np.std(npv_distribution))
        min_fallback = float(np.min(npv_distribution))
        max_fallback = float(np.max(npv_distribution))
    else:
        mean_fallback = 0.0
        std_fallback = 0.0
        min_fallback = 0.0
        max_fallback = 0.0

    mean_npv = _coerce_float(raw.get("mean_npv"), mean_fallback)
    std_npv = _coerce_float(raw.get("std_npv"), std_fallback)
    min_npv = _coerce_float(raw.get("min_npv"), min_fallback)
    max_npv = _coerce_float(raw.get("max_npv"), max_fallback)

    best_schedule = _coerce_dataframe(raw.get("best_schedule"))
    best_cashflows = _coerce_cashflows(raw.get("best_cashflows"))

    economic_breakdown = _coerce_mapping(raw.get("economic_breakdown"))
    convergence_history = _coerce_convergence(raw.get("convergence_history"))
    best_npv_details_value = raw.get("best_npv_details")
    best_npv_details = dict(best_npv_details_value) if isinstance(best_npv_details_value, Mapping) else {}

    normalized_metadata = dict(metadata or {})

    # === EXTRACT NEW AUDIT COMPLIANCE FIELDS ===
    irr_distribution = _coerce_np_array(raw.get("irr_distribution"))
    mean_irr = _coerce_float(raw.get("mean_irr"), 0.0)
    std_irr = _coerce_float(raw.get("std_irr"), 0.0)
    
    provenance = _coerce_mapping(raw.get("provenance"))
    scenario_cashflows = raw.get("scenario_cashflows")
    multiple_irr_warnings = _coerce_int(raw.get("multiple_irr_warnings"), 0)
    classification_filter_applied = bool(raw.get("classification_filter_applied", False))
    blocks_before_filter = _coerce_int(raw.get("blocks_before_filter"), 0)
    blocks_after_filter = _coerce_int(raw.get("blocks_after_filter"), 0)
    validation_metadata = _coerce_mapping(raw.get("validation_metadata"))

    return IRRResult(
        irr_alpha=irr_alpha or 0.0,
        alpha_target=alpha_target or 0.0,
        satisfaction_rate=satisfaction_rate or 0.0,
        num_scenarios=num_scenarios,
        iterations=iterations,
        npv_distribution=npv_distribution,
        mean_npv=mean_npv,
        std_npv=std_npv,
        min_npv=min_npv,
        max_npv=max_npv,
        irr_distribution=irr_distribution,
        mean_irr=mean_irr,
        std_irr=std_irr,
        best_schedule=best_schedule,
        best_cashflows=best_cashflows,
        economic_breakdown=economic_breakdown,
        convergence_history=convergence_history,
        best_npv_details=best_npv_details,
        metadata=normalized_metadata,
        raw=dict(raw),
        # Audit compliance fields
        provenance=provenance,
        scenario_cashflows=scenario_cashflows,
        multiple_irr_warnings=multiple_irr_warnings,
        classification_filter_applied=classification_filter_applied,
        blocks_before_filter=blocks_before_filter,
        blocks_after_filter=blocks_after_filter,
        validation_metadata=validation_metadata,
    )


# ---------------------------------------------------------------------------
# Helper coercion utilities
# ---------------------------------------------------------------------------

def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value: Any, fallback: int = 0) -> int:
    if value is None:
        return int(fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_np_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    try:
        arr = np.asarray(value, dtype=float)
        return arr
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)


def _coerce_dataframe(value: Any) -> Optional[pd.DataFrame]:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except (ValueError, TypeError):
        return None


def _coerce_cashflows(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=float)
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=float, copy=True)
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float, copy=True)
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_convergence(value: Any) -> Dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, np.ndarray] = {}
    for key, series in value.items():
        if isinstance(series, (Iterable, np.ndarray, pd.Series)):
            normalized[key] = _coerce_np_array(series)
    return normalized


