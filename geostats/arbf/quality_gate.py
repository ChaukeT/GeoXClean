"""
Geostatistical defensibility gate for ARBF estimation.

The gate does not pretend to prove a model is correct. It evaluates
whether the workflow passes a minimum set of explicit checks before the
UI is allowed to present the result as decision-grade or report-grade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .utils import rotation_matrix, scale_matrix


@dataclass
class QualityCheck:
    """Single geostatistical QA finding."""

    code: str
    label: str
    status: str
    detail: str
    value: Optional[float] = None
    threshold: str = ""


@dataclass
class QualityGateResult:
    """Structured panel-facing gate result."""

    overall_status: str
    headline: str
    summary: str
    checks: List[QualityCheck] = field(default_factory=list)
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    info_count: int = 0
    actions: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "headline": self.headline,
            "summary": self.summary,
            "checks": [asdict(check) for check in self.checks],
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "info_count": self.info_count,
            "actions": dict(self.actions),
            "metrics": _convert_numpy(self.metrics),
        }


def evaluate_geostatistical_gate(
    audit_record: Any,
    diagnostics: Optional[Dict[str, Any]] = None,
    *,
    sample_values: Optional[np.ndarray] = None,
    sample_coords: Optional[np.ndarray] = None,
    block_centroids: Optional[np.ndarray] = None,
    domain_column: Optional[str] = None,
    available_domain_columns: Optional[Sequence[str]] = None,
    change_of_support: bool = True,
    estimation_mode: Optional[str] = None,
    simulation_run: bool = False,
) -> QualityGateResult:
    """Evaluate whether an ARBF estimation run is defensible enough for the UI.

    The thresholds are deliberately conservative. A pass means the workflow
    cleared the implemented checks. It does not mean the deposit is fully
    validated for public reporting.
    """

    diagnostics = diagnostics or {}
    available_domain_columns = list(available_domain_columns or [])
    checks: List[QualityCheck] = []
    metrics: Dict[str, Any] = {}

    record = _coerce_mapping(audit_record)
    sample_std = _safe_std(sample_values)

    # Compute sample skewness to adapt gate thresholds for skewed deposits
    # (Cu, Au, etc.). Any smoothing estimator will have slope < 1.0 on
    # right-skewed data; penalising that as a "fail" is misleading.
    sample_skewness = _safe_skewness(sample_values)
    metrics["sample_skewness"] = sample_skewness if sample_skewness is not None else float("nan")
    slope_bands = _skewness_adjusted_slope_bands(sample_skewness)
    bias_scale = _skewness_bias_scale(sample_skewness)
    n_blocks_total = int(_value(record, diagnostics, "n_blocks_total", 0))
    n_blocks_estimated = int(_value(record, diagnostics, "n_blocks_estimated", 0))
    unclassified_blocks = int(_value(record, diagnostics, "unclassified_blocks", 0))
    clip_to_drill_footprint = bool(
        _value(record, diagnostics, "clip_to_drill_footprint", False),
    )
    footprint_buffer_ranges = float(
        _value(record, diagnostics, "footprint_buffer_ranges", 1.0),
    )
    gate_block_centroids = block_centroids

    def add_check(
        code: str,
        label: str,
        status: str,
        detail: str,
        *,
        value: Optional[float] = None,
        threshold: str = "",
    ) -> None:
        checks.append(
            QualityCheck(
                code=code,
                label=label,
                status=status,
                detail=detail,
                value=value,
                threshold=threshold,
            ),
        )

    if clip_to_drill_footprint:
        clipped = _apply_footprint_clip_mask(
            sample_coords=sample_coords,
            block_centroids=block_centroids,
            azimuth=float(_value(record, diagnostics, "azimuth", 0.0)),
            dip=float(_value(record, diagnostics, "dip", 0.0)),
            pitch=float(_value(record, diagnostics, "pitch", 0.0)),
            range_max=float(_value(record, diagnostics, "range_max", 1.0)),
            range_mid=float(_value(record, diagnostics, "range_mid", 1.0)),
            range_min=float(_value(record, diagnostics, "range_min", 1.0)),
            buffer_ranges=footprint_buffer_ranges,
        )
        if clipped is not None:
            gate_block_centroids = clipped

    # Geological domains
    if available_domain_columns and not domain_column:
        add_check(
            "domains",
            "Geological domaining",
            "fail",
            "Domain-like fields were found but no domain column was selected. "
            "Pooling mixed populations is not defensible for resource estimation.",
            threshold="Select and enforce a geological domain when domain fields exist",
        )
    elif domain_column:
        domains_enforced = bool(_value(record, diagnostics, "domains_enforced", False))
        if domains_enforced:
            add_check(
                "domains",
                "Geological domaining",
                "pass",
                f"Domain column '{domain_column}' was selected and hard boundaries were enforced.",
            )
        else:
            add_check(
                "domains",
                "Geological domaining",
                "fail",
                f"Domain column '{domain_column}' was selected but boundaries were not enforced.",
                threshold="Selected geological domains must be enforced during estimation",
            )
    else:
        add_check(
            "domains",
            "Geological domaining",
            "info",
            "No obvious domain field was available in the loaded data. "
            "Geology still needs external validation.",
        )

    # Support consistency
    if change_of_support:
        support_ratio = float(_value(record, diagnostics, "support_ratio", 0.0))
        add_check(
            "support_mode",
            "Support consistency",
            "pass",
            "Block support correction was enabled for block estimation.",
            value=support_ratio if np.isfinite(support_ratio) else None,
            threshold="Block estimates should use block-support logic",
        )
    else:
        add_check(
            "support_mode",
            "Support consistency",
            "fail",
            "Change-of-support was disabled, so point-support behaviour is being used for block estimates.",
            threshold="Enable block-support estimation for decision-scale blocks",
        )

    # Spatial cross-validation presence
    cv_slope = _finite_or_none(_value(record, diagnostics, "cv_slope_of_regression", np.nan))
    cv_r2 = _finite_or_none(_value(record, diagnostics, "cv_r_squared", np.nan))
    cv_me = _finite_or_none(_value(record, diagnostics, "cv_mean_error", np.nan))

    if cv_slope is None:
        add_check(
            "spatial_cv",
            "Spatial cross-validation",
            "fail",
            "No spatial cross-validation result was available.",
            threshold="Spatial CV is required for a defensible estimation run",
        )
    else:
        add_check(
            "spatial_cv",
            "Spatial cross-validation",
            "pass" if cv_r2 is not None and cv_r2 >= 0.0 else "warn",
            (
                f"Spatial CV was run (R²={cv_r2:.3f})."
                if cv_r2 is not None
                else "Spatial CV was run."
            ),
            value=cv_r2,
            threshold="Spatial CV must be present",
        )

    # CV slope — thresholds adapt to sample skewness
    pass_lo, pass_hi = slope_bands["pass"]
    warn_lo, warn_hi = slope_bands["warn"]
    skew_note = ""
    if sample_skewness is not None and sample_skewness > 1.0:
        skew_note = f" (skewness-adjusted; sample skew={sample_skewness:.2f})"
    slope_threshold_str = (
        f"Ideal {pass_lo:.2f}-{pass_hi:.2f}; "
        f"acceptable {warn_lo:.2f}-{warn_hi:.2f}{skew_note}"
    )

    if cv_slope is None:
        pass
    elif pass_lo <= cv_slope <= pass_hi:
        add_check(
            "cv_slope",
            "CV slope of regression",
            "pass",
            f"CV slope is {cv_slope:.3f}, close to the ideal 1.0.{skew_note}",
            value=cv_slope,
            threshold=slope_threshold_str,
        )
    elif warn_lo <= cv_slope <= warn_hi:
        add_check(
            "cv_slope",
            "CV slope of regression",
            "warn",
            f"CV slope is {cv_slope:.3f}, indicating some conditional bias or smoothing.{skew_note}",
            value=cv_slope,
            threshold=slope_threshold_str,
        )
    else:
        add_check(
            "cv_slope",
            "CV slope of regression",
            "fail",
            f"CV slope is {cv_slope:.3f}, outside an acceptable range for defensible estimation.{skew_note}",
            value=cv_slope,
            threshold=slope_threshold_str,
        )

    # CV bias — thresholds scaled by skewness
    bias_pass = 0.05 * bias_scale
    bias_warn = 0.10 * bias_scale
    if cv_me is not None and sample_std is not None:
        rel_me = abs(cv_me) / max(sample_std, 1e-12)
        metrics["cv_mean_error_relative_std"] = rel_me
        if rel_me <= bias_pass:
            add_check(
                "cv_bias",
                "CV global bias",
                "pass",
                f"Mean CV error is {cv_me:.4f}, small relative to the sample dispersion.",
                value=cv_me,
                threshold=f"|ME| <= {bias_pass*100:.0f}% of sample standard deviation",
            )
        elif rel_me <= bias_warn:
            add_check(
                "cv_bias",
                "CV global bias",
                "warn",
                f"Mean CV error is {cv_me:.4f}, which is drifting away from unbiased behaviour.",
                value=cv_me,
                threshold=f"|ME| <= {bias_warn*100:.0f}% of sample standard deviation",
            )
        else:
            add_check(
                "cv_bias",
                "CV global bias",
                "fail",
                f"Mean CV error is {cv_me:.4f}, too large relative to the sample dispersion.",
                value=cv_me,
                threshold=f"|ME| <= {bias_warn*100:.0f}% of sample standard deviation",
            )

    # Conditional bias — skewness-adjusted thresholds
    cb_slope = _finite_or_none(_value(record, diagnostics, "conditional_bias_binned_slope", np.nan))
    cb_max_bin = _finite_or_none(_value(record, diagnostics, "conditional_bias_max_abs_bin_bias", np.nan))
    cb_pass_lo, cb_pass_hi = slope_bands["pass"]
    cb_warn_lo, cb_warn_hi = slope_bands["warn"]
    cb_max_bin_warn = 0.15 * bias_scale
    cb_max_bin_fail = 0.30 * bias_scale
    if cb_slope is None:
        add_check(
            "conditional_bias",
            "Conditional bias",
            "fail",
            "Conditional-bias diagnostics were not available.",
            threshold="Conditional-bias diagnostics are required",
        )
    else:
        if cb_pass_lo <= cb_slope <= cb_pass_hi:
            slope_status = "pass"
        elif cb_warn_lo <= cb_slope <= cb_warn_hi:
            slope_status = "warn"
        else:
            slope_status = "fail"

        detail = f"Binned conditional-bias slope is {cb_slope:.3f}."
        if cb_max_bin is not None and sample_std is not None:
            cb_rel = cb_max_bin / max(sample_std, 1e-12)
            metrics["conditional_bias_max_abs_bin_bias_relative_std"] = cb_rel
            if cb_rel > cb_max_bin_fail:
                slope_status = "fail"
                detail += f" Max bin bias is {cb_max_bin:.4f}, too large relative to sample dispersion."
            elif cb_rel > cb_max_bin_warn and slope_status == "pass":
                slope_status = "warn"
                detail += f" Max bin bias is {cb_max_bin:.4f}, showing residual conditional bias."
        add_check(
            "conditional_bias",
            "Conditional bias",
            slope_status,
            detail,
            value=cb_slope,
            threshold=slope_threshold_str,
        )

    # Support swath
    swath_panels_total = int(_value(record, diagnostics, "support_swath_panels_total", 0))
    swath_panels_with_data = int(_value(record, diagnostics, "support_swath_panels_with_data", 0))
    swath_rmse = _finite_or_none(_value(record, diagnostics, "support_swath_mean_rmse", np.nan))
    swath_bias = _finite_or_none(_value(record, diagnostics, "support_swath_mean_bias", np.nan))
    if change_of_support:
        if swath_panels_total <= 0 or swath_panels_with_data <= 0 or swath_rmse is None:
            add_check(
                "support_swath",
                "Support-scale swaths",
                "fail",
                "Support-scale swath diagnostics were not available.",
                threshold="Support swaths must be present for block-support QA",
            )
        else:
            coverage = swath_panels_with_data / max(swath_panels_total, 1)
            metrics["support_swath_coverage"] = coverage
            status = "pass"
            detail = (
                f"Support swath RMSE is {swath_rmse:.4f} across "
                f"{swath_panels_with_data}/{swath_panels_total} panels."
            )
            if coverage < 0.10:
                status = "fail"
                detail += " Too few panels had support data."
            elif coverage < 0.33:
                status = "warn"
                detail += " Panel coverage is sparse."

            if sample_std is not None and sample_std > 0.0:
                rmse_ratio = swath_rmse / sample_std
                bias_ratio = (
                    abs(swath_bias) / sample_std
                    if swath_bias is not None
                    else 0.0
                )
                metrics["support_swath_rmse_relative_std"] = rmse_ratio
                metrics["support_swath_bias_relative_std"] = bias_ratio
                if rmse_ratio > 1.0 or bias_ratio > 0.20:
                    status = "fail"
                    detail += " Support-scale reproduction is poor relative to the sample variability."
                elif (rmse_ratio > 0.75 or bias_ratio > 0.10) and status == "pass":
                    status = "warn"
                    detail += " Support-scale reproduction is weaker than ideal."

            add_check(
                "support_swath",
                "Support-scale swaths",
                status,
                detail,
                value=swath_rmse,
                threshold="Support swaths should be present with low bias at block scale",
            )

    # Coverage and classification
    if n_blocks_total > 0:
        estimated_ratio = n_blocks_estimated / n_blocks_total
        metrics["estimated_block_fraction"] = estimated_ratio
        if estimated_ratio >= 0.99:
            coverage_status = "pass"
            coverage_detail = f"{n_blocks_estimated}/{n_blocks_total} blocks received estimates."
        elif estimated_ratio >= 0.95:
            coverage_status = "warn"
            coverage_detail = (
                f"Only {n_blocks_estimated}/{n_blocks_total} blocks received estimates."
            )
        else:
            coverage_status = "fail"
            coverage_detail = (
                f"Only {n_blocks_estimated}/{n_blocks_total} blocks received estimates."
            )
        add_check(
            "block_coverage",
            "Block coverage",
            coverage_status,
            coverage_detail,
            value=estimated_ratio,
            threshold="Estimated block fraction should be >= 0.99",
        )

        unclassified_ratio = unclassified_blocks / n_blocks_total
        metrics["unclassified_block_fraction"] = unclassified_ratio
        if unclassified_ratio <= 0.10:
            unclass_status = "pass"
        elif unclassified_ratio <= 0.85:
            unclass_status = "warn"
        else:
            unclass_status = "fail"
        add_check(
            "classification",
            "Classification coverage",
            unclass_status,
            f"Unclassified blocks: {unclassified_blocks}/{n_blocks_total}.",
            value=unclassified_ratio,
            threshold="Unclassified fraction should remain low",
        )

    # Extrapolation risk
    extrapolation_stats = _compute_extrapolation_stats(
        sample_coords=sample_coords,
        block_centroids=gate_block_centroids,
        azimuth=float(_value(record, diagnostics, "azimuth", 0.0)),
        dip=float(_value(record, diagnostics, "dip", 0.0)),
        pitch=float(_value(record, diagnostics, "pitch", 0.0)),
        range_max=float(_value(record, diagnostics, "range_max", 1.0)),
        range_mid=float(_value(record, diagnostics, "range_mid", 1.0)),
        range_min=float(_value(record, diagnostics, "range_min", 1.0)),
    )
    if extrapolation_stats is not None:
        metrics.update(extrapolation_stats)
        frac_beyond = extrapolation_stats["fraction_beyond_one_range"]
        p95 = extrapolation_stats["nearest_distance_p95_in_range_units"]
        if frac_beyond > 0.20 or p95 > 1.50:
            ext_status = "fail"
        elif frac_beyond > 0.05 or p95 > 1.00:
            ext_status = "warn"
        else:
            ext_status = "pass"
        add_check(
            "extrapolation",
            "Boundary / extrapolation risk",
            ext_status,
            (
                f"{frac_beyond * 100:.1f}% of blocks lie more than one anisotropic "
                f"range from the nearest sample; p95 nearest distance is {p95:.2f} ranges."
            ),
            value=frac_beyond,
            threshold="Keep extrapolation fraction low, especially beyond one range",
        )
    else:
        add_check(
            "extrapolation",
            "Boundary / extrapolation risk",
            "info",
            "Extrapolation risk could not be computed from the current geometry inputs.",
        )

    # Numerical stability / local drift fallbacks
    local_drift_fallbacks = int(_value(record, diagnostics, "local_drift_fallbacks", 0))
    if n_blocks_total > 0:
        fallback_rate = local_drift_fallbacks / n_blocks_total
        metrics["local_drift_fallback_rate"] = fallback_rate
        if fallback_rate == 0.0:
            fb_status = "pass"
        elif fallback_rate <= 0.10:
            fb_status = "warn"
        else:
            fb_status = "fail"
        add_check(
            "numerical_stability",
            "Local drift stability",
            fb_status,
            f"{local_drift_fallbacks} block solves fell back to constant drift.",
            value=fallback_rate,
            threshold="Fallbacks should be rare",
        )

    # Workflow scope
    mode_text = estimation_mode or diagnostics.get("estimation_mode") or "estimation"
    if simulation_run:
        add_check(
            "uncertainty_scope",
            "Decision-scale uncertainty",
            "pass",
            "A separate conditional simulation workflow was run alongside estimation.",
        )
    else:
        add_check(
            "uncertainty_scope",
            "Decision-scale uncertainty",
            "info",
            f"This run used {mode_text} only. Conditional simulation and recoverable-resource "
            "logic are still separate workflows.",
        )

    pass_count = sum(check.status == "pass" for check in checks)
    warn_count = sum(check.status == "warn" for check in checks)
    fail_count = sum(check.status == "fail" for check in checks)
    info_count = sum(check.status == "info" for check in checks)

    if fail_count > 0:
        overall = "fail"
        headline = "Not geostatistically defensible"
        summary = (
            f"{fail_count} critical geostatistical gate(s) failed. "
            "Use the result for diagnostics only until the failed checks are addressed."
        )
    elif warn_count > 0:
        overall = "warn"
        headline = "Conditionally defensible with reservations"
        summary = (
            f"No critical gate failed, but {warn_count} warning gate(s) remain. "
            "The estimate should not be treated as report-ready without review."
        )
    else:
        overall = "pass"
        headline = "Geostatistical gate passed"
        summary = (
            "The estimation run passed the implemented gate checks. "
            "This does not replace geological review or competent-person signoff."
        )

    actions = {
        "allow_visualise": True,
        "allow_register": fail_count == 0,
        "allow_csv_export": True,
        "allow_audit_export": True,
        "allow_jorc_export": fail_count == 0 and warn_count == 0,
    }

    return QualityGateResult(
        overall_status=overall,
        headline=headline,
        summary=summary,
        checks=checks,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        info_count=info_count,
        actions=actions,
        metrics=metrics,
    )


def _coerce_mapping(record: Any) -> Dict[str, Any]:
    if record is None:
        return {}
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        try:
            return record.to_dict()
        except Exception:
            pass
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    return {}


def _value(record: Dict[str, Any], diagnostics: Dict[str, Any], key: str, default: Any) -> Any:
    if key in diagnostics:
        return diagnostics.get(key, default)
    return record.get(key, default)


def _safe_std(values: Optional[np.ndarray]) -> Optional[float]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    std = float(np.std(arr))
    return std if np.isfinite(std) and std > 0.0 else None


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _compute_extrapolation_stats(
    *,
    sample_coords: Optional[np.ndarray],
    block_centroids: Optional[np.ndarray],
    azimuth: float,
    dip: float,
    pitch: float,
    range_max: float,
    range_mid: float,
    range_min: float,
) -> Optional[Dict[str, float]]:
    if sample_coords is None or block_centroids is None:
        return None
    samples = np.asarray(sample_coords, dtype=np.float64)
    blocks = np.asarray(block_centroids, dtype=np.float64)
    if samples.ndim != 2 or blocks.ndim != 2 or samples.shape[1] != 3 or blocks.shape[1] != 3:
        return None
    if len(samples) == 0 or len(blocks) == 0:
        return None

    R = rotation_matrix(azimuth, dip, pitch)
    S = scale_matrix(range_max, range_mid, range_min)
    T = S @ R
    samples_t = samples @ T.T
    blocks_t = blocks @ T.T

    tree = cKDTree(samples_t)
    distances, _ = tree.query(blocks_t, k=1)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        return None

    return {
        "nearest_distance_mean_in_range_units": float(np.mean(distances)),
        "nearest_distance_p95_in_range_units": float(np.percentile(distances, 95.0)),
        "fraction_beyond_one_range": float(np.mean(distances > 1.0)),
        "fraction_beyond_two_ranges": float(np.mean(distances > 2.0)),
    }


def _apply_footprint_clip_mask(
    *,
    sample_coords: Optional[np.ndarray],
    block_centroids: Optional[np.ndarray],
    azimuth: float,
    dip: float,
    pitch: float,
    range_max: float,
    range_mid: float,
    range_min: float,
    buffer_ranges: float,
) -> Optional[np.ndarray]:
    """Scope gate geometry to the clipped reporting model when requested."""
    if sample_coords is None or block_centroids is None:
        return None
    samples = np.asarray(sample_coords, dtype=np.float64)
    blocks = np.asarray(block_centroids, dtype=np.float64)
    if samples.ndim != 2 or blocks.ndim != 2 or samples.shape[1] != 3 or blocks.shape[1] != 3:
        return None
    if len(samples) == 0 or len(blocks) == 0:
        return None

    R = rotation_matrix(azimuth, dip, pitch)
    S = scale_matrix(range_max, range_mid, range_min)
    T = S @ R
    samples_t = samples @ T.T
    blocks_t = blocks @ T.T
    tree = cKDTree(samples_t)
    nn_count = tree.query_ball_point(
        blocks_t,
        r=max(float(buffer_ranges), 1e-12),
        return_length=True,
    )
    mask = np.asarray(nn_count, dtype=np.int32) > 0
    if not np.any(mask):
        return None
    return blocks[mask]


def _safe_skewness(values: Optional[np.ndarray]) -> Optional[float]:
    """Compute sample skewness, or None if not enough data."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return None
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _skewness_adjusted_slope_bands(
    skewness: Optional[float],
) -> dict:
    """Return pass/warn slope bands adjusted for sample skewness.

    For symmetric data (|skew| < 1), use the standard bands:
        pass: 0.90–1.10,  warn: 0.80–1.20

    For positively-skewed data (skew > 1), any smoothing estimator will
    underpredict highs and overpredict lows, pulling slope below 1.
    We widen the lower bound proportionally:
        skew=2  → pass_lo=0.80, warn_lo=0.65
        skew=4+ → pass_lo=0.70, warn_lo=0.55

    The upper bounds are slightly relaxed but less so — overshoot on
    skewed data usually signals a different problem.
    """
    if skewness is None or abs(skewness) <= 1.0:
        return {
            "pass": (0.90, 1.10),
            "warn": (0.80, 1.20),
        }

    # Clamp effective skewness so we don't over-relax
    s = min(abs(skewness), 5.0)

    # Linear relaxation: pass_lo drops from 0.90 → 0.70 over skew 1→5
    pass_lo = max(0.70, 0.90 - 0.05 * (s - 1.0))
    warn_lo = max(0.55, 0.80 - 0.0625 * (s - 1.0))

    # Upper bounds widen slightly
    pass_hi = min(1.15, 1.10 + 0.0125 * (s - 1.0))
    warn_hi = min(1.30, 1.20 + 0.025 * (s - 1.0))

    return {
        "pass": (pass_lo, pass_hi),
        "warn": (warn_lo, warn_hi),
    }


def _skewness_bias_scale(skewness: Optional[float]) -> float:
    """Scale factor for bias thresholds based on skewness.

    Skewed distributions naturally produce larger absolute biases in
    tail bins. Multiplying bias thresholds by this factor prevents
    false fails on deposits with legitimate right skew.
    """
    if skewness is None or abs(skewness) <= 1.0:
        return 1.0
    s = min(abs(skewness), 5.0)
    # Scale from 1.0 at skew=1 to 2.0 at skew=5
    return 1.0 + 0.25 * (s - 1.0)


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj
