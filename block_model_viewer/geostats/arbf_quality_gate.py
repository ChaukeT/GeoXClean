"""
Quality gate for ARBF (Adaptive Local RBF) estimation results.

Designed for mineral resource estimation review.  Checks cover
cross-validation diagnostics, grade statistics, and classification
guidance quality.

IMPORTANT: This gate provides GUIDANCE, not JORC-compliant certification.
Final resource classification requires competent person review considering
drill spacing, geological continuity, domain confidence, and reconciliation
evidence — factors that cannot be assessed from the estimation output alone.

The gate produces the same QualityGateResult interface expected by
geostats_controller._prepare_arbf_payload().

Change log:
- v3: Replaced variance-ratio classification with uncertainty-index checks.
  Softened JORC language.  Removed affine correction check (correction
  no longer applied).  Added uncertainty distribution check.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes matching the external quality_gate interface
# ---------------------------------------------------------------------------

@dataclass
class QualityCheck:
    """Single geostatistical QA finding."""
    code: str
    label: str
    status: str          # "PASS", "WARN", "FAIL", "INFO"
    detail: str
    value: Optional[float] = None
    threshold: str = ""


@dataclass
class QualityGateResult:
    """Structured panel-facing gate result."""
    overall_status: str   # "PASS", "WARN", "FAIL"
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
            "checks": [asdict(c) for c in self.checks],
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "info_count": self.info_count,
            "actions": dict(self.actions),
            "metrics": _convert_numpy(self.metrics),
        }


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _finite(val: Any) -> Optional[float]:
    """Return float if finite, else None."""
    if val is None:
        return None
    try:
        fv = float(val)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Thresholds — professional mineral estimation standard
# ---------------------------------------------------------------------------

# CV slope: target 1.0 (no conditional bias)
_CV_SLOPE_PASS_LO = 0.85    # lower bound for PASS
_CV_SLOPE_PASS_HI = 1.15    # upper bound for PASS
_CV_SLOPE_WARN_LO = 0.70    # lower bound for WARN (below = FAIL)
_CV_SLOPE_WARN_HI = 1.30    # upper bound for WARN (above = FAIL)

# CV R²
_CV_R2_PASS = 0.40
_CV_R2_WARN = 0.20

# CV mean error (absolute, as fraction of data std)
_CV_ME_PASS = 0.05    # |ME| ≤ 5% of std → PASS
_CV_ME_WARN = 0.10    # |ME| ≤ 10% of std → WARN

# CV RMSE / data std
_CV_RMSE_RATIO_PASS = 1.00
_CV_RMSE_RATIO_WARN = 1.50

# CV uncertainty calibration
_CV_STD_RESID_VAR_PASS_LO = 0.80
_CV_STD_RESID_VAR_PASS_HI = 1.25
_CV_STD_RESID_VAR_WARN_LO = 0.60
_CV_STD_RESID_VAR_WARN_HI = 1.60
_CV_COVER_95_PASS_LO = 0.90
_CV_COVER_95_PASS_HI = 0.98
_CV_COVER_95_WARN_LO = 0.80
_CV_COVER_95_WARN_HI = 0.995

# Fail ratio
_FAIL_RATIO_PASS = 0.05     # ≤ 5% PASS
_FAIL_RATIO_WARN = 0.15     # ≤ 15% WARN
_FAIL_RATIO_FAIL = 0.30     # > 30% FAIL

# Mean uncertainty index (0=certain, 1=uncertain)
_UNCERTAINTY_PASS = 0.40
_UNCERTAINTY_WARN = 0.60

# Grade inflation: max / median
_MAX_MEDIAN_RATIO_PASS = 50.0
_MAX_MEDIAN_RATIO_WARN = 200.0

# Negative grade fraction (for non-negative variables)
_NEGATIVE_GRADE_PASS = 0.001   # 0.1%
_NEGATIVE_GRADE_WARN = 0.01    # 1%

# Classification: minimum fraction that must be Inferred or Unclassified
_MIN_INFERRED_PLUS_UNCLASS = 0.05  # at least 5% must NOT be Measured/Indicated

# Support metrics: NaN count
_NAN_METRIC_PASS = 0    # no NaN in critical metrics


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

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
    """Evaluate whether an ARBF estimation run is defensible.

    Parameters match the external geostats.arbf.quality_gate interface so the
    controller call-site does not need to change.

    This gate is intentionally strict.  A PASS means the estimation is
    suitable for professional mineral resource reporting.  A WARN means
    significant caveats exist.  A FAIL means the estimation should not
    be used for decision-making without further investigation.
    """
    checks: List[QualityCheck] = []
    diagnostics = diagnostics or {}

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Compute data std for relative thresholds
    data_std = None
    if sample_values is not None:
        sv = np.asarray(sample_values, dtype=float)
        sv = sv[np.isfinite(sv)]
        if sv.size > 1:
            data_std = float(np.std(sv))

    # ── 1. CV SLOPE (two-sided) ───────────────────────────────────────────
    cv_slope = _finite(_get(audit_record, "cv_slope_of_regression"))
    if cv_slope is not None:
        if _CV_SLOPE_PASS_LO <= cv_slope <= _CV_SLOPE_PASS_HI:
            status = "PASS"
            detail = f"Slope = {cv_slope:.3f} (target: {_CV_SLOPE_PASS_LO}–{_CV_SLOPE_PASS_HI})"
        elif _CV_SLOPE_WARN_LO <= cv_slope <= _CV_SLOPE_WARN_HI:
            status = "WARN"
            detail = (f"Slope = {cv_slope:.3f} — conditional bias detected "
                      f"(outside {_CV_SLOPE_PASS_LO}–{_CV_SLOPE_PASS_HI})")
        else:
            status = "FAIL"
            detail = (f"Slope = {cv_slope:.3f} — severe conditional bias "
                      f"(outside {_CV_SLOPE_WARN_LO}–{_CV_SLOPE_WARN_HI})")
        checks.append(QualityCheck(
            "CV_SLOPE", "CV regression slope", status, detail,
            value=cv_slope, threshold=f"{_CV_SLOPE_PASS_LO}–{_CV_SLOPE_PASS_HI}",
        ))
    else:
        checks.append(QualityCheck(
            "CV_SLOPE", "CV regression slope",
            "WARN", "CV not run or slope unavailable — cannot assess conditional bias",
        ))

    # ── 2. CV MEAN ERROR ──────────────────────────────────────────────────
    cv_me = _finite(_get(audit_record, "cv_mean_error"))
    if cv_me is not None and data_std is not None and data_std > 1e-12:
        me_ratio = abs(cv_me) / data_std
        if me_ratio <= _CV_ME_PASS:
            status = "PASS"
            detail = f"|ME|/std = {me_ratio:.4f} (≤ {_CV_ME_PASS})"
        elif me_ratio <= _CV_ME_WARN:
            status = "WARN"
            detail = f"|ME|/std = {me_ratio:.4f} — global bias detected"
        else:
            status = "FAIL"
            detail = f"|ME|/std = {me_ratio:.4f} — significant global bias"
        checks.append(QualityCheck(
            "CV_ME", "CV mean error / data std", status, detail,
            value=me_ratio, threshold=f"≤ {_CV_ME_PASS}",
        ))

    # ── 3. CV R² ──────────────────────────────────────────────────────────
    cv_r2 = _finite(_get(audit_record, "cv_r_squared"))
    if cv_r2 is not None:
        if cv_r2 >= _CV_R2_PASS:
            status = "PASS"
        elif cv_r2 >= _CV_R2_WARN:
            status = "WARN"
        else:
            status = "FAIL"
        checks.append(QualityCheck(
            "CV_R2", "CV R-squared", status, f"R² = {cv_r2:.3f}",
            value=cv_r2, threshold=f"≥ {_CV_R2_PASS}",
        ))

    # ── 4. CV RMSE RATIO ──────────────────────────────────────────────────
    cv_rmse = _finite(_get(audit_record, "cv_rmse"))
    if cv_rmse is not None and data_std is not None and data_std > 1e-12:
        ratio = cv_rmse / data_std
        if ratio <= _CV_RMSE_RATIO_PASS:
            status = "PASS"
        elif ratio <= _CV_RMSE_RATIO_WARN:
            status = "WARN"
        else:
            status = "FAIL"
        checks.append(QualityCheck(
            "CV_RMSE", "CV RMSE / data std", status,
            f"RMSE/std = {ratio:.3f}",
            value=ratio, threshold=f"≤ {_CV_RMSE_RATIO_PASS}",
        ))

    # ── 5. FAIL FLAG RATIO ────────────────────────────────────────────────
    cv_std_resid_var = _finite(_get(audit_record, "cv_std_resid_var"))
    cv_cover_95 = _finite(_get(audit_record, "cv_cover_95"))
    if cv_std_resid_var is not None or cv_cover_95 is not None:
        pass_ok = True
        warn_ok = True
        detail_parts = []

        if cv_std_resid_var is not None:
            pass_ok &= _CV_STD_RESID_VAR_PASS_LO <= cv_std_resid_var <= _CV_STD_RESID_VAR_PASS_HI
            warn_ok &= _CV_STD_RESID_VAR_WARN_LO <= cv_std_resid_var <= _CV_STD_RESID_VAR_WARN_HI
            detail_parts.append(f"std resid var = {cv_std_resid_var:.3f}")

        if cv_cover_95 is not None:
            pass_ok &= _CV_COVER_95_PASS_LO <= cv_cover_95 <= _CV_COVER_95_PASS_HI
            warn_ok &= _CV_COVER_95_WARN_LO <= cv_cover_95 <= _CV_COVER_95_WARN_HI
            detail_parts.append(f"95% cover = {cv_cover_95:.1%}")

        if pass_ok:
            status = "PASS"
            suffix = "uncertainty is well calibrated"
        elif warn_ok:
            status = "WARN"
            suffix = "uncertainty calibration needs review"
        else:
            status = "FAIL"
            suffix = "uncertainty is materially miscalibrated"

        checks.append(QualityCheck(
            "CV_CALIBRATION", "CV uncertainty calibration", status,
            f"{', '.join(detail_parts)} â€” {suffix}",
            value=cv_std_resid_var if cv_std_resid_var is not None else cv_cover_95,
            threshold=(
                f"var {_CV_STD_RESID_VAR_PASS_LO:.2f}â€“{_CV_STD_RESID_VAR_PASS_HI:.2f}, "
                f"cover {_CV_COVER_95_PASS_LO:.0%}â€“{_CV_COVER_95_PASS_HI:.0%}"
            ),
        ))

    fail_ratio = _finite(_get(audit_record, "fail_ratio"))
    if fail_ratio is not None:
        if fail_ratio <= _FAIL_RATIO_PASS:
            status = "PASS"
        elif fail_ratio <= _FAIL_RATIO_WARN:
            status = "WARN"
        elif fail_ratio <= _FAIL_RATIO_FAIL:
            status = "WARN"
            detail = f"{fail_ratio:.1%} blocks failed (>{_FAIL_RATIO_WARN:.0%})"
        else:
            status = "FAIL"
        detail = f"{fail_ratio:.1%} blocks failed"
        checks.append(QualityCheck(
            "FAIL_RATIO", "Block fail ratio", status, detail,
            value=fail_ratio, threshold=f"≤ {_FAIL_RATIO_PASS:.0%}",
        ))

    # ── 6. UNCERTAINTY INDEX ──────────────────────────────────────────────
    mean_uncertainty = _finite(_get(audit_record, "mean_uncertainty_index"))
    if mean_uncertainty is not None:
        if mean_uncertainty <= _UNCERTAINTY_PASS:
            status = "PASS"
        elif mean_uncertainty <= _UNCERTAINTY_WARN:
            status = "WARN"
        else:
            status = "FAIL"
        checks.append(QualityCheck(
            "UNCERTAINTY", "Mean uncertainty index", status,
            f"Mean UI = {mean_uncertainty:.3f} (0=certain, 1=uncertain)",
            value=mean_uncertainty, threshold=f"≤ {_UNCERTAINTY_PASS}",
        ))
    else:
        # Fall back to variance ratio if uncertainty index not available
        var_ratio = _finite(_get(audit_record, "mean_variance_ratio"))
        if var_ratio is not None:
            if var_ratio <= 0.70:
                status = "PASS"
            elif var_ratio <= 0.90:
                status = "WARN"
            else:
                status = "FAIL"
            checks.append(QualityCheck(
                "UNCERTAINTY", "Mean variance ratio (legacy)", status,
                f"Var ratio = {var_ratio:.3f}",
                value=var_ratio, threshold="≤ 0.70",
            ))
        else:
            checks.append(QualityCheck(
                "UNCERTAINTY", "Uncertainty index",
                "WARN", "Uncertainty index unavailable",
            ))

    # ── 7. NaN SUPPORT METRICS ────────────────────────────────────────────
    sigma_block = _finite(_get(audit_record, "sigma_block"))
    support_ratio = _finite(_get(audit_record, "support_ratio"))
    variance_ratio = _finite(_get(audit_record, "variance_ratio"))
    nan_metrics = []
    if sigma_block is None:
        nan_metrics.append("sigma_block")
    if support_ratio is None:
        nan_metrics.append("support_ratio")
    if variance_ratio is None:
        nan_metrics.append("variance_ratio")

    if len(nan_metrics) == 0:
        checks.append(QualityCheck(
            "NAN_METRICS", "Support metric validity",
            "PASS", "All support metrics are finite",
        ))
    else:
        checks.append(QualityCheck(
            "NAN_METRICS", "Support metric validity",
            "WARN", f"NaN in: {', '.join(nan_metrics)} — uncertainty unreliable",
        ))

    # Detect zero-centred data (normal-score residuals, demeaned assays,
    # centred indicators, spectral-GRF benchmarks). These variables can
    # legitimately have a near-zero mean, cross zero into negatives, and
    # have enormous max/median ratios simply because median is near zero.
    # The max/median and negative-grade checks are ore-concentration
    # heuristics — they fire false-positive FAILs on centred data.
    grade_mean = _finite(_get(audit_record, "grade_mean"))
    grade_std = _finite(_get(audit_record, "grade_std"))
    _is_zero_centred = (
        grade_mean is not None and grade_std is not None and grade_std > 0
        and abs(grade_mean) < 0.10 * grade_std
    )

    # ── 8. GRADE INFLATION (max/median ratio) ─────────────────────────────
    grade_max = _finite(_get(audit_record, "grade_max"))
    grade_median = _finite(_get(audit_record, "grade_median"))
    if grade_max is not None and grade_median is not None:
        if _is_zero_centred:
            # Median ≈ 0: ratio is meaningless. Report but don't fail.
            checks.append(QualityCheck(
                "GRADE_INFLATION", "Max / Median grade ratio",
                "PASS",
                f"Skipped (zero-centred variable, median ≈ 0: "
                f"mean={grade_mean:.3g}, std={grade_std:.3g})",
            ))
        elif abs(grade_median) > 1e-12:
            max_med_ratio = abs(grade_max / grade_median)
            if max_med_ratio <= _MAX_MEDIAN_RATIO_PASS:
                status = "PASS"
                detail = f"Max/Median = {max_med_ratio:.1f}"
            elif max_med_ratio <= _MAX_MEDIAN_RATIO_WARN:
                status = "WARN"
                detail = f"Max/Median = {max_med_ratio:.1f} — possible grade inflation"
            else:
                status = "FAIL"
                detail = f"Max/Median = {max_med_ratio:.1f} — extreme grade inflation"
            checks.append(QualityCheck(
                "GRADE_INFLATION", "Max / Median grade ratio", status, detail,
                value=max_med_ratio, threshold=f"≤ {_MAX_MEDIAN_RATIO_PASS}",
            ))

    # ── 9. NEGATIVE GRADES ────────────────────────────────────────────────
    n_negative = _finite(_get(audit_record, "n_negative_grades"))
    n_active = _finite(_get(audit_record, "n_blocks_active"))
    if n_negative is not None and n_active is not None and n_active > 0:
        if _is_zero_centred:
            # A zero-centred variable is expected to have ~50% negatives.
            checks.append(QualityCheck(
                "NEGATIVE_GRADES", "Negative grade fraction",
                "PASS",
                f"Skipped (zero-centred variable, negatives expected: "
                f"{int(n_negative):,} / {int(n_active):,})",
            ))
        else:
            neg_frac = n_negative / n_active
            if neg_frac <= _NEGATIVE_GRADE_PASS:
                status = "PASS"
            elif neg_frac <= _NEGATIVE_GRADE_WARN:
                status = "WARN"
            else:
                status = "FAIL"
            checks.append(QualityCheck(
                "NEGATIVE_GRADES", "Negative grade fraction", status,
                f"{neg_frac:.2%} of blocks have negative grades ({int(n_negative):,})",
                value=neg_frac, threshold=f"≤ {_NEGATIVE_GRADE_PASS:.1%}",
            ))

    # ── 10. CLASSIFICATION GUIDANCE REALISM ──────────────────────────────
    n_measured = _get(audit_record, "measured_blocks", 0) or 0
    n_indicated = _get(audit_record, "indicated_blocks", 0) or 0
    n_inferred = _get(audit_record, "inferred_blocks", 0) or 0
    n_unclassified = _get(audit_record, "unclassified_blocks", 0) or 0
    n_total_class = n_measured + n_indicated + n_inferred + n_unclassified
    if n_total_class > 0:
        frac_mi = (n_measured + n_indicated) / n_total_class
        frac_inferred_unclass = (n_inferred + n_unclassified) / n_total_class
        if frac_inferred_unclass >= _MIN_INFERRED_PLUS_UNCLASS:
            status = "PASS"
            detail = (f"High={n_measured:,}, Moderate={n_indicated:,}, "
                      f"Low={n_inferred:,}, VeryLow={n_unclassified:,}")
        else:
            status = "WARN"
            detail = (f"{frac_mi:.1%} in High+Moderate confidence — "
                      f"review with drill spacing and geological continuity")
        checks.append(QualityCheck(
            "CLASSIFICATION", "Classification guidance realism", status, detail,
        ))

    # ── 11. DOMAIN COVERAGE ───────────────────────────────────────────────
    if domain_column:
        checks.append(QualityCheck(
            "DOMAIN", "Hard domain boundaries",
            "PASS", f"Domain column: {domain_column}",
        ))
    elif available_domain_columns:
        checks.append(QualityCheck(
            "DOMAIN", "Domain boundaries available but not used",
            "WARN",
            f"Available: {', '.join(str(c) for c in available_domain_columns[:3])}",
        ))

    # ── 12. SAMPLE COUNT ──────────────────────────────────────────────────
    n_composites = _get(audit_record, "num_composites", 0)
    if n_composites and n_composites > 0:
        if n_composites >= 50:
            status = "PASS"
        elif n_composites >= 20:
            status = "WARN"
        else:
            status = "FAIL"
        checks.append(QualityCheck(
            "SAMPLE_COUNT", "Composite count", status,
            f"{n_composites} composites",
            value=float(n_composites), threshold="≥ 50",
        ))

    # ── 13. STITCHING VARIANCE ────────────────────────────────────────────
    # If partitioning was used and stitching variance is exactly 0, suspicious
    pum_enabled = _get(diagnostics, "pum_enabled", False)
    n_subdomains = _get(diagnostics, "n_subdomains", 1)
    if pum_enabled or (n_subdomains is not None and n_subdomains > 1):
        stitch_var = _finite(_get(audit_record, "stitching_variance_mean"))
        if stitch_var is not None:
            if stitch_var > 1e-12:
                checks.append(QualityCheck(
                    "STITCH_VAR", "Stitching variance",
                    "PASS", f"Mean stitch variance = {stitch_var:.6f}",
                    value=stitch_var,
                ))
            else:
                checks.append(QualityCheck(
                    "STITCH_VAR", "Stitching variance",
                    "WARN", "Stitching variance = 0.0 — tile boundaries may not blend",
                ))

    # ── 14. BACK-TRANSFORM MEAN RATIO ───────────────────────────────────
    # The affine correction is no longer applied (it distorts local grades).
    # Instead, log the ratio as a diagnostic.  A ratio far from 1.0
    # indicates the back-transform or model needs review.
    affine_factor = _finite(_get(audit_record, "affine_correction_factor"))
    if affine_factor is not None and affine_factor != 1.0:
        if 0.80 <= affine_factor <= 1.20:
            status = "PASS"
            detail = f"Back-transform mean ratio = {affine_factor:.3f} — good"
        elif 0.50 <= affine_factor <= 2.0:
            status = "WARN"
            detail = (f"Back-transform mean ratio = {affine_factor:.3f} — "
                      "estimated mean differs from data mean; review model")
        else:
            status = "WARN"
            detail = (f"Back-transform mean ratio = {affine_factor:.3f} — "
                      "large discrepancy between estimated and data mean")
        checks.append(QualityCheck(
            "BT_MEAN_RATIO", "Back-transform mean ratio", status, detail,
            value=affine_factor, threshold="0.80–1.20",
        ))

    # ── Aggregate ─────────────────────────────────────────────────────────
    pass_count = sum(1 for c in checks if c.status == "PASS")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    info_count = sum(1 for c in checks if c.status == "INFO")

    if fail_count > 0:
        overall = "FAIL"
        headline = "Estimation did not pass quality checks"
    elif warn_count >= 3:
        overall = "FAIL"
        headline = f"Estimation has {warn_count} warnings — review needed"
    elif warn_count > 0:
        overall = "WARN"
        headline = "Estimation passed with warnings — competent person review required"
    else:
        overall = "PASS"
        headline = "Estimation passed quality checks — classification is guidance only"

    summary_parts = []
    if pass_count:
        summary_parts.append(f"{pass_count} passed")
    if warn_count:
        summary_parts.append(f"{warn_count} warnings")
    if fail_count:
        summary_parts.append(f"{fail_count} failed")
    if info_count:
        summary_parts.append(f"{info_count} info")
    summary = ", ".join(summary_parts) if summary_parts else "No checks performed"

    allow_register = overall != "FAIL"
    # Classification is guidance only — never auto-certify as JORC compliant
    allow_jorc = False  # requires CP review regardless of gate result

    return QualityGateResult(
        overall_status=overall,
        headline=headline,
        summary=summary,
        checks=checks,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        info_count=info_count,
        actions={
            "allow_register": allow_register,
            "allow_jorc_export": allow_jorc,
            "allow_visualise": True,
            "allow_visualization": True,
            "allow_csv_export": allow_register,
            "allow_audit_export": True,  # always allow audit export for review
        },
        metrics={
            "cv_slope": cv_slope,
            "cv_r2": cv_r2,
            "cv_rmse": cv_rmse,
            "cv_me": cv_me,
            "cv_std_resid_var": cv_std_resid_var,
            "cv_cover_95": cv_cover_95,
            "fail_ratio": fail_ratio,
            "mean_uncertainty_index": mean_uncertainty,
            "sigma_block": sigma_block,
            "support_ratio": support_ratio,
        },
    )
