"""
ARBF Pre-Estimation Data Analysis & Recommendation Engine.
============================================================

Performs comprehensive geostatistical data analysis before estimation:

1. **Univariate statistics** — distribution type (normal, lognormal, mixed),
   outlier detection (metal-at-risk), top-cut analysis, population splitting
2. **Spatial statistics** — drill spacing regularity, clustering coefficient,
   data density per volume, preferential sampling detection
3. **Declustering** — naive vs cell-declustered statistics comparison,
   optimal cell size search
4. **Directional variography** — auto-compute experimental variograms in
   6 directions (N-S, E-W, NE-SW, NW-SE, horizontal, downhole),
   detect anisotropy axes and ratios, auto-fit model
5. **Stationarity** — proportional effect, moving-window mean/variance,
   trend detection via drift regression
6. **Contact analysis** — grade transition sharpness, domain boundary
   detection via multimodal distribution tests
7. **Compositing assessment** — downhole variogram, optimal composite length
8. **Block model coverage** — footprint overlap, extrapolation fraction,
   range-vs-extent adequacy

References
----------
- Deutsch & Journel (1998). GSLIB: Geostatistical Software Library
- Rossi & Deutsch (2014). Mineral Resource Estimation
- Pyrcz & Deutsch (2014). Geostatistical Reservoir Modeling
- Isaaks & Srivastava (1989). Applied Geostatistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .transforms import detect_already_normal_scored

logger = logging.getLogger(__name__)


# ======================================================================
# Result Container
# ======================================================================

@dataclass
class ARBFRecommendation:
    """Full pre-estimation analysis results and recommendations."""

    settings: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)

    # Detailed analysis sections
    univariate: Dict[str, Any] = field(default_factory=dict)
    spatial: Dict[str, Any] = field(default_factory=dict)
    declustering: Dict[str, Any] = field(default_factory=dict)
    variography: Dict[str, Any] = field(default_factory=dict)
    stationarity: Dict[str, Any] = field(default_factory=dict)
    contact: Dict[str, Any] = field(default_factory=dict)
    compositing: Dict[str, Any] = field(default_factory=dict)
    block_coverage: Dict[str, Any] = field(default_factory=dict)

    def summary_text(self) -> str:
        """Full human-readable analysis report."""
        lines = []

        # ---- Univariate ----
        lines.append("=" * 62)
        lines.append("  1. UNIVARIATE DISTRIBUTION ANALYSIS")
        lines.append("=" * 62)
        u = self.univariate
        if u:
            lines.append(f"  Samples:          {u.get('n', '?')}")
            lines.append(f"  Mean:             {u.get('mean', 0):.4g}")
            lines.append(f"  Median:           {u.get('median', 0):.4g}")
            lines.append(f"  Std:              {u.get('std', 0):.4g}")
            lines.append(f"  CV:               {u.get('cv', 0):.3f}")
            lines.append(f"  Skewness:         {u.get('skewness', 0):.3f}")
            lines.append(f"  Kurtosis:         {u.get('kurtosis', 0):.3f}")
            lines.append(f"  Min:              {u.get('min', 0):.4g}")
            lines.append(f"  P05:              {u.get('p05', 0):.4g}")
            lines.append(f"  P25 (Q1):         {u.get('p25', 0):.4g}")
            lines.append(f"  P75 (Q3):         {u.get('p75', 0):.4g}")
            lines.append(f"  P95:              {u.get('p95', 0):.4g}")
            lines.append(f"  P99:              {u.get('p99', 0):.4g}")
            lines.append(f"  Max:              {u.get('max', 0):.4g}")
            lines.append(f"  IQR:              {u.get('iqr', 0):.4g}")
            lines.append(f"  Distribution:     {u.get('distribution_type', '?')}")
            if u.get("n_populations", 1) > 1:
                lines.append(
                    f"  Populations:      {u['n_populations']} detected "
                    f"(possible mixed domains)"
                )
            if u.get("support_style"):
                lines.append(f"  Support style:    {u.get('support_style')}")
            if u.get("appears_normal_scored"):
                lines.append("  Note:             Values already look normal-scored / standardized")
            # Outliers
            oc = u.get("outlier_analysis", {})
            if oc:
                if oc.get("applicable", True) and oc.get("top_cut") is not None:
                    lines.append(f"  Outlier threshold:{oc.get('top_cut'):.4g}")
                    lines.append(f"  Outliers above:   {oc.get('n_above', 0)}")
                    lines.append(
                        f"  Metal-at-risk:    {oc.get('metal_at_risk_pct', 0):.1f}% "
                        f"of total metal in {oc.get('pct_samples_above', 0):.1f}% of samples"
                    )
                    lines.append(f"  Top-cut method:   {oc.get('method', '?')}")
                elif oc.get("reason"):
                    lines.append(f"  Top-cut review:   {oc.get('reason')}")

        # ---- Spatial ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  2. SPATIAL ANALYSIS")
        lines.append("=" * 62)
        sp = self.spatial
        if sp:
            lines.append(f"  Data extent X:    [{sp.get('x_min',0):.1f}, {sp.get('x_max',0):.1f}]")
            lines.append(f"  Data extent Y:    [{sp.get('y_min',0):.1f}, {sp.get('y_max',0):.1f}]")
            lines.append(f"  Data extent Z:    [{sp.get('z_min',0):.1f}, {sp.get('z_max',0):.1f}]")
            lines.append(f"  Span:             {sp.get('extent_str', '?')}")
            lines.append(f"  Median NN dist:   {sp.get('median_spacing', 0):.2f} m")
            lines.append(f"  Mean NN dist:     {sp.get('mean_spacing', 0):.2f} m")
            lines.append(f"  P10 NN dist:      {sp.get('p10_spacing', 0):.2f} m")
            lines.append(f"  P90 NN dist:      {sp.get('p90_spacing', 0):.2f} m")
            lines.append(f"  Spacing CV:       {sp.get('spacing_cv', 0):.2f}")
            lines.append(f"  Clustering coeff: {sp.get('clustering_coeff', 0):.3f}")
            cl_interp = sp.get("clustering_interpretation", "")
            if cl_interp:
                lines.append(f"  Interpretation:   {cl_interp}")
            lines.append(f"  Data density:     {sp.get('density_per_1000m3', 0):.4f} samples/1000m3")
            lines.append(
                f"  Drill pattern:    {sp.get('drill_pattern', '?')}"
            )
            if sp.get("preferential_sampling"):
                lines.append("  !! PREFERENTIAL SAMPLING DETECTED")
                lines.append(f"     Correlation(grade, density): {sp.get('pref_correlation', 0):.3f}")

        # ---- Declustering ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  3. DECLUSTERING ANALYSIS")
        lines.append("=" * 62)
        dc = self.declustering
        if dc:
            lines.append(f"  Naive mean:       {dc.get('naive_mean', 0):.4g}")
            lines.append(f"  Declustered mean: {dc.get('declustered_mean', 0):.4g}")
            lines.append(f"  Shift:            {dc.get('shift_pct', 0):+.2f}%")
            lines.append(f"  Optimal cell:     {dc.get('optimal_cell_size', 0):.1f} m")
            lines.append(f"  N effective:      {dc.get('n_effective', 0):.0f} "
                         f"(of {dc.get('n_total', 0)})")
            if abs(dc.get("shift_pct", 0)) > 5:
                lines.append("  !! Significant clustering bias detected")

        # ---- Directional variography ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  4. DIRECTIONAL VARIOGRAPHY")
        lines.append("=" * 62)
        vg = self.variography
        if vg:
            lines.append(f"  Lag spacing:      {vg.get('lag_spacing', 0):.1f} m")
            lines.append(f"  N lags:           {vg.get('n_lags', 0)}")
            lines.append(f"  Data variance:    {vg.get('data_variance', 0):.4g}")
            lines.append("")

            for dname, dinfo in vg.get("directions", {}).items():
                lines.append(f"  --- {dname} ---")
                lines.append(
                    f"    Azimuth={dinfo.get('azimuth', 0):.0f}, "
                    f"Dip={dinfo.get('dip', 0):.0f}"
                )
                lines.append(f"    Fitted range:   {dinfo.get('range', 0):.1f} m")
                lines.append(f"    Fitted sill:    {dinfo.get('sill', 0):.4g}")
                lines.append(f"    Nugget:         {dinfo.get('nugget', 0):.4g}")
                lines.append(f"    N pairs (avg):  {dinfo.get('avg_pairs', 0):.0f}")
                if dinfo.get("reliable"):
                    lines.append(f"    Quality:        RELIABLE (min pairs >= 30)")
                else:
                    lines.append(f"    Quality:        LOW CONFIDENCE (sparse pairs)")

            ani = vg.get("anisotropy", {})
            if ani:
                lines.append("")
                lines.append(f"  Major direction:  Az={ani.get('major_azimuth',0):.0f}, "
                             f"Dip={ani.get('major_dip',0):.0f}")
                lines.append(f"  Major range:      {ani.get('range_max', 0):.1f} m")
                lines.append(f"  Semi range:       {ani.get('range_mid', 0):.1f} m")
                lines.append(f"  Minor range:      {ani.get('range_min', 0):.1f} m")
                lines.append(f"  Anisotropy ratio: {ani.get('ratio_max_min', 0):.2f}")
                lines.append(f"  Nugget (global):  {ani.get('nugget', 0):.4g}")
                lines.append(f"  Sill (global):    {ani.get('sill', 0):.4g}")
                lines.append(f"  Nugget fraction:  {ani.get('nugget_fraction', 0):.1%}")

            model = vg.get("recommended_model", {})
            if model:
                lines.append("")
                lines.append(f"  Recommended model: {model.get('type', '?')}")

        # ---- Stationarity ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  5. STATIONARITY CHECKS")
        lines.append("=" * 62)
        st = self.stationarity
        if st:
            lines.append(f"  Trend detected:   {st.get('trend_detected', False)}")
            if st.get("trend_detected"):
                lines.append(f"  Trend direction:  {st.get('trend_direction', '?')}")
                lines.append(f"  Trend R2:         {st.get('trend_r2', 0):.4f}")
                lines.append(f"  Trend slope:      {st.get('trend_slope', 0):.4g}")
            lines.append(f"  Proportional eff: {st.get('proportional_effect', False)}")
            if st.get("proportional_effect"):
                lines.append(
                    f"  Mean-var corr:    {st.get('mean_var_correlation', 0):.3f}"
                )
                lines.append("  !! Proportional effect: variance increases with "
                             "mean grade. Normal-score transform recommended.")
            lines.append(f"  Window mean range:{st.get('window_mean_range', '')}")
            lines.append(f"  Window std range: {st.get('window_std_range', '')}")

        # ---- Contact analysis ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  6. CONTACT / DOMAIN ANALYSIS")
        lines.append("=" * 62)
        ct = self.contact
        if ct:
            lines.append(f"  Distribution test:{ct.get('normality_test', '?')}")
            lines.append(f"  Shapiro p-value:  {ct.get('shapiro_p', 0):.4g}")
            lines.append(f"  Multimodal:       {ct.get('multimodal', False)}")
            if ct.get("multimodal"):
                lines.append(f"  N modes detected: {ct.get('n_modes', 0)}")
                lines.append(
                    "  !! Multiple grade populations suggest geological "
                    "domains. Consider domain-constrained estimation."
                )
            lines.append(f"  Log-normality:    {ct.get('lognormal_test', '?')}")

        # ---- Compositing ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  7. COMPOSITING ASSESSMENT")
        lines.append("=" * 62)
        cp = self.compositing
        if cp:
            lines.append(f"  Sample lengths:   {cp.get('length_summary', '?')}")
            lines.append(f"  Median length:    {cp.get('median_length', 0):.2f} m")
            lines.append(f"  Length CV:        {cp.get('length_cv', 0):.3f}")
            lines.append(f"  Regular:          {cp.get('is_regular', False)}")
            if cp.get("downhole_nugget_ratio") is not None:
                lines.append(
                    f"  Downhole nugget:  {cp.get('downhole_nugget_ratio', 0):.1%} "
                    f"of downhole variance"
                )
            lines.append(f"  Recommendation:   {cp.get('recommendation', '?')}")

        # ---- Block coverage ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  8. BLOCK MODEL COVERAGE")
        lines.append("=" * 62)
        bc = self.block_coverage
        if bc:
            lines.append(f"  Block model:      {bc.get('extent_str', 'not loaded')}")
            lines.append(f"  N blocks:         {bc.get('n_blocks', '?')}")
            lines.append(f"  Drill footprint:  {bc.get('drill_extent_str', '?')}")
            lines.append(f"  Volume ratio:     {bc.get('volume_ratio', 0):.1f}x")
            lines.append(f"  Blocks within 1 range:  {bc.get('pct_within_1_range', 0):.1f}%")
            lines.append(f"  Blocks within 2 ranges: {bc.get('pct_within_2_range', 0):.1f}%")
            lines.append(f"  Extrapolation risk:     {bc.get('extrapolation_risk', '?')}")
            if bc.get("clip_recommended"):
                lines.append(
                    "  !! CLIP RECOMMENDED: block model much larger than "
                    "drilled area"
                )

        # ---- Recommendations ----
        lines.append("")
        lines.append("=" * 62)
        lines.append("  RECOMMENDATIONS")
        lines.append("=" * 62)
        for r in self.reasons:
            lines.append(f"  >> {r}")

        if self.warnings:
            lines.append("")
            lines.append("-" * 62)
            lines.append("  WARNINGS")
            lines.append("-" * 62)
            for w in self.warnings:
                lines.append(f"  !! {w}")

        return "\n".join(lines)


# ======================================================================
# Main entry point
# ======================================================================

def recommend_arbf_settings(
    coords: np.ndarray,
    values: np.ndarray,
    block_centroids: Optional[np.ndarray] = None,
    block_sizes: Optional[np.ndarray] = None,
) -> ARBFRecommendation:
    """Full pre-estimation geostatistical data analysis.

    Performs 8 analysis stages and returns settings + justifications.

    Parameters
    ----------
    coords : (N, 3) array
        Sample coordinates.
    values : (N,) array
        Sample grade values.
    block_centroids : (B, 3) array, optional
        Block model centroids.
    block_sizes : (3,) or (B, 3) array, optional
        Block dimensions.
    """
    rec = ARBFRecommendation()
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    N = len(values)

    if N < 10:
        rec.warnings.append("Too few samples for reliable analysis.")
        return rec

    # 1. Univariate distribution analysis
    _analyse_univariate(rec, values)

    # 2. Spatial analysis
    _analyse_spatial(rec, coords, values)

    # 3. Declustering analysis
    _analyse_declustering(rec, coords, values)

    # 4. Directional variography
    _analyse_variography(rec, coords, values)

    # 5. Stationarity checks
    _analyse_stationarity(rec, coords, values)

    # 6. Contact / domain analysis
    _analyse_contacts(rec, values)

    # 7. Compositing assessment
    _analyse_compositing(rec, coords, values)

    # 8. Block model coverage
    _analyse_block_coverage(
        rec, coords, values, block_centroids, block_sizes,
    )

    # 9. Synthesise recommendations
    _build_recommendations(rec, block_sizes=block_sizes)

    logger.info(
        "ARBF data analysis complete: N=%d, CV=%.2f, "
        "recommended range=%.1f, NS=%s, clip=%s",
        N, rec.univariate.get("cv", 0),
        rec.settings.get("range_max", 0),
        rec.settings.get("use_normal_score"),
        rec.settings.get("clip_to_drill_footprint"),
    )

    return rec


# ======================================================================
# 1. Univariate Distribution Analysis
# ======================================================================

def _analyse_univariate(rec: ARBFRecommendation, values: np.ndarray) -> None:
    """Distribution type, outliers, top-cut, population splitting."""
    N = len(values)
    v_mean = float(np.mean(values))
    v_std = float(np.std(values))
    v_median = float(np.median(values))
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    cv = v_std / max(abs(v_mean), 1e-12)

    # Percentiles
    pcts = np.percentile(values, [5, 10, 25, 50, 75, 90, 95, 97.5, 99])
    p05, p10, p25, p50, p75, p90, p95, p975, p99 = pcts
    iqr = p75 - p25

    # Moments
    centered = values - v_mean
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    skewness = m3 / max(m2 ** 1.5, 1e-30)
    kurtosis = m4 / max(m2 ** 2, 1e-30) - 3.0  # excess kurtosis

    support_profile = _characterise_value_support(
        values=values,
        mean=v_mean,
        median=v_median,
        std=v_std,
        p05=float(p05),
        p95=float(p95),
    )

    # Distribution type detection
    dist_type = "unknown"
    if support_profile["appears_normal_scored"]:
        dist_type = "approximately Gaussian / already normal-scored"
    elif abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
        dist_type = "approximately normal"
    elif skewness > 1.0:
        # Check if log-transform normalises
        pos = values[values > 0]
        if len(pos) > 0.9 * N:
            log_vals = np.log(pos)
            log_skew = float(np.mean((log_vals - np.mean(log_vals)) ** 3)) / \
                       max(float(np.std(log_vals)) ** 3, 1e-30)
            if abs(log_skew) < 1.0:
                dist_type = "lognormal"
            else:
                dist_type = "positively skewed (not lognormal)"
        else:
            dist_type = "positively skewed (contains zeros/negatives)"
    elif skewness < -1.0:
        dist_type = "negatively skewed"
    elif 0.5 <= abs(skewness) <= 1.0:
        dist_type = "moderately skewed"
    else:
        dist_type = "near-symmetric"

    # Population splitting — detect multimodality via kernel density
    n_populations = _detect_populations(values)

    # Outlier analysis — metal-at-risk method (Rossi & Deutsch 2014, Ch.3)
    outlier_analysis = _analyse_outliers(
        values,
        p75,
        iqr,
        v_mean,
        v_std,
        skewness=skewness,
        support_profile=support_profile,
    )

    rec.univariate = {
        "n": N,
        "mean": v_mean,
        "median": v_median,
        "std": v_std,
        "cv": cv,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min": v_min,
        "max": v_max,
        "p05": p05, "p10": p10, "p25": p25,
        "p75": p75, "p90": p90, "p95": p95,
        "p975": p975, "p99": p99,
        "iqr": iqr,
        "distribution_type": dist_type,
        "n_populations": n_populations,
        "outlier_analysis": outlier_analysis,
        **support_profile,
    }


def _characterise_value_support(
    values: np.ndarray,
    mean: float,
    median: float,
    std: float,
    p05: float,
    p95: float,
) -> Dict[str, Any]:
    """Classify whether the variable behaves like raw grade, centred residuals, or standardized data."""
    values = np.asarray(values, dtype=np.float64).ravel()
    positive_fraction = float(np.mean(values > 0))
    negative_fraction = float(np.mean(values < 0))
    zero_tolerance = max(std * 1e-6, 1e-12)
    zero_fraction = float(np.mean(np.abs(values) <= zero_tolerance))

    appears_normal_scored = bool(detect_already_normal_scored(values))
    signed_support = positive_fraction > 0.1 and negative_fraction > 0.1
    centered_signed = (
        signed_support
        and abs(mean) < max(0.25 * std, 1e-6)
        and abs(median) < max(0.20 * std, 1e-6)
    )
    mostly_nonnegative = negative_fraction < 0.05
    has_physical_zero_bound = mostly_nonnegative and p95 > 0.0 and mean > -0.05 * max(std, 1e-12)
    positive_grade_like = has_physical_zero_bound and not appears_normal_scored
    cv_is_meaningful = positive_grade_like and mean > 0.0 and positive_fraction > 0.8

    if appears_normal_scored:
        support_style = "standardized / Gaussian support"
    elif centered_signed:
        support_style = "signed centered support"
    elif positive_grade_like:
        support_style = "positive grade-like support"
    elif mostly_nonnegative:
        support_style = "mostly non-negative support"
    else:
        support_style = "mixed signed support"

    return {
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "zero_fraction": zero_fraction,
        "appears_normal_scored": appears_normal_scored,
        "signed_support": signed_support,
        "centered_signed": centered_signed,
        "has_physical_zero_bound": has_physical_zero_bound,
        "positive_grade_like": positive_grade_like,
        "cv_is_meaningful": cv_is_meaningful,
        "support_style": support_style,
    }


def _detect_populations(values: np.ndarray) -> int:
    """Detect number of grade populations via histogram valley detection."""
    # Use Freedman-Diaconis binning
    N = len(values)
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    if iqr < 1e-12:
        return 1
    bin_width = 2.0 * iqr / (N ** (1.0 / 3.0))
    n_bins = max(10, min(100, int(np.ceil((values.max() - values.min()) / bin_width))))

    hist, edges = np.histogram(values, bins=n_bins)

    # Smooth histogram to reduce noise
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    if len(hist) > len(kernel):
        smoothed = np.convolve(hist, kernel, mode="same")
    else:
        smoothed = hist.astype(float)

    # Count peaks (local maxima with prominence > 10% of max count)
    threshold = 0.10 * smoothed.max()
    peaks = 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] > threshold:
                peaks += 1

    return max(peaks, 1)


def _analyse_outliers(
    values: np.ndarray,
    p75: float,
    iqr: float,
    mean: float,
    std: float,
    *,
    skewness: float,
    support_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Metal-at-risk outlier analysis.

    Computes three candidate top-cuts and selects the most conservative:
    1. Tukey fence: P75 + 3*IQR
    2. Percentile: P97.5
    3. Metal-at-risk: threshold where top X% of samples contain Y% of metal
       and removing them changes the mean by > 10%.
    """
    N = len(values)
    sorted_vals = np.sort(values)
    positive_grade_like = bool(support_profile.get("positive_grade_like", False))
    appears_normal_scored = bool(support_profile.get("appears_normal_scored", False))
    centered_signed = bool(support_profile.get("centered_signed", False))

    if appears_normal_scored:
        return {
            "applicable": False,
            "method": "not_applicable",
            "top_cut": None,
            "n_above": 0,
            "pct_samples_above": 0.0,
            "metal_at_risk_pct": 0.0,
            "tukey_cut": None,
            "pct_cut": None,
            "mar_cut": None,
            "mar_pct_samples": 0.0,
            "mar_pct_metal": 0.0,
            "reason": "Top-cut review skipped because the variable already looks normal-scored.",
        }

    if not positive_grade_like or centered_signed or skewness < 1.0:
        return {
            "applicable": False,
            "method": "not_recommended",
            "top_cut": None,
            "n_above": 0,
            "pct_samples_above": 0.0,
            "metal_at_risk_pct": 0.0,
            "tukey_cut": None,
            "pct_cut": None,
            "mar_cut": None,
            "mar_pct_samples": 0.0,
            "mar_pct_metal": 0.0,
            "reason": (
                "Top-cut review skipped because the variable does not behave like a strongly "
                "positively skewed raw grade distribution."
            ),
        }

    # Tukey upper fence (outlier = beyond 3*IQR above P75)
    tukey_cut = p75 + 3.0 * iqr if iqr > 0 else float("inf")

    # P97.5 cut
    pct_cut = float(np.percentile(values, 97.5))

    # Metal-at-risk: cumulative metal from top
    metal_values = np.clip(values, 0.0, None)
    total_metal = float(np.sum(metal_values))
    if total_metal > 0:
        cumulative_from_top = np.cumsum(np.sort(metal_values)[::-1])
        pct_metal = cumulative_from_top / total_metal
        # Find where top samples contribute > 40% of metal
        mar_idx = np.searchsorted(pct_metal, 0.40)
        if mar_idx < N:
            mar_cut = float(sorted_vals[N - mar_idx - 1])
            mar_pct_samples = 100.0 * (mar_idx + 1) / N
            mar_pct_metal = 100.0 * pct_metal[mar_idx]
        else:
            mar_cut = float("inf")
            mar_pct_samples = 0
            mar_pct_metal = 0
    else:
        mar_cut = float("inf")
        mar_pct_samples = 0
        mar_pct_metal = 0

    # Select most appropriate cut
    candidates = [
        ("tukey_3iqr", tukey_cut),
        ("percentile_97.5", pct_cut),
    ]
    if mar_cut < float("inf"):
        candidates.append(("metal_at_risk_40pct", mar_cut))

    # Use the lowest cut that still retains > 95% of samples
    best_method = "none"
    best_cut = v_max = float(np.max(values))
    for method, cut in sorted(candidates, key=lambda x: x[1]):
        n_above = int(np.sum(values > cut))
        if n_above <= 0.05 * N and cut < best_cut:
            best_cut = cut
            best_method = method

    n_above = int(np.sum(values > best_cut))
    metal_above = float(np.sum(metal_values[values > best_cut]))

    return {
        "applicable": True,
        "method": best_method,
        "top_cut": best_cut,
        "n_above": n_above,
        "pct_samples_above": 100.0 * n_above / max(N, 1),
        "metal_at_risk_pct": 100.0 * metal_above / max(total_metal, 1e-30),
        "tukey_cut": tukey_cut,
        "pct_cut": pct_cut,
        "mar_cut": mar_cut if mar_cut < float("inf") else None,
        "mar_pct_samples": mar_pct_samples,
        "mar_pct_metal": mar_pct_metal,
        "reason": "",
    }


# ======================================================================
# 2. Spatial Analysis
# ======================================================================

def _analyse_spatial(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
) -> None:
    """Drill spacing, clustering, preferential sampling."""
    from scipy.spatial import cKDTree

    N = len(coords)
    tree = cKDTree(coords)

    # Nearest-neighbour distances
    k = min(6, N - 1)
    dists, indices = tree.query(coords, k=k + 1)  # +1 for self
    nn1 = dists[:, 1]  # nearest
    nn5 = dists[:, min(5, k)] if k >= 5 else nn1

    median_spacing = float(np.median(nn1))
    mean_spacing = float(np.mean(nn1))
    p10_spacing = float(np.percentile(nn1, 10))
    p90_spacing = float(np.percentile(nn1, 90))
    spacing_cv = float(np.std(nn1) / max(mean_spacing, 1e-12))

    # Data extent
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent = maxs - mins
    extent_str = f"{extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} m"

    # Data density (samples per 1000 m^3)
    volume = max(np.prod(extent), 1.0)
    density = N / volume * 1000.0

    # Clustering coefficient: ratio of median NN distance to expected
    # uniform spacing.  Values < 0.5 indicate strong clustering.
    # For a uniform Poisson process: E[NN] = 0.5 * (V/N)^(1/3)
    expected_nn = 0.5 * (volume / N) ** (1.0 / 3.0)
    clustering_coeff = median_spacing / max(expected_nn, 1e-12)

    if clustering_coeff < 0.4:
        cl_interp = "STRONG clustering (preferential drilling)"
    elif clustering_coeff < 0.7:
        cl_interp = "Moderate clustering (semi-regular drilling)"
    elif clustering_coeff < 1.3:
        cl_interp = "Near-uniform spacing (regular drill grid)"
    else:
        cl_interp = "Dispersed (unusually even spacing)"

    # Drill pattern detection (regular grid vs random)
    # Check if XY positions cluster along grid lines
    xy = coords[:, :2]
    xy_nn = dists[:, 1]
    regularity = float(np.std(xy_nn) / max(np.mean(xy_nn), 1e-12))
    if regularity < 0.3:
        drill_pattern = "regular grid (CV < 0.3)"
    elif regularity < 0.6:
        drill_pattern = "semi-regular (CV 0.3-0.6)"
    else:
        drill_pattern = "irregular / clustered (CV > 0.6)"

    # Preferential sampling detection: correlation between local
    # grade and local sample density.  High positive correlation
    # means denser drilling in high-grade areas.
    pref_sampling = False
    pref_corr = 0.0
    if N >= 50:
        # Local density = 1 / (mean distance to 5 nearest neighbours)
        local_density = 1.0 / np.maximum(np.mean(dists[:, 1:min(6, k+1)], axis=1), 1e-12)
        if np.std(local_density) > 0 and np.std(values) > 0:
            pref_corr = float(np.corrcoef(values, local_density)[0, 1])
            if abs(pref_corr) > 0.3:
                pref_sampling = True

    rec.spatial = {
        "x_min": float(mins[0]), "x_max": float(maxs[0]),
        "y_min": float(mins[1]), "y_max": float(maxs[1]),
        "z_min": float(mins[2]), "z_max": float(maxs[2]),
        "extent": extent.tolist(),
        "extent_str": extent_str,
        "median_spacing": median_spacing,
        "mean_spacing": mean_spacing,
        "p10_spacing": p10_spacing,
        "p90_spacing": p90_spacing,
        "spacing_cv": spacing_cv,
        "clustering_coeff": clustering_coeff,
        "clustering_interpretation": cl_interp,
        "density_per_1000m3": density,
        "data_volume": float(volume),
        "drill_pattern": drill_pattern,
        "preferential_sampling": pref_sampling,
        "pref_correlation": pref_corr,
    }

    if pref_sampling:
        rec.warnings.append(
            f"Preferential sampling detected (grade-density "
            f"correlation={pref_corr:.2f}). Declustering is critical."
        )


# ======================================================================
# 3. Declustering Analysis
# ======================================================================

def _analyse_declustering(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
) -> None:
    """Naive vs declustered statistics, optimal cell size."""
    N = len(values)
    naive_mean = float(np.mean(values))

    # Cell declustering: sweep cell sizes
    median_spacing = rec.spatial.get("median_spacing", 10.0)
    min_cell = max(median_spacing * 0.5, 1.0)
    max_cell = max(median_spacing * 10.0, 50.0)
    n_sizes = 15
    cell_sizes = np.linspace(min_cell, max_cell, n_sizes)

    best_mean = naive_mean
    best_cell = median_spacing
    means = []

    for cs in cell_sizes:
        weights = _cell_decluster_weights(coords, cs)
        w_mean = float(np.average(values, weights=weights))
        means.append(w_mean)

    means = np.array(means)

    # Optimal cell = cell size that gives the minimum (or maximum)
    # declustered mean, depending on whether data is preferentially
    # sampled in high-grade or low-grade areas.
    # Standard practice: use the cell size that gives the most
    # stable (flattest) mean — look for the plateau.
    if len(means) >= 3:
        # Gradient stability: where the derivative is smallest
        grad = np.abs(np.gradient(means, cell_sizes))
        # Smooth gradient
        if len(grad) > 3:
            grad_smooth = np.convolve(grad, [0.25, 0.5, 0.25], mode="same")
        else:
            grad_smooth = grad
        best_idx = int(np.argmin(grad_smooth[1:-1])) + 1
        best_cell = float(cell_sizes[best_idx])
        best_mean = float(means[best_idx])

    # Compute weights at optimal cell size
    optimal_weights = _cell_decluster_weights(coords, best_cell)
    n_effective = float(1.0 / np.sum(optimal_weights ** 2)) if np.sum(optimal_weights ** 2) > 0 else N

    shift_pct = 100.0 * (best_mean - naive_mean) / max(abs(naive_mean), 1e-12)

    rec.declustering = {
        "naive_mean": naive_mean,
        "declustered_mean": best_mean,
        "shift_pct": shift_pct,
        "optimal_cell_size": best_cell,
        "n_effective": n_effective,
        "n_total": N,
        "cell_sizes_tested": cell_sizes.tolist(),
        "means_by_cell": means.tolist(),
    }


def _cell_decluster_weights(
    coords: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """Simple cell-declustering weights (Deutsch 1989)."""
    N = len(coords)
    mins = coords.min(axis=0)

    # Assign each sample to a cell
    cell_idx = np.floor((coords - mins) / max(cell_size, 1e-12)).astype(int)

    # Count samples per cell
    unique_cells, inverse, counts = np.unique(
        cell_idx, axis=0, return_inverse=True, return_counts=True,
    )
    n_occupied = len(unique_cells)

    # Weight = 1 / (n_in_cell * n_occupied_cells)
    weights = 1.0 / (counts[inverse].astype(float) * n_occupied)
    weights /= weights.sum()  # normalise
    return weights


# ======================================================================
# 4. Directional Variography
# ======================================================================

def _analyse_variography(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
) -> None:
    """Auto-compute experimental variograms in 6 directions + fit."""
    N = len(values)
    data_var = float(np.var(values))
    median_spacing = rec.spatial.get("median_spacing", 10.0)

    # Lag spacing: approximately median NN distance
    lag_spacing = max(median_spacing * 1.0, 1.0)
    max_lag_dist = float(np.max(np.max(coords, axis=0) - np.min(coords, axis=0))) * 0.5
    n_lags = max(8, min(20, int(max_lag_dist / lag_spacing)))

    # Subsample for speed
    max_n = 800
    if N > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_n, replace=False)
        sub_c = coords[idx]
        sub_v = values[idx]
    else:
        sub_c = coords
        sub_v = values

    # 6 search directions:
    # (azimuth, dip, bandwidth_fraction, label)
    directions = [
        (0, 0, "N-S (Az=0)"),
        (45, 0, "NE-SW (Az=45)"),
        (90, 0, "E-W (Az=90)"),
        (135, 0, "NW-SE (Az=135)"),
        (0, 90, "Downhole (vertical)"),
        (0, 0, "Omnidirectional"),  # special: no direction filter
    ]

    dir_results = {}
    fitted_ranges = []

    for azimuth, dip, label in directions:
        is_omni = "Omnidirectional" in label
        gamma, lags, pairs = _compute_directional_variogram(
            sub_c, sub_v, lag_spacing, n_lags,
            azimuth=azimuth, dip=dip,
            angular_tolerance=90.0 if is_omni else 22.5,
            bandwidth=max_lag_dist if is_omni else lag_spacing * 3.0,
        )

        # Auto-fit spherical model
        fit_range, fit_sill, fit_nugget, reliable = _fit_spherical_variogram(
            lags, gamma, pairs, data_var,
        )

        dir_results[label] = {
            "azimuth": azimuth,
            "dip": dip,
            "range": fit_range,
            "sill": fit_sill,
            "nugget": fit_nugget,
            "reliable": reliable,
            "avg_pairs": float(np.mean(pairs)) if len(pairs) > 0 else 0,
            "lags": lags.tolist(),
            "gamma": gamma.tolist(),
            "pairs": pairs.tolist(),
        }

        if reliable and fit_range > 0:
            fitted_ranges.append((label, fit_range, fit_sill, fit_nugget, azimuth, dip))

    # Determine anisotropy from fitted ranges
    anisotropy = {}
    if len(fitted_ranges) >= 2:
        # Sort by range (longest = major direction)
        fitted_ranges.sort(key=lambda x: x[1], reverse=True)
        major = fitted_ranges[0]
        minor = fitted_ranges[-1]

        # Use omnidirectional for nugget/sill if available
        omni = dir_results.get("Omnidirectional", {})
        global_nugget = omni.get("nugget", major[3])
        global_sill = omni.get("sill", major[2])
        total_sill = global_nugget + global_sill
        nugget_frac = global_nugget / max(total_sill, 1e-12)

        range_max = major[1]
        range_min = minor[1]

        # Find semi-major (perpendicular to major in horizontal plane)
        # Use the direction closest to 90 degrees from major azimuth
        semi_ranges = [
            r for r in fitted_ranges
            if abs((r[4] - major[4] + 180) % 180 - 90) < 45 and r[0] != major[0]
        ]
        range_mid = semi_ranges[0][1] if semi_ranges else (range_max + range_min) / 2

        anisotropy = {
            "major_azimuth": major[4],
            "major_dip": major[5],
            "range_max": range_max,
            "range_mid": range_mid,
            "range_min": range_min,
            "ratio_max_min": range_max / max(range_min, 1e-12),
            "nugget": global_nugget,
            "sill": global_sill,
            "nugget_fraction": nugget_frac,
        }
    elif len(fitted_ranges) == 1:
        r = fitted_ranges[0]
        anisotropy = {
            "major_azimuth": r[4],
            "major_dip": r[5],
            "range_max": r[1],
            "range_mid": r[1],
            "range_min": r[1] * 0.5,
            "ratio_max_min": 2.0,
            "nugget": r[3],
            "sill": r[2],
            "nugget_fraction": r[3] / max(r[2] + r[3], 1e-12),
        }

    # Recommended model type
    recommended_model = {}
    if anisotropy:
        nf = anisotropy.get("nugget_fraction", 0)
        if nf > 0.6:
            recommended_model["type"] = "spherical (high nugget — keep finite range)"
        elif anisotropy.get("ratio_max_min", 1) > 3.0:
            recommended_model["type"] = "spherical (strong anisotropy)"
        else:
            recommended_model["type"] = "spherical (standard mining deposit)"

    rec.variography = {
        "lag_spacing": lag_spacing,
        "n_lags": n_lags,
        "data_variance": data_var,
        "directions": dir_results,
        "anisotropy": anisotropy,
        "recommended_model": recommended_model,
    }


def _compute_directional_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    lag_spacing: float,
    n_lags: int,
    azimuth: float = 0.0,
    dip: float = 0.0,
    angular_tolerance: float = 22.5,
    bandwidth: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental semi-variogram in a given direction.

    Returns (gamma, lag_centres, pair_counts).
    """
    N = len(values)

    # Direction vector
    az_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)
    dir_vec = np.array([
        np.sin(az_rad) * np.cos(dip_rad),
        np.cos(az_rad) * np.cos(dip_rad),
        -np.sin(dip_rad),
    ])

    # Random pair sampling for speed
    rng = np.random.default_rng(42)
    max_pairs = 100000
    n_possible = N * (N - 1) // 2

    if n_possible > max_pairs:
        idx_i = rng.integers(0, N, max_pairs)
        idx_j = rng.integers(0, N, max_pairs)
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
    else:
        # All pairs
        ii, jj = np.triu_indices(N, k=1)
        idx_i = ii
        idx_j = jj

    # Separation vectors
    delta = coords[idx_j] - coords[idx_i]
    h = np.sqrt(np.sum(delta ** 2, axis=1))

    # Direction filter (unless omnidirectional)
    if angular_tolerance < 89.0:
        h_nonzero = np.maximum(h, 1e-12)
        cos_angle = np.abs(np.sum(delta * dir_vec, axis=1)) / h_nonzero
        cos_tol = np.cos(np.radians(angular_tolerance))
        dir_mask = cos_angle >= cos_tol
    else:
        dir_mask = np.ones(len(h), dtype=bool)

    # Squared grade differences
    sq_diff = 0.5 * (values[idx_i] - values[idx_j]) ** 2

    # Bin into lags
    gamma = np.zeros(n_lags)
    lags = np.zeros(n_lags)
    pairs = np.zeros(n_lags, dtype=int)

    for lag_idx in range(n_lags):
        h_min = lag_idx * lag_spacing
        h_max = (lag_idx + 1) * lag_spacing
        mask = dir_mask & (h >= h_min) & (h < h_max)
        n_pairs = int(mask.sum())
        pairs[lag_idx] = n_pairs
        lags[lag_idx] = (h_min + h_max) / 2.0
        if n_pairs > 0:
            gamma[lag_idx] = float(np.mean(sq_diff[mask]))

    return gamma, lags, pairs


def _fit_spherical_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    pairs: np.ndarray,
    data_variance: float,
    min_pairs: int = 30,
) -> Tuple[float, float, float, bool]:
    """Fit spherical variogram model to experimental values.

    Returns (range, partial_sill, nugget, reliable).
    """
    # Filter to bins with enough pairs
    valid = pairs >= min_pairs
    if valid.sum() < 3:
        # Try with lower threshold
        valid = pairs >= 10
    if valid.sum() < 3:
        return (0.0, data_variance, 0.0, False)

    lags_v = lags[valid]
    gamma_v = gamma[valid]

    # Nugget estimate: extrapolate first 2 bins to h=0
    if len(lags_v) >= 2 and lags_v[1] > lags_v[0]:
        slope_01 = (gamma_v[1] - gamma_v[0]) / (lags_v[1] - lags_v[0])
        nugget_est = max(0.0, gamma_v[0] - slope_01 * lags_v[0])
    else:
        nugget_est = gamma_v[0] * 0.5

    # Sill estimate: mean of upper 30% of valid lags
    upper_start = max(1, int(len(gamma_v) * 0.7))
    sill_est = float(np.mean(gamma_v[upper_start:]))
    partial_sill = max(sill_est - nugget_est, 1e-12)

    # Range estimate: lag where gamma reaches 95% of sill
    target = nugget_est + 0.95 * partial_sill
    range_est = 0.0
    for i, g in enumerate(gamma_v):
        if g >= target:
            # Interpolate between this lag and previous
            if i > 0 and gamma_v[i] > gamma_v[i - 1]:
                frac = (target - gamma_v[i - 1]) / (gamma_v[i] - gamma_v[i - 1])
                range_est = float(lags_v[i - 1] + frac * (lags_v[i] - lags_v[i - 1]))
            else:
                range_est = float(lags_v[i])
            break

    if range_est == 0.0:
        # Didn't reach sill: use 70% of max lag as estimate
        range_est = float(lags_v[-1] * 0.7)

    # Sanity checks
    median_spacing = float(np.median(np.diff(lags_v))) if len(lags_v) > 1 else lags_v[0]
    if range_est < median_spacing:
        range_est = median_spacing * 2.0

    # Reliability: enough pairs in first half of lags
    first_half = valid[:len(valid) // 2 + 1] if len(valid) > 2 else valid
    reliable = bool(first_half.sum() >= 3 and np.mean(pairs[valid]) >= 30)

    return (
        round(range_est, 2),
        round(partial_sill, 6),
        round(nugget_est, 6),
        reliable,
    )


# ======================================================================
# 5. Stationarity Checks
# ======================================================================

def _analyse_stationarity(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
) -> None:
    """Trend detection, proportional effect, moving-window stats."""
    N = len(values)

    # Linear trend detection in each axis
    trend_detected = False
    trend_direction = ""
    trend_r2 = 0.0
    trend_slope = 0.0

    for axis, name in enumerate(["X", "Y", "Z"]):
        x = coords[:, axis]
        x_centered = x - np.mean(x)
        if np.std(x_centered) < 1e-12:
            continue
        # Simple linear regression: grade = a + b*coord
        b = float(np.sum(x_centered * (values - np.mean(values)))) / \
            float(np.sum(x_centered ** 2))
        pred = np.mean(values) + b * x_centered
        ss_res = float(np.sum((values - pred) ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        if r2 > 0.05 and r2 > trend_r2:
            trend_detected = True
            trend_r2 = r2
            trend_direction = name
            trend_slope = b

    # Proportional effect: correlation between local mean and local
    # variance in moving windows.  Strong positive correlation means
    # variance increases with mean — normal-score transform needed.
    proportional_effect = False
    mean_var_corr = 0.0

    if N >= 100:
        # Divide into spatial windows
        n_windows = min(20, N // 10)
        # K-means-like: assign to spatial bins
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(42)
        centres = coords[rng.choice(N, n_windows, replace=False)]
        tree = cKDTree(centres)
        _, assignments = tree.query(coords)

        window_means = []
        window_stds = []
        for w in range(n_windows):
            mask = assignments == w
            if mask.sum() >= 5:
                window_means.append(float(np.mean(values[mask])))
                window_stds.append(float(np.std(values[mask])))

        if len(window_means) >= 5:
            wm = np.array(window_means)
            ws = np.array(window_stds)
            if np.std(wm) > 0 and np.std(ws) > 0:
                mean_var_corr = float(np.corrcoef(wm, ws)[0, 1])
                if mean_var_corr > 0.5:
                    proportional_effect = True

            wm_range = f"{np.min(wm):.4g} - {np.max(wm):.4g}"
            ws_range = f"{np.min(ws):.4g} - {np.max(ws):.4g}"
        else:
            wm_range = "insufficient windows"
            ws_range = "insufficient windows"
    else:
        wm_range = "N too small"
        ws_range = "N too small"

    rec.stationarity = {
        "trend_detected": trend_detected,
        "trend_direction": trend_direction,
        "trend_r2": trend_r2,
        "trend_slope": trend_slope,
        "proportional_effect": proportional_effect,
        "mean_var_correlation": mean_var_corr,
        "window_mean_range": wm_range,
        "window_std_range": ws_range,
    }

    if trend_detected and trend_r2 > 0.1:
        rec.warnings.append(
            f"Linear trend detected in {trend_direction} (R2={trend_r2:.3f}). "
            f"Consider using linear drift or de-trending before estimation."
        )
    if proportional_effect:
        rec.warnings.append(
            f"Proportional effect detected (mean-variance "
            f"correlation={mean_var_corr:.2f}). Normal-score transform "
            f"is strongly recommended."
        )


# ======================================================================
# 6. Contact / Domain Analysis
# ======================================================================

def _analyse_contacts(
    rec: ARBFRecommendation,
    values: np.ndarray,
) -> None:
    """Multimodality, normality, lognormality tests."""
    N = len(values)

    # Shapiro-Wilk test (on subsample if N > 5000)
    shapiro_p = 0.0
    normality_test = "not run"
    try:
        from scipy import stats
        test_vals = values
        if N > 5000:
            rng = np.random.default_rng(42)
            test_vals = values[rng.choice(N, 5000, replace=False)]
        _, shapiro_p = stats.shapiro(test_vals)
        if shapiro_p > 0.05:
            normality_test = f"NORMAL (p={shapiro_p:.4f})"
        else:
            normality_test = f"NOT NORMAL (p={shapiro_p:.4g})"
    except Exception:
        normality_test = "test failed"

    # Log-normality test
    lognormal_test = "not applicable"
    pos = values[values > 0]
    if len(pos) > 0.9 * N:
        try:
            log_vals = np.log(pos)
            test_log = log_vals
            if len(test_log) > 5000:
                rng = np.random.default_rng(42)
                test_log = log_vals[rng.choice(len(log_vals), 5000, replace=False)]
            _, log_p = stats.shapiro(test_log)
            if log_p > 0.05:
                lognormal_test = f"LOGNORMAL (p={log_p:.4f})"
            else:
                lognormal_test = f"NOT LOGNORMAL (p={log_p:.4g})"
        except Exception:
            lognormal_test = "test failed"

    # Multimodality
    n_modes = _detect_populations(values)
    multimodal = n_modes > 1

    rec.contact = {
        "normality_test": normality_test,
        "shapiro_p": shapiro_p,
        "lognormal_test": lognormal_test,
        "multimodal": multimodal,
        "n_modes": n_modes,
    }

    if multimodal:
        rec.warnings.append(
            f"{n_modes} grade populations detected. Check for "
            f"geological domain boundaries — domain-constrained "
            f"estimation may be needed."
        )


# ======================================================================
# 7. Compositing Assessment
# ======================================================================

def _analyse_compositing(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
) -> None:
    """Sample length regularity, downhole variogram, optimal composite."""
    N = len(coords)

    # Detect vertical sample intervals (downhole spacing)
    # Group samples by similar XY position (same drillhole)
    from scipy.spatial import cKDTree

    xy = coords[:, :2]
    tree_xy = cKDTree(xy)
    # Cluster XY positions: samples within 2m XY are same hole
    groups = tree_xy.query_ball_tree(tree_xy, r=2.0)

    # Find unique holes
    visited = set()
    hole_intervals = []
    hole_ids = []

    for i, group in enumerate(groups):
        if i in visited:
            continue
        hole = sorted(set(group))
        visited.update(hole)
        if len(hole) < 3:
            continue

        z_sorted = np.sort(coords[hole, 2])
        dz = np.diff(z_sorted)
        dz = dz[dz > 0.1]  # filter out duplicates
        hole_intervals.extend(dz.tolist())
        hole_ids.append(hole)

    if len(hole_intervals) > 5:
        intervals = np.array(hole_intervals)
        median_length = float(np.median(intervals))
        length_cv = float(np.std(intervals) / max(np.mean(intervals), 1e-12))
        is_regular = length_cv < 0.15

        length_summary = (
            f"median={median_length:.2f}m, "
            f"mean={np.mean(intervals):.2f}m, "
            f"std={np.std(intervals):.2f}m"
        )

        # Downhole variogram for nugget estimation
        downhole_nugget_ratio = None
        if len(hole_ids) >= 5:
            # Compute downhole semi-variogram at lag=1 composite
            gamma_1 = []
            for hole in hole_ids:
                if len(hole) < 3:
                    continue
                z_order = np.argsort(coords[hole, 2])
                sorted_hole = [hole[j] for j in z_order]
                for k in range(len(sorted_hole) - 1):
                    dv = 0.5 * (values[sorted_hole[k]] - values[sorted_hole[k + 1]]) ** 2
                    gamma_1.append(dv)
            if gamma_1:
                g1 = float(np.mean(gamma_1))
                data_var = float(np.var(values))
                if data_var > 0:
                    downhole_nugget_ratio = min(g1 / data_var, 1.0)

        # Recommendation
        recommendation = "composites appear regular"
        if not is_regular:
            recommendation = (
                f"irregular sample lengths (CV={length_cv:.2f}). "
                f"Re-composite to {median_length:.1f}m before estimation."
            )
        elif median_length > 10:
            recommendation = (
                f"long composites ({median_length:.1f}m). "
                f"Consider shorter compositing for better resolution."
            )
    else:
        median_length = 0.0
        length_cv = 0.0
        is_regular = True
        length_summary = "unable to detect (no clear drillhole structure)"
        downhole_nugget_ratio = None
        recommendation = "unable to assess compositing"

    rec.compositing = {
        "median_length": median_length,
        "length_cv": length_cv,
        "is_regular": is_regular,
        "length_summary": length_summary,
        "downhole_nugget_ratio": downhole_nugget_ratio,
        "recommendation": recommendation,
        "n_holes_detected": len(hole_ids),
    }


# ======================================================================
# 8. Block Model Coverage
# ======================================================================

def _analyse_block_coverage(
    rec: ARBFRecommendation,
    coords: np.ndarray,
    values: np.ndarray,
    block_centroids: Optional[np.ndarray],
    block_sizes: Optional[np.ndarray],
) -> None:
    """Block model vs drill footprint overlap and extrapolation risk."""
    if block_centroids is None or len(block_centroids) == 0:
        rec.block_coverage = {
            "extent_str": "no block model loaded",
            "n_blocks": 0,
        }
        return

    from scipy.spatial import cKDTree

    bc = np.asarray(block_centroids, dtype=np.float64)
    n_blocks = len(bc)
    b_min = bc.min(axis=0)
    b_max = bc.max(axis=0)
    b_extent = b_max - b_min
    extent_str = f"{b_extent[0]:.0f} x {b_extent[1]:.0f} x {b_extent[2]:.0f} m"

    d_extent = coords.max(axis=0) - coords.min(axis=0)
    drill_extent_str = f"{d_extent[0]:.0f} x {d_extent[1]:.0f} x {d_extent[2]:.0f} m"

    b_vol = max(np.prod(b_extent), 1.0)
    d_vol = max(np.prod(d_extent), 1.0)
    volume_ratio = b_vol / d_vol

    # Distance from each block to nearest sample
    tree = cKDTree(coords)
    # Subsample blocks if too many
    if n_blocks > 50000:
        rng = np.random.default_rng(42)
        sample_bc = bc[rng.choice(n_blocks, 50000, replace=False)]
    else:
        sample_bc = bc
    dists, _ = tree.query(sample_bc, k=1)

    # Use variography range if available
    ani = rec.variography.get("anisotropy", {})
    est_range = ani.get("range_max", rec.spatial.get("median_spacing", 50) * 3)

    pct_within_1 = 100.0 * float(np.mean(dists <= est_range))
    pct_within_2 = 100.0 * float(np.mean(dists <= 2 * est_range))

    if pct_within_1 > 90:
        extrap_risk = "LOW (>90% blocks within 1 range)"
    elif pct_within_1 > 60:
        extrap_risk = "MODERATE (60-90% blocks within 1 range)"
    elif pct_within_1 > 30:
        extrap_risk = "HIGH (30-60% blocks within 1 range)"
    else:
        extrap_risk = "CRITICAL (<30% blocks within 1 range)"

    clip_recommended = volume_ratio > 2.0 or pct_within_1 < 60

    rec.block_coverage = {
        "extent_str": extent_str,
        "n_blocks": n_blocks,
        "drill_extent_str": drill_extent_str,
        "volume_ratio": volume_ratio,
        "est_range_used": est_range,
        "pct_within_1_range": pct_within_1,
        "pct_within_2_range": pct_within_2,
        "extrapolation_risk": extrap_risk,
        "clip_recommended": clip_recommended,
        "median_block_to_sample": float(np.median(dists)),
        "p95_block_to_sample": float(np.percentile(dists, 95)),
    }

    if clip_recommended:
        rec.warnings.append(
            f"Block model is {volume_ratio:.1f}x the drilled volume. "
            f"Only {pct_within_1:.0f}% of blocks are within 1 variogram "
            f"range. Clip to drillhole footprint recommended."
        )


# ======================================================================
# 9. Build Final Recommendations
# ======================================================================

def _build_recommendations(
    rec: ARBFRecommendation,
    block_sizes: Optional[np.ndarray] = None,
) -> None:
    """Synthesise all analyses into final settings + reasons."""
    settings = {}
    reasons = rec.reasons  # append to existing

    u = rec.univariate
    sp = rec.spatial
    dc = rec.declustering
    vg = rec.variography
    st = rec.stationarity
    ct = rec.contact
    cp = rec.compositing
    bc = rec.block_coverage
    ani = vg.get("anisotropy", {})

    cv = u.get("cv", 0)
    skewness = u.get("skewness", 0)
    cv_is_meaningful = bool(u.get("cv_is_meaningful", False))
    appears_normal_scored = bool(u.get("appears_normal_scored", False))
    positive_grade_like = bool(u.get("positive_grade_like", False))
    centered_signed = bool(u.get("centered_signed", False))
    negative_fraction = float(u.get("negative_fraction", 0.0))

    # --- Normal-score transform ---
    ns_reasons = []
    use_ns = False

    if appears_normal_scored:
        use_ns = False
        reasons.append("Normal-score: OFF (values already look normal-scored / Gaussian)")
    else:
        if cv_is_meaningful and cv > 1.0:
            use_ns = True
            ns_reasons.append(f"CV={cv:.2f} (>1.0)")
        if abs(skewness) > 1.5:
            use_ns = True
            ns_reasons.append(f"skewness={skewness:.1f}")
        if st.get("proportional_effect"):
            use_ns = True
            ns_reasons.append("proportional effect detected")
        if u.get("distribution_type", "").startswith("lognormal"):
            use_ns = True
            ns_reasons.append("lognormal distribution")
        if cv_is_meaningful and cv > 0.5 and not use_ns:
            use_ns = True
            ns_reasons.append(f"moderate CV={cv:.2f} (>0.5)")

        settings["use_normal_score"] = use_ns
        if use_ns:
            reasons.append(f"Normal-score: ON ({', '.join(ns_reasons)})")
        else:
            if not cv_is_meaningful and centered_signed:
                reasons.append(
                    "Normal-score: OFF (signed, centered variable; CV is not meaningful and distribution is already near-Gaussian)"
                )
            elif not positive_grade_like:
                reasons.append(
                    "Normal-score: OFF (variable does not behave like a skewed positive grade distribution)"
                )
            else:
                reasons.append(f"Normal-score: OFF (CV={cv:.2f}, symmetric distribution)")

    if appears_normal_scored:
        settings["use_normal_score"] = False
    elif "use_normal_score" not in settings:
        settings["use_normal_score"] = False

    # --- Kernel type ---
    settings["kernel_type"] = "spherical"
    reasons.append(
        "Kernel: spherical (finite range — standard for mining deposits)"
    )

    # --- Variogram ranges (from directional analysis) ---
    if ani:
        range_max = ani.get("range_max", 100)
        range_mid = ani.get("range_mid", range_max)
        range_min = ani.get("range_min", range_max * 0.5)
        global_nugget = ani.get("nugget", 0)
        global_sill = ani.get("sill", 1.0)
        nugget_frac = ani.get("nugget_fraction", 0)
        major_az = ani.get("major_azimuth", 0)
        major_dip = ani.get("major_dip", 0)

        settings["range_max"] = round(range_max, 1)
        settings["range_mid"] = round(range_mid, 1)
        settings["range_min"] = round(range_min, 1)
        settings["azimuth"] = major_az
        settings["dip"] = major_dip
        settings["pitch"] = 0.0

        reasons.append(
            f"Ranges: major={range_max:.0f}m, semi={range_mid:.0f}m, "
            f"minor={range_min:.0f}m (from directional variography)"
        )
        if ani.get("ratio_max_min", 1) > 1.5:
            reasons.append(
                f"Anisotropy: {ani['ratio_max_min']:.1f}:1 ratio, "
                f"major direction Az={major_az:.0f}"
            )

        # Nugget / sill
        if use_ns:
            # Report in NS-space
            settings["nugget"] = round(nugget_frac, 4)
            settings["sill"] = round(1.0 - nugget_frac, 4)
            reasons.append(
                f"NS variogram: C0={nugget_frac:.3f}, C1={1-nugget_frac:.3f} "
                f"(nugget ratio={nugget_frac:.0%} from omni variogram)"
            )
        else:
            settings["nugget"] = round(global_nugget, 4)
            settings["sill"] = round(global_sill, 4)
            reasons.append(
                f"Variogram: C0={global_nugget:.4f}, C1={global_sill:.4f} "
                f"(nugget fraction={nugget_frac:.0%})"
            )

        if nugget_frac > 0.5:
            rec.warnings.append(
                f"High nugget fraction ({nugget_frac:.0%}). Consider "
                f"longer compositing or check assay/sampling quality."
            )
    else:
        # Fallback: use 3x median spacing
        fallback = round(3.0 * sp.get("median_spacing", 30), 1)
        settings["range_max"] = fallback
        settings["range_mid"] = fallback
        settings["range_min"] = round(fallback * 0.5, 1)
        settings["nugget"] = 0.1
        settings["sill"] = 0.9
        reasons.append(
            f"Ranges: {fallback:.0f}m (fallback = 3x median spacing). "
            f"Auto-variogram inconclusive — fit properly in Variogram Panel."
        )
        rec.warnings.append(
            "Variogram auto-fit did not converge. Import a fitted "
            "variogram from the Variogram Panel before running estimation."
        )

    # --- Alpha ---
    if settings["kernel_type"] == "spheroidal":
        settings["alpha"] = 1.5
    else:
        settings["alpha"] = 1.0

    # --- Drift ---
    if st.get("trend_detected") and st.get("trend_r2", 0) > 0.1:
        settings["drift_type"] = "linear"
        reasons.append(
            f"Drift: linear (trend in {st['trend_direction']}, "
            f"R2={st['trend_r2']:.3f})"
        )
    else:
        settings["drift_type"] = "constant"
        reasons.append("Drift: constant (no significant trend)")

    # --- Accuracy ---
    settings["accuracy"] = 1e-6

    # --- Search neighbourhood ---
    n_samples = u.get("n", 100)
    median_spacing = sp.get("median_spacing", 30)
    range_max = settings.get("range_max", 100)
    clustering_coeff = sp.get("clustering_coeff", 0.5)

    # Max / min samples: scale with data density relative to range
    # More data within range → can afford more neighbours for stability
    samples_per_range_vol = n_samples * (range_max ** 3) / max(
        sp.get("data_volume", range_max ** 3), 1e-6
    ) if sp.get("data_volume") else n_samples // 5
    max_samples = int(np.clip(samples_per_range_vol, 24, 96))
    tighten_local = bool(
        st.get("trend_detected")
        or st.get("proportional_effect")
        or sp.get("preferential_sampling")
        or cp.get("is_regular") is False
        or bc.get("clip_recommended", False)
    )
    if tighten_local:
        max_samples = min(max_samples, 32)
    settings["max_samples"] = max_samples
    # Min samples: need enough for stable kriging, more if clustered
    min_samples = 6 if tighten_local or clustering_coeff >= 0.6 else 4
    settings["min_samples"] = min_samples
    reasons.append(
        f"Max samples: {max_samples} (data density vs range), "
        f"min samples: {min_samples}"
    )

    # Search radii (in anisotropic search-space, relative to range)
    # Pass 1: well-informed neighbourhood (< 1 range)
    # Pass 2: moderate extrapolation (1-2 ranges)
    # Pass 3: far extrapolation (2-3 ranges)
    # Scale based on how well data covers the model
    pct_1_range = bc.get("pct_within_1_range", 80) if bc else 80
    if pct_1_range > 70:
        # Good coverage — standard multi-pass radii
        sr1, sr2, sr3 = 0.75, 1.50, 3.00
    elif pct_1_range > 40:
        # Moderate coverage — extend second pass
        sr1, sr2, sr3 = 1.00, 2.00, 4.00
    else:
        # Sparse coverage — wider searches needed
        sr1, sr2, sr3 = 1.50, 3.00, 5.00
    settings["search_radius_1"] = sr1
    settings["search_radius_2"] = sr2
    settings["search_radius_3"] = sr3
    if tighten_local:
        settings["search_radius_1"] = min(settings["search_radius_1"], 0.75)
        settings["search_radius_2"] = min(settings["search_radius_2"], 1.50)
        settings["search_radius_3"] = min(settings["search_radius_3"], 3.00)
        sr1 = settings["search_radius_1"]
        sr2 = settings["search_radius_2"]
        sr3 = settings["search_radius_3"]
    reasons.append(
        f"Search radii: [{sr1:.2f}, {sr2:.2f}, {sr3:.2f}] "
        f"(coverage={pct_1_range:.0f}% blocks within 1 range)"
    )
    if tighten_local:
        reasons.append(
            "Neighbourhood kept tight to reduce smoothing under trend, preferential sampling, "
            "irregular composites, or unsupported block coverage."
        )

    # Octant settings — enforce spatial balance when data is clustered
    balanced_octant = clustering_coeff > 0.4 or tighten_local
    settings["balanced_octant"] = balanced_octant
    if balanced_octant:
        # Clustered data: stricter octant requirements
        if clustering_coeff > 0.7:
            min_oct, min_oct_lin, max_per_oct = 4, 5, 3
        else:
            min_oct, min_oct_lin, max_per_oct = 3, 4, 4
    else:
        # Regular spacing: relax octant constraints
        min_oct, min_oct_lin, max_per_oct = 2, 3, 6
    settings["min_octants"] = min_oct
    settings["min_octants_linear"] = min_oct_lin
    settings["max_per_octant"] = max_per_oct
    reasons.append(
        f"Octants: min={min_oct}, min(linear)={min_oct_lin}, "
        f"max/octant={max_per_oct} "
        f"(clustering={clustering_coeff:.2f}, "
        f"balanced={'ON' if balanced_octant else 'OFF'})"
    )

    # Auto drift slope tolerance — tighter if trend detected
    if st.get("trend_detected") and st.get("trend_r2", 0) > 0.15:
        drift_tol = 0.10
        reasons.append(
            f"Drift slope tolerance: {drift_tol:.2f} (tight — "
            f"significant trend R2={st.get('trend_r2', 0):.3f})"
        )
    elif st.get("trend_detected"):
        drift_tol = 0.15
        reasons.append(
            f"Drift slope tolerance: {drift_tol:.2f} (moderate — "
            f"weak trend detected)"
        )
    else:
        drift_tol = 0.25
        reasons.append(
            f"Drift slope tolerance: {drift_tol:.2f} (relaxed — no trend)"
        )
    settings["auto_drift_slope_tol"] = drift_tol

    # --- Discretisation ---
    # Adaptive when blocks are large relative to the range (block support
    # correction matters more for large blocks)
    block_to_range = 1.0
    if block_sizes is not None:
        try:
            bs = np.asarray(block_sizes, dtype=float).ravel()
            max_block_dim = float(np.max(bs[:3]))
            block_to_range = max_block_dim / max(range_max, 1.0)
        except Exception:
            pass

    if block_to_range > 0.3:
        disc_mode = "Adaptive"
        disc_density = 64  # 4x4x4 for large blocks
        reasons.append(
            f"Discretisation: Adaptive 4x4x4 (block/range ratio="
            f"{block_to_range:.2f} — block support correction important)"
        )
    elif block_to_range > 0.15:
        disc_mode = "Fixed"
        disc_density = 27  # 3x3x3 standard
        reasons.append("Discretisation: Fixed 3x3x3 (standard)")
    else:
        disc_mode = "Fixed"
        disc_density = 8  # 2x2x2 for small blocks relative to range
        reasons.append(
            f"Discretisation: Fixed 2x2x2 (small blocks relative to range, "
            f"ratio={block_to_range:.2f})"
        )
    settings["change_of_support"] = True
    settings["discretisation_mode"] = disc_mode
    settings["discretisation_density"] = disc_density

    # --- Cross-validation ---
    settings["run_cv"] = True
    reasons.append("Cross-validation: ON")

    # --- Footprint clipping ---
    clip = bool(bc.get("clip_recommended", False))
    settings["clip_to_drill_footprint"] = clip
    if clip:
        settings["footprint_buffer_ranges"] = 1.5
        reasons.append(
            f"Footprint clip: ON (volume ratio={bc.get('volume_ratio',0):.1f}x, "
            f"only {bc.get('pct_within_1_range',0):.0f}% blocks within 1 range)"
        )
    else:
        settings["clip_to_drill_footprint"] = False
        reasons.append("Footprint clip: OFF (adequate coverage)")

    # --- Grade clipping ---
    oc = u.get("outlier_analysis", {})
    if (
        oc.get("applicable", True)
        and positive_grade_like
        and oc.get("top_cut") is not None
        and oc.get("n_above", 0) > 0
        and oc.get("metal_at_risk_pct", 0) > 20
    ):
        settings["clip_max"] = oc["top_cut"]
        reasons.append(
            f"Top-cut: {oc['top_cut']:.4g} ({oc['method']}, "
            f"{oc['metal_at_risk_pct']:.1f}% metal in "
            f"{oc['pct_samples_above']:.1f}% samples)"
        )
    elif oc.get("reason"):
        reasons.append(f"Top-cut: OFF ({oc['reason']})")

    if positive_grade_like and u.get("min", 0) < 0 and negative_fraction <= 0.05:
        settings["clip_min"] = 0.0
        reasons.append("Bottom clip: 0 (minor negative tail on an otherwise non-negative variable)")

    # --- Variogram mode ---
    settings["variogram_mode"] = "global"

    # --- Compositing warning ---
    if cp.get("is_regular") is False:
        rec.warnings.append(
            f"Irregular composites detected (length CV={cp.get('length_cv',0):.2f}). "
            f"Re-composite to {cp.get('median_length',5):.1f}m before estimation."
        )

    # --- Domain warning ---
    if ct.get("multimodal"):
        rec.warnings.append(
            f"{ct.get('n_modes',0)} grade populations detected. "
            f"Select a domain column for hard-boundary estimation."
        )

    rec.settings = settings
    rec.data_summary = {
        "n_samples": u.get("n"),
        "mean": u.get("mean"),
        "std": u.get("std"),
        "cv": cv,
        "skewness": skewness,
        "median_spacing": sp.get("median_spacing"),
    }
