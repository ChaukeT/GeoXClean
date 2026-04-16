"""
Property name formatting utilities.

Converts raw property names (e.g. ``OK_Cu_estimate``) to human-readable
labels (e.g. ``Cu Estimate (Ordinary Kriging)``) for display in the
property panel, legend, and tooltips.
"""

from __future__ import annotations

import re
from typing import Optional

# Method code → short display name
_METHOD_SHORT = {
    "OK": "OK",
    "SK": "SK",
    "UK": "UK",
    "COK": "CoK",
    "IK": "IK",
    "BAYK": "Bayesian",
    "ARBF": "ARBF",
    "FASTRBF": "FastRBF",
    "SGSIM": "SGSIM",
    "SIS": "SIS",
    "IKSGSIM": "IK-SGSIM",
    "TB": "Turning Bands",
    "DBS": "DBS",
    "GRF": "GRF",
    "COSGSIM": "Co-SGSIM",
    "MPS": "MPS",
    "DIFF": "Difference",
}

# Method code → full display name (for legend titles)
_METHOD_FULL = {
    "OK": "Ordinary Kriging",
    "SK": "Simple Kriging",
    "UK": "Universal Kriging",
    "COK": "Co-Kriging",
    "IK": "Indicator Kriging",
    "BAYK": "Bayesian Kriging",
    "ARBF": "Adaptive RBF",
    "FASTRBF": "FastRBF Interpolation",
    "SGSIM": "Sequential Gaussian Simulation",
    "SIS": "Sequential Indicator Simulation",
    "IKSGSIM": "IK-based SGSIM",
    "TB": "Turning Bands Simulation",
    "DBS": "Direct Block Simulation",
    "GRF": "Gaussian Random Field",
    "COSGSIM": "Co-Simulation",
    "MPS": "Multiple-Point Simulation",
    "DIFF": "Difference Map",
}

# Property suffix → readable label
_SUFFIX_LABELS = {
    "estimate": "Estimate",
    "variance": "Variance",
    "mean": "Mean",
    "std": "Std Dev",
    "p10": "P10",
    "p25": "P25",
    "p50": "P50 (Median)",
    "p75": "P75",
    "p90": "P90",
    "kriging_efficiency": "Kriging Efficiency",
    "slope_of_regression": "Slope of Regression",
    "negative_weight_pct": "Negative Weights %",
    "pct_negative_weights": "Negative Weights %",
    "n_samples": "Sample Count",
    "pass_number": "Search Pass",
    "distance_to_nearest": "Distance to Nearest",
    "classification": "Classification",
    "neff": "Effective Samples",
    "fail_flag": "Fail Flag",
    "stitching_var": "Stitching Variance",
    "blending_var": "Blending Variance",
}

# Pattern: {METHOD}_{variable}_{suffix}
_PATTERN = re.compile(
    r"^(" + "|".join(sorted(_METHOD_SHORT.keys(), key=len, reverse=True)) + r")_(.+)$"
)

# Realisation pattern: {METHOD}_{var}_real_{NNNN}
_REAL_PATTERN = re.compile(r"^(.+)_real_(\d+)$")

# Probability pattern: prob_gt_{cutoff}
_PROB_PATTERN = re.compile(r"^(.+)_prob_gt_(.+)$")


def parse_property_name(raw: str) -> dict:
    """Parse a standardised property name into components.

    Returns dict with keys: method, variable, suffix, readable_suffix.
    Returns empty dict if the name doesn't match the standard pattern.
    """
    m = _PATTERN.match(raw)
    if not m:
        return {"method": "", "variable": "", "suffix": raw, "raw": raw}

    method = m.group(1)
    rest = m.group(2)

    # Check for realisation
    rm = _REAL_PATTERN.match(rest)
    if rm:
        return {
            "method": method,
            "variable": rm.group(1),
            "suffix": f"real_{rm.group(2)}",
            "readable_suffix": f"Realisation {int(rm.group(2))}",
            "raw": raw,
        }

    # Check for probability
    pm = _PROB_PATTERN.match(rest)
    if pm:
        return {
            "method": method,
            "variable": pm.group(1),
            "suffix": f"prob_gt_{pm.group(2)}",
            "readable_suffix": f"P(>{pm.group(2)})",
            "raw": raw,
        }

    # Split variable from suffix — try longest known suffix first
    for suffix, label in sorted(_SUFFIX_LABELS.items(), key=lambda x: len(x[0]), reverse=True):
        if rest.endswith(f"_{suffix}"):
            variable = rest[: -(len(suffix) + 1)]
            return {
                "method": method,
                "variable": variable,
                "suffix": suffix,
                "readable_suffix": label,
                "raw": raw,
            }

    # No known suffix — treat entire rest as variable + implicit estimate
    return {
        "method": method,
        "variable": rest,
        "suffix": "",
        "readable_suffix": "",
        "raw": raw,
    }


def format_property_label(raw: str, unit: str = "") -> str:
    """Convert a raw property name to a human-readable label.

    Examples:
        OK_Cu_estimate        → "Cu Estimate (OK)"
        ARBF_Zn_variance      → "Zn Variance (ARBF)"
        SGSIM_Au_p90          → "Au P90 (SGSIM)"
        SGSIM_Au_real_0003    → "Au Realisation 3 (SGSIM)"
        OK_Cu_kriging_efficiency → "Cu Kriging Efficiency (OK)"
        DOMAIN                → "Domain"
    """
    if raw == "DOMAIN":
        return "Domain"
    if raw == "ARBF_FAIL_FLAG":
        return "Fail Flag (ARBF)"

    parts = parse_property_name(raw)
    method = parts.get("method", "")
    variable = parts.get("variable", "")
    readable = parts.get("readable_suffix", "")

    if not method:
        return raw  # Not a standardised name — return as-is

    method_short = _METHOD_SHORT.get(method, method)

    if readable:
        label = f"{variable} {readable} ({method_short})"
    else:
        label = f"{variable} ({method_short})"

    if unit:
        label += f" [{unit}]"

    return label


def format_legend_title(raw: str, unit: str = "") -> str:
    """Format a property name for the colour bar title.

    Uses the FULL method name for clarity.

    Examples:
        OK_Cu_estimate → "Ordinary Kriging — Cu (%)"
        SGSIM_Au_mean  → "SGSIM — Au Mean (g/t)"
    """
    if raw == "DOMAIN":
        return "Domain"

    parts = parse_property_name(raw)
    method = parts.get("method", "")
    variable = parts.get("variable", "")
    readable = parts.get("readable_suffix", "")

    if not method:
        return raw

    method_full = _METHOD_FULL.get(method, method)

    if readable:
        title = f"{method_full} — {variable} {readable}"
    else:
        title = f"{method_full} — {variable}"

    if unit:
        title += f" ({unit})"

    return title


# ═══════════════════════════════════════════════════════════════════
# Property type classification and default colormaps
# ═══════════════════════════════════════════════════════════════════

def classify_property_type(raw: str) -> str:
    """Classify a property name into a type for auto-colormap selection.

    Returns one of: "estimate", "variance", "qa_metric", "probability",
    "classification", "realisation", "unknown".
    """
    low = raw.lower()

    if low == "domain" or "classification" in low or "fail_flag" in low:
        return "classification"
    if "_prob_" in low or "_probability" in low:
        return "probability"
    if "_real_" in low:
        return "realisation"
    if "_variance" in low or "_var" in low or "_std" in low:
        return "variance"
    if any(k in low for k in (
        "kriging_efficiency", "slope_of_regression",
        "negative_weight", "n_samples", "pass_number", "neff",
        "distance_to_nearest",
    )):
        return "qa_metric"
    if "_estimate" in low or "_mean" in low or "_median" in low or "_p50" in low:
        return "estimate"
    if "_p10" in low or "_p25" in low or "_p75" in low or "_p90" in low:
        return "probability"
    return "unknown"


COLORMAP_DEFAULTS = {
    "estimate": "turbo",
    "variance": "viridis",
    "qa_metric": "RdYlGn",
    "probability": "plasma",
    "classification": "tab20",
    "realisation": "turbo",
    "unknown": "turbo",
}


def get_default_colormap(raw: str) -> str:
    """Return the default colormap for a property based on its type."""
    ptype = classify_property_type(raw)
    return COLORMAP_DEFAULTS.get(ptype, "turbo")


def is_discrete_property(raw: str) -> bool:
    """Return True if this property should use discrete (categorical) mode."""
    return classify_property_type(raw) == "classification"


# ═══════════════════════════════════════════════════════════════════
# Canonical name builders — single source of truth for SGSIM naming
# ═══════════════════════════════════════════════════════════════════

def sgsim_property_name(variable: str, stat: str, method: str = "SGSIM") -> str:
    """Build canonical property name for a simulation summary statistic.

    Pattern: ``{METHOD}_{variable}_{stat}``

    Examples:
        sgsim_property_name("Au", "mean")            → "SGSIM_Au_mean"
        sgsim_property_name("Fe_Pct", "P90")          → "SGSIM_Fe_Pct_p90"
        sgsim_property_name("Cu", "mean", "IKSGSIM") → "IKSGSIM_Cu_mean"
        sgsim_property_name("Cu", "mean", "COSGSIM") → "COSGSIM_Cu_mean"
    """
    stat_lower = stat.lower()
    return f"{method}_{variable}_{stat_lower}"


def sgsim_realisation_name(variable: str, index: int, method: str = "SGSIM") -> str:
    """Build canonical property name for a simulation realisation.

    Pattern: ``{METHOD}_{variable}_real_{NNNN}``

    Examples:
        sgsim_realisation_name("Au", 3)  → "SGSIM_Au_real_0003"
        sgsim_realisation_name("Cu", 42, "IKSGSIM") → "IKSGSIM_Cu_real_0042"
    """
    return f"{method}_{variable}_real_{index:04d}"


def sgsim_probability_name(variable: str, cutoff: str, method: str = "SGSIM") -> str:
    """Build canonical property name for a probability threshold map.

    Pattern: ``{METHOD}_{variable}_prob_gt_{cutoff}``

    Examples:
        sgsim_probability_name("Au", "0.5")  → "SGSIM_Au_prob_gt_0.5"
    """
    return f"{method}_{variable}_prob_gt_{cutoff}"
