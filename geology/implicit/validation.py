"""
Validation — Contact Honouring Quality Control.
================================================

Checks that modelled surfaces pass through contact points
within a specified tolerance.  This is the primary QC metric
for implicit geological models.

Misfit is reported in **spatial metres** using the gradient-based
conversion:

    spatial_misfit = |f(x) - expected_isovalue| / |∇f(x)|

where |∇f(x)| is estimated by central finite differences with a 0.5 m step.
This removes the dependence on isovalue scale: a potential field that
spans 100 isovalue units over 500 m has |∇f| ≈ 0.2/m, so an isovalue
misfit of 2.0 converts to 2.0 / 0.2 = 10 m spatially.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Finite difference step (metres) for gradient estimation
_FD_STEP = 0.5


def _gradient_magnitude(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    coords: np.ndarray,
    step: float = _FD_STEP,
) -> np.ndarray:
    """Estimate |∇f| at each contact point via central finite differences.

    Parameters
    ----------
    evaluate_fn : (N, 3) -> (N,)
    coords      : (N, 3) contact coordinates in metres
    step        : finite difference step in metres

    Returns
    -------
    grad_mag : (N,) gradient magnitude, clipped to ≥ 1e-6
    """
    dx = np.array([step, 0.0, 0.0])
    dy = np.array([0.0, step, 0.0])
    dz = np.array([0.0, 0.0, step])

    gx = (evaluate_fn(coords + dx) - evaluate_fn(coords - dx)) / (2.0 * step)
    gy = (evaluate_fn(coords + dy) - evaluate_fn(coords - dy)) / (2.0 * step)
    gz = (evaluate_fn(coords + dz) - evaluate_fn(coords - dz)) / (2.0 * step)

    return np.sqrt(gx**2 + gy**2 + gz**2).clip(min=1e-6)


def check_contact_honouring(
    contacts: pd.DataFrame,
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    tolerance: float = 5.0,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    surface_name_col: str = "surface_name",
    hole_id_col: str = "hole_id",
    expected_value: float = 0.0,
) -> pd.DataFrame:
    """Check that modelled surfaces pass through contact points.

    For each contact point, evaluates the scalar field and converts
    the isovalue misfit to spatial metres via the gradient magnitude:

        spatial_misfit [m] = |f(x) - expected_value| / |∇f(x)|

    Parameters
    ----------
    contacts : pd.DataFrame
        Must have X, Y, Z columns.
    evaluate_fn : callable (B, 3) -> (B,)
        Scalar field evaluation function.
    tolerance : float
        Acceptable spatial misfit **in metres**.  Default 5 m.
    expected_value : float
        Expected scalar field value at contacts (isovalue for this surface).

    Returns
    -------
    pd.DataFrame
        Columns: surface_name, hole_id, X, Y, Z,
                 expected, actual, misfit_isovalue, misfit_m, honoured
    """
    if contacts.empty:
        return pd.DataFrame(columns=[
            "surface_name", "hole_id", "X", "Y", "Z",
            "expected", "actual", "misfit_isovalue", "misfit_m", "honoured",
        ])

    coords = contacts[[x_col, y_col, z_col]].values.astype(np.float64)
    actual = evaluate_fn(coords)

    misfit_iso = np.abs(actual - expected_value)

    # Gradient-based conversion: isovalue units → metres
    grad_mag = _gradient_magnitude(evaluate_fn, coords)
    misfit_m = misfit_iso / grad_mag

    honoured = misfit_m < tolerance

    result = pd.DataFrame({
        "surface_name": contacts[surface_name_col].values if surface_name_col in contacts.columns else "unknown",
        "hole_id": contacts[hole_id_col].values if hole_id_col in contacts.columns else "",
        "X": coords[:, 0],
        "Y": coords[:, 1],
        "Z": coords[:, 2],
        "expected": expected_value,
        "actual": actual,
        "misfit_isovalue": misfit_iso,
        "grad_mag": grad_mag,
        "misfit_m": misfit_m,
        "honoured": honoured,
    })
    # Keep "misfit" alias so callers using the old column name still work
    result["misfit"] = result["misfit_m"]

    n_total = len(result)
    n_honoured = int(honoured.sum())
    pct = 100.0 * n_honoured / max(n_total, 1)

    logger.info(
        "Contact honouring: %d/%d (%.1f%%) within %.1f m  "
        "(mean misfit %.2f m, |∇f| mean %.4f/m)",
        n_honoured, n_total, pct, tolerance,
        float(misfit_m.mean()), float(grad_mag.mean()),
    )

    if pct < 90.0:
        logger.warning(
            "Contact honouring below 90%%. Mean spatial misfit %.2f m. "
            "Consider adjusting kernel parameters or increasing accuracy.",
            float(misfit_m.mean()),
        )

    return result


def contact_honouring_summary(qc_df: pd.DataFrame) -> dict:
    """Summarise contact honouring QC results.

    Parameters
    ----------
    qc_df : DataFrame from check_contact_honouring()

    Returns
    -------
    dict with keys:
        n_contacts, n_honoured, pct_honoured,
        mean_misfit, max_misfit, median_misfit,
        per_surface: {name: {n, honoured, pct, mean_misfit}}
    """
    if qc_df.empty:
        return {
            "n_contacts": 0, "n_honoured": 0, "pct_honoured": 0.0,
            "mean_misfit": 0.0, "max_misfit": 0.0, "median_misfit": 0.0,
            "mean_misfit_isovalue": 0.0, "mean_grad_mag": 0.0,
            "per_surface": {},
        }

    # Prefer the spatial-metres column; fall back to raw misfit for old results
    misfit_col = "misfit_m" if "misfit_m" in qc_df.columns else "misfit"

    summary = {
        "n_contacts": len(qc_df),
        "n_honoured": int(qc_df["honoured"].sum()),
        "pct_honoured": float(100.0 * qc_df["honoured"].mean()),
        "mean_misfit": float(qc_df[misfit_col].mean()),
        "max_misfit": float(qc_df[misfit_col].max()),
        "median_misfit": float(qc_df[misfit_col].median()),
        # Diagnostic: raw isovalue misfit and gradient magnitude
        "mean_misfit_isovalue": float(qc_df["misfit_isovalue"].mean()) if "misfit_isovalue" in qc_df.columns else 0.0,
        "mean_grad_mag": float(qc_df["grad_mag"].mean()) if "grad_mag" in qc_df.columns else 0.0,
        "per_surface": {},
    }

    if "surface_name" in qc_df.columns:
        for name, group in qc_df.groupby("surface_name"):
            summary["per_surface"][str(name)] = {
                "n": len(group),
                "honoured": int(group["honoured"].sum()),
                "pct": float(100.0 * group["honoured"].mean()),
                "mean_misfit": float(group[misfit_col].mean()),
            }

    return summary


def check_interval_consistency(
    lithology_df: pd.DataFrame,
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    isovalues: List[float],
    unit_names: List[str],
    grouping: Optional[Dict[str, List[str]]] = None,
    sample_spacing: float = 2.0,
    hole_id_col: str = "hole_id",
    lith_col: str = "lith_code",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    depth_from_col: str = "depth_from",
    depth_to_col: str = "depth_to",
) -> pd.DataFrame:
    """Check that interpolated domains are consistent with logged lithology.

    Contacts only verify that surfaces pass near transition points.
    This function also validates the INTERIORS — the intervals between
    contacts — by sampling the field at each interval midpoint and
    comparing the predicted domain to the logged lithology.

    A surface can honour all contacts but still oscillate between holes,
    assigning wrong lithology to significant intervals.  Interval
    consistency catches those oscillations.

    Parameters
    ----------
    lithology_df : pd.DataFrame
        Drillhole intervals with hole_id, X, Y, Z, lith_code,
        depth_from, depth_to columns.
    evaluate_fn : callable (N, 3) -> (N,)
        Scalar field evaluator (same function passed to check_contact_honouring).
    isovalues : list of float
        Sorted isovalue boundaries separating domains.
    unit_names : list of str
        Domain names, len(unit_names) == len(isovalues) + 1.
    grouping : dict, optional
        Maps lithology codes to geological unit names, e.g.
        {"SAND": ["SND", "sand", "SAND_1"]}.
    sample_spacing : float
        Interval sampling step in metres.  Default 2 m.
    hole_id_col, lith_col, x_col, y_col, z_col : str
        Column name overrides.
    depth_from_col, depth_to_col : str
        Depth interval column names.

    Returns
    -------
    pd.DataFrame
        Columns: hole_id, depth_from, depth_to, logged_unit,
                 predicted_unit, field_value, match
        One row per interval.  ``match`` is True when
        predicted_unit == logged_unit (after applying grouping).
    """
    if lithology_df is None or lithology_df.empty:
        return pd.DataFrame(columns=[
            "hole_id", "depth_from", "depth_to",
            "logged_unit", "predicted_unit", "field_value", "match",
        ])

    col_map = {c.lower(): c for c in lithology_df.columns}
    hole_col  = col_map.get(hole_id_col.lower(), hole_id_col)
    lith_col_ = col_map.get(lith_col.lower(), lith_col)
    x_col_    = col_map.get(x_col.lower(), x_col)
    y_col_    = col_map.get(y_col.lower(), y_col)
    z_col_    = col_map.get(z_col.lower(), z_col)

    # Build reverse-lookup: lithology code → geological unit name
    def _map_lith(code: str) -> str:
        if grouping is None:
            return str(code)
        code_str = str(code)
        for unit, codes in grouping.items():
            if code_str in [str(c) for c in codes]:
                return unit
        return code_str

    sorted_iso = sorted(isovalues)

    def _predict_unit(field_val: float) -> str:
        for k, iso in enumerate(sorted_iso):
            if field_val < iso:
                return unit_names[k] if k < len(unit_names) else f"Domain_{k}"
        return unit_names[len(sorted_iso)] if len(sorted_iso) < len(unit_names) else f"Domain_{len(sorted_iso)}"

    rows = []
    required = [hole_col, lith_col_, x_col_, y_col_, z_col_]
    if not all(c in lithology_df.columns for c in required):
        missing = [c for c in required if c not in lithology_df.columns]
        logger.warning(
            "check_interval_consistency: missing columns %s — skipping", missing
        )
        return pd.DataFrame(columns=[
            "hole_id", "depth_from", "depth_to",
            "logged_unit", "predicted_unit", "field_value", "match",
        ])

    for _, row in lithology_df.iterrows():
        try:
            x = float(row[x_col_])
            y = float(row[y_col_])
            z = float(row[z_col_])
        except (ValueError, TypeError):
            continue

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue

        logged_raw = str(row[lith_col_])
        logged_unit = _map_lith(logged_raw)

        coords = np.array([[x, y, z]], dtype=np.float64)
        field_val = float(evaluate_fn(coords)[0])
        predicted_unit = _predict_unit(field_val)

        depth_from = float(row[depth_from_col]) if depth_from_col in row.index else float("nan")
        depth_to   = float(row[depth_to_col])   if depth_to_col   in row.index else float("nan")

        rows.append({
            "hole_id":        str(row[hole_col]),
            "depth_from":     depth_from,
            "depth_to":       depth_to,
            "logged_unit":    logged_unit,
            "predicted_unit": predicted_unit,
            "field_value":    field_val,
            "match":          logged_unit == predicted_unit,
        })

    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "hole_id", "depth_from", "depth_to",
        "logged_unit", "predicted_unit", "field_value", "match",
    ])

    if not result.empty:
        n_total = len(result)
        n_match = int(result["match"].sum())
        n_mismatch = n_total - n_match
        pct_mismatch = 100.0 * n_mismatch / max(n_total, 1)

        logger.info(
            "Interval consistency: %d/%d intervals match (%.1f%% mismatch)",
            n_match, n_total, pct_mismatch,
        )
        if pct_mismatch > 10.0:
            logger.warning(
                "More than 10%% of intervals have domain mismatches (%.1f%%). "
                "Review model surfaces in cross-section — surfaces may be "
                "oscillating between drillholes.",
                pct_mismatch,
            )

    return result


def interval_consistency_summary(consistency_df: pd.DataFrame) -> dict:
    """Summarise interval consistency check results.

    Parameters
    ----------
    consistency_df : DataFrame from check_interval_consistency()

    Returns
    -------
    dict with keys:
        n_intervals, n_match, n_mismatch, pct_mismatch,
        per_unit: {unit_name: {n, match, pct_mismatch}}
    """
    if consistency_df.empty:
        return {
            "n_intervals": 0, "n_match": 0, "n_mismatch": 0,
            "pct_mismatch": 0.0, "per_unit": {},
        }

    n_total   = len(consistency_df)
    n_match   = int(consistency_df["match"].sum())
    n_mismatch = n_total - n_match

    summary: dict = {
        "n_intervals":  n_total,
        "n_match":      n_match,
        "n_mismatch":   n_mismatch,
        "pct_mismatch": float(100.0 * n_mismatch / max(n_total, 1)),
        "per_unit":     {},
    }

    if "logged_unit" in consistency_df.columns:
        for unit, group in consistency_df.groupby("logged_unit"):
            g_n = len(group)
            g_match = int(group["match"].sum())
            summary["per_unit"][str(unit)] = {
                "n": g_n,
                "match": g_match,
                "pct_mismatch": float(100.0 * (g_n - g_match) / max(g_n, 1)),
            }

    return summary
