"""
Reconciliation Checks for Professional Kriging
==============================================

JORC/NI 43-101 require reconciliation between:
- OK mean vs. Declustered mean
- OK mean vs. SGSIM P50
- OK variance vs. SGSIM variance

This module provides professional reconciliation checks and reporting.

Author: GeoX Development Team
Date: 2026-02-07
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


def reconcile_ok_vs_declustered(
    ok_estimates: np.ndarray,
    declustered_mean: float,
    tolerance_pct: float = 5.0
) -> Dict[str, any]:
    """
    Reconcile OK mean against declustered mean.

    JORC/NI 43-101 Standard: OK mean should match declustered mean within ±5%.

    Parameters
    ----------
    ok_estimates : np.ndarray
        OK estimates (may contain NaN)
    declustered_mean : float
        Declustered mean from declustering analysis
    tolerance_pct : float
        Acceptable tolerance percentage (default 5.0%)

    Returns
    -------
    dict
        Reconciliation results with keys:
        - passed: bool
        - ok_mean: float
        - declustered_mean: float
        - difference: float (absolute)
        - difference_pct: float
        - within_tolerance: bool
        - message: str
    """
    # Filter valid estimates
    valid = ~np.isnan(ok_estimates)
    if not np.any(valid):
        return {
            'passed': False,
            'ok_mean': np.nan,
            'declustered_mean': declustered_mean,
            'difference': np.nan,
            'difference_pct': np.nan,
            'within_tolerance': False,
            'message': "No valid OK estimates"
        }

    ok_mean = float(np.mean(ok_estimates[valid]))
    difference = ok_mean - declustered_mean
    difference_pct = abs(difference) * 100.0 / abs(declustered_mean) if declustered_mean != 0 else np.inf

    within_tolerance = difference_pct <= tolerance_pct
    passed = within_tolerance

    if passed:
        message = f"✓ OK mean ({ok_mean:.3f}) reconciles with declustered mean ({declustered_mean:.3f}), diff={difference_pct:.2f}%"
    else:
        message = f"✗ OK mean ({ok_mean:.3f}) differs from declustered mean ({declustered_mean:.3f}) by {difference_pct:.2f}% (> {tolerance_pct}%)"

    return {
        'passed': passed,
        'ok_mean': ok_mean,
        'declustered_mean': declustered_mean,
        'difference': difference,
        'difference_pct': difference_pct,
        'within_tolerance': within_tolerance,
        'message': message
    }


def reconcile_ok_vs_sgsim(
    ok_estimates: np.ndarray,
    sgsim_realizations: np.ndarray,
    tolerance_pct: float = 5.0
) -> Dict[str, any]:
    """
    Reconcile OK mean against SGSIM P50 (median).

    JORC/NI 43-101 Standard: OK estimates should match SGSIM P50 within ±5%.

    Parameters
    ----------
    ok_estimates : np.ndarray
        OK estimates (M,) array
    sgsim_realizations : np.ndarray
        SGSIM realizations (M, N) array where N is number of realizations
    tolerance_pct : float
        Acceptable tolerance percentage (default 5.0%)

    Returns
    -------
    dict
        Reconciliation results
    """
    # Compute SGSIM P50 (median across realizations)
    if sgsim_realizations.ndim == 1:
        # Single realization - use as-is
        sgsim_p50 = sgsim_realizations
    else:
        # Multiple realizations - compute median
        sgsim_p50 = np.median(sgsim_realizations, axis=1)

    # Filter valid blocks (both OK and SGSIM)
    valid_ok = ~np.isnan(ok_estimates)
    valid_sgsim = ~np.isnan(sgsim_p50)
    valid = valid_ok & valid_sgsim

    if not np.any(valid):
        return {
            'passed': False,
            'ok_mean': np.nan,
            'sgsim_p50_mean': np.nan,
            'difference': np.nan,
            'difference_pct': np.nan,
            'within_tolerance': False,
            'message': "No valid blocks for comparison"
        }

    ok_mean = float(np.mean(ok_estimates[valid]))
    sgsim_p50_mean = float(np.mean(sgsim_p50[valid]))

    difference = ok_mean - sgsim_p50_mean
    difference_pct = abs(difference) * 100.0 / abs(sgsim_p50_mean) if sgsim_p50_mean != 0 else np.inf

    within_tolerance = difference_pct <= tolerance_pct
    passed = within_tolerance

    if passed:
        message = f"✓ OK mean ({ok_mean:.3f}) reconciles with SGSIM P50 ({sgsim_p50_mean:.3f}), diff={difference_pct:.2f}%"
    else:
        message = f"✗ OK mean ({ok_mean:.3f}) differs from SGSIM P50 ({sgsim_p50_mean:.3f}) by {difference_pct:.2f}% (> {tolerance_pct}%)"

    return {
        'passed': passed,
        'ok_mean': ok_mean,
        'sgsim_p50_mean': sgsim_p50_mean,
        'difference': difference,
        'difference_pct': difference_pct,
        'within_tolerance': within_tolerance,
        'message': message
    }


def reconcile_ok_vs_composite_mean(
    ok_estimates: np.ndarray,
    composite_values: np.ndarray,
    tolerance_pct: float = 10.0
) -> Dict[str, any]:
    """
    Reconcile OK mean against composite mean.

    Less strict than declustered (±10% tolerance) since composites may be clustered.

    Parameters
    ----------
    ok_estimates : np.ndarray
        OK estimates
    composite_values : np.ndarray
        Composite sample values
    tolerance_pct : float
        Acceptable tolerance percentage (default 10.0%)

    Returns
    -------
    dict
        Reconciliation results
    """
    valid_ok = ~np.isnan(ok_estimates)
    valid_comp = ~np.isnan(composite_values)

    if not np.any(valid_ok):
        return {
            'passed': False,
            'ok_mean': np.nan,
            'composite_mean': np.nan,
            'difference': np.nan,
            'difference_pct': np.nan,
            'within_tolerance': False,
            'message': "No valid OK estimates"
        }

    if not np.any(valid_comp):
        return {
            'passed': False,
            'ok_mean': float(np.mean(ok_estimates[valid_ok])),
            'composite_mean': np.nan,
            'difference': np.nan,
            'difference_pct': np.nan,
            'within_tolerance': False,
            'message': "No valid composite data"
        }

    ok_mean = float(np.mean(ok_estimates[valid_ok]))
    composite_mean = float(np.mean(composite_values[valid_comp]))

    difference = ok_mean - composite_mean
    difference_pct = abs(difference) * 100.0 / abs(composite_mean) if composite_mean != 0 else np.inf

    within_tolerance = difference_pct <= tolerance_pct
    passed = within_tolerance

    if passed:
        message = f"✓ OK mean ({ok_mean:.3f}) reconciles with composite mean ({composite_mean:.3f}), diff={difference_pct:.2f}%"
    else:
        message = f"⚠ OK mean ({ok_mean:.3f}) differs from composite mean ({composite_mean:.3f}) by {difference_pct:.2f}% (> {tolerance_pct}%)"

    return {
        'passed': passed,
        'ok_mean': ok_mean,
        'composite_mean': composite_mean,
        'difference': difference,
        'difference_pct': difference_pct,
        'within_tolerance': within_tolerance,
        'message': message
    }


def run_full_reconciliation(
    ok_estimates: np.ndarray,
    declustered_mean: Optional[float] = None,
    composite_values: Optional[np.ndarray] = None,
    sgsim_realizations: Optional[np.ndarray] = None,
    tolerance_pct: float = 5.0
) -> Dict[str, any]:
    """
    Run complete reconciliation suite.

    Parameters
    ----------
    ok_estimates : np.ndarray
        OK estimates
    declustered_mean : float, optional
        Declustered mean
    composite_values : np.ndarray, optional
        Composite sample values
    sgsim_realizations : np.ndarray, optional
        SGSIM realizations
    tolerance_pct : float
        Tolerance percentage

    Returns
    -------
    dict
        Complete reconciliation report
    """
    results = {
        'ok_mean': float(np.nanmean(ok_estimates)),
        'ok_std': float(np.nanstd(ok_estimates)),
        'n_valid_blocks': int(np.sum(~np.isnan(ok_estimates))),
        'checks': []
    }

    # Check 1: OK vs Declustered
    if declustered_mean is not None:
        check = reconcile_ok_vs_declustered(ok_estimates, declustered_mean, tolerance_pct)
        results['checks'].append({
            'name': 'OK vs Declustered Mean',
            'critical': True,
            **check
        })

    # Check 2: OK vs Composite (less critical)
    if composite_values is not None:
        check = reconcile_ok_vs_composite_mean(ok_estimates, composite_values, tolerance_pct * 2)
        results['checks'].append({
            'name': 'OK vs Composite Mean',
            'critical': False,
            **check
        })

    # Check 3: OK vs SGSIM
    if sgsim_realizations is not None:
        check = reconcile_ok_vs_sgsim(ok_estimates, sgsim_realizations, tolerance_pct)
        results['checks'].append({
            'name': 'OK vs SGSIM P50',
            'critical': True,
            **check
        })

    # Overall pass/fail
    critical_checks = [c for c in results['checks'] if c.get('critical', False)]
    results['all_passed'] = all(c['passed'] for c in critical_checks)
    results['critical_passed'] = all(c['passed'] for c in critical_checks)
    results['n_checks'] = len(results['checks'])
    results['n_passed'] = sum(1 for c in results['checks'] if c['passed'])

    return results


def format_reconciliation_report(results: Dict) -> str:
    """
    Format reconciliation results as human-readable text report.

    Parameters
    ----------
    results : dict
        Results from run_full_reconciliation

    Returns
    -------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RECONCILIATION REPORT (JORC/NI 43-101)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"OK Estimate Summary:")
    lines.append(f"  Mean: {results['ok_mean']:.4f}")
    lines.append(f"  Std Dev: {results['ok_std']:.4f}")
    lines.append(f"  Valid Blocks: {results['n_valid_blocks']:,}")
    lines.append("")

    lines.append(f"Reconciliation Checks ({results['n_passed']}/{results['n_checks']} passed):")
    lines.append("")

    for check in results['checks']:
        critical = "CRITICAL" if check.get('critical') else "INFO"
        status = "PASS" if check['passed'] else "FAIL"
        lines.append(f"[{critical}] [{status}] {check['name']}")
        lines.append(f"  {check['message']}")
        lines.append("")

    if results['all_passed']:
        lines.append("✓✓✓ ALL CRITICAL CHECKS PASSED ✓✓✓")
    else:
        lines.append("✗✗✗ RECONCILIATION FAILED - REVIEW REQUIRED ✗✗✗")

    lines.append("=" * 70)

    return "\n".join(lines)
