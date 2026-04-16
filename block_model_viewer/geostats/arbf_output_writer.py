"""
ARBF Output Writer — expands active-only results to full block model arrays.

Inactive cells receive NaN for float fields and 0 for integer flags.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Fields that are float arrays (get NaN for inactive)
_FLOAT_FIELDS = [
    "ARBF_GRADE", "ARBF_VAR_BLOCK", "ARBF_STD_BLOCK",
    "ARBF_NEFF", "ARBF_CONDNUM", "ARBF_STITCH_VAR",
]

# Fields that are integer arrays (get 0 for inactive, except FAIL_FLAG which gets 1)
_INT_FIELDS = ["ARBF_HIGHVAR_FLAG", "ARBF_FAIL_FLAG"]

# PUM coverage count (float, inactive = 0)
_COUNT_FIELDS = ["ARBF_PUM_COUNT"]


def expand_to_full_model(
    n_total: int,
    active_mask: np.ndarray,
    active_results: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Expand active-cell-only results to full block model size.

    Parameters
    ----------
    n_total : int
        Total number of blocks in the block model.
    active_mask : (n_total,) bool array
        True for active cells.
    active_results : dict
        Results from FastRBFEstimator.estimate_blocks() — arrays of size n_active.

    Returns
    -------
    dict with full-size arrays (n_total,), NaN for inactive float fields.
    """
    out: Dict[str, np.ndarray] = {}

    for key in _FLOAT_FIELDS:
        if key in active_results:
            full = np.full(n_total, np.nan, dtype=float)
            full[active_mask] = active_results[key]
            out[key] = full

    for key in _INT_FIELDS:
        if key in active_results:
            default = 1 if key == "ARBF_FAIL_FLAG" else 0
            full = np.full(n_total, default, dtype=int)
            full[active_mask] = active_results[key]
            out[key] = full

    for key in _COUNT_FIELDS:
        if key in active_results:
            full = np.zeros(n_total, dtype=float)
            full[active_mask] = active_results[key]
            out[key] = full

    return out
