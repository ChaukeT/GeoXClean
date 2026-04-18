"""
Domain Assignment — Assign geological domain codes to block model centroids.
=============================================================================

Given interpolated scalar fields (one per surface or one potential field),
assign every block to a geological domain/unit.

Domain codes are the input for ARBF domain-constrained estimation.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def assign_domains_potential_field(
    block_centroids: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    isovalues: List[float],
    unit_names: List[str],
    batch_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign domains from a single potential field (Eq. 5.1, 11.1).

    For K surfaces with sorted isovalues v_1 < v_2 < ... < v_K,
    there are K+1 units.  Block centroids are classified by which
    interval of the potential field they fall in.

    Parameters
    ----------
    block_centroids : (B, 3)
    evaluate_fn : callable (B, 3) -> (B,)
    isovalues : sorted list of K isovalues
    unit_names : list of K+1 unit names (one more than isovalues)
    batch_size : evaluation batch size

    Returns
    -------
    domain_codes : (B,) int, 0-based domain index
    domain_names : (B,) str, unit name per block
    """
    B = block_centroids.shape[0]
    K = len(isovalues)

    if len(unit_names) != K + 1:
        raise ValueError(
            f"Expected {K + 1} unit names for {K} isovalues, got {len(unit_names)}"
        )

    # Sort isovalues and reorder unit_names to match.
    # With K isovalues and K+1 unit_names, unit_names[k] is the unit
    # BELOW sorted_iso[k] (i.e. field_value < sorted_iso[k]).
    # If the caller passes unsorted isovalues, we must co-sort names.
    sort_order = np.argsort(isovalues)
    sorted_iso = [isovalues[i] for i in sort_order]

    # Reorder unit_names: the K+1 names bracket the K sorted isovalues.
    # Original: unit_names[k] is between isovalues[k-1] and isovalues[k].
    # After sorting: we need names in the order that matches sorted_iso.
    # The name below the smallest isovalue is unit_names[sort_order[0]],
    # between sorted_iso[k-1] and sorted_iso[k] is unit_names[sort_order[k]],
    # above the largest isovalue is the remaining name.
    # Simplest correct approach: require pre-sorted input, warn if not.
    if list(isovalues) != sorted_iso:
        logger.warning(
            "Isovalues were not pre-sorted; sorting internally. "
            "Ensure unit_names matches the sorted isovalue order."
        )

    # Evaluate potential field in batches
    field_values = np.empty(B, dtype=np.float64)
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        field_values[start:end] = evaluate_fn(block_centroids[start:end])

    # Classify
    domain_codes = np.full(B, K, dtype=np.int32)  # default = last unit

    for k in range(K):
        mask = field_values < sorted_iso[k]
        domain_codes[mask] = np.minimum(domain_codes[mask], k)

    # Clip to valid range
    domain_codes = np.clip(domain_codes, 0, K)

    # Map to names
    domain_names = np.array([unit_names[c] for c in domain_codes])

    logger.info(
        "Domain assignment: %d blocks into %d units (%s)",
        B, K + 1,
        ", ".join(f"{name}: {np.sum(domain_codes == i)}" for i, name in enumerate(unit_names)),
    )

    return domain_codes, domain_names


def assign_domains_independent_surfaces(
    block_centroids: np.ndarray,
    surface_fields: List[Callable[[np.ndarray], np.ndarray]],
    surface_isovalues: List[float],
    unit_names: List[str],
    batch_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign domains using independent surface SDFs.

    Each surface divides space into above (f > 0) and below (f < 0).
    Starting from the top surface, blocks above are assigned to the
    topmost unit, blocks below move to the next surface check.

    Parameters
    ----------
    block_centroids : (B, 3)
    surface_fields : list of K evaluation functions
    surface_isovalues : list of K isovalues (usually all 0.0 for SDFs)
    unit_names : list of K+1 unit names

    Returns
    -------
    domain_codes : (B,) int
    domain_names : (B,) str
    """
    B = block_centroids.shape[0]
    K = len(surface_fields)

    if len(unit_names) != K + 1:
        raise ValueError(
            f"Expected {K + 1} unit names for {K} surfaces, got {len(unit_names)}"
        )

    domain_codes = np.full(B, K, dtype=np.int32)  # default = deepest unit
    unassigned = np.ones(B, dtype=bool)

    for k in range(K):
        if not np.any(unassigned):
            break

        # Evaluate surface field for unassigned blocks
        idx = np.where(unassigned)[0]
        pts = block_centroids[idx]

        field_values = np.empty(len(idx), dtype=np.float64)
        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            field_values[start:end] = surface_fields[k](pts[start:end])

        # Blocks above this surface (f > isovalue) → assigned to unit k
        above = field_values > surface_isovalues[k]
        above_idx = idx[above]
        domain_codes[above_idx] = k
        unassigned[above_idx] = False

    domain_names = np.array([unit_names[c] for c in domain_codes])

    logger.info(
        "Domain assignment (independent surfaces): %d blocks into %d units",
        B, K + 1,
    )

    return domain_codes, domain_names


def assign_domains_from_lithology(
    composites_df,
    grouping: Dict[str, List[str]],
    lithology_column: str = "lith_code",
) -> np.ndarray:
    """Assign integer domain codes to composites from lithology grouping.

    This is the simplest domain assignment — directly from the lithology
    manager's grouping, without any interpolation.

    Parameters
    ----------
    composites_df : pd.DataFrame
    grouping : {unit_name: [raw_codes]}
    lithology_column : column containing raw lithology codes

    Returns
    -------
    domain_codes : (N,) int, 0-based
    """
    import pandas as pd

    # Build reverse map: raw_code → unit_name
    code_to_unit = {}
    unit_to_code = {}
    for i, (unit_name, codes) in enumerate(grouping.items()):
        unit_to_code[unit_name] = i
        for code in codes:
            code_to_unit[str(code).strip()] = unit_name

    N = len(composites_df)
    domain_codes = np.full(N, -1, dtype=np.int32)

    for idx in range(N):
        raw = str(composites_df.iloc[idx].get(lithology_column, "")).strip()
        unit = code_to_unit.get(raw)
        if unit is not None:
            domain_codes[idx] = unit_to_code[unit]

    n_unassigned = int(np.sum(domain_codes < 0))
    if n_unassigned > 0:
        logger.warning(
            "%d composites could not be assigned a domain code", n_unassigned,
        )

    return domain_codes
