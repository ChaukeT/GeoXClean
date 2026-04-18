"""
ARBF Resource Classification from Posterior Variance.

Dual-criteria classification combining variance-based thresholds
(from the GPR posterior) with geometric quality checks (sample
count, octant coverage, search pass).  Eq. 10.1, 10.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Classification codes
UNCLASSIFIED = 0
INFERRED = 1
INDICATED = 2
MEASURED = 3

CLASS_NAMES = {
    UNCLASSIFIED: "Unclassified",
    INFERRED: "Inferred",
    INDICATED: "Indicated",
    MEASURED: "Measured",
}


@dataclass
class ClassificationResult:
    """Result of resource classification."""

    classes: np.ndarray            # (B,) int codes (0-3)
    class_names: np.ndarray        # (B,) string names
    variance_classes: np.ndarray   # (B,) variance-based only
    geometric_classes: np.ndarray  # (B,) geometry-based only
    summary: Dict[str, int]        # counts per class


@dataclass
class GeometricCriteria:
    """Geometric quality criteria for resource classification (Eq. 10.2).

    Defaults match JORC Table 1 Section 3 standards.
    """

    measured_min_samples: int = 12
    measured_min_octants: int = 4
    measured_max_pass: int = 1

    indicated_min_samples: int = 8
    indicated_min_octants: int = 3
    indicated_max_pass: int = 2

    inferred_min_samples: int = 4
    inferred_min_octants: int = 2
    inferred_max_pass: int = 3


@dataclass
class VarianceThresholds:
    """Variance thresholds for classification (Eq. 10.1).

    T1 < T2 < T3.  Blocks with s2 < T1 are Measured candidates, etc.
    """

    t1_measured: float = 0.1
    t2_indicated: float = 0.3
    t3_inferred: float = 0.6

    @classmethod
    def from_prior_variance(
        cls,
        prior_variance: float,
        measured_fraction: float = 0.1,
        indicated_fraction: float = 0.3,
        inferred_fraction: float = 0.6,
    ) -> "VarianceThresholds":
        """Compute thresholds as fractions of the prior (sill) variance."""
        return cls(
            t1_measured=prior_variance * measured_fraction,
            t2_indicated=prior_variance * indicated_fraction,
            t3_inferred=prior_variance * inferred_fraction,
        )

    @classmethod
    def from_percentiles(
        cls,
        variances: np.ndarray,
        measured_pct: float = 25.0,
        indicated_pct: float = 50.0,
        inferred_pct: float = 75.0,
    ) -> "VarianceThresholds":
        """Compute thresholds from variance distribution percentiles."""
        finite = np.asarray(variances, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return cls()
        return cls(
            t1_measured=float(np.percentile(finite, measured_pct)),
            t2_indicated=float(np.percentile(finite, indicated_pct)),
            t3_inferred=float(np.percentile(finite, inferred_pct)),
        )


def classify_by_variance(
    posterior_variances: np.ndarray,
    thresholds: VarianceThresholds,
) -> np.ndarray:
    """Classify blocks by posterior variance (Eq. 10.1).

    Parameters
    ----------
    posterior_variances : np.ndarray
        (B,) posterior variances.
    thresholds : VarianceThresholds
        Variance thresholds.

    Returns
    -------
    np.ndarray
        (B,) classification codes (0-3).
    """
    classes = np.full(len(posterior_variances), UNCLASSIFIED, dtype=np.int32)
    classes[posterior_variances <= thresholds.t3_inferred] = INFERRED
    classes[posterior_variances <= thresholds.t2_indicated] = INDICATED
    classes[posterior_variances <= thresholds.t1_measured] = MEASURED
    return classes


def classify_by_geometry(
    sample_counts: np.ndarray,
    octant_counts: np.ndarray,
    search_passes: np.ndarray,
    criteria: GeometricCriteria,
) -> np.ndarray:
    """Classify blocks by geometric quality (Eq. 10.2).

    Parameters
    ----------
    sample_counts : np.ndarray
        (B,) number of samples used per block.
    octant_counts : np.ndarray
        (B,) number of octants with at least one sample.
    search_passes : np.ndarray
        (B,) search pass (1=primary, 2=expanded, 3=double).
    criteria : GeometricCriteria
        Classification criteria.

    Returns
    -------
    np.ndarray
        (B,) classification codes (0-3).
    """
    B = len(sample_counts)
    classes = np.full(B, UNCLASSIFIED, dtype=np.int32)

    # Inferred
    mask_inf = (
        (sample_counts >= criteria.inferred_min_samples)
        & (octant_counts >= criteria.inferred_min_octants)
        & (search_passes <= criteria.inferred_max_pass)
    )
    classes[mask_inf] = INFERRED

    # Indicated (stricter)
    mask_ind = (
        (sample_counts >= criteria.indicated_min_samples)
        & (octant_counts >= criteria.indicated_min_octants)
        & (search_passes <= criteria.indicated_max_pass)
    )
    classes[mask_ind] = INDICATED

    # Measured (strictest)
    mask_meas = (
        (sample_counts >= criteria.measured_min_samples)
        & (octant_counts >= criteria.measured_min_octants)
        & (search_passes <= criteria.measured_max_pass)
    )
    classes[mask_meas] = MEASURED

    return classes


def classify_blocks(
    posterior_variances: np.ndarray,
    sample_counts: np.ndarray,
    octant_counts: np.ndarray,
    search_passes: np.ndarray,
    variance_thresholds: Optional[VarianceThresholds] = None,
    geometric_criteria: Optional[GeometricCriteria] = None,
    prior_variance: Optional[float] = None,
) -> ClassificationResult:
    """Dual-criteria classification — most conservative wins (Eq. 10.2).

    Final = min(variance_class, geometric_class)

    Parameters
    ----------
    posterior_variances : np.ndarray
        (B,) posterior variances from ARBF.
    sample_counts : np.ndarray
        (B,) samples used per block.
    octant_counts : np.ndarray
        (B,) octants with samples per block.
    search_passes : np.ndarray
        (B,) search pass per block.
    variance_thresholds : VarianceThresholds, optional
        Variance thresholds. Auto-computed if None.
    geometric_criteria : GeometricCriteria, optional
        Geometric criteria. Defaults used if None.

    Returns
    -------
    ClassificationResult
        Final classification with diagnostics.
    """
    B = len(posterior_variances)

    if variance_thresholds is None:
        if prior_variance is None or not np.isfinite(prior_variance) or prior_variance <= 0.0:
            raise ValueError(
                "variance_thresholds must be supplied when prior_variance is unavailable",
            )
        variance_thresholds = VarianceThresholds.from_prior_variance(
            float(prior_variance),
        )

    if geometric_criteria is None:
        geometric_criteria = GeometricCriteria()

    # Individual classifications
    var_classes = classify_by_variance(posterior_variances, variance_thresholds)
    geo_classes = classify_by_geometry(
        sample_counts, octant_counts, search_passes, geometric_criteria,
    )

    # Final: most conservative (minimum) wins
    final_classes = np.minimum(var_classes, geo_classes)

    # String names
    class_names = np.array([CLASS_NAMES[c] for c in final_classes])

    # Summary
    summary = {
        "Measured": int(np.sum(final_classes == MEASURED)),
        "Indicated": int(np.sum(final_classes == INDICATED)),
        "Inferred": int(np.sum(final_classes == INFERRED)),
        "Unclassified": int(np.sum(final_classes == UNCLASSIFIED)),
        "Total": B,
    }

    logger.info(
        "Classification: Measured=%d, Indicated=%d, Inferred=%d, Unclassified=%d",
        summary["Measured"],
        summary["Indicated"],
        summary["Inferred"],
        summary["Unclassified"],
    )

    return ClassificationResult(
        classes=final_classes,
        class_names=class_names,
        variance_classes=var_classes,
        geometric_classes=geo_classes,
        summary=summary,
    )
