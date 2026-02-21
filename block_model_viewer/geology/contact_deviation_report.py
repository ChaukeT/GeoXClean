"""
Contact Deviation Report - Model-to-Drillhole Comparison.

Compares modelled geological surfaces to original logged contact depths
and generates deviation reports for JORC/SAMREC compliance auditing.

GeoX Invariant Compliance:
- Compares final model surfaces against original drillhole contacts
- Generates per-formation and per-hole deviation statistics
- Produces audit-ready reports for Competent Person review
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("GeoX_ContactDeviation")


@dataclass
class ContactDeviation:
    """Single contact deviation measurement."""
    hole_id: str
    formation: str  # Formation above the contact
    logged_x: float
    logged_y: float
    logged_z: float  # Z coordinate of logged contact (elevation)
    modelled_z: float  # Z coordinate where model surface is at (X, Y)
    deviation_m: float  # Signed: positive = model is higher than logged
    scalar_value: float  # Target scalar value for this contact
    confidence: float  # Interpolation confidence (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "hole_id": self.hole_id,
            "formation": self.formation,
            "logged_x": self.logged_x,
            "logged_y": self.logged_y,
            "logged_z": self.logged_z,
            "modelled_z": self.modelled_z,
            "deviation_m": self.deviation_m,
            "scalar_value": self.scalar_value,
            "confidence": self.confidence,
        }


@dataclass
class FormationStats:
    """Statistics for a single formation's deviations."""
    formation: str
    count: int
    mean_deviation_m: float
    std_deviation_m: float
    min_deviation_m: float
    max_deviation_m: float
    p90_deviation_m: float
    mean_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formation": self.formation,
            "count": self.count,
            "mean_deviation_m": self.mean_deviation_m,
            "std_deviation_m": self.std_deviation_m,
            "min_deviation_m": self.min_deviation_m,
            "max_deviation_m": self.max_deviation_m,
            "p90_deviation_m": self.p90_deviation_m,
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class HoleStats:
    """Statistics for a single drillhole's deviations."""
    hole_id: str
    count: int
    mean_deviation_m: float
    max_deviation_m: float
    mean_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "count": self.count,
            "mean_deviation_m": self.mean_deviation_m,
            "max_deviation_m": self.max_deviation_m,
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class ContactDeviationReport:
    """Complete contact deviation analysis report."""
    deviations: List[ContactDeviation]
    timestamp: datetime = field(default_factory=datetime.now)

    # Summary statistics
    total_contacts: int = 0
    contacts_evaluated: int = 0
    contacts_failed: int = 0

    mean_deviation_m: float = 0.0
    std_deviation_m: float = 0.0
    min_deviation_m: float = 0.0
    max_deviation_m: float = 0.0
    p90_deviation_m: float = 0.0

    # Per-formation statistics
    by_formation: Dict[str, FormationStats] = field(default_factory=dict)

    # Per-hole statistics
    by_hole: Dict[str, HoleStats] = field(default_factory=dict)

    # Compliance status
    status: str = "NOT_EVALUATED"  # 'ACCEPTABLE', 'NEEDS_REVIEW', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_contacts": self.total_contacts,
            "contacts_evaluated": self.contacts_evaluated,
            "contacts_failed": self.contacts_failed,
            "mean_deviation_m": self.mean_deviation_m,
            "std_deviation_m": self.std_deviation_m,
            "min_deviation_m": self.min_deviation_m,
            "max_deviation_m": self.max_deviation_m,
            "p90_deviation_m": self.p90_deviation_m,
            "status": self.status,
            "by_formation": {k: v.to_dict() for k, v in self.by_formation.items()},
            "by_hole": {k: v.to_dict() for k, v in self.by_hole.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert deviations to DataFrame for export."""
        if not self.deviations:
            return pd.DataFrame(columns=[
                'hole_id', 'formation', 'logged_x', 'logged_y', 'logged_z',
                'modelled_z', 'deviation_m', 'confidence'
            ])
        return pd.DataFrame([d.to_dict() for d in self.deviations])

    def export_csv(self, filepath: str) -> None:
        """Export deviations to CSV file."""
        self.to_dataframe().to_csv(filepath, index=False)
        logger.info(f"Exported contact deviations to {filepath}")


class ContactDeviationAnalyzer:
    """
    Analyzes deviation between modelled surfaces and logged contacts.

    This is critical for JORC/SAMREC compliance - auditors need to see
    how well the model honors the original drillhole data.

    The key difference from ComplianceManager.generate_misfit_report():
    - ComplianceManager uses scalar field residuals (fast, approximate)
    - This analyzer computes actual Z deviation in meters (slower, precise)

    Usage:
        analyzer = ContactDeviationAnalyzer(model, scaler, stratigraphy)
        report = analyzer.compute_deviations(original_contacts_df)
    """

    def __init__(
        self,
        model: Any,  # LoopStructural GeologicalModel
        scaler: "StandardScaler",
        stratigraphy: List[str],
        feature_name: str = "Stratigraphy",
        z_samples: int = 100,
        deviation_p90_threshold_m: float = 5.0,
        deviation_mean_threshold_m: float = 2.0,
        formation_values: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the analyzer.

        Args:
            model: Solved LoopStructural GeologicalModel
            scaler: StandardScaler used to normalize coordinates
            stratigraphy: List of formation names (oldest to youngest)
            feature_name: Name of the stratigraphic feature in the model
            z_samples: Number of samples along vertical line for interpolation
            deviation_p90_threshold_m: P90 threshold for ACCEPTABLE status
            deviation_mean_threshold_m: Mean threshold for ACCEPTABLE status
            formation_values: Optional dict mapping formation names to scalar values.
                             If provided, uses these values instead of integer indices.
                             CRITICAL for correct contact evaluation when proportional
                             scalar spacing is used.
        """
        self.model = model
        self.scaler = scaler
        self.stratigraphy = stratigraphy
        self.feature_name = feature_name
        self.z_samples = z_samples
        self.deviation_p90_threshold = deviation_p90_threshold_m
        self.deviation_mean_threshold = deviation_mean_threshold_m

        # Get the stratigraphic feature from model
        try:
            self.feature = model[feature_name]
        except KeyError:
            # Try alternative names
            for alt in ['Stratigraphy', 'Main_Sequence', 'stratigraphy']:
                try:
                    self.feature = model[alt]
                    logger.warning(f"Using alternative feature name '{alt}'")
                    break
                except KeyError:
                    continue
            else:
                raise ValueError(f"Feature '{feature_name}' not found in model")

        # =================================================================
        # FIX: Use actual formation scalar values if provided
        # =================================================================
        # When proportional spacing is used, formations have non-integer values.
        # We must use the actual values to compute correct contact boundaries.
        # =================================================================
        if formation_values:
            self.formation_to_scalar = formation_values.copy()
            logger.info(f"Using provided formation scalar values: {formation_values}")
        else:
            # Legacy behavior: integer indices (0, 1, 2, ...)
            self.formation_to_scalar = {}
            for i, formation in enumerate(stratigraphy):
                self.formation_to_scalar[formation] = float(i)
            logger.warning(
                "No formation_values provided - using integer indices. "
                "Contact boundaries may be incorrect if proportional spacing was used."
            )

        # Pre-calculate contact boundary values (midpoints between formations)
        # Contact between formation[i] and formation[i+1] is at midpoint of their scalars
        self.contact_boundaries = {}
        for i in range(len(stratigraphy) - 1):
            upper_formation = stratigraphy[i + 1]  # Formation above the contact
            lower_val = self.formation_to_scalar.get(stratigraphy[i], float(i))
            upper_val = self.formation_to_scalar.get(stratigraphy[i + 1], float(i + 1))
            boundary_val = (lower_val + upper_val) / 2.0
            self.contact_boundaries[upper_formation] = boundary_val

        logger.info(
            f"ContactDeviationAnalyzer initialized with {len(stratigraphy)} formations, "
            f"contact boundaries: {self.contact_boundaries}"
        )

    def compute_deviations(
        self,
        original_contacts_df: pd.DataFrame,
        tolerance_m: float = 0.01
    ) -> ContactDeviationReport:
        """
        Compare modelled surface positions to original logged contacts.

        For each contact point, we find where the model's isosurface
        intersects a vertical line at the contact's (X, Y) position,
        then compute the Z difference.

        Args:
            original_contacts_df: DataFrame with columns:
                - X, Y, Z: Contact position in world coordinates
                - formation: Formation name (above the contact)
                - hole_id: Drillhole identifier (optional)
            tolerance_m: Minimum deviation to include in report

        Returns:
            ContactDeviationReport with all deviations and statistics
        """
        # Validate input columns
        required_cols = ['X', 'Y', 'Z', 'formation']
        missing = [c for c in required_cols if c not in original_contacts_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        has_hole_id = 'hole_id' in original_contacts_df.columns

        deviations = []
        failed_count = 0

        # Get model Z extent for vertical sampling
        z_min, z_max = self._get_z_extent()

        logger.info(f"Computing deviations for {len(original_contacts_df)} contacts...")

        for idx, row in original_contacts_df.iterrows():
            try:
                formation = row['formation']

                # Skip if formation not in stratigraphy
                if formation not in self.formation_to_scalar:
                    logger.debug(f"Skipping unknown formation: {formation}")
                    failed_count += 1
                    continue

                # =================================================================
                # FIX: Use pre-calculated contact boundary values
                # =================================================================
                # Contact boundary is at the midpoint between this formation's
                # scalar value and the formation below it. This works correctly
                # with both integer and proportional scalar spacing.
                # =================================================================
                if formation not in self.contact_boundaries:
                    # First formation (oldest) has no lower contact
                    # Use a value slightly below its scalar value
                    first_val = self.formation_to_scalar.get(self.stratigraphy[0], 0.0)
                    target_scalar = first_val - 0.1
                    logger.debug(f"Formation '{formation}' is oldest - using target_scalar={target_scalar:.3f}")
                else:
                    target_scalar = self.contact_boundaries[formation]

                # Find modelled Z at this (X, Y)
                x, y, z_logged = row['X'], row['Y'], row['Z']

                result = self._find_surface_z(x, y, target_scalar, z_min, z_max)

                if result is None:
                    failed_count += 1
                    continue

                z_modelled, confidence = result
                deviation = z_modelled - z_logged

                # Only include if above tolerance
                if abs(deviation) >= tolerance_m:
                    deviations.append(ContactDeviation(
                        hole_id=row['hole_id'] if has_hole_id else "UNKNOWN",
                        formation=formation,
                        logged_x=x,
                        logged_y=y,
                        logged_z=z_logged,
                        modelled_z=z_modelled,
                        deviation_m=deviation,
                        scalar_value=target_scalar,
                        confidence=confidence,
                    ))

            except Exception as e:
                logger.debug(f"Failed to compute deviation for row {idx}: {e}")
                failed_count += 1

        # Build report
        report = self._build_report(
            deviations=deviations,
            total_contacts=len(original_contacts_df),
            failed_count=failed_count
        )

        logger.info(
            f"Contact deviation analysis complete: "
            f"{report.contacts_evaluated}/{report.total_contacts} evaluated, "
            f"mean={report.mean_deviation_m:.2f}m, p90={report.p90_deviation_m:.2f}m, "
            f"status={report.status}"
        )

        return report

    def _get_z_extent(self) -> Tuple[float, float]:
        """Get the Z extent of the model in world coordinates."""
        # Get extent from scaler
        # The scaler was fit on [X, Y, Z] data
        if hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
            z_min = self.scaler.data_min_[2]
            z_max = self.scaler.data_max_[2]
        else:
            # Fallback: use mean and scale
            z_mean = self.scaler.mean_[2] if hasattr(self.scaler, 'mean_') else 0
            z_scale = self.scaler.scale_[2] if hasattr(self.scaler, 'scale_') else 1
            z_min = z_mean - 3 * z_scale
            z_max = z_mean + 3 * z_scale

        return z_min, z_max

    def _find_surface_z(
        self,
        x: float,
        y: float,
        target_scalar: float,
        z_min: float,
        z_max: float
    ) -> Optional[Tuple[float, float]]:
        """
        Find Z where the scalar field equals target_scalar at (X, Y).

        Uses vertical line sampling and linear interpolation.

        Args:
            x, y: World coordinates
            target_scalar: Target scalar field value
            z_min, z_max: Z range in world coordinates

        Returns:
            Tuple of (z_modelled, confidence) or None if not found
        """
        # Create vertical sample line
        z_samples = np.linspace(z_min, z_max, self.z_samples)
        points = np.column_stack([
            np.full(self.z_samples, x),
            np.full(self.z_samples, y),
            z_samples
        ])

        # Transform to scaled coordinates
        try:
            scaled_points = self.scaler.transform(points)
        except Exception as e:
            logger.debug(f"Scaler transform failed: {e}")
            return None

        # Evaluate scalar field
        try:
            values = self.feature.evaluate_value(scaled_points)
        except Exception as e:
            logger.debug(f"Scalar field evaluation failed: {e}")
            return None

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return None

        # Find crossing points
        # We want to find where values crosses target_scalar
        crossings = []
        for i in range(len(values) - 1):
            if not (valid_mask[i] and valid_mask[i + 1]):
                continue

            v0, v1 = values[i], values[i + 1]
            z0, z1 = z_samples[i], z_samples[i + 1]

            # Check if target is between v0 and v1
            if (v0 <= target_scalar <= v1) or (v1 <= target_scalar <= v0):
                # Linear interpolation
                if abs(v1 - v0) < 1e-10:
                    z_interp = (z0 + z1) / 2
                else:
                    t = (target_scalar - v0) / (v1 - v0)
                    z_interp = z0 + t * (z1 - z0)

                # Confidence based on gradient magnitude
                gradient = abs(v1 - v0) / abs(z1 - z0) if abs(z1 - z0) > 1e-10 else 0
                confidence = min(gradient * 10, 1.0)  # Scale to 0-1

                crossings.append((z_interp, confidence))

        if not crossings:
            return None

        # Return the crossing closest to the center of the Z range
        z_center = (z_min + z_max) / 2
        best = min(crossings, key=lambda c: abs(c[0] - z_center))
        return best

    def _build_report(
        self,
        deviations: List[ContactDeviation],
        total_contacts: int,
        failed_count: int
    ) -> ContactDeviationReport:
        """Build the complete deviation report with statistics."""

        report = ContactDeviationReport(
            deviations=deviations,
            total_contacts=total_contacts,
            contacts_evaluated=len(deviations),
            contacts_failed=failed_count,
        )

        if not deviations:
            report.status = "NO_DATA"
            return report

        # Compute absolute deviations for statistics
        abs_devs = np.abs([d.deviation_m for d in deviations])
        signed_devs = np.array([d.deviation_m for d in deviations])

        report.mean_deviation_m = float(np.mean(abs_devs))
        report.std_deviation_m = float(np.std(abs_devs))
        report.min_deviation_m = float(np.min(abs_devs))
        report.max_deviation_m = float(np.max(abs_devs))
        report.p90_deviation_m = float(np.percentile(abs_devs, 90))

        # Per-formation statistics
        formation_groups = {}
        for d in deviations:
            if d.formation not in formation_groups:
                formation_groups[d.formation] = []
            formation_groups[d.formation].append(d)

        for formation, group in formation_groups.items():
            devs = np.abs([d.deviation_m for d in group])
            confs = [d.confidence for d in group]
            report.by_formation[formation] = FormationStats(
                formation=formation,
                count=len(group),
                mean_deviation_m=float(np.mean(devs)),
                std_deviation_m=float(np.std(devs)),
                min_deviation_m=float(np.min(devs)),
                max_deviation_m=float(np.max(devs)),
                p90_deviation_m=float(np.percentile(devs, 90)),
                mean_confidence=float(np.mean(confs)),
            )

        # Per-hole statistics
        hole_groups = {}
        for d in deviations:
            if d.hole_id not in hole_groups:
                hole_groups[d.hole_id] = []
            hole_groups[d.hole_id].append(d)

        for hole_id, group in hole_groups.items():
            devs = np.abs([d.deviation_m for d in group])
            confs = [d.confidence for d in group]
            report.by_hole[hole_id] = HoleStats(
                hole_id=hole_id,
                count=len(group),
                mean_deviation_m=float(np.mean(devs)),
                max_deviation_m=float(np.max(devs)),
                mean_confidence=float(np.mean(confs)),
            )

        # Determine compliance status
        if report.p90_deviation_m <= self.deviation_p90_threshold and \
           report.mean_deviation_m <= self.deviation_mean_threshold:
            report.status = "ACCEPTABLE"
        elif report.p90_deviation_m <= self.deviation_p90_threshold * 2:
            report.status = "NEEDS_REVIEW"
        else:
            report.status = "CRITICAL"

        return report
