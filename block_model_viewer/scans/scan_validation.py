"""
Scan Validation Engine
======================

Validates scan data for fragmentation analysis.
Checks CRS, density, outliers, coordinate consistency, and data quality.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from uuid import UUID
from datetime import datetime

from .scan_models import ScanData, ValidationReport, ValidationViolation

logger = logging.getLogger(__name__)


class ScanValidator:
    """
    Validates scan data for fragmentation analysis.

    Performs comprehensive checks on data quality, coordinate systems,
    and suitability for downstream processing.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Default validation thresholds
        self.min_density_pts_per_m3 = self.config.get('min_density_pts_per_m3', 10.0)
        self.max_outlier_fraction = self.config.get('max_outlier_fraction', 0.05)  # 5%
        self.coordinate_tolerance_sigma = self.config.get('coordinate_tolerance_sigma', 10.0)
        self.min_points_required = self.config.get('min_points_required', 1000)

    def validate_scan(self, scan_data: ScanData, scan_id: UUID) -> ValidationReport:
        """
        Validate scan data comprehensively.

        Args:
            scan_data: ScanData to validate
            scan_id: UUID of the scan

        Returns:
            ValidationReport with detailed findings
        """
        violations = []

        # Basic data checks
        violations.extend(self._check_basic_data_integrity(scan_data))

        # Coordinate system checks
        violations.extend(self._check_coordinate_system(scan_data))

        # Data quality checks
        violations.extend(self._check_data_quality(scan_data))

        # Analysis suitability checks
        violations.extend(self._check_analysis_suitability(scan_data))

        # Compute summary statistics
        coordinate_range = None
        density_estimate = None

        if scan_data.points is not None and len(scan_data.points) > 0:
            bounds = scan_data.bounds()
            if bounds:
                coordinate_range = bounds

            # Estimate point density
            if coordinate_range:
                volume = np.prod(np.abs(coordinate_range[1] - coordinate_range[0]))
                if volume > 0:
                    density_estimate = len(scan_data.points) / volume

        # Count outliers (rough estimate)
        outlier_count = sum(1 for v in violations if "outlier" in v.field.lower())

        report = ValidationReport(
            scan_id=scan_id,
            timestamp=datetime.now(),
            is_valid=all(v.violation_type == "warning" for v in violations),  # Valid if no errors
            violations=violations,
            total_points=scan_data.point_count(),
            coordinate_range=coordinate_range,
            density_estimate=density_estimate,
            outlier_count=outlier_count
        )

        logger.info(f"Validation complete for scan {scan_id}: {len(violations)} violations "
                   f"({report.error_count()} errors, {report.warning_count()} warnings)")

        return report

    def _check_basic_data_integrity(self, scan_data: ScanData) -> List[ValidationViolation]:
        """Check basic data integrity and structure."""
        violations = []

        # Check if points exist
        if scan_data.points is None or len(scan_data.points) == 0:
            violations.append(ValidationViolation(
                violation_type="error",
                field="points",
                message="No point data found in scan",
                details={"point_count": 0}
            ))
            return violations  # Can't continue validation without points

        point_count = len(scan_data.points)

        # Check minimum point count
        if point_count < self.min_points_required:
            violations.append(ValidationViolation(
                violation_type="error",
                field="point_count",
                message=f"Insufficient points for analysis. Need ≥{self.min_points_required}, got {point_count}",
                details={"required": self.min_points_required, "actual": point_count}
            ))

        # Check coordinate dimensions
        if scan_data.points.shape[1] != 3:
            violations.append(ValidationViolation(
                violation_type="error",
                field="coordinates",
                message=f"Points must have 3 coordinates (X,Y,Z), got {scan_data.points.shape[1]}",
                details={"expected_dims": 3, "actual_dims": scan_data.points.shape[1]}
            ))

        # Check for non-finite coordinates
        finite_mask = np.isfinite(scan_data.points).all(axis=1)
        non_finite_count = np.sum(~finite_mask)

        if non_finite_count > 0:
            violations.append(ValidationViolation(
                violation_type="error",
                field="coordinates",
                message=f"Found {non_finite_count} points with non-finite coordinates (NaN or Inf)",
                details={"non_finite_count": int(non_finite_count)}
            ))

        # Check for duplicate points
        if point_count > 1:
            # Use a tolerance-based check for near-duplicates
            from scipy.spatial import cKDTree
            tree = cKDTree(scan_data.points)
            # Find points closer than 1mm
            duplicate_pairs = tree.query_pairs(r=0.001)
            if len(duplicate_pairs) > 0:
                violations.append(ValidationViolation(
                    violation_type="warning",
                    field="duplicates",
                    message=f"Found {len(duplicate_pairs)} near-duplicate point pairs (within 1mm)",
                    details={"duplicate_pairs": len(duplicate_pairs)}
                ))

        return violations

    def _check_coordinate_system(self, scan_data: ScanData) -> List[ValidationViolation]:
        """Check coordinate system and CRS."""
        violations = []

        # Check CRS presence
        if not scan_data.crs:
            violations.append(ValidationViolation(
                violation_type="error",
                field="crs",
                message="No coordinate reference system (CRS) specified. CRS is required for mining analysis.",
                details={"crs": scan_data.crs}
            ))

        # Check units
        valid_units = ["meters", "feet"]
        if scan_data.units not in valid_units:
            violations.append(ValidationViolation(
                violation_type="warning",
                field="units",
                message=f"Units '{scan_data.units}' may not be standard. Expected 'meters' or 'feet'.",
                details={"units": scan_data.units, "valid_units": valid_units}
            ))

        # Check coordinate ranges (rough validation)
        if scan_data.points is not None:
            bounds = scan_data.bounds()
            if bounds:
                min_coords, max_coords = bounds
                coord_range = max_coords - min_coords

                # Check for unreasonably large coordinate ranges
                max_reasonable_range = 10000.0  # 10km in meters
                if np.any(coord_range > max_reasonable_range):
                    violations.append(ValidationViolation(
                        violation_type="warning",
                        field="coordinate_range",
                        message=f"Unusually large coordinate range detected: {coord_range.max():.1f} units. "
                               f"Expected range for mining scans: <{max_reasonable_range} units.",
                        details={"max_range": float(coord_range.max()), "expected_max": max_reasonable_range}
                    ))

                # Check for zero extent in any dimension
                zero_extent_dims = np.where(coord_range == 0)[0]
                if len(zero_extent_dims) > 0:
                    dim_names = ['X', 'Y', 'Z']
                    violations.append(ValidationViolation(
                        violation_type="error",
                        field="coordinate_extent",
                        message=f"All points have zero extent in {len(zero_extent_dims)} dimensions: "
                               f"{[dim_names[i] for i in zero_extent_dims]}. Invalid scan geometry.",
                        details={"zero_extent_dims": zero_extent_dims.tolist()}
                    ))

        return violations

    def _check_data_quality(self, scan_data: ScanData) -> List[ValidationViolation]:
        """Check data quality metrics."""
        violations = []

        if scan_data.points is None:
            return violations

        # Statistical outlier detection using IQR method
        for dim, dim_name in enumerate(['X', 'Y', 'Z']):
            coords = scan_data.points[:, dim]

            # Calculate quartiles
            q1, q3 = np.percentile(coords, [25, 75])
            iqr = q3 - q1

            # Define outlier bounds (1.5 * IQR rule)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (coords < lower_bound) | (coords > upper_bound)
            outlier_count = np.sum(outlier_mask)

            if outlier_count > 0:
                outlier_fraction = outlier_count / len(coords)
                violation_type = "warning" if outlier_fraction <= self.max_outlier_fraction else "error"

                violations.append(ValidationViolation(
                    violation_type=violation_type,
                    field=f"outliers_{dim_name.lower()}",
                    message=f"Found {outlier_count} outlier points in {dim_name} coordinate "
                           f"({outlier_fraction:.1%} of total points)",
                    details={
                        "dimension": dim_name,
                        "outlier_count": int(outlier_count),
                        "outlier_fraction": float(outlier_fraction),
                        "bounds": [float(lower_bound), float(upper_bound)]
                    }
                ))

        return violations

    def _check_analysis_suitability(self, scan_data: ScanData) -> List[ValidationViolation]:
        """Check if data is suitable for fragmentation analysis."""
        violations = []

        if scan_data.points is None:
            return violations

        # Estimate point density
        bounds = scan_data.bounds()
        if bounds:
            min_coords, max_coords = bounds
            volume = np.prod(np.abs(max_coords - min_coords))

            if volume > 0:
                density = len(scan_data.points) / volume

                if density < self.min_density_pts_per_m3:
                    violations.append(ValidationViolation(
                        violation_type="error",
                        field="point_density",
                        message=f"Point density too low for reliable fragmentation analysis. "
                               f"Need ≥{self.min_density_pts_per_m3} pts/m³, got {density:.1f} pts/m³.",
                        details={
                            "required_density": self.min_density_pts_per_m3,
                            "actual_density": float(density),
                            "volume_m3": float(volume)
                        }
                    ))

                # Check for extremely high density (possible data issues)
                max_reasonable_density = 1000000.0  # 1M pts/m³
                if density > max_reasonable_density:
                    violations.append(ValidationViolation(
                        violation_type="warning",
                        field="point_density",
                        message=f"Extremely high point density detected: {density:.1f} pts/m³. "
                               f"May indicate data processing issues.",
                        details={"density": float(density), "max_expected": max_reasonable_density}
                    ))

        # Check for mesh/point cloud consistency
        if scan_data.is_mesh():
            # For meshes, check face quality
            if scan_data.faces is not None:
                face_count = len(scan_data.faces)

                # Check for degenerate faces
                if scan_data.points is not None:
                    # Simple check: faces with duplicate vertex indices
                    unique_vertices_per_face = np.array([len(np.unique(face)) for face in scan_data.faces])
                    degenerate_faces = np.sum(unique_vertices_per_face < 3)

                    if degenerate_faces > 0:
                        violations.append(ValidationViolation(
                            violation_type="warning",
                            field="mesh_quality",
                            message=f"Found {degenerate_faces} degenerate faces in mesh "
                                   f"({degenerate_faces/face_count:.1%} of total)",
                            details={"degenerate_faces": int(degenerate_faces), "total_faces": face_count}
                        ))

        # Check attribute consistency
        if scan_data.colors is not None:
            if len(scan_data.colors) != len(scan_data.points):
                violations.append(ValidationViolation(
                    violation_type="error",
                    field="attributes",
                    message=f"Color array length ({len(scan_data.colors)}) doesn't match point count ({len(scan_data.points)})",
                    details={"color_count": len(scan_data.colors), "point_count": len(scan_data.points)}
                ))

        if scan_data.normals is not None:
            if len(scan_data.normals) != len(scan_data.points):
                violations.append(ValidationViolation(
                    violation_type="warning",
                    field="attributes",
                    message=f"Normal array length ({len(scan_data.normals)}) doesn't match point count ({len(scan_data.points)})",
                    details={"normal_count": len(scan_data.normals), "point_count": len(scan_data.points)}
                ))

        return violations


def validate_scan(scan_data: ScanData, scan_id: UUID, config: Optional[dict] = None) -> ValidationReport:
    """
    Convenience function to validate scan data.

    Args:
        scan_data: ScanData to validate
        scan_id: UUID of the scan
        config: Optional validation configuration

    Returns:
        ValidationReport
    """
    validator = ScanValidator(config)
    return validator.validate_scan(scan_data, scan_id)
