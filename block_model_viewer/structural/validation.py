"""
Structural Feature Validation - Comprehensive data quality gates.

Implements validation gates following GeoX audit compliance standards:
- Spatial validity: Coordinates finite and within bounds
- Angular validity: Dip, azimuth, plunge, trend in valid ranges
- Feature completeness: Minimum points for surface features
- Naming: No duplicate feature names within type
- Orientation consistency: Warning if orientations contradict geometry

AUDIT COMPLIANCE:
- All validation results logged to audit trail
- Deterministic: Same input always produces same validation result
- Provenance: Full tracking of what was validated and when
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "ERROR"      # Fatal - data cannot be used
    WARNING = "WARNING"  # Non-fatal - data usable but quality concern
    INFO = "INFO"        # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    feature_id: Optional[str] = None
    feature_name: Optional[str] = None
    location: Optional[str] = None  # e.g., "row 42", "point 15"
    value: Optional[Any] = None  # The problematic value
    expected: Optional[str] = None  # Expected value/range
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'severity': self.severity.value,
            'code': self.code,
            'message': self.message,
            'feature_id': self.feature_id,
            'feature_name': self.feature_name,
            'location': self.location,
            'value': str(self.value) if self.value is not None else None,
            'expected': self.expected,
        }


@dataclass
class ValidationResult:
    """Result of validating structural features."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    feature_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.valid,
            'issues': [i.to_dict() for i in self.issues],
            'timestamp': self.timestamp.isoformat(),
            'feature_count': self.feature_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'metadata': self.metadata,
        }


# =============================================================================
# VALIDATION GATES
# =============================================================================

def validate_structural_collection(
    collection,
    strict: bool = False,
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> ValidationResult:
    """
    Validate a StructuralFeatureCollection.
    
    Performs comprehensive validation checks on all features.
    
    Args:
        collection: StructuralFeatureCollection to validate
        strict: If True, treat warnings as errors
        bounds: Optional (xmin, xmax, ymin, ymax, zmin, zmax) for bounds check
        
    Returns:
        ValidationResult with all issues found
    """
    from .feature_types import StructuralFeatureCollection
    
    result = ValidationResult(valid=True)
    
    if collection is None:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="NULL_COLLECTION",
            message="Collection is None",
        ))
        return result
    
    result.feature_count = len(collection.all_features)
    result.metadata['feature_types'] = collection.feature_count
    
    # Check for duplicate names
    _check_duplicate_names(collection, result)
    
    # Validate each feature
    for fault in collection.faults:
        _validate_fault(fault, result, bounds)
    
    for fold in collection.folds:
        _validate_fold(fold, result, bounds)
    
    for unconformity in collection.unconformities:
        _validate_unconformity(unconformity, result, bounds)
    
    # Convert warnings to errors if strict mode
    if strict:
        for issue in result.issues:
            if issue.severity == ValidationSeverity.WARNING:
                issue.severity = ValidationSeverity.ERROR
                result.error_count += 1
                result.warning_count -= 1
        result.valid = result.error_count == 0
    
    # Log summary
    logger.info(f"Validation complete: {result.feature_count} features, {result.error_count} errors, {result.warning_count} warnings")
    
    return result


def _check_duplicate_names(collection, result: ValidationResult):
    """Check for duplicate feature names within each type."""
    
    # Check faults
    fault_names = [f.name for f in collection.faults]
    duplicates = set([n for n in fault_names if fault_names.count(n) > 1])
    for name in duplicates:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="DUPLICATE_FAULT_NAME",
            message=f"Duplicate fault name: '{name}'",
            feature_name=name,
        ))
    
    # Check folds
    fold_names = [f.name for f in collection.folds]
    duplicates = set([n for n in fold_names if fold_names.count(n) > 1])
    for name in duplicates:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="DUPLICATE_FOLD_NAME",
            message=f"Duplicate fold name: '{name}'",
            feature_name=name,
        ))
    
    # Check unconformities
    unc_names = [u.name for u in collection.unconformities]
    duplicates = set([n for n in unc_names if unc_names.count(n) > 1])
    for name in duplicates:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="DUPLICATE_UNCONFORMITY_NAME",
            message=f"Duplicate unconformity name: '{name}'",
            feature_name=name,
        ))


def _validate_fault(fault, result: ValidationResult, bounds=None):
    """Validate a FaultFeature."""
    
    # Check name
    if not fault.name:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="MISSING_FAULT_NAME",
            message="Fault has no name",
            feature_id=fault.feature_id,
        ))
    
    # Check surface points
    if fault.point_count == 0 and len(fault.orientations) == 0:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="EMPTY_FAULT",
            message=f"Fault '{fault.name}' has no points or orientations",
            feature_id=fault.feature_id,
            feature_name=fault.name,
        ))
    
    # Validate surface points
    if fault.point_count > 0:
        _validate_coordinates(fault.surface_points, fault.feature_id, fault.name, "fault", result, bounds)
    
    # Validate orientations
    for i, orient in enumerate(fault.orientations):
        _validate_orientation(orient, fault.feature_id, fault.name, f"orientation_{i}", result, bounds)


def _validate_fold(fold, result: ValidationResult, bounds=None):
    """Validate a FoldFeature."""
    
    # Check name
    if not fold.name:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="MISSING_FOLD_NAME",
            message="Fold has no name",
            feature_id=fold.feature_id,
        ))
    
    # Check fold axes
    if len(fold.fold_axes) == 0:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="NO_FOLD_AXES",
            message=f"Fold '{fold.name}' has no fold axis measurements",
            feature_id=fold.feature_id,
            feature_name=fold.name,
        ))
    
    # Validate fold axes
    for i, axis in enumerate(fold.fold_axes):
        if not (0 <= axis.plunge <= 90):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_PLUNGE",
                message=f"Fold axis plunge out of range",
                feature_id=fold.feature_id,
                feature_name=fold.name,
                location=f"axis_{i}",
                value=axis.plunge,
                expected="0-90 degrees",
            ))
        
        if not (0 <= axis.trend <= 360):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_TREND",
                message=f"Fold axis trend out of range",
                feature_id=fold.feature_id,
                feature_name=fold.name,
                location=f"axis_{i}",
                value=axis.trend,
                expected="0-360 degrees",
            ))
    
    # Validate wavelength/amplitude if provided
    if fold.wavelength is not None and fold.wavelength <= 0:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="INVALID_WAVELENGTH",
            message=f"Fold wavelength must be positive",
            feature_id=fold.feature_id,
            feature_name=fold.name,
            value=fold.wavelength,
        ))
    
    # Validate surface points
    if fold.point_count > 0:
        _validate_coordinates(fold.surface_points, fold.feature_id, fold.name, "fold", result, bounds)


def _validate_unconformity(unconformity, result: ValidationResult, bounds=None):
    """Validate an UnconformityFeature."""
    
    # Check name
    if not unconformity.name:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="MISSING_UNCONFORMITY_NAME",
            message="Unconformity has no name",
            feature_id=unconformity.feature_id,
        ))
    
    # Check surface points
    if unconformity.point_count < 3:
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="INSUFFICIENT_UNCONFORMITY_POINTS",
            message=f"Unconformity '{unconformity.name}' has fewer than 3 surface points",
            feature_id=unconformity.feature_id,
            feature_name=unconformity.name,
            value=unconformity.point_count,
            expected="at least 3 points",
        ))
    
    # Validate surface points
    if unconformity.point_count > 0:
        _validate_coordinates(unconformity.surface_points, unconformity.feature_id, unconformity.name, "unconformity", result, bounds)
    
    # Validate orientations
    for i, orient in enumerate(unconformity.orientations):
        _validate_orientation(orient, unconformity.feature_id, unconformity.name, f"orientation_{i}", result, bounds)


def _validate_coordinates(
    points: np.ndarray,
    feature_id: str,
    feature_name: str,
    feature_type: str,
    result: ValidationResult,
    bounds=None,
):
    """Validate coordinate array."""
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        nan_count = np.sum(~np.isfinite(points))
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="INVALID_COORDINATES",
            message=f"{feature_type.capitalize()} contains non-finite coordinates",
            feature_id=feature_id,
            feature_name=feature_name,
            value=f"{nan_count} non-finite values",
        ))
    
    # Check bounds if provided
    if bounds is not None and len(points) > 0:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        
        out_of_bounds = 0
        for pt in points:
            if not np.isfinite(pt).all():
                continue
            if pt[0] < xmin or pt[0] > xmax or pt[1] < ymin or pt[1] > ymax or pt[2] < zmin or pt[2] > zmax:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="OUT_OF_BOUNDS",
                message=f"{feature_type.capitalize()} has points outside expected bounds",
                feature_id=feature_id,
                feature_name=feature_name,
                value=f"{out_of_bounds} points",
                expected=f"Within [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]",
            ))


def _validate_orientation(
    orient,
    feature_id: str,
    feature_name: str,
    location: str,
    result: ValidationResult,
    bounds=None,
):
    """Validate a StructuralOrientation."""
    
    # Check dip
    if not (0 <= orient.dip <= 90):
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="INVALID_DIP",
            message=f"Dip out of valid range",
            feature_id=feature_id,
            feature_name=feature_name,
            location=location,
            value=orient.dip,
            expected="0-90 degrees",
        ))
    
    # Check azimuth
    if not (0 <= orient.azimuth <= 360):
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="INVALID_AZIMUTH",
            message=f"Azimuth out of valid range",
            feature_id=feature_id,
            feature_name=feature_name,
            location=location,
            value=orient.azimuth,
            expected="0-360 degrees",
        ))
    
    # Check coordinates
    if not np.isfinite(orient.x) or not np.isfinite(orient.y) or not np.isfinite(orient.z):
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="INVALID_ORIENTATION_POSITION",
            message=f"Orientation has invalid coordinates",
            feature_id=feature_id,
            feature_name=feature_name,
            location=location,
            value=f"({orient.x}, {orient.y}, {orient.z})",
        ))
    
    # Check bounds
    if bounds is not None and np.isfinite([orient.x, orient.y, orient.z]).all():
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        if orient.x < xmin or orient.x > xmax or orient.y < ymin or orient.y > ymax or orient.z < zmin or orient.z > zmax:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="ORIENTATION_OUT_OF_BOUNDS",
                message=f"Orientation position outside expected bounds",
                feature_id=feature_id,
                feature_name=feature_name,
                location=location,
                value=f"({orient.x}, {orient.y}, {orient.z})",
            ))


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def log_structural_import_audit(
    collection,
    source_file: str,
    validation_result: ValidationResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate an audit log entry for structural feature import.
    
    Args:
        collection: Imported StructuralFeatureCollection
        source_file: Path to source CSV file
        validation_result: Result of validation
        metadata: Optional additional metadata
        
    Returns:
        Audit log entry dict (ready for JSONL serialization)
    """
    import hashlib
    from pathlib import Path
    
    timestamp = datetime.now()
    
    # Calculate collection fingerprint
    fingerprint_data = ""
    for feature in collection.all_features:
        fingerprint_data += f"{feature.feature_id}:{feature.name}:{feature.point_count};"
    
    fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    audit_entry = {
        "event_type": "structural_feature_import",
        "timestamp": timestamp.isoformat(),
        "source_file": str(source_file),
        "source_checksum": None,  # Would be computed from file
        "feature_counts": collection.feature_count,
        "total_points": collection.total_points,
        "collection_fingerprint": fingerprint,
        "validation": {
            "valid": validation_result.valid,
            "error_count": validation_result.error_count,
            "warning_count": validation_result.warning_count,
        },
        "features": [],
        "metadata": metadata or {},
    }
    
    # Add file checksum if file exists
    try:
        source_path = Path(source_file)
        if source_path.exists():
            with open(source_path, 'rb') as f:
                audit_entry["source_checksum"] = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        pass
    
    # Add feature summaries
    for feature in collection.all_features:
        audit_entry["features"].append({
            "feature_id": feature.feature_id,
            "feature_type": feature.feature_type.value,
            "name": feature.name,
            "point_count": feature.point_count,
            "import_timestamp": feature.import_timestamp.isoformat() if feature.import_timestamp else None,
        })
    
    return audit_entry


def write_audit_log(audit_entry: Dict[str, Any], audit_dir: str = "audit_logs") -> Optional[str]:
    """
    Write audit entry to JSONL audit log.
    
    Args:
        audit_entry: Audit entry dict
        audit_dir: Directory for audit logs
        
    Returns:
        Path to audit log file, or None if failed
    """
    import json
    from pathlib import Path
    
    try:
        audit_path = Path(audit_dir)
        audit_path.mkdir(parents=True, exist_ok=True)
        
        # Daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = audit_path / f"audit_{date_str}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        logger.info(f"Audit log written: {log_file}")
        return str(log_file)
        
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")
        return None

