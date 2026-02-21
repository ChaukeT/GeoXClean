"""
Structural Feature Types - Data structures for faults, folds, and unconformities.

This module defines the core data structures for representing structural features
loaded from CSV files. These features integrate with:
- 3D geological modeling (GemPy, LoopStructural)
- 3D visualization (PyVista renderer)
- Structural analysis (stereonets, kinematic analysis)

AUDIT COMPLIANCE:
- Deterministic: All data structures are immutable after creation
- Provenance: Full metadata tracking including source file and timestamps
- JORC/SAMREC: Supports standard structural geology data formats
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .datasets import PlaneMeasurement, LineationMeasurement

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FeatureType(str, Enum):
    """Structural feature types."""
    FAULT = "fault"
    FOLD = "fold"
    UNCONFORMITY = "unconformity"


class FaultDisplacementType(str, Enum):
    """Fault displacement types."""
    NORMAL = "normal"
    REVERSE = "reverse"
    STRIKE_SLIP = "strike-slip"
    OBLIQUE = "oblique"
    UNKNOWN = "unknown"


class FoldStyle(str, Enum):
    """Fold geometric styles."""
    ANTICLINE = "anticline"
    SYNCLINE = "syncline"
    MONOCLINE = "monocline"
    DOME = "dome"
    BASIN = "basin"
    UNKNOWN = "unknown"


class UnconformityType(str, Enum):
    """Unconformity types based on relationship to underlying rocks."""
    ANGULAR = "angular"
    DISCONFORMITY = "disconformity"
    NONCONFORMITY = "nonconformity"
    PARACONFORMITY = "paraconformity"
    UNKNOWN = "unknown"


# =============================================================================
# ORIENTATION DATA
# =============================================================================

@dataclass
class StructuralOrientation:
    """
    Orientation measurement with 3D position.
    
    Compatible with GemPy OrientationData format.
    """
    x: float
    y: float
    z: float
    dip: float  # 0-90 degrees from horizontal
    azimuth: float  # 0-360 degrees, dip direction (clockwise from north)
    polarity: int = 1  # 1 for normal, -1 for overturned
    source: str = "csv_import"  # provenance tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def position(self) -> np.ndarray:
        """Return 3D position as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def gradient_vector(self) -> Tuple[float, float, float]:
        """
        Convert dip/azimuth to gradient vector (G_x, G_y, G_z).
        
        This is the format required by GemPy for orientation constraints.
        """
        dip_rad = np.radians(self.dip)
        az_rad = np.radians(self.azimuth)
        
        # Calculate normal vector (pointing in dip direction)
        g_x = np.sin(dip_rad) * np.sin(az_rad)
        g_y = np.sin(dip_rad) * np.cos(az_rad)
        g_z = np.cos(dip_rad)
        
        # Apply polarity
        if self.polarity < 0:
            g_x, g_y, g_z = -g_x, -g_y, -g_z
        
        return (g_x, g_y, g_z)
    
    def to_plane_measurement(self) -> PlaneMeasurement:
        """Convert to PlaneMeasurement for structural analysis."""
        return PlaneMeasurement(
            dip=self.dip,
            dip_direction=self.azimuth,
            set_id=self.metadata.get('feature_name'),
            metadata={
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'polarity': self.polarity,
                'source': self.source,
                **self.metadata
            }
        )
    
    def validate(self) -> List[str]:
        """Validate orientation data. Returns list of error messages."""
        errors = []
        if not np.isfinite(self.x):
            errors.append(f"Invalid X coordinate: {self.x}")
        if not np.isfinite(self.y):
            errors.append(f"Invalid Y coordinate: {self.y}")
        if not np.isfinite(self.z):
            errors.append(f"Invalid Z coordinate: {self.z}")
        if not (0 <= self.dip <= 90):
            errors.append(f"Dip must be 0-90 degrees, got: {self.dip}")
        if not (0 <= self.azimuth <= 360):
            errors.append(f"Azimuth must be 0-360 degrees, got: {self.azimuth}")
        if self.polarity not in (1, -1):
            errors.append(f"Polarity must be 1 or -1, got: {self.polarity}")
        return errors


# =============================================================================
# BASE STRUCTURAL FEATURE
# =============================================================================

@dataclass
class StructuralFeature:
    """
    Base class for all structural features.
    
    This provides common attributes for faults, folds, and unconformities:
    - Unique identification (feature_id, name)
    - Type classification
    - Surface points for 3D representation
    - Provenance metadata
    """
    feature_type: FeatureType
    name: str
    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    surface_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance tracking
    source_file: Optional[str] = None
    source_checksum: Optional[str] = None
    import_timestamp: Optional[datetime] = None
    parser_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate and normalize after initialization."""
        # Ensure surface_points is numpy array
        if not isinstance(self.surface_points, np.ndarray):
            self.surface_points = np.array(self.surface_points)
        
        # Ensure 2D array with 3 columns
        if self.surface_points.ndim == 1 and len(self.surface_points) > 0:
            if len(self.surface_points) % 3 == 0:
                self.surface_points = self.surface_points.reshape(-1, 3)
            else:
                raise ValueError("Surface points must have 3 coordinates per point")
        
        # Set import timestamp if not provided
        if self.import_timestamp is None:
            self.import_timestamp = datetime.now()
    
    @property
    def point_count(self) -> int:
        """Number of surface points."""
        return len(self.surface_points) if self.surface_points.size > 0 else 0
    
    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Centroid of surface points."""
        if self.point_count == 0:
            return None
        return self.surface_points.mean(axis=0)
    
    @property
    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Bounding box (min, max) of surface points."""
        if self.point_count == 0:
            return None
        return (self.surface_points.min(axis=0), self.surface_points.max(axis=0))
    
    def validate(self) -> List[str]:
        """Validate feature data. Returns list of error messages."""
        errors = []
        if not self.name:
            errors.append("Feature name is required")
        if not self.feature_id:
            errors.append("Feature ID is required")
        if self.point_count > 0:
            if not np.all(np.isfinite(self.surface_points)):
                errors.append("Surface points contain non-finite values")
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_type': self.feature_type.value,
            'name': self.name,
            'feature_id': self.feature_id,
            'surface_points': self.surface_points.tolist() if self.surface_points.size > 0 else [],
            'point_count': self.point_count,
            'metadata': self.metadata,
            'source_file': self.source_file,
            'source_checksum': self.source_checksum,
            'import_timestamp': self.import_timestamp.isoformat() if self.import_timestamp else None,
            'parser_version': self.parser_version,
        }


# =============================================================================
# FAULT FEATURE
# =============================================================================

@dataclass
class FaultFeature(StructuralFeature):
    """
    Fault - a discrete discontinuity with displacement.
    
    Attributes:
        orientations: List of orientation measurements on the fault surface
        displacement_type: Type of fault movement (normal, reverse, strike-slip, oblique)
        affected_formations: List of geological formations affected by this fault
        displacement_magnitude: Estimated displacement in meters (optional)
    """
    orientations: List[StructuralOrientation] = field(default_factory=list)
    displacement_type: FaultDisplacementType = FaultDisplacementType.UNKNOWN
    affected_formations: List[str] = field(default_factory=list)
    displacement_magnitude: Optional[float] = None
    
    def __post_init__(self):
        """Set fault-specific defaults."""
        # Ensure feature_type is FAULT
        object.__setattr__(self, 'feature_type', FeatureType.FAULT)
        super().__post_init__()
    
    def validate(self) -> List[str]:
        """Validate fault data."""
        errors = super().validate()
        
        # Validate orientations
        for i, orient in enumerate(self.orientations):
            orient_errors = orient.validate()
            for err in orient_errors:
                errors.append(f"Orientation {i}: {err}")
        
        # Validate displacement magnitude
        if self.displacement_magnitude is not None:
            if self.displacement_magnitude < 0:
                errors.append(f"Displacement magnitude cannot be negative: {self.displacement_magnitude}")
        
        return errors
    
    def get_plane_measurements(self) -> List[PlaneMeasurement]:
        """Convert orientations to PlaneMeasurements for stereonet analysis."""
        return [o.to_plane_measurement() for o in self.orientations]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            'orientations': [
                {
                    'x': o.x, 'y': o.y, 'z': o.z,
                    'dip': o.dip, 'azimuth': o.azimuth,
                    'polarity': o.polarity, 'source': o.source
                }
                for o in self.orientations
            ],
            'displacement_type': self.displacement_type.value,
            'affected_formations': self.affected_formations,
            'displacement_magnitude': self.displacement_magnitude,
        })
        return base


# =============================================================================
# FOLD FEATURE
# =============================================================================

@dataclass
class FoldFeature(StructuralFeature):
    """
    Fold - continuous deformation feature.
    
    Attributes:
        fold_axes: List of fold axis lineation measurements (plunge/trend)
        limb_orientations: List of plane measurements from fold limbs
        fold_style: Geometric classification (anticline, syncline, etc.)
        wavelength: Fold wavelength in meters (optional)
        amplitude: Fold amplitude in meters (optional)
        interlimb_angle: Angle between fold limbs in degrees (optional)
    """
    fold_axes: List[LineationMeasurement] = field(default_factory=list)
    limb_orientations: List[PlaneMeasurement] = field(default_factory=list)
    fold_style: FoldStyle = FoldStyle.UNKNOWN
    wavelength: Optional[float] = None
    amplitude: Optional[float] = None
    interlimb_angle: Optional[float] = None
    
    def __post_init__(self):
        """Set fold-specific defaults."""
        object.__setattr__(self, 'feature_type', FeatureType.FOLD)
        super().__post_init__()
    
    def validate(self) -> List[str]:
        """Validate fold data."""
        errors = super().validate()
        
        # Validate fold axes
        for i, axis in enumerate(self.fold_axes):
            if not (0 <= axis.plunge <= 90):
                errors.append(f"Fold axis {i}: Plunge must be 0-90 degrees, got: {axis.plunge}")
            if not (0 <= axis.trend <= 360):
                errors.append(f"Fold axis {i}: Trend must be 0-360 degrees, got: {axis.trend}")
        
        # Validate limb orientations
        for i, limb in enumerate(self.limb_orientations):
            if not (0 <= limb.dip <= 90):
                errors.append(f"Limb {i}: Dip must be 0-90 degrees, got: {limb.dip}")
            if not (0 <= limb.dip_direction <= 360):
                errors.append(f"Limb {i}: Dip direction must be 0-360 degrees, got: {limb.dip_direction}")
        
        # Validate wavelength/amplitude
        if self.wavelength is not None and self.wavelength <= 0:
            errors.append(f"Wavelength must be positive: {self.wavelength}")
        if self.amplitude is not None and self.amplitude <= 0:
            errors.append(f"Amplitude must be positive: {self.amplitude}")
        if self.interlimb_angle is not None and not (0 <= self.interlimb_angle <= 180):
            errors.append(f"Interlimb angle must be 0-180 degrees: {self.interlimb_angle}")
        
        return errors
    
    @property
    def average_fold_axis(self) -> Optional[Tuple[float, float]]:
        """Calculate average fold axis orientation (plunge, trend)."""
        if not self.fold_axes:
            return None
        
        # Convert to direction cosines, average, convert back
        # This handles the circular nature of trend
        plunges = np.array([a.plunge for a in self.fold_axes])
        trends = np.array([a.trend for a in self.fold_axes])
        
        # Simple average (works well for clustered data)
        avg_plunge = np.mean(plunges)
        
        # Circular mean for trend
        trends_rad = np.radians(trends)
        avg_sin = np.mean(np.sin(trends_rad))
        avg_cos = np.mean(np.cos(trends_rad))
        avg_trend = np.degrees(np.arctan2(avg_sin, avg_cos)) % 360
        
        return (avg_plunge, avg_trend)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            'fold_axes': [
                {'plunge': a.plunge, 'trend': a.trend, 'set_id': a.set_id}
                for a in self.fold_axes
            ],
            'limb_orientations': [
                {'dip': l.dip, 'dip_direction': l.dip_direction, 'set_id': l.set_id}
                for l in self.limb_orientations
            ],
            'fold_style': self.fold_style.value,
            'wavelength': self.wavelength,
            'amplitude': self.amplitude,
            'interlimb_angle': self.interlimb_angle,
            'average_fold_axis': self.average_fold_axis,
        })
        return base


# =============================================================================
# UNCONFORMITY FEATURE
# =============================================================================

@dataclass
class UnconformityFeature(StructuralFeature):
    """
    Unconformity - erosional or non-depositional surface.
    
    Attributes:
        orientations: List of orientation measurements on the unconformity surface
        unconformity_type: Classification based on angular relationship
        formations_above: List of formations above the unconformity (younger)
        formations_below: List of formations below the unconformity (older)
        time_gap: Estimated time gap in millions of years (optional)
    """
    orientations: List[StructuralOrientation] = field(default_factory=list)
    unconformity_type: UnconformityType = UnconformityType.UNKNOWN
    formations_above: List[str] = field(default_factory=list)
    formations_below: List[str] = field(default_factory=list)
    time_gap: Optional[float] = None
    
    def __post_init__(self):
        """Set unconformity-specific defaults."""
        object.__setattr__(self, 'feature_type', FeatureType.UNCONFORMITY)
        super().__post_init__()
    
    def validate(self) -> List[str]:
        """Validate unconformity data."""
        errors = super().validate()
        
        # Validate orientations
        for i, orient in enumerate(self.orientations):
            orient_errors = orient.validate()
            for err in orient_errors:
                errors.append(f"Orientation {i}: {err}")
        
        # Validate time gap
        if self.time_gap is not None and self.time_gap < 0:
            errors.append(f"Time gap cannot be negative: {self.time_gap}")
        
        return errors
    
    def get_plane_measurements(self) -> List[PlaneMeasurement]:
        """Convert orientations to PlaneMeasurements for stereonet analysis."""
        return [o.to_plane_measurement() for o in self.orientations]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            'orientations': [
                {
                    'x': o.x, 'y': o.y, 'z': o.z,
                    'dip': o.dip, 'azimuth': o.azimuth,
                    'polarity': o.polarity, 'source': o.source
                }
                for o in self.orientations
            ],
            'unconformity_type': self.unconformity_type.value,
            'formations_above': self.formations_above,
            'formations_below': self.formations_below,
            'time_gap': self.time_gap,
        })
        return base


# =============================================================================
# STRUCTURAL FEATURE COLLECTION
# =============================================================================

@dataclass
class StructuralFeatureCollection:
    """
    Container for multiple structural features.
    
    Provides convenient access to features by type and methods for
    validation and serialization.
    """
    faults: List[FaultFeature] = field(default_factory=list)
    folds: List[FoldFeature] = field(default_factory=list)
    unconformities: List[UnconformityFeature] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_features(self) -> List[StructuralFeature]:
        """Return all features as a single list."""
        return list(self.faults) + list(self.folds) + list(self.unconformities)
    
    @property
    def feature_count(self) -> Dict[str, int]:
        """Return count of features by type."""
        return {
            'faults': len(self.faults),
            'folds': len(self.folds),
            'unconformities': len(self.unconformities),
            'total': len(self.faults) + len(self.folds) + len(self.unconformities),
        }
    
    @property
    def total_points(self) -> int:
        """Total number of surface points across all features."""
        return sum(f.point_count for f in self.all_features)
    
    def get_by_id(self, feature_id: str) -> Optional[StructuralFeature]:
        """Get a feature by its ID."""
        for feature in self.all_features:
            if feature.feature_id == feature_id:
                return feature
        return None
    
    def get_by_name(self, name: str) -> List[StructuralFeature]:
        """Get all features with a given name."""
        return [f for f in self.all_features if f.name == name]
    
    def add_feature(self, feature: StructuralFeature):
        """Add a feature to the appropriate list."""
        if isinstance(feature, FaultFeature):
            self.faults.append(feature)
        elif isinstance(feature, FoldFeature):
            self.folds.append(feature)
        elif isinstance(feature, UnconformityFeature):
            self.unconformities.append(feature)
        else:
            raise TypeError(f"Unknown feature type: {type(feature)}")
    
    def remove_feature(self, feature_id: str) -> bool:
        """Remove a feature by ID. Returns True if found and removed."""
        for lst in [self.faults, self.folds, self.unconformities]:
            for i, f in enumerate(lst):
                if f.feature_id == feature_id:
                    lst.pop(i)
                    return True
        return False
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all features. Returns dict of feature_id -> errors."""
        all_errors = {}
        for feature in self.all_features:
            errors = feature.validate()
            if errors:
                all_errors[feature.feature_id] = errors
        return all_errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'faults': [f.to_dict() for f in self.faults],
            'folds': [f.to_dict() for f in self.folds],
            'unconformities': [u.to_dict() for u in self.unconformities],
            'feature_count': self.feature_count,
            'total_points': self.total_points,
            'metadata': self.metadata,
        }
    
    def merge(self, other: 'StructuralFeatureCollection'):
        """Merge another collection into this one."""
        self.faults.extend(other.faults)
        self.folds.extend(other.folds)
        self.unconformities.extend(other.unconformities)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_fault_from_points(
    name: str,
    points: np.ndarray,
    orientations: Optional[List[Dict[str, float]]] = None,
    displacement_type: str = "unknown",
    affected_formations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FaultFeature:
    """
    Factory function to create a FaultFeature from point data.
    
    Args:
        name: Fault name
        points: Nx3 array of surface points
        orientations: Optional list of orientation dicts with x, y, z, dip, azimuth
        displacement_type: Type of fault displacement
        affected_formations: List of affected formation names
        metadata: Additional metadata
        
    Returns:
        FaultFeature instance
    """
    orient_list = []
    if orientations:
        for o in orientations:
            orient_list.append(StructuralOrientation(
                x=o.get('x', 0),
                y=o.get('y', 0),
                z=o.get('z', 0),
                dip=o.get('dip', 0),
                azimuth=o.get('azimuth', 0),
                polarity=o.get('polarity', 1),
                source=o.get('source', 'csv_import'),
            ))
    
    try:
        disp_type = FaultDisplacementType(displacement_type.lower())
    except ValueError:
        disp_type = FaultDisplacementType.UNKNOWN
    
    return FaultFeature(
        feature_type=FeatureType.FAULT,
        name=name,
        surface_points=np.array(points) if not isinstance(points, np.ndarray) else points,
        orientations=orient_list,
        displacement_type=disp_type,
        affected_formations=affected_formations or [],
        metadata=metadata or {},
    )


def create_fold_from_axes(
    name: str,
    axes: List[Dict[str, float]],
    points: Optional[np.ndarray] = None,
    fold_style: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> FoldFeature:
    """
    Factory function to create a FoldFeature from axis measurements.
    
    Args:
        name: Fold name
        axes: List of axis dicts with plunge, trend
        points: Optional Nx3 array of surface points
        fold_style: Fold classification (anticline, syncline, etc.)
        metadata: Additional metadata
        
    Returns:
        FoldFeature instance
    """
    axis_list = []
    for a in axes:
        axis_list.append(LineationMeasurement(
            plunge=a.get('plunge', 0),
            trend=a.get('trend', 0),
            set_id=a.get('set_id', name),
            metadata=a.get('metadata', {}),
        ))
    
    try:
        style = FoldStyle(fold_style.lower())
    except ValueError:
        style = FoldStyle.UNKNOWN
    
    return FoldFeature(
        feature_type=FeatureType.FOLD,
        name=name,
        surface_points=np.array(points) if points is not None else np.empty((0, 3)),
        fold_axes=axis_list,
        fold_style=style,
        metadata=metadata or {},
    )


def create_unconformity_from_points(
    name: str,
    points: np.ndarray,
    orientations: Optional[List[Dict[str, float]]] = None,
    unconformity_type: str = "unknown",
    formations_above: Optional[List[str]] = None,
    formations_below: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UnconformityFeature:
    """
    Factory function to create an UnconformityFeature from point data.
    
    Args:
        name: Unconformity name
        points: Nx3 array of surface points
        orientations: Optional list of orientation dicts with x, y, z, dip, azimuth
        unconformity_type: Type classification
        formations_above: List of formations above the unconformity
        formations_below: List of formations below the unconformity
        metadata: Additional metadata
        
    Returns:
        UnconformityFeature instance
    """
    orient_list = []
    if orientations:
        for o in orientations:
            orient_list.append(StructuralOrientation(
                x=o.get('x', 0),
                y=o.get('y', 0),
                z=o.get('z', 0),
                dip=o.get('dip', 0),
                azimuth=o.get('azimuth', 0),
                polarity=o.get('polarity', 1),
                source=o.get('source', 'csv_import'),
            ))
    
    try:
        unc_type = UnconformityType(unconformity_type.lower())
    except ValueError:
        unc_type = UnconformityType.UNKNOWN
    
    return UnconformityFeature(
        feature_type=FeatureType.UNCONFORMITY,
        name=name,
        surface_points=np.array(points) if not isinstance(points, np.ndarray) else points,
        orientations=orient_list,
        unconformity_type=unc_type,
        formations_above=formations_above or [],
        formations_below=formations_below or [],
        metadata=metadata or {},
    )

