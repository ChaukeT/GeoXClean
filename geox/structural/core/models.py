"""
Structural Analysis Data Models.

Typed inputs/outputs and audit bundles for deterministic, reproducible analysis.

GeoX Invariant Compliance:
- All results include provenance metadata
- Parameters and filters are recorded
- Engine version tracked for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# ENUMERATIONS
# =============================================================================

class NetType(str, Enum):
    """Stereonet projection type."""
    SCHMIDT = "schmidt"  # Equal-area projection
    WULFF = "wulff"      # Equal-angle projection


class Hemisphere(str, Enum):
    """Stereonet hemisphere."""
    LOWER = "lower"
    UPPER = "upper"


class FailureMode(str, Enum):
    """Kinematic failure mode."""
    PLANAR = "planar"
    WEDGE = "wedge"
    TOPPLING = "toppling"
    FLEXURAL_TOPPLING = "flexural_toppling"
    BLOCK_TOPPLING = "block_toppling"


class WeightingMode(str, Enum):
    """Rose diagram weighting mode."""
    COUNT = "count"          # Simple count per bin
    LENGTH = "length"        # Trace length weighted
    AREA = "area"            # Area weighted (from surfaces)
    PERSISTENCE = "persistence"  # Persistence weighted


class FeatureType(str, Enum):
    """Structural feature type."""
    BEDDING = "bedding"
    JOINT = "joint"
    FAULT = "fault"
    FOLIATION = "foliation"
    CLEAVAGE = "cleavage"
    VEIN = "vein"
    SHEAR = "shear"
    CONTACT = "contact"
    UNKNOWN = "unknown"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class OrientationData:
    """
    Container for orientation measurements.
    
    Core representation is ALWAYS unit vectors (Nx3).
    Angles are derived properties, not stored.
    
    Attributes:
        normals: Nx3 array of unit normal vectors (plane poles)
        positions: Optional Nx3 array of measurement positions (x, y, z)
        feature_types: Optional list of feature type strings
        weights: Optional array of weights for each measurement
        metadata: Dict mapping measurement index to metadata
        domain_ids: Optional array of domain identifiers
        set_ids: Optional array of structural set identifiers
    """
    normals: np.ndarray  # Shape (N, 3), unit vectors
    positions: Optional[np.ndarray] = None  # Shape (N, 3)
    feature_types: Optional[List[str]] = None
    weights: Optional[np.ndarray] = None
    metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    domain_ids: Optional[np.ndarray] = None
    set_ids: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and normalize after initialization."""
        # Ensure normals is 2D
        if self.normals.ndim == 1:
            self.normals = self.normals.reshape(1, -1)
        
        # Validate shape
        if self.normals.shape[1] != 3:
            raise ValueError(f"Normals must have 3 columns, got {self.normals.shape[1]}")
        
        # Normalize vectors
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.normals = self.normals / norms
    
    @property
    def n_measurements(self) -> int:
        """Number of measurements."""
        return len(self.normals)
    
    @property
    def dips(self) -> np.ndarray:
        """Derive dip angles from normals."""
        from .orientation_math import normal_to_dip_dipdir
        dips, _ = normal_to_dip_dipdir(self.normals)
        return dips
    
    @property
    def dip_directions(self) -> np.ndarray:
        """Derive dip directions from normals."""
        from .orientation_math import normal_to_dip_dipdir
        _, dip_dirs = normal_to_dip_dipdir(self.normals)
        return dip_dirs
    
    def filter_by_domain(self, domain_id: Any) -> "OrientationData":
        """Return filtered OrientationData for a specific domain."""
        if self.domain_ids is None:
            return self
        
        mask = self.domain_ids == domain_id
        return OrientationData(
            normals=self.normals[mask],
            positions=self.positions[mask] if self.positions is not None else None,
            feature_types=[self.feature_types[i] for i, m in enumerate(mask) if m] if self.feature_types else None,
            weights=self.weights[mask] if self.weights is not None else None,
            metadata={i: self.metadata[orig_i] for i, (orig_i, m) in enumerate(zip(range(len(mask)), mask)) if m and orig_i in self.metadata},
            domain_ids=self.domain_ids[mask],
            set_ids=self.set_ids[mask] if self.set_ids is not None else None,
        )
    
    def filter_by_feature_type(self, feature_type: Union[str, FeatureType]) -> "OrientationData":
        """Return filtered OrientationData for a specific feature type."""
        if self.feature_types is None:
            return self
        
        ft_str = feature_type.value if isinstance(feature_type, FeatureType) else feature_type
        mask = np.array([ft == ft_str for ft in self.feature_types])
        
        return OrientationData(
            normals=self.normals[mask],
            positions=self.positions[mask] if self.positions is not None else None,
            feature_types=[ft for ft, m in zip(self.feature_types, mask) if m],
            weights=self.weights[mask] if self.weights is not None else None,
            metadata={i: self.metadata[orig_i] for i, (orig_i, m) in enumerate(zip(range(len(mask)), mask)) if m and orig_i in self.metadata},
            domain_ids=self.domain_ids[mask] if self.domain_ids is not None else None,
            set_ids=self.set_ids[mask] if self.set_ids is not None else None,
        )
    
    @classmethod
    def from_dip_dipdir(
        cls,
        dips: np.ndarray,
        dip_directions: np.ndarray,
        **kwargs
    ) -> "OrientationData":
        """Create OrientationData from dip/dip-direction angles."""
        from .orientation_math import dip_dipdir_to_normal
        normals = dip_dipdir_to_normal(dips, dip_directions)
        return cls(normals=normals, **kwargs)
    
    @classmethod
    def from_alpha_beta(
        cls,
        alphas: np.ndarray,
        betas: np.ndarray,
        hole_azimuths: np.ndarray,
        hole_dips: np.ndarray,
        **kwargs
    ) -> "OrientationData":
        """Create OrientationData from alpha/beta measurements."""
        from .orientation_math import alpha_beta_to_normal
        normals = alpha_beta_to_normal(alphas, betas, hole_azimuths, hole_dips)
        return cls(normals=normals, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "normals": self.normals.tolist(),
            "positions": self.positions.tolist() if self.positions is not None else None,
            "feature_types": self.feature_types,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "domain_ids": self.domain_ids.tolist() if self.domain_ids is not None else None,
            "set_ids": self.set_ids.tolist() if self.set_ids is not None else None,
            "n_measurements": self.n_measurements,
        }


@dataclass
class StructuralSet:
    """
    A structural set (cluster) of orientations.
    
    Attributes:
        set_id: Unique identifier for this set
        normals: Nx3 array of unit normals belonging to this set
        mean_normal: Unit vector mean direction
        kappa: Fisher concentration parameter
        confidence_cone_95: 95% confidence cone half-angle (degrees)
        dispersion: Angular dispersion (degrees)
        n_members: Number of measurements in set
        metadata: Additional set metadata
    """
    set_id: str
    normals: np.ndarray
    mean_normal: np.ndarray
    kappa: float
    confidence_cone_95: float
    dispersion: float
    n_members: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean_dip(self) -> float:
        """Mean dip angle."""
        from .orientation_math import normal_to_dip_dipdir
        dip, _ = normal_to_dip_dipdir(self.mean_normal.reshape(1, 3))
        return float(dip[0])
    
    @property
    def mean_dip_direction(self) -> float:
        """Mean dip direction."""
        from .orientation_math import normal_to_dip_dipdir
        _, dipdir = normal_to_dip_dipdir(self.mean_normal.reshape(1, 3))
        return float(dipdir[0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "set_id": self.set_id,
            "mean_normal": self.mean_normal.tolist(),
            "mean_dip": self.mean_dip,
            "mean_dip_direction": self.mean_dip_direction,
            "kappa": self.kappa,
            "confidence_cone_95": self.confidence_cone_95,
            "dispersion": self.dispersion,
            "n_members": self.n_members,
            "metadata": self.metadata,
        }


# =============================================================================
# RESULT BUNDLES
# =============================================================================

@dataclass
class StereonetResult:
    """
    Result of stereonet analysis.
    
    Contains projected coordinates and optional density grid.
    """
    # Projected coordinates
    x: np.ndarray  # Projected x coordinates
    y: np.ndarray  # Projected y coordinates
    
    # Configuration
    net_type: NetType
    hemisphere: Hemisphere
    show_planes: bool = False
    
    # Optional density contours
    density_grid: Optional[np.ndarray] = None
    density_x: Optional[np.ndarray] = None
    density_y: Optional[np.ndarray] = None
    density_levels: Optional[np.ndarray] = None
    
    # Great circles for planes
    great_circles: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    
    # Statistics
    n_points: int = 0
    
    # Audit
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "net_type": self.net_type.value,
            "hemisphere": self.hemisphere.value,
            "show_planes": self.show_planes,
            "has_density": self.density_grid is not None,
            "n_points": self.n_points,
            "parameters": self.parameters,
        }


@dataclass
class RoseResult:
    """
    Result of rose diagram analysis.
    
    Contains histogram bins and optional statistics.
    """
    # Histogram data
    bin_edges: np.ndarray  # Bin edges in degrees (0-360 or 0-180 for axial)
    bin_centers: np.ndarray  # Bin centers in degrees
    counts: np.ndarray  # Counts or weighted sums per bin
    
    # Configuration
    n_bins: int
    weighting: WeightingMode
    is_axial: bool  # True for bidirectional/axial data
    
    # Statistics
    mean_direction: float  # Mean direction in degrees
    mean_resultant_length: float  # 0-1, measure of concentration
    circular_variance: float  # 0-1
    
    # Audit
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_centers": self.bin_centers.tolist(),
            "counts": self.counts.tolist(),
            "n_bins": self.n_bins,
            "weighting": self.weighting.value,
            "is_axial": self.is_axial,
            "mean_direction": self.mean_direction,
            "mean_resultant_length": self.mean_resultant_length,
            "circular_variance": self.circular_variance,
            "parameters": self.parameters,
        }


@dataclass
class KinematicFeasibility:
    """Result for a single failure mode check."""
    is_feasible: bool
    failure_mode: FailureMode
    
    # Detailed results
    daylight_condition: bool  # Does the plane/line daylight on the slope?
    friction_condition: bool  # Is dip > friction angle?
    lateral_condition: bool   # Within lateral limits?
    
    # Angular margins (positive = within envelope, negative = outside)
    daylight_margin_deg: float
    friction_margin_deg: float
    lateral_margin_deg: Optional[float] = None
    
    # Limiting vectors (for visualization)
    limiting_vectors: Optional[Dict[str, np.ndarray]] = None
    
    # Input measurement
    measurement_index: Optional[int] = None
    measurement_normal: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_feasible": self.is_feasible,
            "failure_mode": self.failure_mode.value,
            "daylight_condition": self.daylight_condition,
            "friction_condition": self.friction_condition,
            "lateral_condition": self.lateral_condition,
            "daylight_margin_deg": self.daylight_margin_deg,
            "friction_margin_deg": self.friction_margin_deg,
            "lateral_margin_deg": self.lateral_margin_deg,
            "measurement_index": self.measurement_index,
        }


@dataclass
class KinematicResult:
    """
    Complete kinematic analysis result.
    """
    # Slope parameters used
    slope_dip: float
    slope_dip_direction: float
    slope_normal: np.ndarray
    friction_angle: float
    lateral_limits: Optional[float] = None  # Degrees either side of dip direction
    tension_cutoff: Optional[float] = None  # Degrees from vertical
    
    # Results by failure mode
    planar_results: List[KinematicFeasibility] = field(default_factory=list)
    wedge_results: List[KinematicFeasibility] = field(default_factory=list)
    toppling_results: List[KinematicFeasibility] = field(default_factory=list)
    
    # Summary statistics
    n_planar_feasible: int = 0
    n_wedge_feasible: int = 0
    n_toppling_feasible: int = 0
    n_total_measurements: int = 0
    
    # Envelopes for plotting
    daylight_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None
    friction_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None
    toppling_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Audit
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def planar_feasible_fraction(self) -> float:
        """Fraction of measurements feasible for planar sliding."""
        if self.n_total_measurements == 0:
            return 0.0
        return self.n_planar_feasible / self.n_total_measurements
    
    @property
    def wedge_feasible_fraction(self) -> float:
        """Fraction of measurement pairs feasible for wedge sliding."""
        n_pairs = len(self.wedge_results)
        if n_pairs == 0:
            return 0.0
        return self.n_wedge_feasible / n_pairs
    
    @property
    def toppling_feasible_fraction(self) -> float:
        """Fraction of measurements feasible for toppling."""
        if self.n_total_measurements == 0:
            return 0.0
        return self.n_toppling_feasible / self.n_total_measurements
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "slope_dip": self.slope_dip,
            "slope_dip_direction": self.slope_dip_direction,
            "friction_angle": self.friction_angle,
            "lateral_limits": self.lateral_limits,
            "tension_cutoff": self.tension_cutoff,
            "n_planar_feasible": self.n_planar_feasible,
            "n_wedge_feasible": self.n_wedge_feasible,
            "n_toppling_feasible": self.n_toppling_feasible,
            "n_total_measurements": self.n_total_measurements,
            "planar_feasible_fraction": self.planar_feasible_fraction,
            "wedge_feasible_fraction": self.wedge_feasible_fraction,
            "toppling_feasible_fraction": self.toppling_feasible_fraction,
            "planar_results": [r.to_dict() for r in self.planar_results],
            "wedge_results": [r.to_dict() for r in self.wedge_results],
            "toppling_results": [r.to_dict() for r in self.toppling_results],
            "parameters": self.parameters,
        }


@dataclass
class AnalysisBundle:
    """
    Complete analysis bundle with audit trail.
    
    Every analysis run produces this bundle for reproducibility.
    """
    # Analysis identification
    analysis_id: str
    analysis_type: str  # "stereonet", "rose", "kinematic", "clustering"
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Input data summary
    n_measurements: int = 0
    input_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Results (one of these will be populated)
    stereonet_result: Optional[StereonetResult] = None
    rose_result: Optional[RoseResult] = None
    kinematic_result: Optional[KinematicResult] = None
    structural_sets: Optional[List[StructuralSet]] = None
    
    # Audit metadata
    engine_version: str = "1.0.0"
    random_seed: Optional[int] = None
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete bundle to dictionary for JSON export."""
        result = {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp.isoformat(),
            "n_measurements": self.n_measurements,
            "input_filters": self.input_filters,
            "parameters": self.parameters,
            "engine_version": self.engine_version,
            "random_seed": self.random_seed,
            "execution_time_ms": self.execution_time_ms,
        }
        
        if self.stereonet_result:
            result["stereonet_result"] = self.stereonet_result.to_dict()
        if self.rose_result:
            result["rose_result"] = self.rose_result.to_dict()
        if self.kinematic_result:
            result["kinematic_result"] = self.kinematic_result.to_dict()
        if self.structural_sets:
            result["structural_sets"] = [s.to_dict() for s in self.structural_sets]
        
        return result
    
    def to_json(self) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

