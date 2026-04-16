"""
Fragment Dataset Data Model
============================

Core data structures for the Fragmentation Analysis Module.
A FragmentDataset represents one analysed muck pile survey from drone-derived
LiDAR + RGB data. It holds all fragments detected from one scan session and
is the unit of persistence, provenance, and export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SegmentationMethod(Enum):
    """Source segmentation method for a fragment."""
    GEOMETRIC = "geometric"
    WATERSHED = "watershed"
    DL = "deep_learning"
    HYBRID = "hybrid"


class ValidationStatus(Enum):
    """Dataset validation lifecycle state."""
    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"


# ---------------------------------------------------------------------------
# Fragment Size Distribution
# ---------------------------------------------------------------------------

@dataclass
class FSD:
    """
    Fragment Size Distribution — cumulative passing curve.

    Stores the sorted equivalent diameters and cumulative passing fractions,
    plus fitted distribution parameters.
    """
    diameters_sorted: np.ndarray  # Sorted equiv diameters (ascending), m
    cumulative_passing: np.ndarray  # Fraction passing at each diameter [0..1]

    # Key percentiles (metres)
    d10: float = 0.0
    d50: float = 0.0
    d80: float = 0.0
    d90: float = 0.0

    # Rosin-Rammler fit: P(x) = 1 - exp(-(x / x_c)^n)
    rr_n: Optional[float] = None
    rr_xc: Optional[float] = None

    # Swebrec (Ouchterlony) fit parameters
    swebrec_xmax: Optional[float] = None
    swebrec_x50: Optional[float] = None
    swebrec_b: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diameters_sorted": self.diameters_sorted.tolist(),
            "cumulative_passing": self.cumulative_passing.tolist(),
            "d10": self.d10, "d50": self.d50,
            "d80": self.d80, "d90": self.d90,
            "rr_n": self.rr_n, "rr_xc": self.rr_xc,
            "swebrec_xmax": self.swebrec_xmax,
            "swebrec_x50": self.swebrec_x50,
            "swebrec_b": self.swebrec_b,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FSD:
        return cls(
            diameters_sorted=np.array(d["diameters_sorted"]),
            cumulative_passing=np.array(d["cumulative_passing"]),
            d10=d.get("d10", 0.0), d50=d.get("d50", 0.0),
            d80=d.get("d80", 0.0), d90=d.get("d90", 0.0),
            rr_n=d.get("rr_n"), rr_xc=d.get("rr_xc"),
            swebrec_xmax=d.get("swebrec_xmax"),
            swebrec_x50=d.get("swebrec_x50"),
            swebrec_b=d.get("swebrec_b"),
        )


# ---------------------------------------------------------------------------
# Oriented Bounding Box
# ---------------------------------------------------------------------------

@dataclass
class OBB:
    """Oriented bounding box (PCA-aligned)."""
    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) — rows are principal axes
    half_extents: np.ndarray  # (3,) — half-lengths along each axis

    @property
    def volume(self) -> float:
        return float(8.0 * np.prod(self.half_extents))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "half_extents": self.half_extents.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OBB:
        return cls(
            center=np.array(d["center"]),
            axes=np.array(d["axes"]),
            half_extents=np.array(d["half_extents"]),
        )


# ---------------------------------------------------------------------------
# Fragment Record
# ---------------------------------------------------------------------------

@dataclass
class FragmentRecord:
    """
    Individual rock fragment with geometry and derived metrics.

    Each record stores the fragment's spatial footprint, size metrics, shape
    descriptors, and segmentation provenance.
    """
    fragment_id: int
    centroid_xyz: np.ndarray  # (3,) — 3D centroid

    # Point membership
    point_indices: np.ndarray  # Indices into fused_cloud array

    # Geometry
    bbox_3d: Optional[OBB] = None
    mask_polygon: Optional[Any] = None  # shapely Polygon (2D projected outline)

    # Size metrics (metres / m^2 / m^3)
    equiv_diameter: float = 0.0  # From convex hull volume
    feret_max: float = 0.0
    feret_min: float = 0.0
    projected_area: float = 0.0  # 2D projected area
    volume_estimate: float = 0.0  # Convex hull volume
    surface_area: float = 0.0

    # Shape descriptors
    aspect_ratio: float = 1.0  # feret_max / feret_min
    sphericity: float = 0.0  # Wadell sphericity index
    elongation: float = 1.0

    # Provenance
    source_method: SegmentationMethod = SegmentationMethod.GEOMETRIC
    confidence: float = 0.0  # [0..1]

    @property
    def point_count(self) -> int:
        return len(self.point_indices)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fragment_id": self.fragment_id,
            "centroid_xyz": self.centroid_xyz.tolist(),
            "point_indices_count": len(self.point_indices),
            "equiv_diameter": self.equiv_diameter,
            "feret_max": self.feret_max,
            "feret_min": self.feret_min,
            "projected_area": self.projected_area,
            "volume_estimate": self.volume_estimate,
            "surface_area": self.surface_area,
            "aspect_ratio": self.aspect_ratio,
            "sphericity": self.sphericity,
            "elongation": self.elongation,
            "source_method": self.source_method.value,
            "confidence": self.confidence,
        }
        if self.bbox_3d is not None:
            d["bbox_3d"] = self.bbox_3d.to_dict()
        return d


# ---------------------------------------------------------------------------
# Provenance Log
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceEntry:
    """Single provenance record for an operation in the fragmentation pipeline."""
    operation: str  # e.g. 'statistical_outlier_removal', 'run_watershed'
    parameters: Dict[str, Any]
    input_hash: Optional[str] = None  # SHA-256 of input array
    output_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    software_version: str = "GeoX 1.0"
    random_seed: Optional[int] = None
    # DL model metadata (if applicable)
    model_file_hash: Optional[str] = None
    model_training_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "operation": self.operation,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "software_version": self.software_version,
            "output_summary": self.output_summary,
        }
        if self.input_hash:
            d["input_hash"] = self.input_hash
        if self.random_seed is not None:
            d["random_seed"] = self.random_seed
        if self.model_file_hash:
            d["model_file_hash"] = self.model_file_hash
        if self.model_training_metadata:
            d["model_training_metadata"] = self.model_training_metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProvenanceEntry:
        return cls(
            operation=d["operation"],
            parameters=d["parameters"],
            input_hash=d.get("input_hash"),
            output_summary=d.get("output_summary", {}),
            timestamp=d.get("timestamp", ""),
            software_version=d.get("software_version", "GeoX 1.0"),
            random_seed=d.get("random_seed"),
            model_file_hash=d.get("model_file_hash"),
            model_training_metadata=d.get("model_training_metadata"),
        )


@dataclass
class ProvenanceLog:
    """Full provenance chain for a FragmentDataset."""
    entries: List[ProvenanceEntry] = field(default_factory=list)

    def record(
        self,
        operation: str,
        parameters: Dict[str, Any],
        input_array: Optional[np.ndarray] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        model_file_hash: Optional[str] = None,
        model_training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a pipeline operation."""
        input_hash = None
        if input_array is not None:
            input_hash = hashlib.sha256(input_array.tobytes()[:1_000_000]).hexdigest()

        entry = ProvenanceEntry(
            operation=operation,
            parameters=parameters,
            input_hash=input_hash,
            output_summary=output_summary or {},
            random_seed=random_seed,
            model_file_hash=model_file_hash,
            model_training_metadata=model_training_metadata,
        )
        self.entries.append(entry)
        logger.info("Provenance: %s (%d params)", operation, len(parameters))

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> ProvenanceLog:
        return cls(entries=[ProvenanceEntry.from_dict(d) for d in data])

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# FragmentDataset
# ---------------------------------------------------------------------------

@dataclass
class FragmentDataset:
    """
    Central data structure for one analysed muck pile survey.

    Holds the fused point cloud, all detected fragments, global FSD,
    spatial grid, and full provenance chain.
    """
    # Identity
    dataset_id: UUID = field(default_factory=uuid4)
    name: str = ""  # e.g. 'Lift2_Blast_2026-03-15'

    # Source files (never modified — read-only references)
    source_lidar_path: Optional[Path] = None
    source_rgb_paths: List[Path] = field(default_factory=list)

    # Coordinate system
    crs: Optional[str] = None  # EPSG code string, e.g. "EPSG:32633"
    acquisition_date: Optional[datetime] = None

    # Fused point cloud: columns = X,Y,Z,Nx,Ny,Nz,R,G,B,Curvature,Intensity
    fused_cloud: Optional[np.ndarray] = None

    # Segmentation results
    fragment_labels: Optional[np.ndarray] = None  # (N,) per-point labels, -1 = noise
    fragments: List[FragmentRecord] = field(default_factory=list)

    # Size distribution
    fsd_global: Optional[FSD] = None

    # Spatial grid (standard GeoX BlockModel or 2D grid)
    spatial_grid: Optional[Any] = None  # BlockModel instance

    # Provenance
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
    validation_status: ValidationStatus = ValidationStatus.DRAFT

    # --- Convenience properties ---

    @property
    def point_count(self) -> int:
        if self.fused_cloud is None:
            return 0
        return self.fused_cloud.shape[0]

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    @property
    def has_fused_cloud(self) -> bool:
        return self.fused_cloud is not None and self.fused_cloud.shape[0] > 0

    @property
    def has_fragments(self) -> bool:
        return len(self.fragments) > 0

    @property
    def has_spatial_grid(self) -> bool:
        return self.spatial_grid is not None

    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Axis-aligned bounding box of fused cloud (min_xyz, max_xyz)."""
        if self.fused_cloud is None:
            return None
        xyz = self.fused_cloud[:, :3]
        return xyz.min(axis=0), xyz.max(axis=0)

    def get_fragment(self, fragment_id: int) -> Optional[FragmentRecord]:
        """Lookup a fragment by ID."""
        for f in self.fragments:
            if f.fragment_id == fragment_id:
                return f
        return None

    def summary(self) -> Dict[str, Any]:
        """Quick summary dict for UI display."""
        s: Dict[str, Any] = {
            "dataset_id": str(self.dataset_id),
            "name": self.name,
            "point_count": self.point_count,
            "fragment_count": self.fragment_count,
            "has_spatial_grid": self.has_spatial_grid,
            "validation_status": self.validation_status.value,
            "provenance_steps": len(self.provenance.entries),
        }
        if self.fsd_global:
            s["d50"] = self.fsd_global.d50
            s["d80"] = self.fsd_global.d80
        if self.acquisition_date:
            s["acquisition_date"] = self.acquisition_date.isoformat()
        return s
