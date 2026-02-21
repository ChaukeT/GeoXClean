"""
Structural Datasets - Planar and linear structural measurements.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class PlaneMeasurement:
    """Planar structural measurement (e.g., joint, fault, bedding)."""
    dip: float  # Dip angle in degrees (0-90)
    dip_direction: float  # Dip direction in degrees (0-360, azimuth)
    set_id: Optional[str] = None  # Optional structural set identifier
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineationMeasurement:
    """Linear structural measurement (e.g., lineation, fold axis)."""
    plunge: float  # Plunge angle in degrees (0-90)
    trend: float  # Trend in degrees (0-360, azimuth)
    set_id: Optional[str] = None  # Optional structural set identifier
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuralDataset:
    """Container for structural measurements."""
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize structural dataset.
        
        Args:
            metadata: Optional metadata dictionary
        """
        self.planes: List[PlaneMeasurement] = []
        self.lineations: List[LineationMeasurement] = []
        self.metadata: Dict[str, Any] = metadata or {}
    
    def add_plane(self, dip: float, dip_direction: float, set_id: Optional[str] = None, **metadata):
        """Add a plane measurement."""
        self.planes.append(PlaneMeasurement(
            dip=dip,
            dip_direction=dip_direction,
            set_id=set_id,
            metadata=metadata
        ))
    
    def add_lineation(self, plunge: float, trend: float, set_id: Optional[str] = None, **metadata):
        """Add a lineation measurement."""
        self.lineations.append(LineationMeasurement(
            plunge=plunge,
            trend=trend,
            set_id=set_id,
            metadata=metadata
        ))
    
    def get_planes_by_set(self, set_id: str) -> List[PlaneMeasurement]:
        """Get all planes in a specific set."""
        return [p for p in self.planes if p.set_id == set_id]
    
    def get_lineations_by_set(self, set_id: str) -> List[LineationMeasurement]:
        """Get all lineations in a specific set."""
        return [l for l in self.lineations if l.set_id == set_id]

