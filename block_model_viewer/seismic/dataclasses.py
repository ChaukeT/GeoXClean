"""
Seismic data structures and dataclasses.

Core entities for seismic events, catalogues, hazard volumes, and rockburst analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np


@dataclass
class SeismicEvent:
    """
    Single seismic event record.
    
    Attributes:
        id: Unique event identifier
        time: Event timestamp
        x, y, z: Event coordinates
        magnitude: Magnitude (ML or Mw)
        energy: Seismic energy (Joules, optional)
        moment: Seismic moment (N·m, optional)
        mechanism: Failure mechanism (e.g., "shear", "tensile")
        quality: Data quality/confidence (0-1)
    """
    id: str
    time: datetime
    x: float
    y: float
    z: float
    magnitude: float
    energy: Optional[float] = None
    moment: Optional[float] = None
    mechanism: Optional[str] = None
    quality: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'time': self.time.isoformat() if isinstance(self.time, datetime) else str(self.time),
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'magnitude': self.magnitude,
            'energy': self.energy,
            'moment': self.moment,
            'mechanism': self.mechanism,
            'quality': self.quality
        }


@dataclass
class SeismicCatalogue:
    """
    Collection of seismic events.
    
    Attributes:
        events: List of SeismicEvent instances
        metadata: Additional metadata (source, format, etc.)
    """
    events: List[SeismicEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return number of events."""
        return len(self.events)
    
    def get_coordinates(self) -> np.ndarray:
        """Get array of event coordinates."""
        coords = np.array([[e.x, e.y, e.z] for e in self.events])
        return coords
    
    def get_magnitudes(self) -> np.ndarray:
        """Get array of event magnitudes."""
        mags = np.array([e.magnitude for e in self.events])
        return mags
    
    def get_times(self) -> List[datetime]:
        """Get list of event times."""
        return [e.time for e in self.events]
    
    def get_time_range(self) -> Tuple[datetime, datetime]:
        """Get time range of catalogue."""
        if not self.events:
            raise ValueError("Catalogue is empty")
        times = self.get_times()
        return (min(times), max(times))
    
    def get_spatial_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get spatial bounds as (xmin, xmax, ymin, ymax, zmin, zmax)."""
        if not self.events:
            raise ValueError("Catalogue is empty")
        coords = self.get_coordinates()
        return (
            float(np.min(coords[:, 0])), float(np.max(coords[:, 0])),
            float(np.min(coords[:, 1])), float(np.max(coords[:, 1])),
            float(np.min(coords[:, 2])), float(np.max(coords[:, 2]))
        )


@dataclass
class HazardVolume:
    """
    3D grid of seismic hazard indices.
    
    Attributes:
        grid_definition: Grid definition dict (aligned to block model or independent)
        hazard_index: Hazard index array (n_cells,)
        time_window: Time window tuple (start, end)
        magnitude_range: Magnitude range tuple (min, max)
        metadata: Additional metadata
    """
    grid_definition: Dict[str, Any]
    hazard_index: np.ndarray
    time_window: Tuple[datetime, datetime]
    magnitude_range: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hazard_at_point(self, x: float, y: float, z: float) -> float:
        """
        Get hazard index at a point (interpolated from grid).
        
        Args:
            x, y, z: Point coordinates
            
        Returns:
            Interpolated hazard index
        """
        # Simple nearest-neighbor lookup (can be enhanced with interpolation)
        coords = self.grid_definition.get('coordinates')
        if coords is None:
            return 0.0
        
        # Find nearest grid point
        from scipy.spatial.distance import cdist
        point = np.array([[x, y, z]])
        distances = cdist(point, coords)
        nearest_idx = np.argmin(distances[0])
        
        return float(self.hazard_index[nearest_idx])


@dataclass
class RockburstIndexResult:
    """
    Rockburst index result for a location.
    
    Attributes:
        location: Location tuple (x, y, z)
        index_value: Computed rockburst index (0-1)
        index_class: Classification (Low, Moderate, High, Extreme)
        contributing_events: Number of contributing seismic events
        notes: Additional notes
    """
    location: Tuple[float, float, float]
    index_value: float
    index_class: str
    contributing_events: int
    notes: str = ""
    
    INDEX_CLASSES = {
        'LOW': 'Low',
        'MODERATE': 'Moderate',
        'HIGH': 'High',
        'EXTREME': 'Extreme'
    }
    
    @classmethod
    def classify_index(cls, value: float) -> str:
        """
        Classify rockburst index value.
        
        Args:
            value: Index value (0-1)
            
        Returns:
            Classification string
        """
        if value < 0.25:
            return cls.INDEX_CLASSES['LOW']
        elif value < 0.5:
            return cls.INDEX_CLASSES['MODERATE']
        elif value < 0.75:
            return cls.INDEX_CLASSES['HIGH']
        else:
            return cls.INDEX_CLASSES['EXTREME']


@dataclass
class SeismicMCResult:
    """
    Result of Monte Carlo seismic analysis.
    
    Attributes:
        realisations: List of HazardVolume realisations
        summary_stats: Summary statistics dict
        exceedance_curve: Exceedance curve dict (threshold -> probability)
        n_realizations: Number of realizations
    """
    realisations: List[HazardVolume]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    exceedance_curve: Dict[float, float] = field(default_factory=dict)
    n_realizations: int = 0
    
    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from realisations."""
        if not self.realisations:
            return {}
        
        # Extract hazard indices from all realisations
        all_indices = []
        for vol in self.realisations:
            all_indices.extend(vol.hazard_index.flatten())
        
        all_indices = np.array(all_indices)
        
        stats = {
            'hazard_index': {
                'mean': float(np.mean(all_indices)),
                'std': float(np.std(all_indices)),
                'min': float(np.min(all_indices)),
                'max': float(np.max(all_indices)),
                'p10': float(np.percentile(all_indices, 10)),
                'p50': float(np.percentile(all_indices, 50)),
                'p90': float(np.percentile(all_indices, 90))
            }
        }
        
        self.summary_stats = stats
        return stats
    
    def compute_exceedance_curve(self, thresholds: Optional[List[float]] = None) -> Dict[float, float]:
        """
        Compute exceedance curve for hazard indices.
        
        Args:
            thresholds: Optional list of threshold values
            
        Returns:
            Dict mapping threshold -> exceedance probability
        """
        if not self.realisations:
            return {}
        
        # Extract all hazard indices
        all_indices = []
        for vol in self.realisations:
            all_indices.extend(vol.hazard_index.flatten())
        
        all_indices = np.array(all_indices)
        
        if thresholds is None:
            # Auto-generate thresholds
            thresholds = np.linspace(np.min(all_indices), np.max(all_indices), 50).tolist()
        
        exceedance = {}
        n_total = len(all_indices)
        
        for threshold in thresholds:
            exceedance_prob = np.sum(all_indices >= threshold) / n_total
            exceedance[float(threshold)] = float(exceedance_prob)
        
        self.exceedance_curve = exceedance
        return exceedance

