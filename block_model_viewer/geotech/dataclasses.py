"""
Geotechnical data structures and dataclasses.

Core entities for rock-mass properties, stope stability, and slope risk analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class RockMassPoint:
    """
    Single point measurement of rock-mass properties.
    
    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        rqd: Rock Quality Designation (0-100)
        q: Q-value (Barton's Q)
        rmr: Rock Mass Rating (0-100)
        gsi: Geological Strength Index (0-100)
        domain: Rock domain identifier (string)
        confidence: Measurement confidence (0-1)
    """
    x: float
    y: float
    z: float
    rqd: Optional[float] = None
    q: Optional[float] = None
    rmr: Optional[float] = None
    gsi: Optional[float] = None
    domain: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'rqd': self.rqd,
            'q': self.q,
            'rmr': self.rmr,
            'gsi': self.gsi,
            'domain': self.domain,
            'confidence': self.confidence
        }


@dataclass
class RockMassGrid:
    """
    3D grid of rock-mass properties aligned with block model.
    
    Attributes:
        grid_definition: Grid definition dict (same structure as BlockModel)
        rqd: RQD array (n_blocks,)
        q: Q-value array (n_blocks,)
        rmr: RMR array (n_blocks,)
        gsi: GSI array (n_blocks,)
        quality_category: Derived quality category codes (n_blocks,)
    """
    grid_definition: Dict[str, Any]
    rqd: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    rmr: Optional[np.ndarray] = None
    gsi: Optional[np.ndarray] = None
    quality_category: Optional[np.ndarray] = None
    
    def get_property(self, name: str) -> Optional[np.ndarray]:
        """Get property array by name."""
        prop_map = {
            'RQD': self.rqd,
            'Q': self.q,
            'RMR': self.rmr,
            'GSI': self.gsi,
            'QUALITY': self.quality_category
        }
        return prop_map.get(name.upper())
    
    def compute_quality_categories(self) -> np.ndarray:
        """
        Compute quality category codes based on RMR.
        
        Categories:
        0: Very Poor (RMR < 20)
        1: Poor (20 <= RMR < 40)
        2: Fair (40 <= RMR < 60)
        3: Good (60 <= RMR < 80)
        4: Very Good (RMR >= 80)
        """
        if self.rmr is None:
            return np.zeros(self.grid_definition.get('n_blocks', 0), dtype=np.int8)
        
        categories = np.zeros_like(self.rmr, dtype=np.int8)
        categories[self.rmr >= 80] = 4  # Very Good
        categories[(self.rmr >= 60) & (self.rmr < 80)] = 3  # Good
        categories[(self.rmr >= 40) & (self.rmr < 60)] = 2  # Fair
        categories[(self.rmr >= 20) & (self.rmr < 40)] = 1  # Poor
        categories[self.rmr < 20] = 0  # Very Poor
        
        self.quality_category = categories
        return categories


@dataclass
class StopeStabilityInput:
    """
    Input parameters for Mathews Stability Graph analysis.
    
    Attributes:
        span: Stope span (m)
        height: Stope height (m)
        hydraulic_radius: Hydraulic radius = span * height / (2 * (span + height))
        q_prime: Modified Q-value (Q')
        stress_factor: Stress factor (A)
        joint_orientation_factor: Joint orientation factor (B)
        gravity_factor: Gravity factor (C)
        dilution_allowance: Dilution allowance factor
        rock_mass_properties: Optional rock mass properties dict
    """
    span: float
    height: float
    hydraulic_radius: Optional[float] = None
    q_prime: float = 1.0
    stress_factor: float = 1.0
    joint_orientation_factor: float = 1.0
    gravity_factor: float = 1.0
    dilution_allowance: float = 0.0
    rock_mass_properties: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Compute hydraulic radius if not provided."""
        if self.hydraulic_radius is None:
            if self.span > 0 and self.height > 0:
                self.hydraulic_radius = (self.span * self.height) / (2.0 * (self.span + self.height))
            else:
                self.hydraulic_radius = 0.0


@dataclass
class StopeStabilityResult:
    """
    Result of Mathews Stability Graph analysis.
    
    Attributes:
        stability_number: Computed stability number N
        factor_of_safety: Approximate factor of safety
        probability_of_instability: Probability of instability (0-1)
        stability_class: Classification (Stable, Transition, Caving)
        recommended_support_class: Recommended support class
        notes: Additional notes or warnings
    """
    stability_number: float
    factor_of_safety: float
    probability_of_instability: float
    stability_class: str
    recommended_support_class: str
    notes: str = ""
    
    STABILITY_CLASSES = {
        'STABLE': 'Stable',
        'TRANSITION': 'Transition',
        'CAVING': 'Caving/Overbreak'
    }
    
    SUPPORT_CLASSES = {
        'NONE': 'No Support',
        'LIGHT': 'Light Support',
        'MODERATE': 'Moderate Support',
        'HEAVY': 'Heavy Support'
    }


@dataclass
class SlopeRiskInput:
    """
    Input parameters for slope risk assessment.
    
    Attributes:
        bench_height: Bench height (m)
        overall_slope_angle: Overall slope angle (degrees)
        pit_wall_orientation: Wall orientation (azimuth, degrees)
        rock_mass_properties: Rock mass properties dict (RMR, Q, etc.)
        water_present: Water/infill flag
        structural_features: Structural feature flags (simplified)
    """
    bench_height: float
    overall_slope_angle: float
    pit_wall_orientation: float
    rock_mass_properties: Dict[str, float]
    water_present: bool = False
    structural_features: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SlopeRiskResult:
    """
    Result of slope risk assessment.
    
    Attributes:
        risk_index: Computed risk index (0-100, higher = more risk)
        qualitative_class: Qualitative risk class (Low, Moderate, High, Very High)
        probability_of_failure: Indicative probability of failure (0-1)
        notes: Additional notes or recommendations
    """
    risk_index: float
    qualitative_class: str
    probability_of_failure: float
    notes: str = ""
    
    RISK_CLASSES = {
        'LOW': 'Low Risk',
        'MODERATE': 'Moderate Risk',
        'HIGH': 'High Risk',
        'VERY_HIGH': 'Very High Risk'
    }


@dataclass
class GeotechMCResult:
    """
    Result of Monte Carlo geotechnical analysis.
    
    Attributes:
        input_params: Original input parameters
        n_realizations: Number of realizations
        stability_numbers: Array of stability numbers (n_realizations,)
        risk_indices: Array of risk indices (n_realizations,)
        stability_classes: Array of stability class strings
        risk_classes: Array of risk class strings
        summary_stats: Summary statistics dict
        exceedance_curves: Exceedance curve data
    """
    input_params: Dict[str, Any]
    n_realizations: int
    stability_numbers: Optional[np.ndarray] = None
    risk_indices: Optional[np.ndarray] = None
    stability_classes: Optional[List[str]] = None
    risk_classes: Optional[List[str]] = None
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    exceedance_curves: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from realizations."""
        stats = {}
        
        if self.stability_numbers is not None:
            stats['stability_number'] = {
                'mean': float(np.mean(self.stability_numbers)),
                'std': float(np.std(self.stability_numbers)),
                'min': float(np.min(self.stability_numbers)),
                'max': float(np.max(self.stability_numbers)),
                'p10': float(np.percentile(self.stability_numbers, 10)),
                'p50': float(np.percentile(self.stability_numbers, 50)),
                'p90': float(np.percentile(self.stability_numbers, 90))
            }
        
        if self.risk_indices is not None:
            stats['risk_index'] = {
                'mean': float(np.mean(self.risk_indices)),
                'std': float(np.std(self.risk_indices)),
                'min': float(np.min(self.risk_indices)),
                'max': float(np.max(self.risk_indices)),
                'p10': float(np.percentile(self.risk_indices, 10)),
                'p50': float(np.percentile(self.risk_indices, 50)),
                'p90': float(np.percentile(self.risk_indices, 90))
            }
        
        # Class distributions
        if self.stability_classes:
            from collections import Counter
            class_counts = Counter(self.stability_classes)
            stats['stability_class_distribution'] = {
                k: v / len(self.stability_classes) 
                for k, v in class_counts.items()
            }
        
        if self.risk_classes:
            from collections import Counter
            class_counts = Counter(self.risk_classes)
            stats['risk_class_distribution'] = {
                k: v / len(self.risk_classes)
                for k, v in class_counts.items()
            }
        
        self.summary_stats = stats
        return stats

