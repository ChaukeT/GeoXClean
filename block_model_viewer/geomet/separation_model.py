"""
Separation Models (STEP 28)

Translate feed size + liberation + chemistry into recovery & concentrate grade.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .liberation_model import LiberationCurve


@dataclass
class SeparationConfig:
    """
    Separation circuit configuration.
    
    Attributes:
        method: Separation method ("magnetic", "gravity", "flotation")
        ore_type_code: Ore type code
        target_grade: Target concentrate grade (optional)
        residence_time: Residence time (minutes)
        reagent_scheme: Reagent scheme identifier
        operating_point: Dictionary with operating parameters
    """
    method: str  # "magnetic", "gravity", "flotation"
    ore_type_code: str
    target_grade: Optional[float] = None
    residence_time: Optional[float] = None
    reagent_scheme: Optional[str] = None
    operating_point: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default operating_point if None."""
        if self.operating_point is None:
            self.operating_point = {}


@dataclass
class SeparationResponse:
    """
    Result from separation prediction.
    
    Attributes:
        recovery: Recovery to concentrate (0-1)
        concentrate_grade: Concentrate grade (%)
        tailings_grade: Tailings grade (%)
        mass_pull: Mass pull to concentrate (0-1)
        byproduct_recovery: Optional dictionary of byproduct recoveries
    """
    recovery: float  # to concentrate
    concentrate_grade: float
    tailings_grade: float
    mass_pull: float
    byproduct_recovery: Optional[Dict[str, float]] = None


def predict_separation_response(
    liberation: LiberationCurve | np.ndarray,
    chemistry: Dict[str, float],
    config: SeparationConfig
) -> SeparationResponse:
    """
    Predict separation response from liberation, chemistry, and config.
    
    Args:
        liberation: LiberationCurve or array of liberation fractions
        chemistry: Dictionary with grades/element ratios (Fe, SiO2, Al2O3, P, etc.)
        config: SeparationConfig
        
    Returns:
        SeparationResponse
    """
    # Extract feed grade (primary element)
    # Assume first element in chemistry dict is primary
    primary_element = list(chemistry.keys())[0] if chemistry else "Fe"
    feed_grade = chemistry.get(primary_element, 50.0)  # Default 50%
    
    # Convert liberation to average liberation if array
    if isinstance(liberation, np.ndarray):
        avg_liberation = np.mean(liberation)
    else:
        # Use average liberation from curve
        avg_liberation = np.mean(liberation.liberation_fraction)
    
    # Method-specific recovery prediction (simplified)
    if config.method == "magnetic":
        # Magnetic separation: recovery depends on liberation and magnetic properties
        # Simplified: recovery increases with liberation and feed grade
        base_recovery = 0.85
        liberation_factor = 0.15 * avg_liberation
        grade_factor = min(0.1 * (feed_grade / 60.0), 0.1)  # Cap at 10% bonus
        recovery = min(base_recovery + liberation_factor + grade_factor, 0.98)
        
        # Concentrate grade: improves with liberation
        concentrate_grade = feed_grade * (1.0 + 0.3 * avg_liberation)
        concentrate_grade = min(concentrate_grade, 68.0)  # Typical max for magnetic
        
    elif config.method == "gravity":
        # Gravity separation: sensitive to liberation and size
        base_recovery = 0.70
        liberation_factor = 0.25 * avg_liberation
        recovery = min(base_recovery + liberation_factor, 0.95)
        
        concentrate_grade = feed_grade * (1.0 + 0.4 * avg_liberation)
        concentrate_grade = min(concentrate_grade, 65.0)
        
    elif config.method == "flotation":
        # Flotation: depends on liberation, chemistry, reagents
        base_recovery = 0.80
        liberation_factor = 0.20 * avg_liberation
        
        # Chemistry effects (simplified)
        sio2 = chemistry.get("SiO2", 5.0)
        al2o3 = chemistry.get("Al2O3", 2.0)
        chemistry_penalty = min(0.1 * (sio2 / 10.0 + al2o3 / 5.0), 0.15)
        
        recovery = min(base_recovery + liberation_factor - chemistry_penalty, 0.95)
        
        concentrate_grade = feed_grade * (1.0 + 0.35 * avg_liberation)
        concentrate_grade = min(concentrate_grade, 67.0)
        
    else:
        # Unknown method: use default
        recovery = 0.75
        concentrate_grade = feed_grade * 1.2
    
    # Calculate tailings grade
    # Mass balance: feed = concentrate + tailings
    # Assuming mass_pull from recovery relationship
    mass_pull = recovery * (feed_grade / concentrate_grade) if concentrate_grade > 0 else 0.1
    mass_pull = min(mass_pull, 0.5)  # Cap mass pull
    
    tailings_grade = (feed_grade - recovery * concentrate_grade * mass_pull) / (1.0 - mass_pull)
    tailings_grade = max(tailings_grade, 0.0)  # Ensure non-negative
    
    return SeparationResponse(
        recovery=recovery,
        concentrate_grade=concentrate_grade,
        tailings_grade=tailings_grade,
        mass_pull=mass_pull,
        byproduct_recovery=None  # Can be extended
    )

