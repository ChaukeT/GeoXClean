"""
Geotechnical Material Properties - Store and estimate material properties.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeotechMaterial:
    """Geotechnical material properties."""
    name: str
    unit_weight: float  # γ (kN/m³)
    friction_angle: float  # φ (degrees)
    cohesion: float  # c (kPa)
    tensile_strength: Optional[float] = None  # σ_t (kPa)
    hoek_brown_mb: Optional[float] = None  # Hoek-Brown m_b
    hoek_brown_s: Optional[float] = None  # Hoek-Brown s
    hoek_brown_a: Optional[float] = None  # Hoek-Brown a
    water_condition: Optional[str] = None  # "dry", "wet", "saturated"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeotechMaterialLibrary:
    """Library of geotechnical materials."""
    
    def __init__(self, name: str = "default"):
        """Initialize material library."""
        self.name = name
        self.materials: Dict[str, GeotechMaterial] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_material(self, material: GeotechMaterial):
        """Add a material to the library."""
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[GeotechMaterial]:
        """Get material by name."""
        return self.materials.get(name)
    
    def list_materials(self) -> list[str]:
        """List all material names."""
        return sorted(self.materials.keys())


def estimate_material_from_rock_mass(
    RMR: Optional[float] = None,
    Q: Optional[float] = None,
    GSI: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> GeotechMaterial:
    """
    Estimate geotechnical material properties from rock mass classification.
    
    Links to geotech/rock_mass_model.py (STEP 19) but does not depend on Qt.
    
    Args:
        RMR: Rock Mass Rating (0-100)
        Q: Q-value (0.001-1000)
        GSI: Geological Strength Index (0-100)
        config: Optional configuration dictionary
    
    Returns:
        GeotechMaterial with estimated properties
    """
    config = config or {}
    
    # Determine primary classification system
    if GSI is not None:
        primary = "GSI"
        value = GSI
    elif RMR is not None:
        primary = "RMR"
        value = RMR
    elif Q is not None:
        primary = "Q"
        value = Q
    else:
        # Default to moderate rock mass
        logger.warning("No rock mass classification provided, using default moderate properties")
        return GeotechMaterial(
            name="default_moderate",
            unit_weight=25.0,
            friction_angle=35.0,
            cohesion=100.0,
            notes="Default moderate rock mass"
        )
    
    # Estimate properties based on classification
    # These are simplified empirical relationships - real implementations would use
    # more sophisticated correlations (e.g., Hoek-Brown, Bieniawski correlations)
    
    if primary == "GSI":
        # GSI-based estimation (simplified)
        if GSI >= 75:
            # Very good rock
            friction_angle = 45.0
            cohesion = 500.0
            unit_weight = 27.0
        elif GSI >= 50:
            # Good rock
            friction_angle = 40.0
            cohesion = 300.0
            unit_weight = 26.0
        elif GSI >= 25:
            # Fair rock
            friction_angle = 35.0
            cohesion = 150.0
            unit_weight = 25.0
        else:
            # Poor rock
            friction_angle = 30.0
            cohesion = 50.0
            unit_weight = 24.0
        
        # Estimate Hoek-Brown parameters (simplified)
        hoek_brown_mb = 0.1 * GSI if GSI < 25 else 0.5 + 0.01 * GSI
        hoek_brown_s = 0.001 if GSI < 25 else 0.01 * (GSI - 25) / 75
        hoek_brown_a = 0.5 if GSI < 25 else 0.5 - 0.01 * (GSI - 25) / 75
        
    elif primary == "RMR":
        # RMR-based estimation (simplified Bieniawski correlations)
        if RMR >= 80:
            friction_angle = 45.0
            cohesion = 400.0
            unit_weight = 27.0
        elif RMR >= 60:
            friction_angle = 40.0
            cohesion = 250.0
            unit_weight = 26.0
        elif RMR >= 40:
            friction_angle = 35.0
            cohesion = 150.0
            unit_weight = 25.0
        else:
            friction_angle = 30.0
            cohesion = 75.0
            unit_weight = 24.0
        
        # Convert RMR to approximate GSI for Hoek-Brown
        GSI_approx = RMR - 5  # Simplified conversion
        hoek_brown_mb = 0.1 * GSI_approx if GSI_approx < 25 else 0.5 + 0.01 * GSI_approx
        hoek_brown_s = 0.001 if GSI_approx < 25 else 0.01 * (GSI_approx - 25) / 75
        hoek_brown_a = 0.5 if GSI_approx < 25 else 0.5 - 0.01 * (GSI_approx - 25) / 75
        
    else:  # Q-based
        # Q-based estimation (simplified Barton correlations)
        if Q >= 100:
            friction_angle = 45.0
            cohesion = 400.0
            unit_weight = 27.0
        elif Q >= 10:
            friction_angle = 40.0
            cohesion = 250.0
            unit_weight = 26.0
        elif Q >= 1:
            friction_angle = 35.0
            cohesion = 150.0
            unit_weight = 25.0
        else:
            friction_angle = 30.0
            cohesion = 75.0
            unit_weight = 24.0
        
        # Approximate Hoek-Brown from Q
        GSI_approx = 10 * (Q ** 0.5) if Q > 0 else 20
        GSI_approx = min(100, max(0, GSI_approx))
        hoek_brown_mb = 0.1 * GSI_approx if GSI_approx < 25 else 0.5 + 0.01 * GSI_approx
        hoek_brown_s = 0.001 if GSI_approx < 25 else 0.01 * (GSI_approx - 25) / 75
        hoek_brown_a = 0.5 if GSI_approx < 25 else 0.5 - 0.01 * (GSI_approx - 25) / 75
    
    # Apply config overrides
    friction_angle = config.get("friction_angle", friction_angle)
    cohesion = config.get("cohesion", cohesion)
    unit_weight = config.get("unit_weight", unit_weight)
    
    material_name = config.get("name", f"estimated_{primary.lower()}_{value:.1f}")
    
    return GeotechMaterial(
        name=material_name,
        unit_weight=unit_weight,
        friction_angle=friction_angle,
        cohesion=cohesion,
        hoek_brown_mb=hoek_brown_mb,
        hoek_brown_s=hoek_brown_s,
        hoek_brown_a=hoek_brown_a,
        water_condition=config.get("water_condition", "dry"),
        notes=f"Estimated from {primary}={value:.1f}",
        metadata={
            "RMR": RMR,
            "Q": Q,
            "GSI": GSI,
            "estimation_method": primary
        }
    )

