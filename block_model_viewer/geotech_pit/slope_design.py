"""
Slope Design Rules & Bench Layout - Translate geotech into design recommendations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import logging

from .limit_equilibrium_2d import LEM2DResult
from .limit_equilibrium_3d import LEM3DResult
from .slope_probabilistic import ProbSlopeResult
from ..geotech_common.material_properties import GeotechMaterial

logger = logging.getLogger(__name__)


@dataclass
class BenchDesignRule:
    """Design rule for bench geometry."""
    domain_code: str
    rock_mass_class: str  # e.g., "Good", "Fair", "Poor"
    bench_height: float  # Height of individual benches (m)
    berm_width: float  # Width of berms between benches (m)
    overall_slope_angle: float  # Overall slope angle (degrees from horizontal)
    max_fos_target: float  # Maximum acceptable FOS (for optimization)
    min_fos_target: float  # Minimum acceptable FOS (safety requirement)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchDesignSet:
    """Collection of bench design rules."""
    
    def __init__(self, name: str = "default"):
        """Initialize bench design set."""
        self.name = name
        self.rules: List[BenchDesignRule] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_rule(self, rule: BenchDesignRule):
        """Add a design rule."""
        self.rules.append(rule)
    
    def get_rule(self, domain_code: str, rock_mass_class: str) -> Optional[BenchDesignRule]:
        """Get rule for domain and rock mass class."""
        for rule in self.rules:
            if rule.domain_code == domain_code and rule.rock_mass_class == rock_mass_class:
                return rule
        return None
    
    def list_rules(self) -> List[BenchDesignRule]:
        """List all rules."""
        return self.rules.copy()


def suggest_bench_design(
    domain_code: str,
    material: GeotechMaterial,
    rock_mass_class: str,
    constraints: Dict[str, Any]
) -> BenchDesignRule:
    """
    Suggest bench design based on material properties and constraints.
    
    Args:
        domain_code: Geological domain code
        material: Geotechnical material properties
        rock_mass_class: Rock mass classification ("Excellent", "Good", "Fair", "Poor", "Very Poor")
        constraints: Dictionary with constraints:
            - min_fos_target: Minimum FOS requirement (default 1.3)
            - max_height: Maximum overall height (m)
            - available_width: Available width for slope (m)
    
    Returns:
        BenchDesignRule with suggested geometry
    """
    min_fos_target = constraints.get("min_fos_target", 1.3)
    max_height = constraints.get("max_height", 100.0)
    available_width = constraints.get("available_width", None)
    
    # Estimate bench height based on rock mass class
    bench_height_map = {
        "Excellent": 15.0,
        "Good": 12.0,
        "Fair": 10.0,
        "Poor": 8.0,
        "Very Poor": 6.0
    }
    bench_height = bench_height_map.get(rock_mass_class, 10.0)
    
    # Estimate berm width (typically 5-10% of bench height, minimum 3-5m)
    berm_width = max(3.0, bench_height * 0.08)
    
    # Estimate overall slope angle based on material properties
    # Simplified empirical relationship
    phi_rad = math.radians(material.friction_angle)
    
    # Base angle from friction angle (simplified)
    base_angle = material.friction_angle * 0.7  # Conservative factor
    
    # Adjust for cohesion
    if material.cohesion > 200:
        base_angle += 5.0
    elif material.cohesion < 50:
        base_angle -= 5.0
    
    # Adjust for rock mass class
    class_adjustment = {
        "Excellent": 5.0,
        "Good": 2.0,
        "Fair": 0.0,
        "Poor": -3.0,
        "Very Poor": -5.0
    }
    base_angle += class_adjustment.get(rock_mass_class, 0.0)
    
    # Ensure reasonable bounds
    overall_slope_angle = max(25.0, min(55.0, base_angle))
    
    # If width constraint exists, adjust angle
    if available_width is not None and max_height > 0:
        required_angle = math.degrees(math.atan(max_height / available_width))
        if required_angle < overall_slope_angle:
            overall_slope_angle = required_angle
            logger.info(f"Adjusted slope angle to {overall_slope_angle:.1f}° due to width constraint")
    
    return BenchDesignRule(
        domain_code=domain_code,
        rock_mass_class=rock_mass_class,
        bench_height=bench_height,
        berm_width=berm_width,
        overall_slope_angle=overall_slope_angle,
        max_fos_target=min_fos_target + 0.2,  # Slightly higher for optimization
        min_fos_target=min_fos_target,
        metadata={
            "material_friction_angle": material.friction_angle,
            "material_cohesion": material.cohesion,
            "estimated_from": "empirical"
        }
    )


def evaluate_design_against_results(
    rule: BenchDesignRule,
    results: List[LEM2DResult | LEM3DResult | ProbSlopeResult]
) -> Dict[str, Any]:
    """
    Evaluate design rule against stability analysis results.
    
    Args:
        rule: Bench design rule to evaluate
        results: List of stability analysis results
    
    Returns:
        Dictionary with pass/fail status and margins
    """
    if not results:
        return {
            "status": "unknown",
            "reason": "No analysis results provided",
            "min_fos": None,
            "max_fos": None,
            "mean_fos": None,
            "meets_min_target": None,
            "safety_margin": None
        }
    
    # Extract FOS values
    fos_values = []
    for result in results:
        if isinstance(result, ProbSlopeResult):
            fos_values.extend(result.fos_samples.tolist())
        elif isinstance(result, (LEM2DResult, LEM3DResult)):
            fos_values.append(result.fos)
    
    if not fos_values:
        return {
            "status": "unknown",
            "reason": "No FOS values found in results",
            "min_fos": None,
            "max_fos": None,
            "mean_fos": None,
            "meets_min_target": None,
            "safety_margin": None
        }
    
    min_fos = min(fos_values)
    max_fos = max(fos_values)
    mean_fos = sum(fos_values) / len(fos_values)
    
    # Check against targets
    meets_min_target = min_fos >= rule.min_fos_target
    safety_margin = min_fos - rule.min_fos_target
    
    status = "pass" if meets_min_target else "fail"
    
    return {
        "status": status,
        "reason": f"Min FOS {min_fos:.3f} {'meets' if meets_min_target else 'below'} target {rule.min_fos_target:.3f}",
        "min_fos": min_fos,
        "max_fos": max_fos,
        "mean_fos": mean_fos,
        "meets_min_target": meets_min_target,
        "safety_margin": safety_margin,
        "n_results": len(results),
        "n_fos_values": len(fos_values)
    }

