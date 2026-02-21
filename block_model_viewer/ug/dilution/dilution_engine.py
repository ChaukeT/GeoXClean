"""
Dilution Engine (STEP 37)

Calculate dilution and ore loss for stopes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

from ..slos.slos_geometry import StopeInstance

logger = logging.getLogger(__name__)


@dataclass
class DilutionModel:
    """
    Model for dilution calculation.
    
    Attributes:
        overbreak_m: Overbreak per wall in meters
        contact_grade: Contact grade dict (HW/FW grades)
        in_stope_recovery: In-stope recovery fraction
    """
    overbreak_m: float = 1.0
    contact_grade: Dict[str, float] = field(default_factory=dict)
    in_stope_recovery: float = 0.95


@dataclass
class DilutionResult:
    """
    Result from dilution calculation.
    
    Attributes:
        diluted_tonnes: Total diluted tonnes
        diluted_grade_by_element: Diluted grades by element
        ore_loss_fraction: Fraction of ore lost
    """
    diluted_tonnes: float
    diluted_grade_by_element: Dict[str, float] = field(default_factory=dict)
    ore_loss_fraction: float = 0.0


def apply_dilution(
    stope: StopeInstance,
    model: DilutionModel
) -> DilutionResult:
    """
    Apply dilution model to a stope.
    
    Args:
        stope: StopeInstance
        model: DilutionModel
    
    Returns:
        DilutionResult
    """
    logger.debug(f"Applying dilution to stope {stope.id}")
    
    # Calculate dilution volume (simplified)
    # Assume rectangular stope with overbreak on all walls
    # In practice, would use actual stope geometry from template
    
    # Estimate stope dimensions (simplified)
    # Would get from template in production
    base_volume_m3 = stope.tonnes / 2.7  # Assume density 2.7 t/m³
    
    # Calculate overbreak volume
    # Simplified: assume overbreak adds volume proportional to surface area
    # In practice, would calculate based on actual geometry
    overbreak_factor = 1.0 + (model.overbreak_m * 0.1)  # Simplified
    diluted_volume_m3 = base_volume_m3 * overbreak_factor
    
    # Calculate diluted tonnes
    diluted_tonnes = diluted_volume_m3 * 2.7  # Assume same density
    
    # Calculate dilution tonnes
    dilution_tonnes = diluted_tonnes - stope.tonnes
    
    # Calculate diluted grades
    # Mix original ore with contact grade (dilution)
    diluted_grades = {}
    
    for element, grade in stope.grade_by_element.items():
        # Weighted average: original ore + dilution
        original_metal = stope.tonnes * grade
        dilution_metal = dilution_tonnes * model.contact_grade.get(element, 0.0)
        diluted_grades[element] = (original_metal + dilution_metal) / diluted_tonnes if diluted_tonnes > 0 else grade
    
    # Calculate ore loss
    ore_loss_fraction = 1.0 - model.in_stope_recovery
    
    logger.debug(f"Stope {stope.id}: {stope.tonnes:.0f} t -> {diluted_tonnes:.0f} t (dilution: {dilution_tonnes:.0f} t)")
    
    return DilutionResult(
        diluted_tonnes=diluted_tonnes,
        diluted_grade_by_element=diluted_grades,
        ore_loss_fraction=ore_loss_fraction
    )

