"""
Ore/Waste Marking & SMU Metrics (STEP 29)

Classify GC blocks and compute tonnage/grade metrics per dig polygon.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging

from .digpolygons import DiglineSet
from .support_model import GCModel

logger = logging.getLogger(__name__)


@dataclass
class OreWasteCutoffRule:
    """
    Cutoff rule for ore/waste classification.
    
    Attributes:
        property_name: Property name (e.g., "Fe")
        cutoff: Cutoff value
        direction: Direction ("<=" is ore", ">=" is ore", etc.)
    """
    property_name: str
    cutoff: float
    direction: str = ">="  # ">=" is ore", "<=" is ore", etc.


@dataclass
class OreWasteClassificationResult:
    """
    Result from ore/waste classification.
    
    Attributes:
        classification: Array of classification codes (0=waste, 1=ore, 2=marginal)
        tonnage_by_class: Dictionary mapping class code -> tonnage
        grade_by_class: Dictionary mapping class code -> average grade
    """
    classification: np.ndarray
    tonnage_by_class: Dict[int, float] = field(default_factory=dict)
    grade_by_class: Dict[int, float] = field(default_factory=dict)


def classify_gc_blocks(
    gc_model: GCModel,
    cutoff_rules: List[OreWasteCutoffRule]
) -> OreWasteClassificationResult:
    """
    Classify GC blocks into ore/marginal/waste based on cutoff rules.
    
    Args:
        gc_model: GCModel
        cutoff_rules: List of OreWasteCutoffRule
        
    Returns:
        OreWasteClassificationResult
    """
    n_blocks = gc_model.grid.get_block_count()
    classification = np.zeros(n_blocks, dtype=int)  # 0 = waste, 1 = ore, 2 = marginal
    
    # Process each cutoff rule
    for rule in cutoff_rules:
        prop_values = gc_model.get_property(rule.property_name)
        if prop_values is None:
            continue
        
        # Apply cutoff
        if rule.direction == ">=":
            meets_cutoff = prop_values >= rule.cutoff
        elif rule.direction == "<=":
            meets_cutoff = prop_values <= rule.cutoff
        elif rule.direction == ">":
            meets_cutoff = prop_values > rule.cutoff
        elif rule.direction == "<":
            meets_cutoff = prop_values < rule.cutoff
        else:
            meets_cutoff = np.zeros(n_blocks, dtype=bool)
        
        # Mark as ore where cutoff is met
        classification[meets_cutoff] = 1
    
    # Calculate tonnage and grade by class
    block_volume = gc_model.grid.dx * gc_model.grid.dy * gc_model.grid.dz
    
    # Get density property
    density_prop = gc_model.get_property("density")
    if density_prop is None:
        density_prop = np.full(n_blocks, 2.7)  # Default density
    
    tonnage_by_class = {}
    grade_by_class = {}
    
    for class_code in [0, 1, 2]:
        mask = classification == class_code
        if np.any(mask):
            # Calculate tonnage
            volumes = np.full(np.sum(mask), block_volume)
            densities = density_prop[mask]
            tonnage = np.sum(volumes * densities)
            tonnage_by_class[class_code] = tonnage
            
            # Calculate average grade (for primary property)
            if cutoff_rules:
                primary_prop = cutoff_rules[0].property_name
                primary_values = gc_model.get_property(primary_prop)
                if primary_values is not None:
                    class_grades = primary_values[mask]
                    valid_grades = class_grades[~np.isnan(class_grades)]
                    if len(valid_grades) > 0:
                        grade_by_class[class_code] = np.mean(valid_grades)
                    else:
                        grade_by_class[class_code] = 0.0
                else:
                    grade_by_class[class_code] = 0.0
            else:
                grade_by_class[class_code] = 0.0
    
    logger.info(f"Classified {n_blocks} GC blocks: "
                f"Ore={np.sum(classification == 1)}, "
                f"Waste={np.sum(classification == 0)}, "
                f"Marginal={np.sum(classification == 2)}")
    
    return OreWasteClassificationResult(
        classification=classification,
        tonnage_by_class=tonnage_by_class,
        grade_by_class=grade_by_class
    )


def summarise_by_digpolygon(
    gc_model: GCModel,
    diglines: DiglineSet,
    ore_waste_result: OreWasteClassificationResult,
    density_property: str = "density"
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize tonnes and grade per dig polygon.
    
    Args:
        gc_model: GCModel
        diglines: DiglineSet
        ore_waste_result: OreWasteClassificationResult
        density_property: Name of density property
        
    Returns:
        Dictionary mapping polygon_id -> summary dict
    """
    from .digpolygons import blocks_within_polygon
    
    block_volume = gc_model.grid.dx * gc_model.grid.dy * gc_model.grid.dz
    density_values = gc_model.get_property(density_property)
    if density_values is None:
        density_values = np.full(gc_model.grid.get_block_count(), 2.7)
    
    summaries = {}
    
    for polygon in diglines.polygons:
        # Find blocks within polygon
        block_mask = blocks_within_polygon(gc_model.grid, polygon)
        
        if not np.any(block_mask):
            summaries[polygon.id] = {
                "polygon_id": polygon.id,
                "bench_code": polygon.bench_code,
                "ore_flag": polygon.ore_flag,
                "tonnes_total": 0.0,
                "tonnes_ore": 0.0,
                "tonnes_waste": 0.0,
                "grade_ore": 0.0,
                "grade_waste": 0.0,
                "ore_percent": 0.0
            }
            continue
        
        # Calculate tonnage
        volumes = np.full(np.sum(block_mask), block_volume)
        densities = density_values[block_mask]
        tonnes_total = np.sum(volumes * densities)
        
        # Get classification for blocks in polygon
        classification = ore_waste_result.classification[block_mask]
        
        # Separate ore and waste
        ore_mask = classification == 1
        waste_mask = classification == 0
        
        tonnes_ore = np.sum(volumes[ore_mask] * densities[ore_mask]) if np.any(ore_mask) else 0.0
        tonnes_waste = np.sum(volumes[waste_mask] * densities[waste_mask]) if np.any(waste_mask) else 0.0
        
        # Calculate grades
        grade_ore = 0.0
        grade_waste = 0.0
        
        # Get primary property for grade calculation
        prop_names = gc_model.get_property_names()
        grade_prop = None
        for prop_name in prop_names:
            if prop_name not in ["density", "x", "y", "z"]:
                grade_prop = prop_name
                break
        
        if grade_prop:
            prop_values = gc_model.get_property(grade_prop)
            if prop_values is not None:
                if np.any(ore_mask):
                    ore_grades = prop_values[block_mask][ore_mask]
                    valid_ore = ore_grades[~np.isnan(ore_grades)]
                    if len(valid_ore) > 0:
                        grade_ore = np.mean(valid_ore)
                
                if np.any(waste_mask):
                    waste_grades = prop_values[block_mask][waste_mask]
                    valid_waste = waste_grades[~np.isnan(waste_grades)]
                    if len(valid_waste) > 0:
                        grade_waste = np.mean(valid_waste)
        
        ore_percent = (tonnes_ore / tonnes_total * 100.0) if tonnes_total > 0 else 0.0
        
        summaries[polygon.id] = {
            "polygon_id": polygon.id,
            "bench_code": polygon.bench_code,
            "ore_flag": polygon.ore_flag,
            "tonnes_total": tonnes_total,
            "tonnes_ore": tonnes_ore,
            "tonnes_waste": tonnes_waste,
            "grade_ore": grade_ore,
            "grade_waste": grade_waste,
            "ore_percent": ore_percent
        }
    
    logger.info(f"Summarized {len(summaries)} dig polygons")
    return summaries

