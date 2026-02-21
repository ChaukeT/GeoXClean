"""
SLOS Geometry Engine (STEP 37)

Generate stope layouts from block model and templates.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StopeTemplate:
    """
    Template for SLOS stope geometry.
    
    Attributes:
        id: Template identifier
        strike_length_m: Strike length in meters
        dip_length_m: Dip length in meters
        height_m: Stope height in meters
        min_mining_width_m: Minimum mining width
        crown_pillar_m: Crown pillar thickness
        sill_pillar_m: Sill pillar thickness
        orientation: Orientation dict with 'dip' and 'dip_azimuth'
    """
    id: str
    strike_length_m: float
    dip_length_m: float
    height_m: float
    min_mining_width_m: float
    crown_pillar_m: float
    sill_pillar_m: float
    orientation: Dict[str, float] = field(default_factory=lambda: {'dip': 0.0, 'dip_azimuth': 0.0})


@dataclass
class StopeInstance:
    """
    Instance of a stope in the mine.
    
    Attributes:
        id: Stope identifier
        template_id: Reference to StopeTemplate
        center: Center coordinates (x, y, z)
        level: Level identifier
        tonnes: Tonnes in stope
        grade_by_element: Grades by element
        dilution_tonnes: Dilution tonnes
        dilution_grade_by_element: Dilution grades
        ore_loss_fraction: Fraction of ore lost
        predecessors: List of stope IDs that must be mined before this one
    """
    id: str
    template_id: str
    center: Tuple[float, float, float]
    level: str
    tonnes: float = 0.0
    grade_by_element: Dict[str, float] = field(default_factory=dict)
    dilution_tonnes: float = 0.0
    dilution_grade_by_element: Dict[str, float] = field(default_factory=dict)
    ore_loss_fraction: float = 0.0
    predecessors: List[str] = field(default_factory=list)


def generate_stopes_from_block_model(
    block_model: Any,
    template: StopeTemplate,
    level_spacing_m: float,
    strike_panel_length_m: float,
    ore_domain_property: Optional[str] = None,
    min_grade_cutoff: Optional[Dict[str, float]] = None
) -> List[StopeInstance]:
    """
    Generate regular SLOS panels from block model geometry + ore domain.
    
    Args:
        block_model: BlockModel instance
        template: StopeTemplate to use
        level_spacing_m: Spacing between levels
        strike_panel_length_m: Length of strike panels
        ore_domain_property: Optional property name for ore domain filtering
        min_grade_cutoff: Optional minimum grade cutoff dict
    
    Returns:
        List of StopeInstance objects
    """
    logger.info(f"Generating SLOS stopes from block model using template '{template.id}'")
    
    stopes = []
    
    # Get block model data
    if hasattr(block_model, 'to_dataframe'):
        df = block_model.to_dataframe()
    else:
        logger.error("Block model does not support to_dataframe()")
        return stopes
    
    if df.empty:
        logger.warning("Block model is empty")
        return stopes
    
    # Get positions
    if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
        positions = df[['X', 'Y', 'Z']].values
    elif 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        positions = df[['x', 'y', 'z']].values
    else:
        logger.error("Block model missing coordinate columns (X/Y/Z or x/y/z)")
        return stopes
    
    # Filter by ore domain if specified
    if ore_domain_property and ore_domain_property in df.columns:
        ore_mask = df[ore_domain_property].notna() & (df[ore_domain_property] > 0)
        positions = positions[ore_mask]
        df = df[ore_mask]
    
    if len(positions) == 0:
        logger.warning("No blocks match ore domain criteria")
        return stopes
    
    # Get grade properties
    grade_properties = {}
    for col in df.columns:
        if col not in ['X', 'Y', 'Z', 'x', 'y', 'z', 'tonnage', 'TONNAGE']:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                grade_properties[col] = df[col].values
    
    # Apply grade cutoff if specified
    if min_grade_cutoff:
        valid_mask = np.ones(len(positions), dtype=bool)
        for element, cutoff in min_grade_cutoff.items():
            if element in grade_properties:
                valid_mask &= grade_properties[element] >= cutoff
        positions = positions[valid_mask]
        for key in grade_properties:
            grade_properties[key] = grade_properties[key][valid_mask]
    
    if len(positions) == 0:
        logger.warning("No blocks match grade cutoff criteria")
        return stopes
    
    # Get tonnage
    if 'tonnage' in df.columns:
        tonnages = df['tonnage'].values
    elif 'TONNAGE' in df.columns:
        tonnages = df['TONNAGE'].values
    else:
        # Estimate from dimensions
        if hasattr(block_model, 'dimensions') and block_model.dimensions is not None:
            volumes = np.prod(block_model.dimensions, axis=1)
            density = 2.7  # Default density
            tonnages = volumes * density
        else:
            tonnages = np.ones(len(positions)) * 1000.0
    
    # Group blocks by level (Z coordinate)
    z_coords = positions[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    # Create levels
    levels = []
    current_z = z_min
    level_index = 0
    while current_z <= z_max:
        levels.append((current_z, f"L{level_index:03d}"))
        current_z += level_spacing_m
        level_index += 1
    
    # Generate stopes for each level
    stope_index = 0
    for level_z, level_id in levels:
        # Find blocks in this level (within half level spacing)
        level_mask = np.abs(z_coords - level_z) <= level_spacing_m / 2
        if not np.any(level_mask):
            continue
        
        level_positions = positions[level_mask]
        level_tonnages = tonnages[level_mask]
        level_grades = {k: v[level_mask] for k, v in grade_properties.items()}
        
        # Group into strike panels
        # Simple approach: divide by strike_panel_length_m
        y_coords = level_positions[:, 1]
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        y_current = y_min
        panel_index = 0
        while y_current < y_max:
            panel_mask = (y_coords >= y_current) & (y_coords < y_current + strike_panel_length_m)
            if np.any(panel_mask):
                panel_positions = level_positions[panel_mask]
                panel_tonnages = level_tonnages[panel_mask]
                panel_grades = {k: v[panel_mask] for k, v in level_grades.items()}
                
                # Calculate center
                center = (
                    float(np.mean(panel_positions[:, 0])),
                    float(np.mean(panel_positions[:, 1])),
                    float(level_z)
                )
                
                # Calculate total tonnes
                total_tonnes = float(np.sum(panel_tonnages))
                
                # Calculate average grades
                avg_grades = {}
                for element, grades in panel_grades.items():
                    if len(grades) > 0:
                        avg_grades[element] = float(np.nanmean(grades))
                
                # Create stope instance
                stope_id = f"STOPE_{level_id}_P{panel_index:03d}"
                stope = StopeInstance(
                    id=stope_id,
                    template_id=template.id,
                    center=center,
                    level=level_id,
                    tonnes=total_tonnes,
                    grade_by_element=avg_grades
                )
                
                stopes.append(stope)
                stope_index += 1
            
            y_current += strike_panel_length_m
            panel_index += 1
    
    logger.info(f"Generated {len(stopes)} SLOS stopes")
    return stopes

