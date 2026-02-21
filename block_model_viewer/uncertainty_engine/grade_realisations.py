"""
Grade Realisation Manager

Index and manage realisations stored in BlockModel for downstream uncertainty analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import re

logger = logging.getLogger(__name__)


@dataclass
class GradeRealisationSet:
    """Represents a set of grade realisations for a property."""
    property_name: str
    realisation_names: List[str]
    n_realizations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def register_realisations(
    block_model: Any,
    property_name: str,
    names: List[str]
) -> GradeRealisationSet:
    """
    Register a set of realisations for a property.
    
    Args:
        block_model: BlockModel instance
        property_name: Base property name (e.g., "Fe")
        names: List of realization property names
    
    Returns:
        GradeRealisationSet instance
    """
    if not hasattr(block_model, 'properties'):
        raise ValueError("BlockModel must have 'properties' attribute")
    
    # Verify all names exist in block model
    missing = [name for name in names if name not in block_model.properties]
    if missing:
        raise ValueError(f"Realisation properties not found: {missing}")
    
    return GradeRealisationSet(
        property_name=property_name,
        realisation_names=names,
        n_realizations=len(names),
        metadata={
            'source': 'registered',
            'block_model_size': len(block_model.coordinates) if hasattr(block_model, 'coordinates') else None
        }
    )


def get_realisation_values(
    block_model: Any,
    grade_set: GradeRealisationSet,
    idx: int
) -> np.ndarray:
    """
    Get values for a specific realisation index.
    
    Args:
        block_model: BlockModel instance
        grade_set: GradeRealisationSet instance
        idx: Realisation index (0-based)
    
    Returns:
        Array of values for the realisation
    """
    if idx < 0 or idx >= grade_set.n_realizations:
        raise IndexError(f"Realisation index {idx} out of range [0, {grade_set.n_realizations})")
    
    realisation_name = grade_set.realisation_names[idx]
    
    if not hasattr(block_model, 'properties'):
        raise ValueError("BlockModel must have 'properties' attribute")
    
    if realisation_name not in block_model.properties:
        raise ValueError(f"Realisation '{realisation_name}' not found in block model")
    
    return np.asarray(block_model.properties[realisation_name])


def list_grade_realisations(block_model: Any) -> Dict[str, GradeRealisationSet]:
    """
    List all grade realisation sets found in block model using naming patterns.
    
    Patterns detected:
    - ik_sim_<property>_<number>
    - cosim_<property>_<number>
    - sgsim_<property>_<number>
    - <property>_sim<number>
    - <property>_real<number>
    
    Args:
        block_model: BlockModel instance
    
    Returns:
        Dict mapping property_name -> GradeRealisationSet
    """
    if not hasattr(block_model, 'properties'):
        return {}
    
    realisation_sets = {}
    
    # Pattern: <prefix>_<property>_<number>
    patterns = [
        (r'^(ik_sim|cosim|sgsim)_([^_]+)_(\d+)$', 1, 2),  # ik_sim_Fe_0001 -> Fe
        (r'^([^_]+)_sim(\d+)$', 0, 1),  # Fe_sim1 -> Fe
        (r'^([^_]+)_real(\d+)$', 0, 1),  # Fe_real1 -> Fe
    ]
    
    # Group by property name
    property_groups = {}
    
    for prop_name in block_model.properties.keys():
        matched = False
        
        for pattern, prefix_idx, prop_idx in patterns:
            match = re.match(pattern, prop_name)
            if match:
                property_name = match.group(prop_idx + 1)  # +1 because group 0 is full match
                
                if property_name not in property_groups:
                    property_groups[property_name] = []
                
                property_groups[property_name].append(prop_name)
                matched = True
                break
        
        if not matched:
            # Try generic pattern: ends with _<number>
            match = re.match(r'^(.+)_(\d+)$', prop_name)
            if match:
                base_name = match.group(1)
                # Check if base_name looks like a property (not too long, alphanumeric)
                if len(base_name) < 50 and re.match(r'^[A-Za-z0-9_]+$', base_name):
                    if base_name not in property_groups:
                        property_groups[base_name] = []
                    property_groups[base_name].append(prop_name)
    
    # Create GradeRealisationSet for each property
    for property_name, realisation_names in property_groups.items():
        # Sort by number suffix
        def extract_number(name):
            match = re.search(r'(\d+)$', name)
            return int(match.group(1)) if match else 0
        
        sorted_names = sorted(realisation_names, key=extract_number)
        
        realisation_sets[property_name] = GradeRealisationSet(
            property_name=property_name,
            realisation_names=sorted_names,
            n_realizations=len(sorted_names),
            metadata={
                'source': 'auto_detected',
                'pattern': 'auto'
            }
        )
    
    logger.info(f"Found {len(realisation_sets)} grade realisation sets")
    
    return realisation_sets


def get_all_realisation_values(
    block_model: Any,
    grade_set: GradeRealisationSet
) -> np.ndarray:
    """
    Get all realisation values as a 2D array.
    
    Args:
        block_model: BlockModel instance
        grade_set: GradeRealisationSet instance
    
    Returns:
        Array of shape (n_realizations, n_blocks)
    """
    n_realizations = grade_set.n_realizations
    n_blocks = len(block_model.coordinates) if hasattr(block_model, 'coordinates') else 0
    
    if n_blocks == 0:
        # Try to infer from first realisation
        first_name = grade_set.realisation_names[0]
        first_values = block_model.properties[first_name]
        n_blocks = len(first_values)
    
    all_values = np.zeros((n_realizations, n_blocks))
    
    for idx, realisation_name in enumerate(grade_set.realisation_names):
        values = block_model.properties[realisation_name]
        all_values[idx, :] = np.asarray(values)
    
    return all_values

