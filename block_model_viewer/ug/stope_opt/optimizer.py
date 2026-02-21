"""
Stope Optimization Engine

Implements algorithms for underground stope shape optimization:
1. Maximum closure (graph-based)
2. Floating stope with geometry constraints
3. Contact dilution modeling

Author: BlockModelViewer Team
Date: 2025-11-06
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Union, TYPE_CHECKING
from dataclasses import dataclass
import networkx as nx

from ..dataclasses import Stope

if TYPE_CHECKING:
    from ...models.block_model import BlockModel

logger = logging.getLogger(__name__)


@dataclass
class StopeOptConfig:
    """
    Configuration for stope optimization.
    
    Attributes:
        min_length: Minimum stope length (m)
        max_length: Maximum stope length (m)
        min_width: Minimum stope width (m)
        max_width: Maximum stope width (m)
        min_height: Minimum stope height (m)
        max_height: Maximum stope height (m)
        dilation_skin_m: Dilution skin thickness (m)
        crown_pillar_m: Crown pillar thickness (m)
        rib_pillar_m: Rib pillar thickness (m)
        waste_grade_default: Default grade for dilution material
        min_nsr_diluted: Minimum NSR after dilution ($/t)
        algorithm: Optimization algorithm ('max_closure', 'floating', 'morphological')
    """
    min_length: float = 15.0
    max_length: float = 50.0
    min_width: float = 4.0
    max_width: float = 20.0
    min_height: float = 10.0
    max_height: float = 40.0
    dilation_skin_m: float = 0.5
    crown_pillar_m: float = 6.0
    rib_pillar_m: float = 4.0
    waste_grade_default: Dict[str, float] = None
    min_nsr_diluted: float = 0.0
    algorithm: str = 'max_closure'
    
    def __post_init__(self):
        if self.waste_grade_default is None:
            self.waste_grade_default = {}


def diluted_grade(grade_ore: float, tonnes_ore: float, 
                  grade_waste: float, tonnes_waste: float) -> float:
    """
    Calculate diluted grade after mixing ore and waste.
    
    Args:
        grade_ore: Ore grade (%)
        tonnes_ore: Ore tonnage
        grade_waste: Waste grade (%)
        tonnes_waste: Waste tonnage
        
    Returns:
        Diluted grade (%)
    """
    total_tonnes = tonnes_ore + tonnes_waste
    if total_tonnes < 1e-9:
        return 0.0
    return (grade_ore * tonnes_ore + grade_waste * tonnes_waste) / total_tonnes


def calculate_stope_geometry(voxels: List[int], block_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate stope geometry dimensions from voxel list.
    
    Args:
        voxels: List of block IDs in the stope
        block_df: DataFrame with columns [x, y, z, dx, dy, dz]
        
    Returns:
        Dictionary with length, width, height, volume
    """
    if not voxels:
        return {'length': 0, 'width': 0, 'height': 0, 'volume': 0}
    
    # Get bounding box
    subset = block_df[block_df.index.isin(voxels)]
    
    x_min, x_max = subset['x'].min(), subset['x'].max()
    y_min, y_max = subset['y'].min(), subset['y'].max()
    z_min, z_max = subset['z'].min(), subset['z'].max()
    
    # Add half block size
    dx = subset['dx'].iloc[0] if 'dx' in subset else 5.0
    dy = subset['dy'].iloc[0] if 'dy' in subset else 5.0
    dz = subset['dz'].iloc[0] if 'dz' in subset else 5.0
    
    length = (x_max - x_min) + dx
    width = (y_max - y_min) + dy
    height = (z_max - z_min) + dz
    volume = len(voxels) * dx * dy * dz
    
    return {
        'length': length,
        'width': width,
        'height': height,
        'volume': volume
    }


def apply_contact_dilution(stope_voxels: Set[int], block_df: pd.DataFrame,
                           skin_thickness: float, waste_grade: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Apply contact dilution around stope boundaries.
    
    Args:
        stope_voxels: Set of block IDs in the stope
        block_df: Block model DataFrame
        skin_thickness: Dilution skin thickness (m)
        waste_grade: Grade dictionary for dilution material
        
    Returns:
        Tuple of (dilution_tonnes, diluted_grade_dict)
    """
    # Simplified dilution model: assume uniform skin
    # In production, use spatial index to find actual contact blocks
    
    stope_blocks = block_df[block_df.index.isin(stope_voxels)]
    
    # Estimate surface area (simplified as rectangular prism)
    geom = calculate_stope_geometry(list(stope_voxels), block_df)
    L, W, H = geom['length'], geom['width'], geom['height']
    surface_area = 2 * (L*W + L*H + W*H)
    
    # Dilution volume = surface_area × skin_thickness
    dilution_volume = surface_area * skin_thickness
    
    # Assume density ~2.7 t/m³
    density = block_df['density'].mean() if 'density' in block_df else 2.7
    dilution_tonnes = dilution_volume * density
    
    # Calculate diluted grades
    ore_tonnes = stope_blocks['tonnes'].sum() if 'tonnes' in stope_blocks else len(stope_voxels) * density * 125
    
    diluted_grades = {}
    for grade_col in waste_grade.keys():
        if grade_col in stope_blocks.columns:
            ore_grade = stope_blocks[grade_col].mean()
            waste_g = waste_grade.get(grade_col, 0.0)
            diluted_grades[grade_col] = diluted_grade(ore_grade, ore_tonnes, waste_g, dilution_tonnes)
        else:
            diluted_grades[grade_col] = 0.0
    
    return dilution_tonnes, diluted_grades


def max_closure_stope(
    block_df: Union['BlockModel', pd.DataFrame], 
    value_col: str = 'value'
) -> Set[int]:
    """
    Maximum closure algorithm for stope optimization.
    
    ✅ NEW STANDARD API: Accepts BlockModel or DataFrame (backward compatible)
    
    Uses graph min-cut to find economically optimal stope shape.
    
    Args:
        block_df: BlockModel instance (preferred) or DataFrame (legacy) with 'value' column (NSR or profit)
        value_col: Column name for block value
        
    Returns:
        Set of block IDs in optimal stope
    """
    from ...models.block_model import BlockModel
    
    # Handle BlockModel input
    if isinstance(block_df, BlockModel):
        # ✅ STANDARD API: Convert to DataFrame for processing
        block_df_work = block_df.to_dataframe()
        logger.info(f"✅ BlockModel API: Extracted {block_df.block_count} blocks for max closure")
    else:
        block_df_work = block_df
        logger.info(f"⚠️ DataFrame API (legacy): {len(block_df_work)} blocks")
    
    G = nx.DiGraph()
    
    # Add source and sink
    G.add_node('source')
    G.add_node('sink')
    
    # Add blocks with capacities based on value
    for idx, row in block_df_work.iterrows():
        value = row[value_col]
        
        if value > 0:
            # Positive value: connect to source
            G.add_edge('source', idx, capacity=value)
        else:
            # Negative value: connect to sink
            G.add_edge(idx, 'sink', capacity=-value)
    
    # Add precedence edges (simplified: vertical stacking)
    # In production, use actual spatial precedence
    if 'z' in block_df_work.columns:
        block_df_sorted = block_df_work.sort_values('z')
        for i in range(len(block_df_sorted) - 1):
            upper = block_df_sorted.index[i]
            lower = block_df_sorted.index[i + 1]
            if block_df_sorted.iloc[i]['z'] > block_df_sorted.iloc[i + 1]['z']:
                # Upper block must be mined before lower
                G.add_edge(upper, lower, capacity=float('inf'))
    
    # Find minimum cut using preflow_push (faster than default Edmonds-Karp)
    # preflow_push: O(V^2 * sqrt(E)) - significantly faster for sparse graphs
    try:
        cut_value, (reachable, non_reachable) = nx.minimum_cut(
            G, 'source', 'sink',
            flow_func=nx.algorithms.flow.preflow_push
        )
        stope_blocks = {n for n in reachable if n not in ('source', 'sink')}
        logger.info(f"Max closure found {len(stope_blocks)} blocks, cut value: {cut_value:.2f}")
        return stope_blocks
    except Exception as e:
        logger.error(f"Max closure failed: {e}")
        return set()


def optimize_stopes(
    blocks: Union['BlockModel', pd.DataFrame], 
    params: Dict
) -> List[Stope]:
    """
    Main stope optimization function.
    
    ✅ NEW STANDARD API: Accepts BlockModel or DataFrame (backward compatible)
    
    Args:
        blocks: BlockModel instance (preferred) or DataFrame (legacy) with columns [x, y, z, dx, dy, dz, tonnes, grade_*, nsr, ...]
        params: Configuration dictionary with optimization parameters
        
    Returns:
        List of optimized Stope objects
    """
    from ...models.block_model import BlockModel
    
    # Handle BlockModel input
    if isinstance(blocks, BlockModel):
        # ✅ STANDARD API: Convert to DataFrame for processing
        blocks_df = blocks.to_dataframe()
        logger.info(f"✅ BlockModel API: Extracted {blocks.block_count} blocks for stope optimization")
    else:
        blocks_df = blocks
        logger.info(f"⚠️ DataFrame API (legacy): {len(blocks_df)} blocks")
    
    logger.info(f"Starting stope optimization with {len(blocks_df)} blocks")
    
    # Build configuration
    config = StopeOptConfig(
        min_length=params.get('min_length', 15.0),
        max_length=params.get('max_length', 50.0),
        min_width=params.get('min_width', 4.0),
        max_width=params.get('max_width', 20.0),
        min_height=params.get('min_height', 10.0),
        max_height=params.get('max_height', 40.0),
        dilation_skin_m=params.get('dilation_skin_m', 0.5),
        crown_pillar_m=params.get('crown_pillar_m', 6.0),
        rib_pillar_m=params.get('rib_pillar_m', 4.0),
        waste_grade_default=params.get('waste_grade_default', {}),
        min_nsr_diluted=params.get('min_nsr_diluted', 0.0),
        algorithm=params.get('algorithm', 'max_closure')
    )
    
    stopes = []
    
    if config.algorithm == 'max_closure':
        # Run maximum closure on entire domain
        stope_voxels = max_closure_stope(blocks, value_col='nsr')
        
        if len(stope_voxels) == 0:
            logger.warning("No viable stopes found")
            return []
        
        # Calculate geometry
        geom = calculate_stope_geometry(list(stope_voxels), blocks)
        
        # Check geometry constraints
        if not (config.min_length <= geom['length'] <= config.max_length and
                config.min_width <= geom['width'] <= config.max_width and
                config.min_height <= geom['height'] <= config.max_height):
            logger.warning(f"Stope geometry out of bounds: L={geom['length']:.1f}, W={geom['width']:.1f}, H={geom['height']:.1f}")
            return []
        
        # Calculate raw tonnage and grade
        stope_blocks = blocks[blocks.index.isin(stope_voxels)]
        tonnes_raw = stope_blocks['tonnes'].sum() if 'tonnes' in stope_blocks else len(stope_voxels) * 2.7 * 125
        
        grade_cols = [c for c in stope_blocks.columns if c.startswith('grade_') or c in ['Fe', 'Cu', 'Au', 'Ag']]
        grade_raw = {col: stope_blocks[col].mean() for col in grade_cols if col in stope_blocks}
        
        # Apply dilution
        dilution_tonnes, grade_dil = apply_contact_dilution(
            stope_voxels, blocks, config.dilation_skin_m, config.waste_grade_default
        )
        tonnes_dil = tonnes_raw + dilution_tonnes
        
        # Calculate diluted NSR (simplified)
        nsr_dil = stope_blocks['nsr'].mean() if 'nsr' in stope_blocks else 0.0
        nsr_dil = nsr_dil * (tonnes_raw / tonnes_dil)  # Dilution reduces NSR
        
        # Check economic viability
        if nsr_dil < config.min_nsr_diluted:
            logger.warning(f"Stope NSR too low: {nsr_dil:.2f} < {config.min_nsr_diluted:.2f}")
            return []
        
        # Determine level
        level = int(stope_blocks['z'].mean()) if 'z' in stope_blocks else 0
        
        # Create stope object
        geom['crown_pillar'] = config.crown_pillar_m
        geom['rib_pillar'] = config.rib_pillar_m
        
        stope = Stope(
            id=f"ST_{level}_001",
            level=level,
            voxels=list(stope_voxels),
            tonnes_raw=tonnes_raw,
            grade_raw=grade_raw,
            tonnes_dil=tonnes_dil,
            grade_dil=grade_dil,
            nsr_dil=nsr_dil,
            geom=geom,
            parents=[],
            risk_score=0.0
        )
        
        stopes.append(stope)
        logger.info(f"Created stope {stope.id}: {tonnes_dil:.0f}t @ NSR ${nsr_dil:.2f}/t")
    
    else:
        logger.error(f"Algorithm '{config.algorithm}' not yet implemented")
    
    logger.info(f"Optimization complete: {len(stopes)} stopes")
    return stopes
