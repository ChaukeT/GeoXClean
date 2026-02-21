"""
Cave Footprint Builder (STEP 37)

Build cave footprints from block model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaveCell:
    """
    Individual cell in a cave footprint.
    
    Attributes:
        id: Cell identifier
        x: X coordinate
        y: Y coordinate
        level: Level elevation
        tonnage: Tonnes in cell
        grade_by_element: Grades by element
        footprint_id: Footprint identifier
    """
    id: str
    x: float
    y: float
    level: float
    tonnage: float = 0.0
    grade_by_element: Dict[str, float] = field(default_factory=dict)
    footprint_id: str = ""


@dataclass
class CaveFootprint:
    """
    Cave footprint definition.
    
    Attributes:
        id: Footprint identifier
        cells: List of CaveCell objects
        drawbell_ids: List of drawbell identifiers
    """
    id: str
    cells: List[CaveCell] = field(default_factory=list)
    drawbell_ids: List[str] = field(default_factory=list)


def build_cave_footprint_from_block_model(
    block_model: Any,
    footprint_polygon: Optional[List[Tuple[float, float]]] = None,
    levels: Optional[List[float]] = None,
    cell_size_m: float = 25.0
) -> CaveFootprint:
    """
    Build cave footprint from block model.
    
    Args:
        block_model: BlockModel instance
        footprint_polygon: Optional list of (x, y) tuples defining footprint boundary
        levels: Optional list of level elevations
        cell_size_m: Cell size in meters
    
    Returns:
        CaveFootprint
    """
    logger.info("Building cave footprint from block model")
    
    # Get block model data
    if hasattr(block_model, 'to_dataframe'):
        df = block_model.to_dataframe()
    else:
        logger.error("Block model does not support to_dataframe()")
        return CaveFootprint(id="default", cells=[])
    
    if df.empty:
        logger.warning("Block model is empty")
        return CaveFootprint(id="default", cells=[])
    
    # Get positions
    if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
        positions = df[['X', 'Y', 'Z']].values
    elif 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        positions = df[['x', 'y', 'z']].values
    else:
        logger.error("Block model missing coordinate columns")
        return CaveFootprint(id="default", cells=[])
    
    # Filter by polygon if provided
    if footprint_polygon:
        from shapely.geometry import Point, Polygon
        poly = Polygon(footprint_polygon)
        mask = np.array([poly.contains(Point(x, y)) for x, y in positions[:, :2]])
        positions = positions[mask]
        df = df[mask]
    
    if len(positions) == 0:
        logger.warning("No blocks within footprint polygon")
        return CaveFootprint(id="default", cells=[])
    
    # Determine levels if not provided
    if levels is None:
        z_coords = positions[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        levels = np.arange(z_min, z_max + cell_size_m, cell_size_m).tolist()
    
    # Get tonnage
    if 'tonnage' in df.columns:
        tonnages = df['tonnage'].values
    elif 'TONNAGE' in df.columns:
        tonnages = df['TONNAGE'].values
    else:
        tonnages = np.ones(len(positions)) * 1000.0
    
    # Get grades
    grade_properties = {}
    for col in df.columns:
        if col not in ['X', 'Y', 'Z', 'x', 'y', 'z', 'tonnage', 'TONNAGE']:
            if df[col].dtype in [np.float64, np.float32]:
                grade_properties[col] = df[col].values
    
    # Create cells by discretizing space
    cells = []
    cell_index = 0
    
    for level in levels:
        # Find blocks near this level
        level_mask = np.abs(positions[:, 2] - level) <= cell_size_m / 2
        if not np.any(level_mask):
            continue
        
        level_positions = positions[level_mask]
        level_tonnages = tonnages[level_mask]
        level_grades = {k: v[level_mask] for k, v in grade_properties.items()}
        
        # Discretize into cells
        x_coords = level_positions[:, 0]
        y_coords = level_positions[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        x_cells = int(np.ceil((x_max - x_min) / cell_size_m))
        y_cells = int(np.ceil((y_max - y_min) / cell_size_m))
        
        for i in range(x_cells):
            for j in range(y_cells):
                cell_x = x_min + (i + 0.5) * cell_size_m
                cell_y = y_min + (j + 0.5) * cell_size_m
                
                # Find blocks in this cell
                cell_mask = (
                    (x_coords >= cell_x - cell_size_m / 2) &
                    (x_coords < cell_x + cell_size_m / 2) &
                    (y_coords >= cell_y - cell_size_m / 2) &
                    (y_coords < cell_y + cell_size_m / 2)
                )
                
                if np.any(cell_mask):
                    cell_tonnage = float(np.sum(level_tonnages[cell_mask]))
                    cell_grades = {}
                    for element, grades in level_grades.items():
                        if len(grades[cell_mask]) > 0:
                            cell_grades[element] = float(np.nanmean(grades[cell_mask]))
                    
                    cell = CaveCell(
                        id=f"CELL_{cell_index:05d}",
                        x=cell_x,
                        y=cell_y,
                        level=level,
                        tonnage=cell_tonnage,
                        grade_by_element=cell_grades,
                        footprint_id="default"
                    )
                    cells.append(cell)
                    cell_index += 1
    
    logger.info(f"Built cave footprint with {len(cells)} cells")
    
    return CaveFootprint(
        id="default",
        cells=cells,
        drawbell_ids=[]  # Would be populated from drawbell design
    )

