"""
3D Stope Visualization for Underground Mining

Creates PyVista meshes for stopes, supporting:
- Color-coding by period, NSR, grade, or custom attributes
- Wireframe edges
- Transparency controls
- Individual or batch rendering
"""

import logging
from typing import List, Optional, Dict, Union
import numpy as np
import pandas as pd

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

logger = logging.getLogger(__name__)


class StopeVisualizer:
    """Handles 3D visualization of underground stopes."""
    
    def __init__(self):
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - stope visualization disabled")
        
        self.stope_actors = {}  # Cache of stope actors
        
    def create_stope_mesh(self, stope, blocks_df: pd.DataFrame) -> Optional[pv.UnstructuredGrid]:
        """
        Create a PyVista mesh for a single stope from its constituent blocks.
        
        Args:
            stope: Stope object with block_ids, x_center, y_center, z_center
            blocks_df: DataFrame with block coordinates (x, y, z, dx, dy, dz)
            
        Returns:
            PyVista UnstructuredGrid representing the stope solid
        """
        if not PYVISTA_AVAILABLE:
            logger.error("PyVista not available")
            return None
        
        try:
            # Get blocks belonging to this stope (support block_indices or block_ids)
            ids = getattr(stope, 'block_indices', getattr(stope, 'block_ids', []))
            stope_blocks = blocks_df[blocks_df.index.isin(ids)].copy()
            
            if len(stope_blocks) == 0:
                logger.warning(f"No blocks found for stope {stope.id}")
                return None
            
            # Ensure coordinate columns exist
            coord_cols = ['x', 'y', 'z']
            size_cols = ['dx', 'dy', 'dz']
            
            # Try to find coordinate columns (case-insensitive)
            for col in coord_cols:
                if col not in stope_blocks.columns:
                    # Try uppercase
                    if col.upper() in stope_blocks.columns:
                        stope_blocks[col] = stope_blocks[col.upper()]
                    else:
                        logger.error(f"Missing coordinate column: {col}")
                        return None
            
            # Try to find size columns
            for col in size_cols:
                if col not in stope_blocks.columns:
                    # Use default block size if not available
                    default_size = 10.0  # meters
                    stope_blocks[col] = default_size
                    logger.debug(f"Using default {col}={default_size}m")
            
            # Create individual block meshes
            blocks = []
            for idx, row in stope_blocks.iterrows():
                # Create block as hexahedron (8-vertex rectangular solid)
                x, y, z = row['x'], row['y'], row['z']
                dx, dy, dz = row['dx'], row['dy'], row['dz']
                
                # Define 8 vertices of the block
                vertices = np.array([
                    [x - dx/2, y - dy/2, z - dz/2],  # 0: bottom-back-left
                    [x + dx/2, y - dy/2, z - dz/2],  # 1: bottom-back-right
                    [x + dx/2, y + dy/2, z - dz/2],  # 2: bottom-front-right
                    [x - dx/2, y + dy/2, z - dz/2],  # 3: bottom-front-left
                    [x - dx/2, y - dy/2, z + dz/2],  # 4: top-back-left
                    [x + dx/2, y - dy/2, z + dz/2],  # 5: top-back-right
                    [x + dx/2, y + dy/2, z + dz/2],  # 6: top-front-right
                    [x - dx/2, y + dy/2, z + dz/2],  # 7: top-front-left
                ])
                
                # Create hexahedron cell (VTK_HEXAHEDRON = 12)
                cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
                cell_types = np.array([12])  # VTK_HEXAHEDRON
                
                block_mesh = pv.UnstructuredGrid(cells, cell_types, vertices)
                blocks.append(block_mesh)
            
            # Merge all blocks into a single mesh
            if len(blocks) == 1:
                stope_mesh = blocks[0]
            else:
                stope_mesh = blocks[0]
                for block in blocks[1:]:
                    stope_mesh = stope_mesh.merge(block)
            
            # Add stope metadata as cell data
            stope_mesh.cell_data['stope_id'] = np.full(stope_mesh.n_cells, stope.id)
            stope_mesh.cell_data['nsr'] = np.full(stope_mesh.n_cells, stope.nsr)
            stope_mesh.cell_data['grade'] = np.full(stope_mesh.n_cells, stope.grade)
            stope_mesh.cell_data['tonnes'] = np.full(stope_mesh.n_cells, stope.tonnes / stope_mesh.n_cells)
            
            # Add diluted values if available
            if hasattr(stope, 'diluted_grade'):
                stope_mesh.cell_data['diluted_grade'] = np.full(stope_mesh.n_cells, stope.diluted_grade)
            if hasattr(stope, 'diluted_tonnes'):
                stope_mesh.cell_data['diluted_tonnes'] = np.full(stope_mesh.n_cells, stope.diluted_tonnes / stope_mesh.n_cells)
            
            logger.info(f"Created stope mesh for {stope.id}: {stope_mesh.n_cells} cells")
            return stope_mesh
            
        except Exception as e:
            logger.error(f"Error creating stope mesh: {e}", exc_info=True)
            return None
    
    def create_stopes_multiblock(self, stopes: List, blocks_df: pd.DataFrame, 
                                 schedule: Optional[List] = None) -> Optional[pv.MultiBlock]:
        """
        Create a PyVista MultiBlock dataset containing all stopes.
        
        Args:
            stopes: List of Stope objects
            blocks_df: DataFrame with block coordinates
            schedule: Optional list of PeriodKPI objects to assign mining periods
            
        Returns:
            PyVista MultiBlock with named stope meshes
        """
        if not PYVISTA_AVAILABLE:
            return None
        
        try:
            multiblock = pv.MultiBlock()
            
            # Create period mapping if schedule provided
            period_map = {}
            if schedule:
                for period in schedule:
                    if hasattr(period, 'stopes_mined'):
                        for stope_id in period.stopes_mined:
                            period_map[stope_id] = period.t
            
            # Create mesh for each stope
            for i, stope in enumerate(stopes):
                mesh = self.create_stope_mesh(stope, blocks_df)
                
                if mesh is not None:
                    # Add period information if available
                    if stope.id in period_map:
                        mesh.cell_data['period'] = np.full(mesh.n_cells, period_map[stope.id])
                    
                    # Add to multiblock with name
                    multiblock.append(mesh, name=f"Stope_{stope.id}")
            
            logger.info(f"Created MultiBlock with {len(multiblock)} stopes")
            return multiblock
            
        except Exception as e:
            logger.error(f"Error creating stopes multiblock: {e}", exc_info=True)
            return None
    
    def create_colored_stopes_mesh(self, stopes: List, blocks_df: pd.DataFrame,
                                   color_by: str = 'nsr',
                                   schedule: Optional[List] = None) -> Optional[pv.UnstructuredGrid]:
        """
        Create a single merged mesh of all stopes with color-coding.
        
        Args:
            stopes: List of Stope objects
            blocks_df: DataFrame with block coordinates
            color_by: Attribute to color by ('nsr', 'grade', 'period', 'diluted_grade')
            schedule: Optional schedule for period coloring
            
        Returns:
            Single PyVista UnstructuredGrid with colored stopes
        """
        if not PYVISTA_AVAILABLE:
            return None
        
        try:
            # Create period mapping if needed
            period_map = {}
            if schedule and color_by == 'period':
                for period in schedule:
                    if hasattr(period, 'stopes_mined'):
                        for stope_id in period.stopes_mined:
                            period_map[stope_id] = period.t
            
            # Create individual stope meshes
            meshes = []
            color_values = []
            
            for stope in stopes:
                mesh = self.create_stope_mesh(stope, blocks_df)
                
                if mesh is not None:
                    meshes.append(mesh)
                    
                    # Determine color value
                    if color_by == 'nsr':
                        value = stope.nsr
                    elif color_by == 'grade':
                        value = stope.grade
                    elif color_by == 'diluted_grade' and hasattr(stope, 'diluted_grade'):
                        value = stope.diluted_grade
                    elif color_by == 'period':
                        value = period_map.get(stope.id, 0)
                    else:
                        value = 0
                    
                    color_values.extend([value] * mesh.n_cells)
            
            if not meshes:
                logger.warning("No stope meshes created")
                return None
            
            # Merge all meshes
            merged_mesh = meshes[0]
            for mesh in meshes[1:]:
                merged_mesh = merged_mesh.merge(mesh)
            
            # Add color values
            merged_mesh.cell_data[color_by] = np.array(color_values)
            
            logger.info(f"Created merged stopes mesh: {merged_mesh.n_cells} cells, colored by {color_by}")
            return merged_mesh
            
        except Exception as e:
            logger.error(f"Error creating colored stopes mesh: {e}", exc_info=True)
            return None
    
    def add_stopes_to_plotter(self, plotter: 'pv.Plotter', stopes: List, 
                              blocks_df: pd.DataFrame,
                              color_by: str = 'nsr',
                              schedule: Optional[List] = None,
                              show_edges: bool = True,
                              opacity: float = 0.8,
                              cmap: str = 'viridis') -> Optional[str]:
        """
        Add stopes to a PyVista plotter.
        
        Args:
            plotter: PyVista Plotter instance
            stopes: List of Stope objects
            blocks_df: DataFrame with block coordinates
            color_by: Attribute to color by
            schedule: Optional schedule for period coloring
            show_edges: Show wireframe edges
            opacity: Stope transparency (0-1)
            cmap: Matplotlib colormap name
            
        Returns:
            Actor name if successful, None otherwise
        """
        if not PYVISTA_AVAILABLE:
            return None
        
        try:
            # Create colored mesh
            mesh = self.create_colored_stopes_mesh(
                stopes, blocks_df, color_by, schedule
            )
            
            if mesh is None:
                return None
            
            # Add to plotter
            actor_name = f"Stopes_ColoredBy_{color_by}"
            
            # VIOLATION FIX: Removed scalar_bar_args - LegendManager handles legends
            actor = plotter.add_mesh(
                mesh,
                scalars=color_by,
                cmap=cmap,
                show_edges=show_edges,
                edge_color='black',
                line_width=1,
                opacity=opacity,
                name=actor_name,
                show_scalar_bar=False  # LegendManager handles scalar bars
            )
            
            # NOTE: Legend should be updated via LegendManager.set_continuous() or set_categorical()
            # by the calling code after adding stopes to the scene
            
            logger.info(f"Added stopes to plotter: {actor_name}")
            return actor_name
            
        except Exception as e:
            logger.error(f"Error adding stopes to plotter: {e}", exc_info=True)
            return None
    
    def export_stopes_to_vtk(self, stopes: List, blocks_df: pd.DataFrame,
                             filepath: str, schedule: Optional[List] = None) -> bool:
        """
        Export stopes to VTK file format for use in external software.
        
        Args:
            stopes: List of Stope objects
            blocks_df: DataFrame with block coordinates
            filepath: Output VTK file path
            schedule: Optional schedule for period metadata
            
        Returns:
            True if successful
        """
        if not PYVISTA_AVAILABLE:
            return False
        
        try:
            # Create multiblock dataset
            multiblock = self.create_stopes_multiblock(stopes, blocks_df, schedule)
            
            if multiblock is None:
                return False
            
            # Save to file
            multiblock.save(filepath)
            
            logger.info(f"Exported {len(multiblock)} stopes to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting stopes to VTK: {e}", exc_info=True)
            return False
    
    def calculate_stope_bounds(self, stope, blocks_df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate bounding box for a stope.
        
        Args:
            stope: Stope object
            blocks_df: DataFrame with block coordinates
            
        Returns:
            Dictionary with xmin, xmax, ymin, ymax, zmin, zmax
        """
        try:
            stope_blocks = blocks_df[blocks_df.index.isin(stope.block_ids)]
            
            if len(stope_blocks) == 0:
                return None
            
            # Get coordinate columns (case-insensitive)
            x_col = 'x' if 'x' in stope_blocks.columns else 'X'
            y_col = 'y' if 'y' in stope_blocks.columns else 'Y'
            z_col = 'z' if 'z' in stope_blocks.columns else 'Z'
            
            bounds = {
                'xmin': float(stope_blocks[x_col].min()),
                'xmax': float(stope_blocks[x_col].max()),
                'ymin': float(stope_blocks[y_col].min()),
                'ymax': float(stope_blocks[y_col].max()),
                'zmin': float(stope_blocks[z_col].min()),
                'zmax': float(stope_blocks[z_col].max())
            }
            
            return bounds
            
        except Exception as e:
            logger.error(f"Error calculating stope bounds: {e}", exc_info=True)
            return None


# Convenience functions for common use cases

def visualize_stopes_by_period(stopes: List, blocks_df: pd.DataFrame, 
                               schedule: List, plotter: 'pv.Plotter') -> Optional[str]:
    """
    Quick function to visualize stopes colored by mining period.
    
    Args:
        stopes: List of Stope objects
        blocks_df: DataFrame with block coordinates
        schedule: List of PeriodKPI objects
        plotter: PyVista Plotter instance
        
    Returns:
        Actor name if successful
    """
    viz = StopeVisualizer()
    return viz.add_stopes_to_plotter(
        plotter, stopes, blocks_df,
        color_by='period',
        schedule=schedule,
        show_edges=True,
        opacity=0.8,
        cmap='tab20'  # Categorical colormap for periods
    )


def visualize_stopes_by_nsr(stopes: List, blocks_df: pd.DataFrame,
                            plotter: 'pv.Plotter') -> Optional[str]:
    """
    Quick function to visualize stopes colored by NSR.
    
    Args:
        stopes: List of Stope objects
        blocks_df: DataFrame with block coordinates
        plotter: PyVista Plotter instance
        
    Returns:
        Actor name if successful
    """
    viz = StopeVisualizer()
    return viz.add_stopes_to_plotter(
        plotter, stopes, blocks_df,
        color_by='nsr',
        show_edges=True,
        opacity=0.8,
        cmap='RdYlGn'  # Red-Yellow-Green for economic value
    )


def visualize_stopes_by_grade(stopes: List, blocks_df: pd.DataFrame,
                              plotter: 'pv.Plotter') -> Optional[str]:
    """
    Quick function to visualize stopes colored by grade.
    
    Args:
        stopes: List of Stope objects
        blocks_df: DataFrame with block coordinates
        plotter: PyVista Plotter instance
        
    Returns:
        Actor name if successful
    """
    viz = StopeVisualizer()
    return viz.add_stopes_to_plotter(
        plotter, stopes, blocks_df,
        color_by='grade',
        show_edges=True,
        opacity=0.8,
        cmap='plasma'  # High contrast for grade
    )
