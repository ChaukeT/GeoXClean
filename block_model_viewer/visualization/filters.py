"""
Filtering and slicing operations for block models.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

from ..models.block_model import BlockModel
from ..utils.profiling import profile_section

logger = logging.getLogger(__name__)


class Filters:
    """
    Handles filtering and slicing operations for block models.
    
    Supports spatial slicing, property-based filtering, and cross-sections.
    """
    
    def __init__(self, block_model: Optional[BlockModel] = None):
        self.block_model = block_model
        self.current_filters = {}
        self.slice_planes = {}  # 'x', 'y', 'z' -> position
        self.property_filters = {}  # property_name -> (min, max)
        
    def set_block_model(self, block_model) -> None:
        """Set the block model to filter."""
        self.block_model = block_model
        self.current_filters.clear()
        self.slice_planes.clear()
        self.property_filters.clear()

        # Determine block count based on object type
        if hasattr(block_model, 'block_count'):
            # BlockModel object
            block_count = block_model.block_count
        elif isinstance(block_model, pd.DataFrame):
            # DataFrame - count rows
            block_count = len(block_model)
        else:
            # Unknown type - try to get length
            try:
                block_count = len(block_model)
            except:
                block_count = 0

        logger.info(f"Set block model for filtering: {block_count} blocks")
    
    def apply_spatial_slice(self, axis: str, position: float, 
                           keep_side: str = 'above') -> np.ndarray:
        """
        Apply spatial slicing along an axis.
        
        Args:
            axis: Axis to slice along ('x', 'y', 'z')
            position: Position along the axis
            keep_side: Which side to keep ('above', 'below', 'both')
            
        Returns:
            Array of block indices that pass the filter
        """
        if self.block_model is None:
            return np.array([])
        
        positions = self.block_model.positions
        if positions is None:
            return np.array([])
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis.lower(), 0)
        
        # Get block centers
        block_centers = positions[:, axis_idx]
        
        # Apply slice filter
        if keep_side == 'above':
            mask = block_centers >= position
        elif keep_side == 'below':
            mask = block_centers <= position
        else:  # 'both' - no filtering
            mask = np.ones(len(block_centers), dtype=bool)
        
        # Update slice plane
        self.slice_planes[axis.lower()] = position
        
        filtered_indices = np.where(mask)[0]
        logger.info(f"Applied {axis} slice at {position}: {len(filtered_indices)} blocks")
        
        return filtered_indices
    
    def apply_property_filter(self, property_name: str, min_value: Optional[float] = None,
                             max_value: Optional[float] = None) -> np.ndarray:
        """
        Apply property-based filtering.
        
        Args:
            property_name: Name of property to filter on
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            Array of block indices that pass the filter
        """
        if self.block_model is None:
            return np.array([])
        
        property_values = self.block_model.get_property(property_name)
        if property_values is None:
            logger.warning(f"Property '{property_name}' not found")
            return np.array([])
        
        # Create filter mask
        mask = np.ones(len(property_values), dtype=bool)
        
        if min_value is not None:
            mask &= property_values >= min_value
        if max_value is not None:
            mask &= property_values <= max_value
        
        # Update property filter
        self.property_filters[property_name] = (min_value, max_value)
        
        filtered_indices = np.where(mask)[0]
        logger.info(f"Applied property filter '{property_name}': {len(filtered_indices)} blocks")
        
        return filtered_indices
    
    def apply_combined_filters(self) -> np.ndarray:
        """
        Apply all active filters and return combined result.
        
        Returns:
            Array of block indices that pass all filters
        """
        if self.block_model is None:
            return np.array([])
        
        with profile_section("filters.apply_combined_filters", min_duration_ms=5):
            mask = np.ones(self.block_model.block_count, dtype=bool)
            
            positions = self.block_model.positions
            if positions is not None:
                for axis, position in self.slice_planes.items():
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    block_centers = positions[:, axis_idx]
                    mask &= block_centers >= position
            
            for property_name, (min_val, max_val) in self.property_filters.items():
                property_values = self.block_model.get_property(property_name)
                if property_values is not None:
                    if min_val is not None:
                        mask &= property_values >= min_val
                    if max_val is not None:
                        mask &= property_values <= max_val
            
            filtered_indices = np.where(mask)[0]
            logger.info(f"Applied combined filters: {len(filtered_indices)} blocks remain")
            return filtered_indices
    
    def clear_filters(self) -> None:
        """Clear all active filters."""
        self.slice_planes.clear()
        self.property_filters.clear()
        self.current_filters.clear()
        logger.info("Cleared all filters")
    
    def clear_spatial_slices(self) -> None:
        """Clear all spatial slice filters."""
        self.slice_planes.clear()
        logger.info("Cleared spatial slices")
    
    def clear_property_filters(self) -> None:
        """Clear all property-based filters."""
        self.property_filters.clear()
        logger.info("Cleared property filters")
    
    def get_cross_section(self, axis: str, position: float, 
                         thickness: float = 0.1) -> np.ndarray:
        """
        Get blocks within a cross-section slice.
        
        Args:
            axis: Axis perpendicular to slice ('x', 'y', 'z')
            position: Center position of the slice
            thickness: Thickness of the slice
            
        Returns:
            Array of block indices within the slice
        """
        if self.block_model is None:
            return np.array([])
        
        positions = self.block_model.positions
        if positions is None:
            return np.array([])
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis.lower(), 0)
        
        # Get block centers
        block_centers = positions[:, axis_idx]
        
        # Find blocks within slice thickness
        half_thickness = thickness / 2.0
        mask = np.abs(block_centers - position) <= half_thickness
        
        slice_indices = np.where(mask)[0]
        logger.info(f"Cross-section {axis} at {position}±{half_thickness}: {len(slice_indices)} blocks")
        
        return slice_indices
    
    def get_statistics_for_filtered(self, property_name: str, 
                                   filtered_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics for a property on filtered blocks.
        
        Args:
            property_name: Name of property
            filtered_indices: Block indices to include (uses all if None)
            
        Returns:
            Dictionary with statistics
        """
        if self.block_model is None:
            return {}
        
        property_values = self.block_model.get_property(property_name)
        if property_values is None:
            return {}
        
        with profile_section("filters.statistics", min_duration_ms=5):
            if filtered_indices is not None:
                values = property_values[filtered_indices]
            else:
                values = property_values
            
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) == 0:
                return {'count': len(values), 'valid_count': 0, 'all_nan': True}
            
            stats = {
                'count': len(values),
                'valid_count': len(valid_values),
                'nan_count': len(values) - len(valid_values),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'median': float(np.median(valid_values)),
                'sum': float(np.sum(valid_values))
            }
            
            return stats
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of all active filters."""
        summary = {
            'spatial_slices': dict(self.slice_planes),
            'property_filters': dict(self.property_filters),
            'total_blocks': self.block_model.block_count if self.block_model else 0
        }
        
        if self.block_model:
            filtered_indices = self.apply_combined_filters()
            summary['filtered_blocks'] = len(filtered_indices)
            summary['filter_percentage'] = (len(filtered_indices) / self.block_model.block_count * 100) if self.block_model.block_count > 0 else 0
        
        return summary
    
    def export_filtered_data(self, filtered_indices: np.ndarray, 
                           filename: str, format: str = 'csv') -> None:
        """
        Export filtered block data to file.
        
        Args:
            filtered_indices: Block indices to export
            filename: Output filename
            format: Output format ('csv', 'vtk')
        """
        if self.block_model is None:
            logger.warning("No block model loaded for export")
            return
        
        if format.lower() == 'csv':
            self._export_csv(filtered_indices, filename)
        elif format.lower() == 'vtk':
            self._export_vtk(filtered_indices, filename)
        else:
            logger.warning(f"Unsupported export format: {format}")
    
    def _export_csv(self, filtered_indices: np.ndarray, filename: str) -> None:
        """Export filtered data as CSV."""
        import pandas as pd
        
        if self.block_model is None:
            return
        
        # Get filtered data
        positions = self.block_model.positions[filtered_indices]
        dimensions = self.block_model.dimensions[filtered_indices]
        
        # Create DataFrame
        data = {
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'dx': dimensions[:, 0],
            'dy': dimensions[:, 1],
            'dz': dimensions[:, 2]
        }
        
        # Add properties
        for prop_name, prop_values in self.block_model.properties.items():
            data[prop_name] = prop_values[filtered_indices]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(filtered_indices)} blocks to CSV: {filename}")
    
    def _export_vtk(self, filtered_indices: np.ndarray, filename: str) -> None:
        """Export filtered data as VTK."""
        import pyvista as pv
        
        if self.block_model is None:
            return
        
        # Get filtered data
        positions = self.block_model.positions[filtered_indices]
        dimensions = self.block_model.dimensions[filtered_indices]
        
        # Create block meshes
        corners = self.block_model.get_block_corners(filtered_indices)
        
        # Combine all blocks into single mesh
        all_points = []
        all_faces = []
        point_offset = 0
        
        for i, block_corners in enumerate(corners):
            # Add points
            all_points.extend(block_corners)
            
            # Add faces (6 faces per block)
            faces = np.array([
                [4, 0, 1, 2, 3],  # Bottom
                [4, 4, 7, 6, 5],  # Top
                [4, 0, 4, 5, 1],  # Front
                [4, 2, 6, 7, 3],  # Back
                [4, 0, 3, 7, 4],  # Left
                [4, 1, 5, 6, 2]   # Right
            ])
            
            # Offset face indices
            faces[:, 1:] += point_offset
            all_faces.extend(faces)
            
            point_offset += 8
        
        # Create PyVista mesh
        mesh = pv.PolyData(np.array(all_points), np.array(all_faces))
        
        # Add properties
        for prop_name, prop_values in self.block_model.properties.items():
            mesh[prop_name] = prop_values[filtered_indices]
        
        # Save VTK file
        mesh.save(filename)
        logger.info(f"Exported {len(filtered_indices)} blocks to VTK: {filename}")
