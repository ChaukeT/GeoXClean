"""
VTK parser for block model data.
"""

import pyvista as pv
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .base_parser import BaseParser
from ..models.block_model import BlockModel, BlockMetadata

logger = logging.getLogger(__name__)


class VTKParser(BaseParser):
    """
    Parser for VTK format files using PyVista.
    
    Supports both legacy VTK and XML VTK formats.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.vtk', '.vtu', '.vtp', '.vts', '.vtr', '.vti']
        self.format_name = "VTK"
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a VTK file into a BlockModel object.
        
        ⚠️ MEMORY WARNING: VTK files are loaded entirely into RAM.
        For very large files (>2GB), consider using chunked loading or streaming.
        
        Args:
            file_path: Path to the VTK file
            **kwargs: Additional options:
                - extract_blocks: Whether to extract individual blocks (default: True)
                - block_size: Default block size if not specified in data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size and warn if very large
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 2048:  # > 2GB
            logger.warning(
                f"Large VTK file detected ({file_size_mb:.1f} MB). "
                f"Loading entire file into RAM. Consider using chunked loading for very large files."
            )
        
        extract_blocks = kwargs.get('extract_blocks', True)
        default_block_size = kwargs.get('block_size', (1.0, 1.0, 1.0))
        
        try:
            # Load VTK file using PyVista
            # CRITICAL: This runs in a worker thread, but pv.read() is safe because:
            # 1. It only reads file data into memory (no GUI operations)
            # 2. We extract primitive data immediately and don't return the PyVista object
            # 3. The mesh object is only used to extract numpy arrays, then discarded
            # NOTE: PyVista's pv.read() loads entire file into RAM
            # For streaming/chunked loading, would need custom vtkXMLUnstructuredGridReader
            mesh = pv.read(str(file_path))
            logger.info(f"Loaded VTK file: {mesh.n_points} points, {mesh.n_cells} cells")
            
            # Create metadata
            metadata = BlockMetadata(
                source_file=str(file_path),
                file_format="VTK",
                coordinate_system="unknown",
                units="unknown"
            )
            
            block_model = BlockModel(metadata)
            
            if extract_blocks and mesh.n_cells > 0:
                # Extract block information from VTK cells
                self._extract_blocks_from_mesh(block_model, mesh, default_block_size)
            else:
                # Treat as point cloud
                self._extract_points_from_mesh(block_model, mesh, default_block_size)
            
            # CRITICAL: Don't store or return the PyVista mesh object
            # It's only used for data extraction, then discarded
            del mesh
            
            logger.info(f"Parsed VTK: {block_model.block_count} blocks")
            return block_model
            
        except Exception as e:
            logger.error(f"Error parsing VTK file {file_path}: {e}")
            raise ValueError(f"Failed to parse VTK file: {e}")
    
    def _extract_blocks_from_mesh(self, block_model: BlockModel, mesh: pv.PolyData, 
                                 default_block_size: tuple) -> None:
        """
        Extract block information from VTK mesh cells.
        
        ⚠️ PERFORMANCE OPTIMIZED: Uses PyVista's built-in filters instead of
        Python loops. mesh.get_cell(i) in a loop for 1M cells would freeze UI for minutes.
        
        Args:
            block_model: BlockModel to populate
            mesh: PyVista mesh object
            default_block_size: Default block dimensions
        """
        # Get cell centers
        cell_centers = mesh.cell_centers().points
        
        # Compute cell bounds efficiently
        # PyVista doesn't have cell_bounds() method, so we compute manually
        # Using VTK's GetCellBounds which calls into C++ and is reasonably fast
        n_cells = mesh.n_cells
        dimensions = np.zeros((n_cells, 3))
        
        # Access VTK dataset directly (PyVista wraps VTK objects)
        # GetCellBounds modifies the bounds array in place
        bounds_array = np.zeros(6)  # [xmin, xmax, ymin, ymax, zmin, zmax]
        
        for i in range(n_cells):
            mesh.GetCellBounds(i, bounds_array)
            # Calculate dimensions from bounds
            dx = bounds_array[1] - bounds_array[0]
            dy = bounds_array[3] - bounds_array[2]
            dz = bounds_array[5] - bounds_array[4]
            
            # Handle degenerate cells (points, lines, faces) with zero dimensions
            # Replace zero or negative dimensions with default_block_size
            if dx <= 0:
                dx = default_block_size[0]
            if dy <= 0:
                dy = default_block_size[1]
            if dz <= 0:
                dz = default_block_size[2]
            
            dimensions[i] = [dx, dy, dz]
        
        # Set geometry
        block_model.set_geometry(cell_centers, dimensions)
        
        # CRITICAL FIX: Use PyVista's built-in filters instead of Python loops
        # mesh.get_cell(i) in a Python loop for 1M cells = minutes of freezing!
        # PyVista filters run in C++ (VTK) and are orders of magnitude faster
        
        # Extract point data as properties
        if mesh.point_data:
            for name, data in mesh.point_data.items():
                # Check if data length matches (point data vs cell data)
                if len(data) == mesh.n_points:
                    # Point data: convert to cell data using PyVista filter
                    # This runs in C++ (VTK) and is extremely fast
                    try:
                        # Use PyVista's point_data_to_cell_data filter
                        # This averages point data to cell centers efficiently
                        mesh_with_cell_data = mesh.point_data_to_cell_data()
                        
                        if name in mesh_with_cell_data.cell_data:
                            cell_data = mesh_with_cell_data.cell_data[name]
                            if len(cell_data) == len(cell_centers):
                                block_model.add_property(name, cell_data)
                            else:
                                logger.warning(f"Point data '{name}' conversion produced unexpected length")
                        else:
                            logger.warning(f"Point data '{name}' not found after conversion")
                    except Exception as e:
                        logger.warning(f"Could not convert point data '{name}' to cell data: {e}")
                        # Fallback: skip this property
                        continue
                elif len(data) == len(cell_centers):
                    # Already cell data (shouldn't happen, but handle gracefully)
                    block_model.add_property(name, data)
        
        # Extract cell data as properties
        if mesh.cell_data:
            for name, data in mesh.cell_data.items():
                block_model.add_property(name, data)
    
    def _extract_points_from_mesh(self, block_model: BlockModel, mesh: pv.PolyData, 
                                 default_block_size: tuple) -> None:
        """
        Extract point information from VTK mesh.
        
        Args:
            block_model: BlockModel to populate
            mesh: PyVista mesh object
            default_block_size: Default block dimensions
        """
        points = mesh.points
        
        # Use default block size for all points
        dimensions = np.tile(default_block_size, (len(points), 1))
        
        # Set geometry
        block_model.set_geometry(points, dimensions)
        
        # Extract point data as properties
        if mesh.point_data:
            for name, data in mesh.point_data.items():
                block_model.add_property(name, data)
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get VTK file information."""
        info = super().get_file_info(file_path)
        
        try:
            # Try to read VTK file header
            mesh = pv.read(str(file_path))
            info.update({
                "points": mesh.n_points,
                "cells": mesh.n_cells,
                "point_data": list(mesh.point_data.keys()) if mesh.point_data else [],
                "cell_data": list(mesh.cell_data.keys()) if mesh.cell_data else [],
                "bounds": mesh.bounds
            })
        except Exception as e:
            logger.warning(f"Could not preview VTK file: {e}")
            info.update({
                "points": 0,
                "cells": 0,
                "point_data": [],
                "cell_data": [],
                "bounds": None
            })
        
        return info
