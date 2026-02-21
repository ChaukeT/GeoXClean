"""
Professional Visualization Module for Block Models

Provides standardized, memory-efficient visualization functions for geostatistical results.
Uses PyVista ImageData (UniformGrid) instead of RectilinearGrid for optimal performance.

Standard Shape Convention: (nz, ny, nx) - Z-Y-X (Geological standard for NumPy)
"""

import numpy as np
import logging

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

logger = logging.getLogger(__name__)


def create_block_model(
    values: np.ndarray,
    origin: tuple,
    spacing: tuple,
    dims: tuple,
    name: str = "Grade",
    no_data_value: float = -9999.0
) -> 'pv.ImageData':
    """
    Creates a highly optimized VTK UniformGrid (ImageData).
    
    Uses ImageData instead of RectilinearGrid for memory efficiency:
    - RectilinearGrid stores coordinates for every slice (wasteful)
    - ImageData only stores origin, spacing, and dimensions (3 numbers)
    - Much faster to load in Paraview/GeoX
    
    Parameters
    ----------
    values : np.ndarray
        The data array. 
        MUST be shaped (nz, ny, nx) -> Z-Y-X (Geological standard for NumPy).
    origin : tuple (x, y, z)
        Bottom-left-corner coordinates.
    spacing : tuple (dx, dy, dz)
        Block sizes.
    dims : tuple (nx, ny, nz)
        Number of blocks in each direction.
    name : str
        Name of the property/attribute
    no_data_value : float
        Value to use for NaN replacement (optional)
        
    Returns
    -------
    pv.ImageData
        PyVista ImageData grid ready for visualization/export
        
    Raises
    ------
    ImportError
        If PyVista is not available
    ValueError
        If array shape doesn't match dimensions
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for visualization. Install with: pip install pyvista")
    
    nx, ny, nz = dims
    dx, dy, dz = spacing
    ox, oy, oz = origin

    # 1. Validation: Ensure array matches dimensions
    # We expect Z, Y, X shape (numpy default)
    if values.shape != (nz, ny, nx):
        logger.warning(
            f"Shape Mismatch! Grid expects (nz={nz}, ny={ny}, nx={nx}) "
            f"but got data shape {values.shape}. Attempting reshape..."
        )
        try:
            # Try to force fit (dangerous if axis order is wrong)
            values = values.reshape((nz, ny, nx))
        except ValueError:
            logger.error("Could not reshape data to fit grid dimensions.")
            raise ValueError(
                f"Cannot reshape array of shape {values.shape} to ({nz}, {ny}, {nx}). "
                "Ensure data is in (nz, ny, nx) format."
            )

    # 2. Create UniformGrid (ImageData)
    # This uses zero memory for coordinates (implicit geometry)
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)  # VTK requires points = cells + 1
    grid.origin = (ox, oy, oz)
    grid.spacing = (dx, dy, dz)

    # 3. Add Data
    # VTK uses C-ordering on ZYX data (iterates X fastest, then Y, then Z)
    # We handle NaNs by masking them or setting a NoData value
    flat_data = values.ravel(order='C')
    
    # Handle NaNs for visualization (Paraview doesn't like NaNs in some filters)
    if np.isnan(flat_data).any():
        # Optionally replace NaN with no_data_value
        # For now, we keep NaN as PyVista handles it reasonably well
        # Uncomment below if you need explicit NoData handling:
        # flat_data = np.where(np.isnan(flat_data), no_data_value, flat_data)
        pass

    grid.cell_data[name] = flat_data
    
    return grid


def export_to_vti(grid: 'pv.ImageData', filename: str):
    """
    Saves as .vti (XML Image Data). 
    
    Smaller and faster than .vtk legacy format.
    Modern format supported by Paraview, GeoX, and most visualization tools.
    
    Parameters
    ----------
    grid : pv.ImageData
        PyVista ImageData grid to export
    filename : str
        Output filename (should end with .vti)
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for export. Install with: pip install pyvista")
    
    grid.save(filename)
    logger.info(f"Exported optimized model: {filename} (ImageData format)")


def add_property_to_grid(grid: 'pv.ImageData', values: np.ndarray, name: str):
    """
    Add an additional property to an existing ImageData grid.
    
    Parameters
    ----------
    grid : pv.ImageData
        Existing PyVista ImageData grid
    values : np.ndarray
        Property values, must match grid dimensions (nz, ny, nx)
    name : str
        Name of the property
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required. Install with: pip install pyvista")
    
    # Get grid dimensions from existing grid
    dims = grid.dimensions
    nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1  # Convert from points to cells
    
    # Validate shape
    if values.shape != (nz, ny, nx):
        logger.warning(f"Property shape {values.shape} doesn't match grid ({nz}, {ny}, {nx}). Reshaping...")
        values = values.reshape((nz, ny, nx))
    
    # Add property
    grid.cell_data[name] = values.ravel(order='C')

