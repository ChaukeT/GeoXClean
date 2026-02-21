"""
Cross-section slicing utilities for drillhole visualization.
"""

import pyvista as pv
import numpy as np
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def render_cross_section(
    mesh: pv.PolyData,
    normal: Tuple[float, float, float] = (1, 0, 0),
    origin: Optional[Tuple[float, float, float]] = None,
    scalar: str = "Fe",
    cmap: str = "viridis",
    invert: bool = False,
    plotter: Optional[pv.Plotter] = None,
    show_scalar_bar: bool = False,  # Disabled: use custom LegendWidget instead
    overlay: bool = True,
    full_mesh_opacity: float = 0.25,
    full_mesh_color: str = "lightgrey"
) -> Optional[pv.PolyData]:
    """
    Slice a merged drillhole mesh by a plane for sectional viewing.
    
    Args:
        mesh: PyVista mesh to slice (e.g., combined drillhole mesh)
        normal: Unit vector normal to the plane (e.g., (1,0,0) for N-S section)
        origin: Point the plane passes through (defaults to mesh center)
        scalar: Scalar field name for coloring
        cmap: Colormap name
        invert: If True, show the opposite side of the plane
        plotter: Optional plotter to add mesh to (if None, creates new plotter)
        show_scalar_bar: Whether to show scalar bar (DISABLED - use custom LegendWidget)
        overlay: If True, show full mesh as grey context behind colored slice
        full_mesh_opacity: Opacity of full mesh when overlay=True (0-1)
        full_mesh_color: Color of full mesh when overlay=True
        
    Returns:
        Sliced mesh, or None if slicing failed
    """
    if mesh is None or mesh.n_points == 0:
        logger.warning("No valid mesh to slice")
        return None
    
    # Normalize the normal vector
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    
    # Use mesh center if origin not provided
    if origin is None:
        origin = np.array(mesh.center)
    else:
        origin = np.array(origin, dtype=float)
    
    try:
        # Perform slice
        # If invert is True, negate the normal to show opposite side
        slice_normal = -normal if invert else normal
        sliced = mesh.slice(normal=slice_normal, origin=origin)
        
        if sliced.n_points == 0:
            logger.warning("Plane did not intersect mesh; try shifting origin.")
            return None
        
        # If plotter provided, add to it
        if plotter is not None:
            # Remove existing cross-section and full mesh if present
            try:
                plotter.remove_actor('cross_section', render=False)
                plotter.remove_actor('cross_section_full_mesh', render=False)
            except Exception:
                pass
            
            # If overlay mode, add full mesh as grey context
            if overlay:
                plotter.add_mesh(
                    mesh,
                    color=full_mesh_color,
                    opacity=full_mesh_opacity,
                    show_edges=False,
                    show_scalar_bar=False,
                    name='cross_section_full_mesh'
                )
            
            # Check if scalar exists
            if scalar not in sliced.array_names:
                logger.warning(f"Scalar field '{scalar}' not found in sliced mesh, using default")
                scalar = None
            
            # Add the colored slice (PyVista scalar bar disabled)
            plotter.add_mesh(
                sliced,
                scalars=scalar,
                cmap=cmap,
                show_scalar_bar=False,  # Use custom LegendWidget instead
                name='cross_section',
                opacity=1.0
            )
        
        logger.info(f"Created cross-section with normal {normal}, origin {origin}")
        return sliced
        
    except Exception as e:
        logger.error(f"Error creating cross-section: {e}")
        return None


def export_cross_section_data(
    sliced_mesh: pv.PolyData,
    scalar_field: str = "Fe",
    strike: Optional[float] = None,
    dip: Optional[float] = None,
    orientation: str = "unknown",
    save_dir: Union[str, Path] = "exports"
) -> Dict[str, Path]:
    """
    Export cross-section intersection data to CSV.
    
    Args:
        sliced_mesh: The sliced PolyData mesh from cross-section
        scalar_field: Name of scalar field to export
        strike: Strike angle in degrees (for geological sections)
        dip: Dip angle in degrees (for geological sections)
        orientation: Orientation description (e.g., "north-south", "strike-dip")
        save_dir: Directory to save exports
        
    Returns:
        Dictionary with 'csv' key pointing to saved CSV file path
    """
    if sliced_mesh is None or sliced_mesh.n_points == 0:
        logger.warning("No data to export: empty sliced mesh")
        return {}
    
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for CSV export. Install with: pip install pandas")
        return {}
    
    # Create export directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    points = sliced_mesh.points
    df_data = {
        "X": points[:, 0],
        "Y": points[:, 1],
        "Z": points[:, 2]
    }
    
    # Add scalar field if available
    if scalar_field and scalar_field in sliced_mesh.array_names:
        df_data[scalar_field] = sliced_mesh[scalar_field]
        logger.info(f"Exporting scalar field '{scalar_field}' with {len(sliced_mesh[scalar_field])} values")
    else:
        logger.warning(f"Scalar field '{scalar_field}' not found in mesh. Available fields: {sliced_mesh.array_names}")
    
    # Add any other point/cell data arrays
    for array_name in sliced_mesh.array_names:
        if array_name not in df_data:
            df_data[array_name] = sliced_mesh[array_name]
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if strike is not None and dip is not None:
        filename = f"section_strike{strike:.0f}_dip{dip:.0f}_{timestamp}.csv"
    elif orientation != "unknown":
        filename = f"section_{orientation.replace('-', '_')}_{timestamp}.csv"
    else:
        filename = f"section_{timestamp}.csv"
    
    csv_path = save_path / filename
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Exported cross-section data to CSV: {csv_path} ({len(df)} points)")
    
    return {"csv": csv_path}


def export_cross_section_screenshot(
    plotter: pv.Plotter,
    strike: Optional[float] = None,
    dip: Optional[float] = None,
    orientation: str = "unknown",
    save_dir: Union[str, Path] = "exports"
) -> Dict[str, Path]:
    """
    Export a screenshot of the current plotter scene (with cross-section).
    
    Args:
        plotter: PyVista plotter with rendered scene
        strike: Strike angle in degrees (for geological sections)
        dip: Dip angle in degrees (for geological sections)
        orientation: Orientation description (e.g., "north-south", "strike-dip")
        save_dir: Directory to save exports
        
    Returns:
        Dictionary with 'image' key pointing to saved image file path
    """
    if plotter is None:
        logger.warning("No plotter provided for screenshot export")
        return {}
    
    try:
        # Create export directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if strike is not None and dip is not None:
            filename = f"section_strike{strike:.0f}_dip{dip:.0f}_{timestamp}.png"
        elif orientation != "unknown":
            filename = f"section_{orientation.replace('-', '_')}_{timestamp}.png"
        else:
            filename = f"section_{timestamp}.png"
        
        img_path = save_path / filename
        
        # Render and save screenshot
        plotter.render()  # Ensure scene is rendered
        plotter.screenshot(str(img_path))
        
        logger.info(f"Exported cross-section screenshot: {img_path}")
        
        return {"image": img_path}
    except Exception as e:
        logger.error(f"Error exporting screenshot: {e}")
        return {}


def get_orientation_normal(orientation: str, azimuth: Optional[float] = None, 
                          strike: Optional[float] = None, dip: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Get normal vector for common orientations.
    
    Args:
        orientation: One of 'north-south', 'east-west', 'custom', 'strike-dip'
        azimuth: For 'custom', azimuth in degrees (0 = North, clockwise)
        strike: For 'strike-dip', strike in degrees (0 = North, clockwise)
        dip: For 'strike-dip', dip in degrees (90 = vertical, <90 = dipping)
        
    Returns:
        Normal vector tuple (nx, ny, nz)
    """
    orientation = orientation.lower()
    
    if orientation == 'north-south' or orientation == 'n-s':
        return (1.0, 0.0, 0.0)  # Normal to E-W plane (shows N-S section)
    elif orientation == 'east-west' or orientation == 'e-w':
        return (0.0, 1.0, 0.0)  # Normal to N-S plane (shows E-W section)
    elif orientation == 'strike-dip' and strike is not None and dip is not None:
        # Convert strike/dip to plane normal in 3D (right-hand rule)
        # Strike measured clockwise from north, dip toward right of strike
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Plane normal components
        nx = np.sin(dip_rad) * np.sin(strike_rad)
        ny = np.sin(dip_rad) * np.cos(strike_rad)
        nz = np.cos(dip_rad)
        normal = (nx, ny, nz)
        
        # Normalize
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        return tuple(normal)
    elif orientation == 'custom' and azimuth is not None:
        # Convert azimuth to normal vector
        # Azimuth 0 = North, clockwise
        # Normal should be perpendicular to the section line
        azimuth_rad = np.radians(azimuth)
        # For a vertical plane with azimuth, normal is perpendicular to the dip direction
        normal_x = np.cos(azimuth_rad + np.pi/2)  # Perpendicular to azimuth
        normal_y = np.sin(azimuth_rad + np.pi/2)
        return (normal_x, normal_y, 0.0)
    else:
        logger.warning(f"Unknown orientation '{orientation}', defaulting to N-S")
        return (1.0, 0.0, 0.0)

