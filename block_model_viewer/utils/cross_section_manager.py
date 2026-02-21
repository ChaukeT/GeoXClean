"""
Cross-Section Manager

Manages named cross-sections (plane specs + thickness), quick rendering, and exports.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionSpec:
    """Specification for a cross-section plane."""
    name: str
    plane_type: str  # 'X', 'Y', 'Z', 'arbitrary'
    position: float  # For axis-aligned planes
    thickness: float  # Slice thickness
    normal: Tuple[float, float, float]  # For arbitrary planes
    origin: Tuple[float, float, float]  # For arbitrary planes
    description: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrossSectionSpec':
        """Deserialize from dictionary."""
        return cls(**data)


class CrossSectionManager:
    """
    Manages cross-sections for block model visualization.
    
    Features:
    - Named cross-sections (save, load, delete)
    - Axis-aligned and arbitrary plane support
    - Quick section rendering
    - Export section data and images
    """
    
    def __init__(self):
        self.block_df: Optional[pd.DataFrame] = None
        self.grid_spec: Optional[Dict] = None
        self.sections: Dict[str, CrossSectionSpec] = {}
        logger.info("Initialized CrossSectionManager")
    
    def set_block_model(self, block_df: pd.DataFrame, grid_spec: Optional[Dict] = None):
        """Set the block model data."""
        if block_df is None:
            logger.warning("set_block_model called with None — skipping")
            return
        self.block_df = block_df.copy()
        self.grid_spec = grid_spec
        logger.info(f"Set block model: {len(self.block_df)} blocks")
    
    def create_section(self, name: str, 
                      plane_type: str,
                      position: float = 0,
                      thickness: float = 10,
                      normal: Optional[Tuple[float, float, float]] = None,
                      origin: Optional[Tuple[float, float, float]] = None,
                      description: str = "") -> bool:
        """
        Create a new cross-section specification.
        
        Args:
            name: Name for the section
            plane_type: 'X', 'Y', 'Z', or 'arbitrary'
            position: Position along axis (for axis-aligned planes)
            thickness: Slice thickness
            normal: Plane normal vector (for arbitrary planes)
            origin: Point on plane (for arbitrary planes)
            description: Optional description
        
        Returns:
            True if successful
        """
        if plane_type.upper() in ['X', 'Y', 'Z']:
            # Axis-aligned plane
            if plane_type.upper() == 'X':
                normal = (1, 0, 0)
                origin = (position, 0, 0)
            elif plane_type.upper() == 'Y':
                normal = (0, 1, 0)
                origin = (0, position, 0)
            else:  # Z
                normal = (0, 0, 1)
                origin = (0, 0, position)
        else:
            # Arbitrary plane
            if normal is None or origin is None:
                logger.error("Arbitrary plane requires normal and origin")
                return False
        
        spec = CrossSectionSpec(
            name=name,
            plane_type=plane_type.upper(),
            position=position,
            thickness=thickness,
            normal=normal,
            origin=origin,
            description=description,
            created_at=pd.Timestamp.now().isoformat()
        )
        
        self.sections[name] = spec
        logger.info(f"Created cross-section '{name}': {plane_type} at {position}, thickness={thickness}")
        return True
    
    def delete_section(self, name: str) -> bool:
        """Delete a named section."""
        if name in self.sections:
            del self.sections[name]
            logger.info(f"Deleted cross-section '{name}'")
            return True
        return False
    
    def get_section(self, name: str) -> Optional[CrossSectionSpec]:
        """Get a section specification."""
        return self.sections.get(name)
    
    def slice_blocks(self, section_name: str) -> Optional[pd.DataFrame]:
        """
        Get blocks within a cross-section slice.
        
        Args:
            section_name: Name of the section
        
        Returns:
            DataFrame of blocks within the slice, or None
        """
        if self.block_df is None:
            logger.warning("No block model set")
            return None
        
        if section_name not in self.sections:
            logger.warning(f"Section '{section_name}' not found")
            return None
        
        spec = self.sections[section_name]
        
        # Get coordinate columns
        x_col = self._find_coordinate_column(['X', 'XC', 'x', 'xc'])
        y_col = self._find_coordinate_column(['Y', 'YC', 'y', 'yc'])
        z_col = self._find_coordinate_column(['Z', 'ZC', 'z', 'zc'])
        
        if not all([x_col, y_col, z_col]):
            logger.error("Could not find coordinate columns")
            return None
        
        # Calculate distance from plane
        points = np.column_stack([
            self.block_df[x_col].values,
            self.block_df[y_col].values,
            self.block_df[z_col].values
        ])
        
        origin = np.array(spec.origin)
        normal = np.array(spec.normal)
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Distance from plane (signed)
        distances = np.dot(points - origin, normal)
        
        # Select blocks within thickness
        half_thick = spec.thickness / 2
        mask = np.abs(distances) <= half_thick
        
        sliced_df = self.block_df[mask].copy()
        sliced_df['_distance_from_plane'] = distances[mask]
        
        logger.info(f"Sliced section '{section_name}': {len(sliced_df)} blocks")
        return sliced_df
    
    def export_section_csv(self, section_name: str, filepath: Path) -> bool:
        """
        Export cross-section data to CSV.
        
        Args:
            section_name: Name of the section
            filepath: Output file path
        
        Returns:
            True if successful
        """
        sliced_df = self.slice_blocks(section_name)
        
        if sliced_df is None or sliced_df.empty:
            logger.warning(f"No data in section '{section_name}' to export")
            return False
        
        try:
            sliced_df.to_csv(filepath, index=False)
            logger.info(f"Exported section '{section_name}' ({len(sliced_df)} blocks) to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting section to CSV: {e}", exc_info=True)
            return False
    
    def export_section_image(self, section_name: str, 
                            filepath: Path,
                            property_name: str,
                            dpi: int = 150,
                            figsize: Tuple[int, int] = (10, 8)) -> bool:
        """
        Export cross-section as image.
        
        Args:
            section_name: Name of the section
            filepath: Output image path
            property_name: Property to visualize
            dpi: Image DPI
            figsize: Figure size in inches
        
        Returns:
            True if successful
        """
        sliced_df = self.slice_blocks(section_name)
        
        if sliced_df is None or sliced_df.empty:
            logger.warning(f"No data in section '{section_name}' to export")
            return False
        
        if property_name not in sliced_df.columns:
            logger.error(f"Property '{property_name}' not found in section data")
            return False
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm
            
            spec = self.sections[section_name]
            
            # Get coordinate columns
            x_col = self._find_coordinate_column(['X', 'XC', 'x', 'xc'])
            y_col = self._find_coordinate_column(['Y', 'YC', 'y', 'yc'])
            z_col = self._find_coordinate_column(['Z', 'ZC', 'z', 'zc'])
            
            # Determine which coordinates to plot based on plane type
            if spec.plane_type == 'X':
                plot_x, plot_y = y_col, z_col
                xlabel, ylabel = 'Y', 'Z'
            elif spec.plane_type == 'Y':
                plot_x, plot_y = x_col, z_col
                xlabel, ylabel = 'X', 'Z'
            else:  # Z or arbitrary
                plot_x, plot_y = x_col, y_col
                xlabel, ylabel = 'X', 'Y'
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Scatter plot
            scatter = ax.scatter(
                sliced_df[plot_x],
                sliced_df[plot_y],
                c=sliced_df[property_name],
                cmap='viridis',
                s=20,
                alpha=0.7
            )
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{section_name} - {property_name}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(property_name)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Exported section '{section_name}' image to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting section image: {e}", exc_info=True)
            return False
    
    def get_section_mesh(self, section_name: str, property_name: Optional[str] = None, as_cubes: bool = True):
        """
        Build a PyVista UnstructuredGrid of hexahedral cells for the cross-section,
        matching the block model's cube geometry. Always returns cubes, not points.
        
        Args:
            section_name: Name of the section
            property_name: Optional property to attach as cell scalars
        
        Returns:
            PyVista UnstructuredGrid or None
        """
        sliced_df = self.slice_blocks(section_name)
        if sliced_df is None or sliced_df.empty:
            return None

        try:
            import pyvista as pv

            # Coordinate columns (centers)
            x_col = self._find_coordinate_column(['X', 'XC', 'x', 'xc'])
            y_col = self._find_coordinate_column(['Y', 'YC', 'y', 'yc'])
            z_col = self._find_coordinate_column(['Z', 'ZC', 'z', 'zc'])
            if not all([x_col, y_col, z_col]):
                logger.error("Could not find coordinate columns for section mesh")
                return None

            # Optional point cloud mode
            if not as_cubes:
                points = np.column_stack([
                    sliced_df[x_col].values,
                    sliced_df[y_col].values,
                    sliced_df[z_col].values
                ])
                mesh = pv.PolyData(points)
                if property_name and property_name in sliced_df.columns:
                    try:
                        mesh[property_name] = sliced_df[property_name].values
                    except Exception:
                        pass
                return mesh

            # Dimension columns (per-block size)
            def find_dim_col(candidates):
                for c in candidates:
                    if c in sliced_df.columns:
                        return c
                return None

            dx_col = find_dim_col(['DX', 'dx', 'XINC', 'xinc'])
            dy_col = find_dim_col(['DY', 'dy', 'YINC', 'yinc'])
            dz_col = find_dim_col(['DZ', 'dz', 'ZINC', 'zinc'])

            # Fall back to grid_spec increments if columns missing
            def fallback_inc(axis):
                try:
                    if self.grid_spec and f"{axis}inc" in self.grid_spec:
                        return float(self.grid_spec[f"{axis}inc"])
                except Exception:
                    pass
                # As a last resort, estimate from nearest-neighbor spacing along axis
                try:
                    vals = np.asarray(sliced_df[{'x': x_col, 'y': y_col, 'z': z_col}[axis]].values)
                    if len(vals) > 1:
                        diffs = np.diff(np.sort(vals))
                        diffs = diffs[diffs > 0]
                        if diffs.size > 0:
                            return float(np.median(diffs))
                except Exception:
                    pass
                return 1.0

            dx_vals = sliced_df[dx_col].values if dx_col else np.full(len(sliced_df), fallback_inc('x'))
            dy_vals = sliced_df[dy_col].values if dy_col else np.full(len(sliced_df), fallback_inc('y'))
            dz_vals = sliced_df[dz_col].values if dz_col else np.full(len(sliced_df), fallback_inc('z'))

            cx = sliced_df[x_col].values
            cy = sliced_df[y_col].values
            cz = sliced_df[z_col].values
            hx = dx_vals / 2.0
            hy = dy_vals / 2.0
            hz = dz_vals / 2.0

            # Build corners for each block (8 points per cell)
            corners = np.stack([
                np.column_stack([cx - hx, cy - hy, cz - hz]),
                np.column_stack([cx + hx, cy - hy, cz - hz]),
                np.column_stack([cx + hx, cy + hy, cz - hz]),
                np.column_stack([cx - hx, cy + hy, cz - hz]),
                np.column_stack([cx - hx, cy - hy, cz + hz]),
                np.column_stack([cx + hx, cy - hy, cz + hz]),
                np.column_stack([cx + hx, cy + hy, cz + hz]),
                np.column_stack([cx - hx, cy + hy, cz + hz]),
            ], axis=1)  # shape (n, 8, 3)

            n = corners.shape[0]
            points = corners.reshape(n * 8, 3)
            base = (np.arange(n, dtype=np.int64) * 8)[:, None]
            local = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)[None, :]
            cell_conn = (base + local).reshape(n, 8)
            cells = np.hstack((np.full((n, 1), 8, dtype=np.int64), cell_conn)).ravel()
            cell_types = np.full(n, int(pv.CellType.HEXAHEDRON), dtype=np.uint8)

            grid = pv.UnstructuredGrid(cells, cell_types, points)

            # Attach property as cell data if provided
            if property_name and property_name in sliced_df.columns:
                try:
                    vals = np.asarray(sliced_df[property_name].values)
                    if len(vals) == n:
                        grid.cell_data[property_name] = vals
                except Exception:
                    pass

            return grid

        except Exception as e:
            logger.error(f"Error creating section mesh: {e}", exc_info=True)
            return None
    
    def _find_coordinate_column(self, candidates: List[str]) -> Optional[str]:
        """Find first matching coordinate column name."""
        if self.block_df is None:
            return None
        
        for candidate in candidates:
            if candidate in self.block_df.columns:
                return candidate
        return None
    
    def save_sections_to_file(self, filepath: Path) -> bool:
        """Save all sections to JSON file."""
        try:
            import json
            
            data = {
                name: spec.to_dict()
                for name, spec in self.sections.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.sections)} sections to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sections: {e}", exc_info=True)
            return False
    
    def load_sections_from_file(self, filepath: Path) -> bool:
        """Load sections from JSON file."""
        try:
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.sections.clear()
            
            for name, spec_dict in data.items():
                self.sections[name] = CrossSectionSpec.from_dict(spec_dict)
            
            logger.info(f"Loaded {len(self.sections)} sections from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sections: {e}", exc_info=True)
            return False
