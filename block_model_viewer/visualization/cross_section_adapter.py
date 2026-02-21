"""
Cross-Section Adapter - Converts CrossSectionPayload to PyVista geometries.

This adapter handles all PyVista cross-section creation logic.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Optional

from .render_payloads import CrossSectionPayload

logger = logging.getLogger(__name__)


class CrossSectionAdapter:
    """Adapter for converting CrossSectionPayload to PyVista geometries."""
    
    def build_section(self, payload: CrossSectionPayload) -> pv.PolyData:
        """
        Build PyVista PolyData from cross-section payload.
        
        Args:
            payload: Cross-section payload
            
        Returns:
            PyVista PolyData
        """
        try:
            points = np.asarray(payload.points, dtype=np.float64)
            
            if payload.lines is not None:
                # Polyline with connectivity
                lines = np.asarray(payload.lines, dtype=np.int32)
                mesh = pv.PolyData(points, lines=lines)
            else:
                # Simple point cloud or line strip
                mesh = pv.PolyData(points)
            
            # Add scalars if provided
            if payload.scalars is not None:
                scalars = np.asarray(payload.scalars)
                if len(scalars) == len(points):
                    mesh[payload.name] = scalars
            
            # Extrude if thickness > 0
            if payload.thickness > 0:
                # Extrude along normal (simplified - assumes Z-up)
                normal = np.array([0, 0, 1])
                mesh = mesh.extrude(normal * payload.thickness)
            
            return mesh
            
        except Exception as e:
            logger.error(f"Error building cross-section: {e}", exc_info=True)
            raise
    
    def create_actor(self, payload: CrossSectionPayload, plotter: pv.Plotter) -> pv.Actor:
        """
        Create and add actor to plotter from payload.
        
        Args:
            payload: Cross-section payload
            plotter: PyVista plotter
            
        Returns:
            PyVista actor
        """
        mesh = self.build_section(payload)
        
        # Determine color
        color = payload.color if payload.color is not None else payload.metadata.get('color', 'white')
        
        # Determine scalar field
        scalar_name = payload.name if payload.scalars is not None else None
        
        # Add to plotter
        actor = plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            color=color if scalar_name is None else None,
            opacity=payload.opacity,
            line_width=payload.metadata.get('line_width', 2.0),
            show_scalar_bar=False  # LegendManager handles scalar bars
        )
        
        # Set visibility
        actor.SetVisibility(1 if payload.visible else 0)
        
        return actor

