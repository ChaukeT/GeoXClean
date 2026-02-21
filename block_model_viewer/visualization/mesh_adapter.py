"""
Mesh Adapter - Converts MeshPayload to PyVista meshes.

This adapter handles all PyVista mesh creation logic.
"""

import logging
import numpy as np
import pyvista as pv
from typing import Tuple, Optional

from .render_payloads import MeshPayload

logger = logging.getLogger(__name__)


class MeshAdapter:
    """Adapter for converting MeshPayload to PyVista PolyData."""
    
    def to_pv_mesh(self, payload: MeshPayload) -> Tuple[pv.PolyData, Optional[np.ndarray]]:
        """
        Convert MeshPayload to PyVista PolyData.
        
        Args:
            payload: Mesh payload containing vertices, faces, scalars
            
        Returns:
            Tuple of (pv.PolyData mesh, scalars array)
        """
        try:
            vertices = np.asarray(payload.vertices, dtype=np.float64)
            
            if payload.faces is not None and len(payload.faces) > 0:
                # Create mesh with faces
                faces = np.asarray(payload.faces)
                
                # CRITICAL FIX: Convert (N, 3) face array to PyVista format [3, i, j, k, 3, ...]
                # PyVista expects faces as a flat array with face counts prepended
                # Check if already in PyVista format (1D array) or in (N, 3) format
                if faces.ndim == 2:
                    # Convert from (N, 3) to PyVista format [3, i0, i1, i2, 3, j0, j1, j2, ...]
                    n_faces = len(faces)
                    n_verts_per_face = faces.shape[1]  # Usually 3 for triangles, 4 for quads
                    
                    # Prepend the vertex count to each face
                    faces_pv = np.hstack([
                        np.full((n_faces, 1), n_verts_per_face, dtype=np.int64),
                        faces.astype(np.int64)
                    ]).flatten()
                    
                    logger.debug(f"Converted {n_faces} faces from (N,{n_verts_per_face}) to PyVista format")
                else:
                    # Assume already in PyVista format
                    faces_pv = faces.astype(np.int64)
                
                mesh = pv.PolyData(vertices, faces_pv)
                logger.debug(f"Created mesh with {mesh.n_points} points, {mesh.n_faces} faces")
            else:
                # Point cloud
                mesh = pv.PolyData(vertices)
                logger.debug(f"Created point cloud with {mesh.n_points} points")
            
            # Add scalars if provided
            scalars = None
            if payload.scalars is not None:
                scalars = np.asarray(payload.scalars)
                if len(scalars) == len(vertices):
                    mesh[payload.name] = scalars
                else:
                    logger.warning(f"Scalar length {len(scalars)} doesn't match vertex count {len(vertices)}")
            
            # Add colors if provided
            if payload.colors is not None:
                colors = np.asarray(payload.colors)
                if colors.shape == (len(vertices), 3):
                    mesh['colors'] = colors
            
            return mesh, scalars
            
        except Exception as e:
            logger.error(f"Error converting mesh payload to PyVista: {e}", exc_info=True)
            raise
    
    def create_actor(self, payload: MeshPayload, plotter: pv.Plotter) -> pv.Actor:
        """
        Create and add actor to plotter from payload.
        
        Args:
            payload: Mesh payload
            plotter: PyVista plotter
            
        Returns:
            PyVista actor
        """
        mesh, scalars = self.to_pv_mesh(payload)
        
        # Determine scalar field name
        scalar_name = payload.name if scalars is not None else None
        
        # Add to plotter
        actor = plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            opacity=payload.opacity,
            show_edges=False,
            cmap=payload.metadata.get('colormap', 'viridis'),
            show_scalar_bar=False  # LegendManager handles scalar bars
        )
        
        # Set visibility
        actor.SetVisibility(1 if payload.visible else 0)
        
        return actor

