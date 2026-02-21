"""
Pit Adapter - Converts PitShellPayload to PyVista pit geometries.

This adapter handles all PyVista pit shell creation logic.
"""

import logging
import numpy as np
import pyvista as pv
from typing import List

from .render_payloads import PitShellPayload, MeshPayload
from .mesh_adapter import MeshAdapter

logger = logging.getLogger(__name__)


class PitAdapter:
    """Adapter for converting PitShellPayload to PyVista pit geometries."""
    
    def __init__(self):
        self.mesh_adapter = MeshAdapter()
    
    def build_pit_shell(self, payload: PitShellPayload) -> List[pv.Actor]:
        """
        Build PyVista actors from pit shell payload.
        
        Args:
            payload: Pit shell payload
            
        Returns:
            List of PyVista actors
        """
        actors = []
        
        try:
            # Build bench meshes
            for bench in payload.benches:
                mesh, _ = self.mesh_adapter.to_pv_mesh(bench)
                actors.append(mesh)
            
            # Build phase meshes
            for phase in payload.phases:
                mesh, _ = self.mesh_adapter.to_pv_mesh(phase)
                actors.append(mesh)
            
            return actors
            
        except Exception as e:
            logger.error(f"Error building pit shell: {e}", exc_info=True)
            raise
    
    def create_actors(self, payload: PitShellPayload, plotter: pv.Plotter) -> List[pv.Actor]:
        """
        Create and add actors to plotter from payload.
        
        Args:
            payload: Pit shell payload
            plotter: PyVista plotter
            
        Returns:
            List of PyVista actors
        """
        actors = []
        
        try:
            # Build and add bench meshes
            for bench in payload.benches:
                actor = self.mesh_adapter.create_actor(bench, plotter)
                if payload.wireframe:
                    actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetOpacity(payload.opacity)
                actors.append(actor)
            
            # Build and add phase meshes
            for phase in payload.phases:
                actor = self.mesh_adapter.create_actor(phase, plotter)
                if payload.wireframe:
                    actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetOpacity(payload.opacity)
                actors.append(actor)
            
            return actors
            
        except Exception as e:
            logger.error(f"Error creating pit shell actors: {e}", exc_info=True)
            raise

