"""
Mesh parser for OBJ and GLTF files using trimesh.
"""

import trimesh
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .base_parser import BaseParser
from ..models.block_model import BlockModel, BlockMetadata

logger = logging.getLogger(__name__)


class MeshParser(BaseParser):
    """
    Parser for mesh files (OBJ, GLTF) using trimesh.
    
    Converts mesh data into block model representation.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.obj', '.gltf', '.glb', '.ply', '.stl', '.off']
        self.format_name = "Mesh"
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a mesh file into a BlockModel object.
        
        Args:
            file_path: Path to the mesh file
            **kwargs: Additional options:
                - voxel_size: Size of voxels for voxelization (default: auto)
                - voxelize: Whether to voxelize the mesh (default: True)
                - sample_points: Number of points to sample for point cloud mode
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        voxelize = kwargs.get('voxelize', True)
        voxel_size = kwargs.get('voxel_size', None)
        sample_points = kwargs.get('sample_points', 10000)
        
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(file_path))
            logger.info(f"Loaded mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
            
            # Create metadata
            metadata = BlockMetadata(
                source_file=str(file_path),
                file_format="Mesh",
                coordinate_system="unknown",
                units="unknown"
            )
            
            block_model = BlockModel(metadata)
            
            if voxelize and hasattr(mesh, 'voxelized'):
                # Voxelize the mesh
                self._extract_voxels(block_model, mesh, voxel_size)
            else:
                # Sample points from mesh surface
                self._extract_surface_points(block_model, mesh, sample_points)
            
            logger.info(f"Parsed mesh: {block_model.block_count} blocks")
            return block_model
            
        except Exception as e:
            logger.error(f"Error parsing mesh file {file_path}: {e}")
            raise ValueError(f"Failed to parse mesh file: {e}")
    
    def _extract_voxels(self, block_model: BlockModel, mesh: trimesh.Trimesh, 
                       voxel_size: Optional[float]) -> None:
        """
        Extract voxelized representation of the mesh.
        
        Args:
            block_model: BlockModel to populate
            mesh: Trimesh object
            voxel_size: Size of voxels (None for auto)
        """
        try:
            # Voxelize the mesh
            if voxel_size is None:
                # Auto-determine voxel size based on mesh bounds
                bounds = mesh.bounds
                diagonal = np.linalg.norm(bounds[1] - bounds[0])
                voxel_size = diagonal / 50  # 50 voxels along diagonal
            
            voxels = mesh.voxelized(pitch=voxel_size)
            
            # Get voxel centers
            voxel_centers = voxels.points
            
            # All voxels have the same size
            dimensions = np.tile([voxel_size, voxel_size, voxel_size], (len(voxel_centers), 1))
            
            # Set geometry
            block_model.set_geometry(voxel_centers, dimensions)
            
            # Add voxel density as property
            if hasattr(voxels, 'fill'):
                block_model.add_property('density', voxels.fill.astype(float))
            
        except Exception as e:
            logger.warning(f"Voxelization failed: {e}, falling back to surface sampling")
            self._extract_surface_points(block_model, mesh, 10000)
    
    def _extract_surface_points(self, block_model: BlockModel, mesh: trimesh.Trimesh, 
                               sample_points: int) -> None:
        """
        Extract surface points from the mesh.
        
        Args:
            block_model: BlockModel to populate
            mesh: Trimesh object
            sample_points: Number of points to sample
        """
        # Sample points from mesh surface
        points, face_indices = mesh.sample(sample_points, return_index=True)
        
        # Calculate point normals for additional properties
        if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
            normals = mesh.face_normals[face_indices]
            block_model.add_property('normal_x', normals[:, 0])
            block_model.add_property('normal_y', normals[:, 1])
            block_model.add_property('normal_z', normals[:, 2])
        
        # Estimate local block size based on mesh density
        # Use a simple heuristic: find nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points))).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_distance = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
        
        # Use average distance as block size
        dimensions = np.column_stack([avg_distance, avg_distance, avg_distance])
        
        # Set geometry
        block_model.set_geometry(points, dimensions)
        
        # Add distance-based properties
        block_model.add_property('local_density', 1.0 / (avg_distance + 1e-6))
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get mesh file information."""
        info = super().get_file_info(file_path)
        
        try:
            # Try to load mesh for preview
            mesh = trimesh.load(str(file_path))
            info.update({
                "vertices": mesh.vertices.shape[0],
                "faces": mesh.faces.shape[0],
                "bounds": mesh.bounds.tolist(),
                "volume": float(mesh.volume) if hasattr(mesh, 'volume') else 0.0,
                "surface_area": float(mesh.surface_area) if hasattr(mesh, 'surface_area') else 0.0,
                "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else False
            })
        except Exception as e:
            logger.warning(f"Could not preview mesh file: {e}")
            info.update({
                "vertices": 0,
                "faces": 0,
                "bounds": None,
                "volume": 0.0,
                "surface_area": 0.0,
                "is_watertight": False
            })
        
        return info
