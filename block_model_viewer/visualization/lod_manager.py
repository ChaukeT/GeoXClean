"""
Level of Detail (LOD) Manager for performance optimization.

Manages LOD selection and mesh simplification based on camera distance,
cell count, and user quality settings.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pyvista as pv

from .scene_layer import SceneLayer

logger = logging.getLogger(__name__)


class LODManager:
    """
    Manages Level of Detail rendering for performance optimization.
    
    Selects appropriate LOD level based on:
    - Number of cells/triangles
    - Camera distance/zoom level
    - User quality settings
    """
    
    # LOD levels: 0 = highest detail, higher = lower detail
    LOD_LEVELS = {
        0: {"name": "Full", "reduction": 0.0},      # No reduction
        1: {"name": "High", "reduction": 0.3},      # 30% reduction
        2: {"name": "Medium", "reduction": 0.6},   # 60% reduction
        3: {"name": "Low", "reduction": 0.85},     # 85% reduction
        4: {"name": "Very Low", "reduction": 0.95} # 95% reduction
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LOD Manager.
        
        Args:
            config: Configuration dict with performance settings
        """
        self.config = config or {}
        
        # Quality setting: 0.0 = low quality (aggressive LOD), 1.0 = high quality (minimal LOD)
        self.quality_setting = self.config.get('lod_quality', 0.7)
        
        # Thresholds for LOD switching
        self.cell_count_thresholds = {
            0: 50_000,      # Full detail up to 50k cells
            1: 200_000,     # High detail up to 200k cells
            2: 500_000,     # Medium detail up to 500k cells
            3: 1_000_000,   # Low detail up to 1M cells
            4: float('inf') # Very low for >1M cells
        }
        
        # Camera distance thresholds (normalized to scene bounds)
        self.distance_thresholds = {
            0: 0.1,   # Close: full detail
            1: 0.3,   # Medium-close: high detail
            2: 0.6,   # Medium: medium detail
            3: 0.9,   # Far: low detail
            4: 1.0    # Very far: very low detail
        }
        
        # Cache for LOD meshes to avoid recomputation
        self._lod_cache: Dict[Tuple[str, int], pv.PolyData] = {}
        
        # Hysteresis to prevent LOD thrashing
        self._current_lod_levels: Dict[str, int] = {}
        self._lod_hysteresis = 0.1  # 10% threshold before switching
        
    def select_lod(self, layer: SceneLayer, camera_metadata: Dict[str, Any]) -> int:
        """
        Select appropriate LOD level for a layer.
        
        Args:
            layer: SceneLayer to evaluate
            camera_metadata: Camera info dict with 'distance', 'position', etc.
            
        Returns:
            LOD level (0-4)
        """
        try:
            # Get mesh data
            mesh = None
            if hasattr(layer, 'data') and layer.data is not None:
                if isinstance(layer.data, pv.PolyData) or isinstance(layer.data, pv.UnstructuredGrid):
                    mesh = layer.data
                elif isinstance(layer.data, dict) and 'mesh' in layer.data:
                    mesh = layer.data['mesh']
            
            if mesh is None:
                return 0  # No mesh, use full detail
            
            # Get cell count
            cell_count = mesh.n_cells if hasattr(mesh, 'n_cells') else 0
            
            # Get camera distance (normalized to scene bounds)
            distance = camera_metadata.get('distance', 0.0)
            scene_bounds = camera_metadata.get('scene_bounds')
            
            normalized_distance = 0.0
            if scene_bounds and len(scene_bounds) == 6:
                # Calculate scene diagonal
                dx = scene_bounds[1] - scene_bounds[0]
                dy = scene_bounds[3] - scene_bounds[2]
                dz = scene_bounds[5] - scene_bounds[4]
                diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
                if diagonal > 0:
                    normalized_distance = min(distance / diagonal, 1.0)
            
            # Determine LOD based on cell count
            lod_by_cells = 0
            for level, threshold in sorted(self.cell_count_thresholds.items()):
                if cell_count <= threshold:
                    lod_by_cells = level
                    break
            
            # Determine LOD based on camera distance
            lod_by_distance = 0
            for level, threshold in sorted(self.distance_thresholds.items()):
                if normalized_distance <= threshold:
                    lod_by_distance = level
                    break
            
            # Use the higher (more aggressive) LOD level
            suggested_lod = max(lod_by_cells, lod_by_distance)
            
            # Apply quality setting adjustment
            # High quality (1.0) reduces LOD by 1 level, low quality (0.0) increases by 1
            quality_adjustment = int((1.0 - self.quality_setting) * 2)
            suggested_lod = min(4, max(0, suggested_lod + quality_adjustment))
            
            # Apply hysteresis to prevent thrashing
            layer_key = layer.name
            current_lod = self._current_lod_levels.get(layer_key, suggested_lod)
            
            # Only switch if difference is significant
            if abs(suggested_lod - current_lod) >= 2:
                # Large difference, switch immediately
                final_lod = suggested_lod
            elif abs(suggested_lod - current_lod) == 1:
                # Small difference, use hysteresis
                if suggested_lod > current_lod:
                    # Moving to lower detail - switch if significantly worse
                    threshold = current_lod + self._lod_hysteresis
                    final_lod = suggested_lod if suggested_lod >= threshold else current_lod
                else:
                    # Moving to higher detail - switch if significantly better
                    threshold = current_lod - self._lod_hysteresis
                    final_lod = suggested_lod if suggested_lod <= threshold else current_lod
            else:
                final_lod = current_lod
            
            self._current_lod_levels[layer_key] = final_lod
            return final_lod
            
        except Exception as e:
            logger.warning(f"Error selecting LOD for layer {layer.name}: {e}", exc_info=True)
            return 0  # Fallback to full detail
    
    def get_lod_mesh(self, layer: SceneLayer, lod_level: int) -> Optional[pv.PolyData]:
        """
        Get LOD mesh for a layer at specified level.
        
        Uses cache to avoid recomputation. If not cached, returns None
        (caller should compute and cache it).
        
        Args:
            layer: SceneLayer
            lod_level: LOD level (0-4)
            
        Returns:
            Decimated mesh or None if not cached
        """
        cache_key = (layer.name, lod_level)
        return self._lod_cache.get(cache_key)
    
    def cache_lod_mesh(self, layer: SceneLayer, lod_level: int, mesh: pv.PolyData) -> None:
        """
        Cache a computed LOD mesh.
        
        Args:
            layer: SceneLayer
            lod_level: LOD level
            mesh: Decimated mesh to cache
        """
        cache_key = (layer.name, lod_level)
        self._lod_cache[cache_key] = mesh
    
    def clear_cache(self, layer_name: Optional[str] = None) -> None:
        """
        Clear LOD cache.
        
        Args:
            layer_name: If provided, clear only this layer's cache. Otherwise clear all.
        """
        if layer_name:
            keys_to_remove = [k for k in self._lod_cache.keys() if k[0] == layer_name]
            for key in keys_to_remove:
                del self._lod_cache[key]
        else:
            self._lod_cache.clear()
    
    def set_quality(self, quality: float) -> None:
        """
        Set quality setting (0.0 = low quality, 1.0 = high quality).
        
        Args:
            quality: Quality value in [0.0, 1.0]
        """
        self.quality_setting = max(0.0, min(1.0, quality))
        # Clear cache when quality changes
        self.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_meshes': len(self._lod_cache),
            'layers': len(set(k[0] for k in self._lod_cache.keys())),
            'total_memory_mb': sum(
                mesh.n_points * 3 * 4 + mesh.n_cells * 8  # Rough estimate
                for mesh in self._lod_cache.values()
                if hasattr(mesh, 'n_points')
            ) / (1024 * 1024) if self._lod_cache else 0
        }

