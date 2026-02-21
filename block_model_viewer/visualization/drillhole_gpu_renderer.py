"""
GPU-Optimized Drillhole Renderer

High-performance drillhole visualization using:
- GPU instancing for efficient rendering of thousands of intervals
- Batched geometry updates (no full scene rebuilds)
- GPU picking via color ID buffer
- Shader-based hover/selection highlighting
- LOD switching and frustum culling

Performance targets:
- 2k holes (30k intervals): 60 FPS
- 5k holes (80k intervals): 40-60 FPS
- 10k holes (150k intervals): 30-45 FPS
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pyvista as pv
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================

class RenderQuality(IntEnum):
    """LOD quality levels for drillhole rendering."""
    LOW = 4       # 4 sides - for distant holes
    MEDIUM = 8    # 8 sides - for mid-distance
    HIGH = 16     # 16 sides - for close-up (default)
    ULTRA = 32    # 32 sides - for very close inspection


class SelectionState(IntEnum):
    """Selection state for intervals."""
    NONE = 0
    HOVERED = 1
    SELECTED = 2
    HIDDEN = 3


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DrillholeInterval:
    """Represents a single drillhole interval for rendering."""
    hole_id: str
    interval_index: int
    start_point: np.ndarray  # (x, y, z)
    end_point: np.ndarray    # (x, y, z)
    depth_from: float
    depth_to: float
    radius: float
    color_id: int            # Unique ID for GPU picking
    lith_code: Optional[str] = None
    assay_value: Optional[float] = None
    declust_weight: Optional[float] = None  # Declustering weight for statistical analysis
    selection_state: SelectionState = SelectionState.NONE
    
    @property
    def length(self) -> float:
        return abs(self.depth_to - self.depth_from)
    
    @property
    def center(self) -> np.ndarray:
        return (self.start_point + self.end_point) / 2
    
    @property
    def direction(self) -> np.ndarray:
        vec = self.end_point - self.start_point
        length = np.linalg.norm(vec)
        return vec / length if length > 1e-6 else np.array([0, 0, -1])


@dataclass
class DrillholeRenderState:
    """Complete render state for all drillholes."""
    intervals: List[DrillholeInterval] = field(default_factory=list)
    visible_holes: Set[str] = field(default_factory=set)
    selected_intervals: Set[int] = field(default_factory=set)  # color_id set
    hovered_interval: Optional[int] = None  # color_id
    color_property: str = "lithology"  # 'lithology', assay field name, or 'declust_weight'
    colormap_name: str = "tab10"
    camera_position: Optional[np.ndarray] = None
    clip_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    
    # Caches
    _id_to_interval: Dict[int, DrillholeInterval] = field(default_factory=dict)
    _hole_to_intervals: Dict[str, List[DrillholeInterval]] = field(default_factory=dict)
    
    def build_indices(self):
        """Build lookup indices for fast access."""
        self._id_to_interval = {iv.color_id: iv for iv in self.intervals}
        self._hole_to_intervals = {}
        for iv in self.intervals:
            self._hole_to_intervals.setdefault(iv.hole_id, []).append(iv)
    
    def get_interval_by_id(self, color_id: int) -> Optional[DrillholeInterval]:
        return self._id_to_interval.get(color_id)
    
    def get_intervals_by_hole(self, hole_id: str) -> List[DrillholeInterval]:
        return self._hole_to_intervals.get(hole_id, [])


# =============================================================================
# Event System
# =============================================================================

class DrillholeEventBus(QObject):
    """
    Central event bus for drillhole visualization.
    Allows decoupled communication between renderer, UI, and analysis panels.
    """
    
    # Interval events
    intervalSelected = pyqtSignal(object)      # DrillholeInterval or None
    intervalHovered = pyqtSignal(object)       # DrillholeInterval or None
    intervalDoubleClicked = pyqtSignal(object) # DrillholeInterval
    
    # Visibility events
    visibilityChanged = pyqtSignal(set)        # Set of visible hole_ids
    
    # Camera events
    cameraChanged = pyqtSignal(object)         # Camera state dict
    
    # Clipping events
    clipPlaneChanged = pyqtSignal(object)      # Clip plane parameters
    
    # Scene events
    sceneLoaded = pyqtSignal(int)              # Number of intervals loaded
    sceneCleared = pyqtSignal()
    
    # Render events
    renderStarted = pyqtSignal()
    renderCompleted = pyqtSignal(float)        # Render time in ms
    renderError = pyqtSignal(str)              # Error message
    
    # Color events
    colorPropertyChanged = pyqtSignal(str)     # Property name
    colormapChanged = pyqtSignal(str)          # Colormap name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._listeners: Dict[str, List[Callable]] = {}
    
    def emit_interval_selected(self, interval: Optional[DrillholeInterval]):
        """Emit interval selection event."""
        self.intervalSelected.emit(interval)
        logger.debug(f"Event: intervalSelected -> {interval.hole_id if interval else None}")
    
    def emit_interval_hovered(self, interval: Optional[DrillholeInterval]):
        """Emit interval hover event (throttled internally)."""
        self.intervalHovered.emit(interval)
    
    def emit_visibility_changed(self, visible_holes: Set[str]):
        """Emit visibility change event."""
        self.visibilityChanged.emit(visible_holes)
        logger.debug(f"Event: visibilityChanged -> {len(visible_holes)} holes visible")


# Global event bus instance
_event_bus: Optional[DrillholeEventBus] = None

def get_drillhole_event_bus() -> DrillholeEventBus:
    """Get or create the global drillhole event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = DrillholeEventBus()
    return _event_bus


# =============================================================================
# GPU Geometry Builder
# =============================================================================

class DrillholeGeometryBuilder:
    """
    Builds optimized geometry for GPU-instanced drillhole rendering.
    
    Key optimizations:
    - Single merged mesh for all intervals (GPU batching)
    - Color ID buffer for GPU picking
    - LOD-aware geometry generation
    - Minimal Python involvement in geometry updates
    """
    
    def __init__(self, quality: RenderQuality = RenderQuality.HIGH):
        self.quality = quality
        self._cylinder_template: Optional[pv.PolyData] = None
        self._build_cylinder_template()
    
    def _build_cylinder_template(self):
        """Pre-build cylinder template for instancing."""
        # Unit cylinder along Z-axis, centered at origin
        self._cylinder_template = pv.Cylinder(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            radius=1.0,
            height=1.0,
            resolution=int(self.quality),
            capping=True
        )
    
    def build_batched_mesh(
        self,
        intervals: List[DrillholeInterval],
        include_picking_ids: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pv.PolyData:
        """
        Build a single batched mesh for all intervals.
        
        OPTIMIZED: Uses pv.merge() with list instead of sequential merge (O(n) vs O(n²)).
        For very large datasets (>5000), uses simplified tube geometry.
        
        Args:
            intervals: List of intervals to render
            include_picking_ids: Whether to include color_id for GPU picking
            progress_callback: Optional callback for progress updates
            
        Returns:
            Single merged PolyData mesh with all intervals
        """
        import time
        start_time = time.perf_counter()
        
        def _progress(frac: float, msg: str):
            if progress_callback:
                try:
                    progress_callback(frac, msg)
                except Exception:
                    pass
        
        if not intervals:
            return pv.PolyData()
        
        n_intervals = len(intervals)
        logger.info(f"Building batched mesh for {n_intervals} intervals...")
        _progress(0.0, f"Building mesh for {n_intervals} intervals")
        
        # For very large datasets, use lower quality to prevent freezing
        effective_quality = self.quality
        if n_intervals > 10000:
            effective_quality = RenderQuality.LOW  # 4 sides
            logger.info(f"Large dataset ({n_intervals}): using LOW quality for performance")
        elif n_intervals > 5000:
            effective_quality = RenderQuality.MEDIUM  # 8 sides
            logger.info(f"Medium dataset ({n_intervals}): using MEDIUM quality")
        
        meshes = []
        color_ids = []
        lith_indices = []
        assay_values = []
        selection_states = []
        
        # Unique lithology mapping
        unique_liths = sorted(set(iv.lith_code or "Unknown" for iv in intervals))
        lith_to_idx = {lith: i for i, lith in enumerate(unique_liths)}
        
        # Build cylinders in chunks with progress
        chunk_size = 500
        visible_intervals = [iv for iv in intervals if iv.selection_state != SelectionState.HIDDEN]
        n_visible = len(visible_intervals)
        
        for i, iv in enumerate(visible_intervals):
            # Build cylinder for this interval
            cylinder = self._build_interval_cylinder_fast(iv, int(effective_quality))
            if cylinder is not None and cylinder.n_cells > 0:
                meshes.append(cylinder)
                
                # Store per-cell data for the cylinder
                n_cells = cylinder.n_cells
                color_ids.extend([iv.color_id] * n_cells)
                lith_indices.extend([lith_to_idx.get(iv.lith_code, 0)] * n_cells)
                assay_values.extend([iv.assay_value or 0.0] * n_cells)
                selection_states.extend([int(iv.selection_state)] * n_cells)
            
            # Progress update every chunk
            if (i + 1) % chunk_size == 0 or i == n_visible - 1:
                progress = 0.1 + 0.5 * ((i + 1) / n_visible)
                _progress(progress, f"Building geometry: {i + 1}/{n_visible}")
                
                # Allow Qt event processing to prevent freeze
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
        
        if not meshes:
            return pv.PolyData()
        
        logger.info(f"Built {len(meshes)} meshes, now merging...")
        _progress(0.6, f"Merging {len(meshes)} meshes...")
        
        # OPTIMIZED: Use pv.merge() with list - O(n) instead of O(n²)
        try:
            if len(meshes) == 1:
                merged = meshes[0]
            else:
                # Single merge call with all meshes - MUCH faster
                merged = pv.merge(meshes)
        except Exception as e:
            logger.warning(f"Batch merge failed: {e}, trying chunked merge")
            # Fallback: chunked merge in groups of 500
            merged = self._chunked_merge(meshes, chunk_size=500, progress_callback=progress_callback)
        
        _progress(0.9, "Adding cell data...")
        
        # Add cell data for rendering and picking
        if merged.n_cells > 0:
            if include_picking_ids:
                merged.cell_data["color_id"] = np.array(color_ids, dtype=np.int32)
            merged.cell_data["lith_idx"] = np.array(lith_indices, dtype=np.int32)
            merged.cell_data["assay"] = np.array(assay_values, dtype=np.float32)
            merged.cell_data["selection_state"] = np.array(selection_states, dtype=np.int8)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Built batched mesh: {merged.n_cells} cells, {merged.n_points} points in {elapsed:.0f}ms")
        _progress(1.0, f"Mesh ready ({merged.n_cells} cells)")
        
        return merged
    
    def _chunked_merge(
        self,
        meshes: List[pv.PolyData],
        chunk_size: int = 500,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pv.PolyData:
        """Merge meshes in chunks to prevent memory issues."""
        if len(meshes) <= chunk_size:
            return pv.merge(meshes)
        
        # Merge in chunks
        chunks = []
        for i in range(0, len(meshes), chunk_size):
            chunk = meshes[i:i + chunk_size]
            merged_chunk = pv.merge(chunk)
            chunks.append(merged_chunk)
            
            if progress_callback:
                progress = 0.6 + 0.3 * ((i + chunk_size) / len(meshes))
                progress_callback(progress, f"Merging chunk {len(chunks)}/{(len(meshes) + chunk_size - 1) // chunk_size}")
            
            # Allow Qt event processing
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        
        # Final merge of chunks
        return pv.merge(chunks)
    
    def _build_interval_cylinder_fast(self, interval: DrillholeInterval, resolution: int = 8) -> Optional[pv.PolyData]:
        """Build a cylinder mesh for a single interval with specified resolution."""
        start = interval.start_point
        end = interval.end_point
        
        vec = end - start
        length = np.linalg.norm(vec)
        
        if length < 1e-6:
            return None
        
        direction = vec / length
        center = (start + end) / 2
        
        try:
            cylinder = pv.Cylinder(
                center=center,
                direction=direction,
                radius=interval.radius,
                height=length,
                resolution=resolution,
                capping=True
            )
            return cylinder
        except Exception as e:
            logger.debug(f"Failed to build cylinder: {e}")
            return None
    
    def _build_interval_cylinder(self, interval: DrillholeInterval) -> Optional[pv.PolyData]:
        """Build a cylinder mesh for a single interval."""
        start = interval.start_point
        end = interval.end_point
        
        vec = end - start
        length = np.linalg.norm(vec)
        
        if length < 1e-6:
            return None
        
        direction = vec / length
        center = (start + end) / 2
        
        try:
            cylinder = pv.Cylinder(
                center=center,
                direction=direction,
                radius=interval.radius,
                height=length,
                resolution=int(self.quality),
                capping=True
            )
            return cylinder
        except Exception as e:
            logger.debug(f"Failed to build cylinder for interval: {e}")
            return None
    
    def build_with_glyphs(
        self,
        intervals: List[DrillholeInterval],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pv.PolyData:
        """
        Build drillholes using VTK glyph3D for O(1) radius updates.
        
        Phase 5.2 Implementation: Instead of building N cylinders, we create
        N transforms applied to a single cylinder template. This allows:
        - Instant radius changes (just update the scale array)
        - Better GPU memory usage
        - Faster initial build for large datasets
        
        Note: Picking is harder with glyphs - use build_batched_mesh for
        datasets where picking is important.
        
        Args:
            intervals: List of DrillholeInterval objects
            progress_callback: Progress callback (progress, message)
        
        Returns:
            PyVista PolyData with glyph-instanced cylinders
        """
        import vtk
        import time
        
        start_time = time.perf_counter()
        
        def _progress(frac: float, msg: str):
            if progress_callback:
                progress_callback(frac, msg)
        
        _progress(0.0, "Initializing glyph rendering...")
        
        # Filter visible intervals
        visible = [iv for iv in intervals if iv.selection_state != SelectionState.HIDDEN]
        n_intervals = len(visible)
        
        if n_intervals == 0:
            return pv.PolyData()
        
        _progress(0.1, f"Processing {n_intervals} intervals...")
        
        # Create template cylinder (unit radius, unit height, centered at origin)
        template = pv.Cylinder(
            center=(0, 0, 0),
            direction=(0, 0, 1),  # Z-up
            radius=1.0,
            height=1.0,
            resolution=int(self.quality),
            capping=True
        )
        
        # Prepare arrays for glyph transform data
        centers = np.zeros((n_intervals, 3), dtype=np.float64)
        scales = np.zeros((n_intervals, 3), dtype=np.float64)
        orientations = np.zeros((n_intervals, 3), dtype=np.float64)  # direction vectors
        
        # Per-interval data
        lith_indices = np.zeros(n_intervals, dtype=np.int32)
        assay_values = np.zeros(n_intervals, dtype=np.float32)
        
        # Build unique lith mapping
        unique_liths = sorted(set(iv.lith_code for iv in visible if iv.lith_code))
        if not unique_liths:
            unique_liths = ["UNKNOWN"]
        lith_to_idx = {lith: i for i, lith in enumerate(unique_liths)}
        
        _progress(0.2, "Computing transforms...")
        
        for i, iv in enumerate(visible):
            # Compute center, direction, and scale
            start = iv.start_point
            end = iv.end_point
            vec = end - start
            length = np.linalg.norm(vec)
            
            if length < 1e-6:
                length = 1e-6
                direction = np.array([0, 0, 1])
            else:
                direction = vec / length
            
            center = (start + end) / 2
            
            centers[i] = center
            scales[i] = [iv.radius, iv.radius, length]  # x=radius, y=radius, z=height
            orientations[i] = direction
            lith_indices[i] = lith_to_idx.get(iv.lith_code, 0)
            assay_values[i] = iv.assay_value or 0.0
            
            if (i + 1) % 1000 == 0:
                _progress(0.2 + 0.4 * (i / n_intervals), f"Processed {i + 1}/{n_intervals}")
        
        _progress(0.6, "Creating glyph points...")
        
        # Create point cloud with glyph data
        points = pv.PolyData(centers)
        points["scale"] = scales
        points["orientation"] = orientations
        points["lith_idx"] = lith_indices
        points["assay"] = assay_values
        
        _progress(0.7, "Applying glyph filter...")
        
        # Use VTK glyph3D for instancing
        try:
            glyph_filter = vtk.vtkGlyph3D()
            glyph_filter.SetInputData(points)
            glyph_filter.SetSourceData(template)
            
            # Set scaling mode
            glyph_filter.SetScaleModeToScaleByVectorComponents()
            glyph_filter.SetScaleFactor(1.0)
            
            # Set orientation mode
            glyph_filter.SetVectorModeToUseVector()
            glyph_filter.OrientOn()
            
            # Set input arrays
            points.GetPointData().SetActiveScalars("scale")
            points.GetPointData().SetActiveVectors("orientation")
            
            glyph_filter.Update()
            
            result = pv.wrap(glyph_filter.GetOutput())
            
        except Exception as e:
            logger.warning(f"VTK glyph3D failed: {e}, falling back to batch merge")
            return self.build_batched_mesh(intervals, progress_callback=progress_callback)
        
        # Transfer point data to cell data (for coloring)
        _progress(0.9, "Finalizing mesh...")
        
        # Note: Glyph filter expands point data to cell data
        # We need to map the interval data to cells
        if result.n_cells > 0:
            cells_per_glyph = template.n_cells
            
            # Expand arrays to match number of cells
            result.cell_data["lith_idx"] = np.repeat(lith_indices, cells_per_glyph)[:result.n_cells]
            result.cell_data["assay"] = np.repeat(assay_values, cells_per_glyph)[:result.n_cells]
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Built glyph mesh: {result.n_cells} cells, {result.n_points} points in {elapsed:.0f}ms")
        _progress(1.0, f"Glyph mesh ready ({result.n_cells} cells)")
        
        return result
    
    def update_selection_buffer(
        self,
        mesh: pv.PolyData,
        selected_ids: Set[int],
        hovered_id: Optional[int]
    ) -> np.ndarray:
        """
        Update selection state buffer without rebuilding geometry.
        
        This is the key to fast hover/selection updates:
        we only update a small array, not the entire mesh.
        
        Returns:
            Updated selection state array
        """
        if "color_id" not in mesh.cell_data:
            return np.zeros(mesh.n_cells, dtype=np.int8)
        
        color_ids = mesh.cell_data["color_id"]
        states = np.zeros(len(color_ids), dtype=np.int8)
        
        # Mark selected
        for cid in selected_ids:
            mask = color_ids == cid
            states[mask] = int(SelectionState.SELECTED)
        
        # Mark hovered (overrides selected for visual priority)
        if hovered_id is not None:
            mask = color_ids == hovered_id
            states[mask] = int(SelectionState.HOVERED)
        
        return states


# =============================================================================
# GPU Picking System
# =============================================================================

class GPUPicker:
    """
    GPU-based picking for instant interval selection.
    
    Uses color ID buffer technique:
    1. Render scene with each interval as unique color
    2. Read pixel under cursor
    3. Map color back to interval ID
    
    Target: <5ms pick time for 100k intervals
    """
    
    def __init__(self, plotter: pv.Plotter, state: DrillholeRenderState):
        self.plotter = plotter
        self.state = state
        self._pick_buffer_dirty = True
        self._last_pick_time = 0.0
    
    def pick_at_position(
        self,
        screen_x: int,
        screen_y: int
    ) -> Optional[Tuple[DrillholeInterval, Tuple[float, float, float]]]:
        """
        Pick interval at screen position using GPU-accelerated method.
        
        Returns tuple of (interval, world_position) or None.
        
        Falls back to VTK picking if GPU picking unavailable.
        """
        import time
        start = time.perf_counter()
        
        try:
            # Use VTK's built-in cell picker which is GPU-accelerated
            picked = self.plotter.pick_click_position()
            
            if picked is not None:
                # Try to get the picked cell's color_id
                mesh = picked.get("mesh")
                cell_id = picked.get("cell")
                world_pos = picked.get("world_position")
                
                if mesh is not None and cell_id is not None:
                    if "color_id" in mesh.cell_data:
                        color_id = int(mesh.cell_data["color_id"][cell_id])
                        interval = self.state.get_interval_by_id(color_id)
                        
                        # Get world position
                        if world_pos is not None:
                            world_position = tuple(world_pos[:3])
                        else:
                            # Fallback: use interval center
                            world_position = tuple(interval.center)
                        
                        self._last_pick_time = (time.perf_counter() - start) * 1000
                        logger.debug(f"GPU pick completed in {self._last_pick_time:.2f}ms")
                        return (interval, world_position)
        except Exception as e:
            logger.debug(f"GPU pick failed: {e}")
        
        self._last_pick_time = (time.perf_counter() - start) * 1000
        return None
    
    def pick_at_world_position(
        self,
        world_pos: Tuple[float, float, float],
        tolerance: float = 1.0
    ) -> Optional[DrillholeInterval]:
        """Pick interval nearest to world position."""
        if not self.state.intervals:
            return None
        
        world_point = np.array(world_pos)
        min_dist = float('inf')
        nearest = None
        
        for interval in self.state.intervals:
            if interval.selection_state == SelectionState.HIDDEN:
                continue
            
            # Distance to line segment
            dist = self._point_to_segment_distance(
                world_point,
                interval.start_point,
                interval.end_point
            )
            
            if dist < min_dist and dist < tolerance + interval.radius:
                min_dist = dist
                nearest = interval
        
        return nearest
    
    @staticmethod
    def _point_to_segment_distance(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray
    ) -> float:
        """Calculate distance from point to line segment."""
        v = seg_end - seg_start
        w = point - seg_start
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(np.linalg.norm(point - seg_start))
        
        c2 = np.dot(v, v)
        if c2 <= c1:
            return float(np.linalg.norm(point - seg_end))
        
        b = c1 / c2
        pb = seg_start + b * v
        return float(np.linalg.norm(point - pb))


# =============================================================================
# Main GPU Renderer
# =============================================================================

class DrillholeGPURenderer:
    """
    High-performance GPU-optimized drillhole renderer.
    
    Features:
    - GPU instancing for efficient rendering
    - Batched geometry updates
    - GPU picking
    - Shader-based highlighting
    - LOD management
    - Frustum culling
    
    Usage:
        renderer = DrillholeGPURenderer(plotter)
        renderer.load_intervals(intervals)
        renderer.set_color_property("Fe")
        renderer.render()
    """
    
    def __init__(
        self,
        plotter: pv.Plotter,
        quality: RenderQuality = RenderQuality.HIGH,
        enable_gpu_picking: bool = True
    ):
        self.plotter = plotter
        self.quality = quality
        self.enable_gpu_picking = enable_gpu_picking
        
        # State
        self.state = DrillholeRenderState()
        self.event_bus = get_drillhole_event_bus()
        
        # Components
        self.geometry_builder = DrillholeGeometryBuilder(quality)
        self.picker: Optional[GPUPicker] = None
        
        # Rendering
        self._main_mesh: Optional[pv.PolyData] = None
        self._main_actor = None
        self._collar_actors: Dict[str, Any] = {}
        self._highlight_actor = None
        
        # Performance tracking
        self._render_times: List[float] = []
        self._last_hover_time = 0.0
        
        # Hover throttling (10-20 Hz)
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(50)  # 20 Hz
        self._hover_timer.timeout.connect(self._process_hover_queue)
        self._pending_hover_pos: Optional[Tuple[int, int]] = None
        
        # Tooltip state
        self._current_tooltip: Optional[str] = None
        self._current_tooltip_interval: Optional[DrillholeInterval] = None
    
    def get_current_tooltip(self) -> Optional[str]:
        """Get current hover tooltip text."""
        return self._current_tooltip
    
    def load_intervals(self, intervals: List[DrillholeInterval]):
        """
        Load intervals for rendering.
        
        This triggers a full geometry rebuild, which is expensive.
        Use update_*() methods for incremental changes.
        """
        import time
        start = time.perf_counter()
        
        self.state.intervals = intervals
        self.state.visible_holes = {iv.hole_id for iv in intervals}
        self.state.build_indices()
        
        # Build batched mesh
        self._main_mesh = self.geometry_builder.build_batched_mesh(
            intervals,
            include_picking_ids=self.enable_gpu_picking
        )
        
        # Initialize picker
        if self.enable_gpu_picking:
            self.picker = GPUPicker(self.plotter, self.state)
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Loaded {len(intervals)} intervals in {elapsed:.1f}ms")
        
        self.event_bus.sceneLoaded.emit(len(intervals))
    
    def render(self, colormap: str = "tab10", show_collars: bool = True):
        """
        Render all loaded intervals.
        
        This should be called after load_intervals() or when colormap changes.
        For selection/hover updates, use the faster update methods.
        """
        import time
        start = time.perf_counter()
        
        self.event_bus.renderStarted.emit()
        
        try:
            # Remove old actor
            if self._main_actor is not None:
                try:
                    self.plotter.remove_actor(self._main_actor)
                except Exception:
                    pass
            
            if self._main_mesh is None or self._main_mesh.n_cells == 0:
                logger.warning("No mesh to render")
                return
            
            # Determine coloring
            scalar_name = "lith_idx" if self.state.color_property == "lithology" else "assay"
            
            # Add main mesh
            self._main_actor = self.plotter.add_mesh(
                self._main_mesh,
                scalars=scalar_name,
                cmap=colormap,
                show_scalar_bar=False,
                smooth_shading=True,
                pbr=False,
                lighting=True,
                specular=0.3,
                specular_power=15,
                ambient=0.3,
                diffuse=0.8,
                reset_camera=False,
                name="drillholes_gpu"
            )
            
            # Add collar markers
            if show_collars:
                self._add_collar_markers()
            
            elapsed = (time.perf_counter() - start) * 1000
            self._render_times.append(elapsed)
            
            logger.info(f"Rendered {len(self.state.intervals)} intervals in {elapsed:.1f}ms")
            self.event_bus.renderCompleted.emit(elapsed)
            
        except Exception as e:
            logger.error(f"Render failed: {e}", exc_info=True)
            self.event_bus.renderError.emit(str(e))
    
    def _add_collar_markers(self):
        """Add collar disc markers for each unique hole."""
        # Get unique collars
        collar_positions = {}
        for iv in self.state.intervals:
            if iv.hole_id not in collar_positions:
                # Use first interval's start point as collar
                collar_positions[iv.hole_id] = iv.start_point
        
        # Clear old collars
        for actor in self._collar_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._collar_actors.clear()
        
        # Batch collar creation for performance
        if len(collar_positions) > 100:
            # For many holes, create merged collar mesh
            collar_meshes = []
            radius = self.state.intervals[0].radius * 1.8 if self.state.intervals else 1.0
            
            for hid, pos in collar_positions.items():
                disc = pv.Disc(
                    center=tuple(pos),
                    inner=0,
                    outer=radius,
                    normal=(0, 0, 1),
                    c_res=8
                )
                collar_meshes.append(disc)
            
            if collar_meshes:
                merged_collars = pv.merge(collar_meshes)
                actor = self.plotter.add_mesh(
                    merged_collars,
                    color="#FFD700",
                    show_scalar_bar=False,
                    reset_camera=False,
                    lighting=True,
                    ambient=0.5,
                    name="collars_gpu"
                )
                self._collar_actors["_merged"] = actor
        else:
            # For fewer holes, individual markers (allows per-hole visibility)
            radius = self.state.intervals[0].radius * 1.8 if self.state.intervals else 1.0
            for hid, pos in collar_positions.items():
                disc = pv.Disc(
                    center=tuple(pos),
                    inner=0,
                    outer=radius,
                    normal=(0, 0, 1),
                    c_res=8
                )
                actor = self.plotter.add_mesh(
                    disc,
                    color="#FFD700",
                    show_scalar_bar=False,
                    reset_camera=False,
                    lighting=True,
                    ambient=0.5,
                    name=f"collar_{hid}"
                )
                self._collar_actors[hid] = actor
    
    # -------------------------------------------------------------------------
    # Fast Update Methods (no geometry rebuild)
    # -------------------------------------------------------------------------
    
    def update_selection(self, selected_ids: Set[int]):
        """
        Update selection state without rebuilding geometry.
        Target: <5ms update time.
        """
        import time
        start = time.perf_counter()
        
        self.state.selected_intervals = selected_ids
        
        if self._main_mesh is not None and self._main_actor is not None:
            # Update selection buffer only
            new_states = self.geometry_builder.update_selection_buffer(
                self._main_mesh,
                selected_ids,
                self.state.hovered_interval
            )
            self._main_mesh.cell_data["selection_state"] = new_states
            
            # Force actor update
            try:
                mapper = self._main_actor.GetMapper()
                if mapper:
                    mapper.Modified()
            except Exception:
                pass
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"Selection update: {elapsed:.2f}ms")
        
        # Emit events for selected intervals
        for cid in selected_ids:
            interval = self.state.get_interval_by_id(cid)
            if interval:
                self.event_bus.emit_interval_selected(interval)
    
    def update_hover(self, screen_x: int, screen_y: int):
        """
        Queue hover update (throttled to 10-20 Hz).
        """
        self._pending_hover_pos = (screen_x, screen_y)
        if not self._hover_timer.isActive():
            self._hover_timer.start()
    
    def _process_hover_queue(self):
        """Process pending hover (called by timer)."""
        if self._pending_hover_pos is None:
            self._hover_timer.stop()
            return
        
        x, y = self._pending_hover_pos
        self._pending_hover_pos = None
        
        # Perform pick
        if self.picker:
            result = self.picker.pick_at_position(x, y)
            if result:
                interval, world_pos = result
                old_hover = self.state.hovered_interval
                new_hover = interval.color_id
                
                if new_hover != old_hover:
                    self.state.hovered_interval = new_hover
                    self._update_highlight(interval)
                    
                    # Emit hover event with full metadata
                    self.event_bus.emit_interval_hovered(interval)
                    
                    # Update tooltip (if tooltip system is connected)
                    self._update_hover_tooltip(interval, world_pos)
            else:
                # Clear hover
                if self.state.hovered_interval is not None:
                    self.state.hovered_interval = None
                    self._update_highlight(None)
                    self._update_hover_tooltip(None, None)
    
    def _update_hover_tooltip(self, interval: Optional[DrillholeInterval], world_pos: Optional[Tuple[float, float, float]]):
        """Update hover tooltip with interval information."""
        if interval is None:
            # Clear tooltip
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                try:
                    # Clear VTK tooltip if exists
                    pass
                except Exception:
                    pass
            return
        
        # Build tooltip text
        tooltip_lines = [
            f"Hole ID: {interval.hole_id}",
            f"Depth: {interval.depth_from:.2f} - {interval.depth_to:.2f} m",
            f"Length: {interval.length:.2f} m",
        ]
        
        if interval.lith_code:
            tooltip_lines.append(f"Lithology: {interval.lith_code}")
        
        if interval.assay_value is not None:
            tooltip_lines.append(f"Assay: {interval.assay_value:.4f}")
        
        if world_pos:
            tooltip_lines.append(f"Position: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})")
        
        tooltip_text = "\n".join(tooltip_lines)
        
        # Store tooltip for external systems (e.g., Qt tooltip)
        self._current_tooltip = tooltip_text
        self._current_tooltip_interval = interval
        
        # Emit tooltip update event
        self.event_bus.intervalHovered.emit(interval)
    
    def _update_highlight(self, interval: Optional[DrillholeInterval]):
        """
        Update hover highlight visualization using VTK shader-based approach.
        
        Uses two strategies for optimal performance:
        1. Fast path: Modify scalar array to highlight cells (no remesh)
        2. Fallback: Wireframe overlay for hover (if fast path fails)
        """
        import time
        start = time.perf_counter()
        
        # Try fast shader-based highlighting first
        if self._try_shader_highlight(interval):
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"Shader highlight update: {elapsed:.2f}ms")
            return
        
        # Fallback to wireframe overlay (slower but more visible)
        self._update_wireframe_highlight(interval)
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"Wireframe highlight update: {elapsed:.2f}ms")
    
    def _try_shader_highlight(self, interval: Optional[DrillholeInterval]) -> bool:
        """
        Attempt to use VTK's lookup table for instant highlighting.
        
        This modifies the existing mesh's color mapping without creating
        new geometry - target is <1ms update time.
        
        Returns:
            True if shader highlighting succeeded, False for fallback
        """
        if self._main_mesh is None or self._main_actor is None:
            return False
        
        if "color_id" not in self._main_mesh.cell_data:
            return False
        
        try:
            # Get current color IDs
            color_ids = self._main_mesh.cell_data["color_id"]
            
            # Update selection_state array in-place
            states = self._main_mesh.cell_data.get("selection_state")
            if states is None:
                states = np.zeros(len(color_ids), dtype=np.int8)
            else:
                states = np.array(states, dtype=np.int8)  # Make writable copy
            
            # Reset all to NONE first (clear previous hover)
            states[states == int(SelectionState.HOVERED)] = int(SelectionState.NONE)
            
            # Keep SELECTED states
            selected_mask = np.isin(color_ids, list(self.state.selected_intervals))
            states[selected_mask] = int(SelectionState.SELECTED)
            
            # Mark hovered (if any)
            if interval is not None:
                hover_mask = color_ids == interval.color_id
                states[hover_mask] = int(SelectionState.HOVERED)
            
            # Update mesh data
            self._main_mesh.cell_data["selection_state"] = states
            
            # Force VTK mapper update
            mapper = self._main_actor.GetMapper()
            if mapper:
                # Update lookup table to include selection colors
                self._apply_selection_lut(mapper)
                mapper.Modified()
            
            # Also update actor property for additional visual feedback
            prop = self._main_actor.GetProperty()
            if prop:
                prop.Modified()
            
            return True
            
        except Exception as e:
            logger.debug(f"Shader highlight failed: {e}")
            return False
    
    def _apply_selection_lut(self, mapper):
        """
        Apply selection-aware lookup table modification.
        
        This creates visual distinction for:
        - Normal cells: Original colormap
        - Hovered cells: Brighter/highlighted
        - Selected cells: Outlined/emphasized
        """
        try:
            lut = mapper.GetLookupTable()
            if lut is None:
                return
            
            # Note: For more advanced shader-based effects, we would need
            # to use VTK's shader replacement mechanism. For now, we use
            # a simple approach that works with existing pipeline.
            
            # Trigger LUT rebuild by marking modified
            lut.Modified()
            
        except Exception as e:
            logger.debug(f"LUT modification failed: {e}")
    
    def _update_wireframe_highlight(self, interval: Optional[DrillholeInterval]):
        """
        Fallback: Create wireframe overlay for hover visualization.
        
        This creates new geometry but is still fast (<5ms).
        """
        # Remove old highlight
        if self._highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_actor)
            except Exception:
                pass
            self._highlight_actor = None
        
        if interval is None:
            return
        
        # Create highlight geometry (slight larger cylinder with outline)
        try:
            cylinder = pv.Cylinder(
                center=tuple(interval.center),
                direction=tuple(interval.direction),
                radius=interval.radius * 1.2,  # Slightly larger
                height=interval.length * 1.05,
                resolution=int(self.quality),
                capping=True
            )
            
            # Add as wireframe overlay
            self._highlight_actor = self.plotter.add_mesh(
                cylinder,
                style='wireframe',
                color='yellow',
                line_width=2,
                show_scalar_bar=False,
                reset_camera=False,
                name="drillhole_highlight"
            )
        except Exception as e:
            logger.debug(f"Failed to create wireframe highlight: {e}")
    
    def update_selection_colors(self, force_render: bool = True):
        """
        Update visual representation of selected intervals.
        
        This uses the selection_state cell data to modify colors
        without rebuilding geometry.
        
        Target: <5ms for any number of selections
        """
        import time
        start = time.perf_counter()
        
        if self._main_mesh is None or self._main_actor is None:
            return
        
        if "color_id" not in self._main_mesh.cell_data:
            return
        
        try:
            color_ids = self._main_mesh.cell_data["color_id"]
            
            # Build selection state array
            states = np.zeros(len(color_ids), dtype=np.int8)
            
            # Mark selected intervals
            for cid in self.state.selected_intervals:
                mask = color_ids == cid
                states[mask] = int(SelectionState.SELECTED)
            
            # Mark hovered (if any)
            if self.state.hovered_interval is not None:
                mask = color_ids == self.state.hovered_interval
                states[mask] = int(SelectionState.HOVERED)
            
            # Update mesh data
            self._main_mesh.cell_data["selection_state"] = states
            
            # Force update
            mapper = self._main_actor.GetMapper()
            if mapper:
                mapper.Modified()
            
            if force_render:
                try:
                    self.plotter.render()
                except Exception:
                    pass
            
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"Selection color update: {elapsed:.2f}ms for {len(self.state.selected_intervals)} selected")
            
        except Exception as e:
            logger.warning(f"Failed to update selection colors: {e}")
    
    def update_visibility(self, visible_holes: Set[str]):
        """
        Update hole visibility without full rebuild.
        
        This updates the visibility flag per-interval and
        triggers a partial geometry update.
        """
        self.state.visible_holes = visible_holes
        
        # Mark hidden intervals
        for iv in self.state.intervals:
            if iv.hole_id in visible_holes:
                if iv.selection_state == SelectionState.HIDDEN:
                    iv.selection_state = SelectionState.NONE
            else:
                iv.selection_state = SelectionState.HIDDEN
        
        # Rebuild mesh (TODO: optimize to toggle visibility flags only)
        self._main_mesh = self.geometry_builder.build_batched_mesh(
            self.state.intervals,
            include_picking_ids=self.enable_gpu_picking
        )
        
        # Update collar visibility
        for hid, actor in self._collar_actors.items():
            if hid == "_merged":
                continue
            if hid in visible_holes:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        
        self.event_bus.emit_visibility_changed(visible_holes)
    
    def update_colormap(self, colormap: str):
        """Update colormap without rebuilding geometry."""
        self.state.colormap_name = colormap
        
        if self._main_actor is not None:
            try:
                # Update lookup table
                mapper = self._main_actor.GetMapper()
                if mapper:
                    import matplotlib.cm as cm
                    cmap_obj = cm.get_cmap(colormap)
                    
                    lut = mapper.GetLookupTable()
                    if lut:
                        n_colors = lut.GetNumberOfTableValues()
                        for i in range(n_colors):
                            rgba = cmap_obj(i / (n_colors - 1))
                            lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], 1.0)
                        lut.Modified()
                        mapper.Modified()
            except Exception as e:
                logger.warning(f"Failed to update colormap: {e}")
        
        self.event_bus.colormapChanged.emit(colormap)
    
    def update_color_property(self, property_name: str):
        """Switch between lithology and assay coloring."""
        self.state.color_property = property_name
        
        scalar_name = "lith_idx" if property_name == "lithology" else "assay"
        
        if self._main_actor is not None and self._main_mesh is not None:
            try:
                mapper = self._main_actor.GetMapper()
                if mapper:
                    mapper.SelectColorArray(scalar_name)
                    mapper.SetScalarModeToUseCellData()
                    mapper.Modified()
            except Exception as e:
                logger.warning(f"Failed to switch color property: {e}")
        
        self.event_bus.colorPropertyChanged.emit(property_name)
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def clear(self):
        """Clear all drillhole visualization with proper memory release."""
        # Stop timer
        self._hover_timer.stop()
        
        # Remove actors
        if self._main_actor is not None:
            try:
                self.plotter.remove_actor(self._main_actor)
            except Exception as e:
                logger.debug(f"Could not remove main drillhole actor: {e}")
        
        if self._highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_actor)
            except Exception as e:
                logger.debug(f"Could not remove highlight actor: {e}")
        
        for actor in self._collar_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception as e:
                logger.debug(f"Could not remove collar actor: {e}")
        
        # Clear state with proper memory release
        # Clear intervals list which can be large
        if hasattr(self.state, 'intervals') and self.state.intervals:
            self.state.intervals.clear()
        if hasattr(self.state, 'visible_holes') and self.state.visible_holes:
            self.state.visible_holes.clear()
        if hasattr(self.state, 'selected_intervals') and self.state.selected_intervals:
            self.state.selected_intervals.clear()
        
        self.state = DrillholeRenderState()
        
        # Clear mesh reference
        self._main_mesh = None
        self._main_actor = None
        self._highlight_actor = None
        self._collar_actors.clear()
        
        # Clear render time history
        if hasattr(self, '_render_times') and self._render_times:
            self._render_times.clear()
        
        # Force garbage collection for large datasets
        import gc
        gc.collect()
        
        self.event_bus.sceneCleared.emit()
        logger.info("DrillholeGPURenderer cleared with memory released")
    
    # -------------------------------------------------------------------------
    # Performance Stats
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "interval_count": len(self.state.intervals),
            "visible_holes": len(self.state.visible_holes),
            "selected_intervals": len(self.state.selected_intervals),
            "mesh_cells": self._main_mesh.n_cells if self._main_mesh else 0,
            "mesh_points": self._main_mesh.n_points if self._main_mesh else 0,
            "avg_render_time_ms": np.mean(self._render_times) if self._render_times else 0,
            "last_pick_time_ms": self.picker._last_pick_time if self.picker else 0,
            "quality": self.quality.name,
        }


# =============================================================================
# LOD Manager - Level of Detail Switching
# =============================================================================

@dataclass
class LODLevel:
    """Defines a Level of Detail configuration."""
    name: str
    quality: RenderQuality
    min_distance: float  # Minimum camera distance for this LOD
    max_distance: float  # Maximum camera distance for this LOD


class DrillholeLODManager:
    """
    Manages Level of Detail for drillhole rendering.
    
    Automatically switches geometry quality based on camera distance:
    - ULTRA (32 sides): < 50m from camera
    - HIGH (16 sides): 50-200m from camera  
    - MEDIUM (8 sides): 200-500m from camera
    - LOW (4 sides): > 500m from camera
    
    Also implements frustum culling to skip rendering intervals
    outside the camera's view frustum.
    """
    
    # Default LOD distance thresholds (can be customized)
    DEFAULT_LOD_LEVELS = [
        LODLevel("ultra", RenderQuality.ULTRA, 0, 50),
        LODLevel("high", RenderQuality.HIGH, 50, 200),
        LODLevel("medium", RenderQuality.MEDIUM, 200, 500),
        LODLevel("low", RenderQuality.LOW, 500, float('inf')),
    ]
    
    def __init__(
        self,
        plotter: pv.Plotter,
        lod_levels: Optional[List[LODLevel]] = None,
        enable_frustum_culling: bool = True,
        frustum_margin: float = 1.2,  # 20% margin beyond frustum
    ):
        self.plotter = plotter
        self.lod_levels = lod_levels or self.DEFAULT_LOD_LEVELS
        self.enable_frustum_culling = enable_frustum_culling
        self.frustum_margin = frustum_margin
        
        # Caches
        self._last_camera_position: Optional[np.ndarray] = None
        self._hole_lod_cache: Dict[str, RenderQuality] = {}
        self._hole_visible_cache: Dict[str, bool] = {}
        
        # Statistics
        self._culled_count = 0
        self._lod_distribution: Dict[str, int] = {}
        
        logger.info("DrillholeLODManager initialized")
    
    def get_lod_for_distance(self, distance: float) -> RenderQuality:
        """Get appropriate LOD quality for a given distance."""
        for level in self.lod_levels:
            if level.min_distance <= distance < level.max_distance:
                return level.quality
        return RenderQuality.LOW  # Default to low for very far
    
    def update_lods(
        self,
        intervals: List[DrillholeInterval],
        camera_position: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, RenderQuality]:
        """
        Update LOD assignments for all holes based on camera position.
        
        Args:
            intervals: List of intervals to evaluate
            camera_position: Camera position (auto-detected if None)
            
        Returns:
            Dictionary mapping hole_id to recommended LOD quality
        """
        # Get camera position
        if camera_position is None:
            try:
                cam = self.plotter.renderer.GetActiveCamera()
                if cam:
                    camera_position = cam.GetPosition()
            except Exception:
                pass
        
        if camera_position is None:
            return {}
        
        camera_pos = np.array(camera_position)
        
        # Check if camera moved significantly
        if self._last_camera_position is not None:
            movement = np.linalg.norm(camera_pos - self._last_camera_position)
            if movement < 1.0:  # Less than 1m movement, skip update
                return self._hole_lod_cache
        
        self._last_camera_position = camera_pos
        
        # Calculate LOD per hole (use hole center, not every interval)
        hole_centers: Dict[str, np.ndarray] = {}
        for iv in intervals:
            if iv.hole_id not in hole_centers:
                hole_centers[iv.hole_id] = iv.center
        
        # Update LOD cache
        self._hole_lod_cache.clear()
        self._lod_distribution = {"ultra": 0, "high": 0, "medium": 0, "low": 0}
        
        for hole_id, center in hole_centers.items():
            distance = float(np.linalg.norm(center - camera_pos))
            lod = self.get_lod_for_distance(distance)
            self._hole_lod_cache[hole_id] = lod
            
            # Track distribution
            if lod == RenderQuality.ULTRA:
                self._lod_distribution["ultra"] += 1
            elif lod == RenderQuality.HIGH:
                self._lod_distribution["high"] += 1
            elif lod == RenderQuality.MEDIUM:
                self._lod_distribution["medium"] += 1
            else:
                self._lod_distribution["low"] += 1
        
        return self._hole_lod_cache
    
    def get_frustum_planes(self) -> Optional[np.ndarray]:
        """
        Extract frustum planes from the camera.
        
        Returns 6 planes (near, far, left, right, top, bottom) as 
        (a, b, c, d) coefficients where ax + by + cz + d = 0.
        """
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return None
            
            # Get camera parameters
            position = np.array(cam.GetPosition())
            focal_point = np.array(cam.GetFocalPoint())
            view_up = np.array(cam.GetViewUp())
            
            # Calculate view direction
            view_dir = focal_point - position
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Calculate right vector
            right = np.cross(view_dir, view_up)
            right = right / np.linalg.norm(right)
            
            # Recalculate up vector (ensure orthogonal)
            up = np.cross(right, view_dir)
            up = up / np.linalg.norm(up)
            
            # Get clipping range and view angle
            near, far = cam.GetClippingRange()
            fov = math.radians(cam.GetViewAngle())
            aspect = 1.0  # Assume square for simplicity
            
            try:
                # Get actual aspect ratio from renderer
                w, h = self.plotter.window_size
                if h > 0:
                    aspect = w / h
            except Exception:
                pass
            
            # Calculate frustum half-sizes at near plane
            half_height = near * math.tan(fov / 2)
            half_width = half_height * aspect
            
            # Expand frustum by margin
            half_height *= self.frustum_margin
            half_width *= self.frustum_margin
            
            # Define frustum planes (normal pointing inward)
            planes = []
            
            # Near plane
            planes.append(np.array([*view_dir, -np.dot(view_dir, position + near * view_dir)]))
            
            # Far plane
            planes.append(np.array([*(-view_dir), np.dot(view_dir, position + far * view_dir)]))
            
            # Left plane
            left_normal = np.cross(up, view_dir - right * (half_width / near))
            left_normal = left_normal / np.linalg.norm(left_normal)
            planes.append(np.array([*left_normal, -np.dot(left_normal, position)]))
            
            # Right plane  
            right_normal = np.cross(view_dir + right * (half_width / near), up)
            right_normal = right_normal / np.linalg.norm(right_normal)
            planes.append(np.array([*right_normal, -np.dot(right_normal, position)]))
            
            # Top plane
            top_normal = np.cross(view_dir + up * (half_height / near), right)
            top_normal = top_normal / np.linalg.norm(top_normal)
            planes.append(np.array([*top_normal, -np.dot(top_normal, position)]))
            
            # Bottom plane
            bottom_normal = np.cross(right, view_dir - up * (half_height / near))
            bottom_normal = bottom_normal / np.linalg.norm(bottom_normal)
            planes.append(np.array([*bottom_normal, -np.dot(bottom_normal, position)]))
            
            return np.array(planes)
            
        except Exception as e:
            logger.debug(f"Failed to extract frustum planes: {e}")
            return None
    
    def is_point_in_frustum(self, point: np.ndarray, planes: np.ndarray) -> bool:
        """Check if a point is inside the frustum."""
        for plane in planes:
            # Distance = ax + by + cz + d
            dist = np.dot(plane[:3], point) + plane[3]
            if dist < 0:  # Point is outside this plane
                return False
        return True
    
    def is_interval_in_frustum(
        self,
        interval: DrillholeInterval,
        planes: np.ndarray
    ) -> bool:
        """
        Check if an interval is potentially visible in the frustum.
        
        Uses conservative bounding sphere test.
        """
        center = interval.center
        # Bounding sphere radius = half length + tube radius
        radius = interval.length / 2 + interval.radius
        
        for plane in planes:
            # Signed distance from center to plane
            dist = np.dot(plane[:3], center) + plane[3]
            if dist < -radius:  # Sphere entirely outside this plane
                return False
        return True
    
    def filter_visible_intervals(
        self,
        intervals: List[DrillholeInterval],
    ) -> Tuple[List[DrillholeInterval], int]:
        """
        Filter intervals to only those visible in the frustum.
        
        Args:
            intervals: All intervals
            
        Returns:
            Tuple of (visible intervals, culled count)
        """
        if not self.enable_frustum_culling:
            return intervals, 0
        
        planes = self.get_frustum_planes()
        if planes is None:
            return intervals, 0
        
        visible = []
        culled = 0
        
        for iv in intervals:
            if self.is_interval_in_frustum(iv, planes):
                visible.append(iv)
            else:
                culled += 1
        
        self._culled_count = culled
        
        if culled > 0:
            logger.debug(f"Frustum culling: {culled} intervals culled, {len(visible)} visible")
        
        return visible, culled
    
    def get_grouped_by_lod(
        self,
        intervals: List[DrillholeInterval],
    ) -> Dict[RenderQuality, List[DrillholeInterval]]:
        """
        Group intervals by their LOD level.
        
        This allows building separate meshes per LOD for mixed-quality rendering.
        """
        # Ensure LOD cache is current
        if not self._hole_lod_cache:
            self.update_lods(intervals)
        
        grouped: Dict[RenderQuality, List[DrillholeInterval]] = {
            RenderQuality.ULTRA: [],
            RenderQuality.HIGH: [],
            RenderQuality.MEDIUM: [],
            RenderQuality.LOW: [],
        }
        
        for iv in intervals:
            lod = self._hole_lod_cache.get(iv.hole_id, RenderQuality.MEDIUM)
            grouped[lod].append(iv)
        
        return grouped
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LOD manager statistics."""
        return {
            "lod_distribution": self._lod_distribution.copy(),
            "culled_intervals": self._culled_count,
            "frustum_culling_enabled": self.enable_frustum_culling,
            "frustum_margin": self.frustum_margin,
        }


# =============================================================================
# Enhanced GPU Renderer with LOD Support
# =============================================================================

class DrillholeGPURendererWithLOD(DrillholeGPURenderer):
    """
    Enhanced GPU renderer with automatic LOD switching and frustum culling.
    
    Extends DrillholeGPURenderer with:
    - Distance-based LOD quality adjustment
    - Frustum culling for off-screen intervals
    - Per-LOD mesh batching for optimal performance
    - Camera-aware updates
    """
    
    def __init__(
        self,
        plotter: pv.Plotter,
        enable_lod: bool = True,
        enable_frustum_culling: bool = True,
        lod_update_threshold: float = 10.0,  # Update LODs when camera moves this far
        **kwargs
    ):
        super().__init__(plotter, **kwargs)
        
        self.enable_lod = enable_lod
        self.lod_manager = DrillholeLODManager(
            plotter,
            enable_frustum_culling=enable_frustum_culling,
        )
        self.lod_update_threshold = lod_update_threshold
        
        # LOD-specific meshes and actors
        self._lod_meshes: Dict[RenderQuality, pv.PolyData] = {}
        self._lod_actors: Dict[RenderQuality, Any] = {}
        
        # Camera tracking for LOD updates
        self._last_lod_camera_pos: Optional[np.ndarray] = None
        
        # Connect to camera changes (if available)
        self._setup_camera_observer()
    
    def _setup_camera_observer(self):
        """Setup observer for camera changes."""
        try:
            # VTK camera observer for LOD updates
            cam = self.plotter.renderer.GetActiveCamera()
            if cam:
                # Note: VTK observers require more complex setup
                # For now, we'll check on each render
                pass
        except Exception:
            pass
    
    def render(self, colormap: str = "tab10", show_collars: bool = True):
        """
        Render with LOD support.
        
        If LOD is enabled, builds separate meshes per quality level.
        Otherwise, falls back to standard single-mesh rendering.
        """
        if not self.enable_lod:
            return super().render(colormap, show_collars)
        
        import time
        start = time.perf_counter()
        
        self.event_bus.renderStarted.emit()
        
        try:
            # Clear old LOD actors
            for actor in self._lod_actors.values():
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self._lod_actors.clear()
            self._lod_meshes.clear()
            
            # Also clear main actor if exists
            if self._main_actor is not None:
                try:
                    self.plotter.remove_actor(self._main_actor)
                except Exception:
                    pass
                self._main_actor = None
            
            # Update LODs based on camera
            self.lod_manager.update_lods(self.state.intervals)
            
            # Apply frustum culling
            visible_intervals, culled = self.lod_manager.filter_visible_intervals(
                self.state.intervals
            )
            
            if not visible_intervals:
                logger.warning("No visible intervals after culling")
                return
            
            # Group by LOD
            grouped = self.lod_manager.get_grouped_by_lod(visible_intervals)
            
            # Build and render mesh per LOD level
            scalar_name = "lith_idx" if self.state.color_property == "lithology" else "assay"
            
            for lod_quality, intervals in grouped.items():
                if not intervals:
                    continue
                
                # Build mesh with appropriate quality
                builder = DrillholeGeometryBuilder(quality=lod_quality)
                mesh = builder.build_batched_mesh(intervals, include_picking_ids=True)
                
                if mesh.n_cells == 0:
                    continue
                
                self._lod_meshes[lod_quality] = mesh
                
                # Add mesh actor
                actor = self.plotter.add_mesh(
                    mesh,
                    scalars=scalar_name,
                    cmap=colormap,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    pbr=False,
                    lighting=True,
                    specular=0.3,
                    specular_power=15,
                    ambient=0.3,
                    diffuse=0.8,
                    reset_camera=False,
                    name=f"drillholes_lod_{lod_quality.name}"
                )
                
                self._lod_actors[lod_quality] = actor
            
            # Add collars
            if show_collars:
                self._add_collar_markers()
            
            elapsed = (time.perf_counter() - start) * 1000
            self._render_times.append(elapsed)
            
            lod_stats = self.lod_manager.get_stats()
            logger.info(
                f"LOD Render: {len(visible_intervals)} visible "
                f"(culled {culled}), LOD dist: {lod_stats['lod_distribution']}, "
                f"time: {elapsed:.1f}ms"
            )
            
            self.event_bus.renderCompleted.emit(elapsed)
            
        except Exception as e:
            logger.error(f"LOD render failed: {e}", exc_info=True)
            self.event_bus.renderError.emit(str(e))
    
    def update_lods_on_camera_move(self):
        """
        Check if camera moved enough to warrant LOD update.
        Called periodically or on camera interaction end.
        """
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            
            current_pos = np.array(cam.GetPosition())
            
            if self._last_lod_camera_pos is not None:
                movement = np.linalg.norm(current_pos - self._last_lod_camera_pos)
                if movement < self.lod_update_threshold:
                    return  # Not enough movement
            
            self._last_lod_camera_pos = current_pos
            
            # Re-render with updated LODs
            self.render(colormap=self.state.colormap_name)
            
        except Exception as e:
            logger.debug(f"LOD update check failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extended statistics including LOD info."""
        stats = super().get_stats()
        stats["lod_enabled"] = self.enable_lod
        stats.update(self.lod_manager.get_stats())
        return stats


# =============================================================================
# Factory Functions
# =============================================================================

def create_intervals_from_polyline_data(
    polyline_data: Dict[str, Any],
    radius: float = 1.0
) -> List[DrillholeInterval]:
    """
    Convert polyline data (from build_drillhole_polylines) to GPU intervals.
    
    CRITICAL: Now uses persistent GLOBAL_INTERVAL_ID from DataRegistry for stable GPU picking.
    
    Args:
        polyline_data: Output from build_drillhole_polylines()
        radius: Tube radius for rendering
        registry: DataRegistry instance for looking up GLOBAL_INTERVAL_IDs (optional but recommended)
        
    Returns:
        List of DrillholeInterval objects with persistent color_ids
    """
    intervals = []
    
    # Try to get persistent IDs from registry
    use_persistent_ids = False
    id_mapping = {}  # Maps (hole_id, segment_idx) -> GLOBAL_INTERVAL_ID
    registry = polyline_data.get('_registry')  # Registry can be passed in polyline_data
    
    if registry is not None:
        try:
            import pandas as pd
            drillhole_data = registry.get_drillhole_data(copy_data=False)
            if drillhole_data:
                # Build mapping from all interval DataFrames
                for df_name in ['assays', 'lithology', 'composites']:
                    df = drillhole_data.get(df_name)
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    
                    if 'GLOBAL_INTERVAL_ID' in df.columns and 'hole_id' in df.columns:
                        # Group by hole_id and map intervals by index within each hole
                        for hole_id, group in df.groupby('hole_id'):
                            for idx, (row_idx, row) in enumerate(group.iterrows()):
                                global_id = int(row['GLOBAL_INTERVAL_ID'])
                                key = (str(hole_id), idx)
                                id_mapping[key] = global_id
                        use_persistent_ids = True
                
                if use_persistent_ids:
                    logger.info(f"GPU Renderer: Using {len(id_mapping)} persistent interval IDs from DataRegistry")
                else:
                    logger.warning("GPU Renderer: DataRegistry has no GLOBAL_INTERVAL_ID - using temporary IDs (UNSAFE)")
        except Exception as e:
            logger.warning(f"GPU Renderer: Could not load persistent IDs: {e}")
    else:
        logger.warning("GPU Renderer: No registry provided - using temporary counter (UNSAFE for picking after reload)")
    
    # Fallback counter
    temp_id_counter = 1  # 0 reserved for background
    
    hole_polys = polyline_data.get("hole_polys", {})
    hole_segment_lith = polyline_data.get("hole_segment_lith", {})
    hole_segment_assay = polyline_data.get("hole_segment_assay", {})
    
    for hid, poly in hole_polys.items():
        if poly is None or poly.n_points < 2:
            continue
        
        points = np.asarray(poly.points)
        liths = hole_segment_lith.get(hid, [])
        assays = hole_segment_assay.get(hid, [])
        
        # Build intervals from line segments
        if hasattr(poly, 'lines') and poly.lines is not None:
            lines = np.asarray(poly.lines)
            i = 0
            segment_idx = 0
            
            while i < len(lines):
                n_pts = lines[i]
                if n_pts >= 2:
                    for j in range(n_pts - 1):
                        idx1 = lines[i + 1 + j]
                        idx2 = lines[i + 1 + j + 1]
                        
                        start = points[idx1]
                        end = points[idx2]
                        
                        lith = liths[segment_idx] if segment_idx < len(liths) else "Unknown"
                        assay = assays[segment_idx] if segment_idx < len(assays) else 0.0
                        
                        # CRITICAL: Use persistent ID if available
                        key = (hid, segment_idx)
                        if use_persistent_ids and key in id_mapping:
                            color_id = id_mapping[key]
                        else:
                            color_id = temp_id_counter
                            temp_id_counter += 1
                        
                        interval = DrillholeInterval(
                            hole_id=hid,
                            interval_index=segment_idx,
                            start_point=start,
                            end_point=end,
                            depth_from=0.0,  # Would need depth info
                            depth_to=np.linalg.norm(end - start),
                            radius=radius,
                            color_id=color_id,  # NOW PERSISTENT!
                            lith_code=lith,
                            assay_value=assay,
                        )
                        intervals.append(interval)
                        segment_idx += 1
                
                i += n_pts + 1
    
    status = "PERSISTENT" if use_persistent_ids else "TEMPORARY (UNSAFE)"
    logger.info(f"Created {len(intervals)} GPU intervals with {status} IDs")
    return intervals

