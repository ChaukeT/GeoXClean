"""
Picking Controller - Single authority for all picking state and LOD decisions.

This module implements a Picking LOD (Level of Detail) system that:
- Prevents interaction logic from running when no data is loaded
- Auto-degrades based on dataset size and interaction cost
- Enforces performance thresholds
- Eliminates redundant picking mechanisms

LOD Levels:
- LOD-P0 (Navigation): No hover, no click - enabled when no data loaded
- LOD-P1 (Actor-level): Hover highlights actor only, no cell data access
- LOD-P2 (Cell-level): Cell picking on explicit click only
- LOD-P3 (Analytical): Debug mode with full metadata (disabled by default)

Performance Thresholds:
- Hover: Target <2ms, disable if >5ms
- Click: Target <50ms, warn if >150ms, degrade if >250ms
"""

from enum import IntEnum
from typing import Optional, Callable, Dict, Any, Tuple
import time
import logging
import weakref

logger = logging.getLogger(__name__)


class PickingLOD(IntEnum):
    """Picking Level of Detail levels."""
    P0_NAVIGATION = 0   # No picking - navigation only
    P1_ACTOR = 1        # Actor-level interaction only
    P2_CELL = 2         # Cell-level picking on click
    P3_ANALYTICAL = 3   # Full metadata (debug only)


class PickingPerformanceThresholds:
    """Performance thresholds for picking operations (in milliseconds)."""
    # Hover thresholds
    HOVER_TARGET_MS = 2.0
    HOVER_DISABLE_MS = 5.0
    
    # Click thresholds
    CLICK_TARGET_MS = 50.0
    CLICK_WARN_MS = 150.0
    CLICK_DEGRADE_MS = 250.0
    
    # Cell count thresholds for LOD selection
    CELL_COUNT_P1_THRESHOLD = 200_000  # >200k cells -> LOD-P1
    CELL_COUNT_P2_THRESHOLD = 50_000   # <50k cells -> LOD-P2 with hover


class PickingMetrics:
    """Tracks picking performance metrics."""
    
    def __init__(self, window_size: int = 10):
        self._hover_times: list[float] = []
        self._click_times: list[float] = []
        self._window_size = window_size
        self._hover_disabled_count = 0
        self._click_degraded_count = 0
    
    def record_hover(self, elapsed_ms: float) -> bool:
        """
        Record hover timing and return True if hover should be disabled.
        """
        self._hover_times.append(elapsed_ms)
        if len(self._hover_times) > self._window_size:
            self._hover_times.pop(0)
        
        if elapsed_ms > PickingPerformanceThresholds.HOVER_DISABLE_MS:
            self._hover_disabled_count += 1
            return True
        return False
    
    def record_click(self, elapsed_ms: float) -> Tuple[bool, bool]:
        """
        Record click timing and return (should_warn, should_degrade).
        """
        self._click_times.append(elapsed_ms)
        if len(self._click_times) > self._window_size:
            self._click_times.pop(0)
        
        should_warn = elapsed_ms > PickingPerformanceThresholds.CLICK_WARN_MS
        should_degrade = elapsed_ms > PickingPerformanceThresholds.CLICK_DEGRADE_MS
        
        if should_degrade:
            self._click_degraded_count += 1
        
        return should_warn, should_degrade
    
    @property
    def avg_hover_ms(self) -> float:
        if not self._hover_times:
            return 0.0
        return sum(self._hover_times) / len(self._hover_times)
    
    @property
    def avg_click_ms(self) -> float:
        if not self._click_times:
            return 0.0
        return sum(self._click_times) / len(self._click_times)
    
    def reset(self):
        """Reset all metrics."""
        self._hover_times.clear()
        self._click_times.clear()
        self._hover_disabled_count = 0
        self._click_degraded_count = 0


class PickingController:
    """
    Single authority for all picking state and LOD decisions.
    
    This controller owns:
    - Picking state (enabled/disabled)
    - LOD decisions based on dataset size
    - Performance monitoring and auto-degradation
    - Enable/disable logic
    
    Usage:
        controller = PickingController()
        controller.on_data_loaded(cell_count=150000, has_drillholes=True)
        
        # In hover handler:
        if controller.hover_allowed:
            with controller.timed_hover() as timer:
                # perform hover pick
                pass
        
        # In click handler:
        if controller.click_allowed:
            with controller.timed_click() as timer:
                # perform click pick
                pass
    """
    
    def __init__(self):
        self._lod = PickingLOD.P0_NAVIGATION
        self._data_loaded = False
        self._cell_count = 0
        self._has_block_model = False
        self._has_drillholes = False
        
        # Hover state
        self._hover_enabled = False
        self._hover_suppressed = False  # Temporarily suppressed during navigation
        
        # Click state
        self._click_enabled = False
        
        # Performance tracking
        self._metrics = PickingMetrics()
        
        # VTK pickers - SEPARATE by responsibility (created lazily)
        # _prop_picker: Actor-level (cheap), for ALL hover and P1 clicks
        # _cell_picker: Cell-level (expensive), for P2+ clicks ONLY
        self._prop_picker = None
        self._cell_picker = None
        
        # Callbacks for LOD transitions
        self._lod_change_callbacks: list[Callable[[PickingLOD, PickingLOD], None]] = []
        
        # Navigation state
        self._is_navigating = False
        
        logger.info("PickingController initialized (LOD-P0: Navigation only)")
    
    # =========================================================================
    # Data State Management
    # =========================================================================
    
    def on_data_loaded(
        self, 
        cell_count: int = 0,
        has_block_model: bool = False,
        has_drillholes: bool = False
    ) -> None:
        """
        Called when renderable data is added to the scene.
        
        Args:
            cell_count: Total number of cells in block model
            has_block_model: Whether a block model is loaded
            has_drillholes: Whether drillholes are loaded
        """
        self._data_loaded = True
        self._cell_count = cell_count
        self._has_block_model = has_block_model
        self._has_drillholes = has_drillholes
        
        self._compute_lod()
        
        logger.info(
            f"PickingController: Data loaded - cells={cell_count}, "
            f"block_model={has_block_model}, drillholes={has_drillholes}, "
            f"LOD={self._lod.name}"
        )
    
    def on_data_cleared(self) -> None:
        """Called when all data is removed from the scene."""
        old_lod = self._lod
        
        self._data_loaded = False
        self._cell_count = 0
        self._has_block_model = False
        self._has_drillholes = False
        self._lod = PickingLOD.P0_NAVIGATION
        self._hover_enabled = False
        self._click_enabled = False
        self._metrics.reset()
        
        if old_lod != self._lod:
            self._notify_lod_change(old_lod, self._lod)
        
        logger.info("PickingController: Data cleared - LOD-P0 (Navigation only)")
    
    def update_cell_count(self, cell_count: int) -> None:
        """Update cell count and recompute LOD (e.g., after filtering)."""
        if cell_count != self._cell_count:
            self._cell_count = cell_count
            self._compute_lod()
    
    # =========================================================================
    # LOD Computation
    # =========================================================================
    
    def _compute_lod(self) -> None:
        """Compute appropriate LOD level based on current state."""
        old_lod = self._lod
        
        if not self._data_loaded:
            self._lod = PickingLOD.P0_NAVIGATION
            self._hover_enabled = False
            self._click_enabled = False
        elif self._cell_count > PickingPerformanceThresholds.CELL_COUNT_P1_THRESHOLD:
            # Large dataset: actor-level only
            self._lod = PickingLOD.P1_ACTOR
            self._hover_enabled = True  # Actor-level hover is cheap
            self._click_enabled = True  # Actor-level click only
        elif self._cell_count > PickingPerformanceThresholds.CELL_COUNT_P2_THRESHOLD:
            # Medium dataset: cell-level click, no hover
            self._lod = PickingLOD.P2_CELL
            self._hover_enabled = False  # Hover disabled for performance
            self._click_enabled = True
        else:
            # Small dataset: cell-level with hover
            self._lod = PickingLOD.P2_CELL
            self._hover_enabled = True
            self._click_enabled = True
        
        if old_lod != self._lod:
            self._notify_lod_change(old_lod, self._lod)
            logger.info(f"PickingController: LOD transition {old_lod.name} -> {self._lod.name}")
    
    def _notify_lod_change(self, old_lod: PickingLOD, new_lod: PickingLOD) -> None:
        """Notify registered callbacks of LOD change."""
        for callback in self._lod_change_callbacks:
            try:
                callback(old_lod, new_lod)
            except Exception as e:
                logger.debug(f"LOD change callback failed: {e}")
    
    def add_lod_change_callback(self, callback: Callable[[PickingLOD, PickingLOD], None]) -> None:
        """Register callback for LOD transitions."""
        self._lod_change_callbacks.append(callback)
    
    # =========================================================================
    # Query Properties
    # =========================================================================
    
    @property
    def lod(self) -> PickingLOD:
        """Current picking LOD level."""
        return self._lod
    
    @property
    def data_loaded(self) -> bool:
        """Whether any pickable data is loaded."""
        return self._data_loaded
    
    @property
    def hover_allowed(self) -> bool:
        """Whether hover picking is currently allowed."""
        return (
            self._data_loaded and 
            self._hover_enabled and 
            not self._hover_suppressed and
            not self._is_navigating
        )
    
    @property
    def click_allowed(self) -> bool:
        """Whether click picking is currently allowed."""
        return self._data_loaded and self._click_enabled
    
    @property
    def cell_level_allowed(self) -> bool:
        """Whether cell-level data access is allowed."""
        return self._lod >= PickingLOD.P2_CELL
    
    @property
    def metrics(self) -> PickingMetrics:
        """Access to performance metrics."""
        return self._metrics
    
    # =========================================================================
    # Navigation State
    # =========================================================================
    
    def on_navigation_start(self) -> None:
        """Called when camera navigation (rotate/pan/zoom) starts."""
        self._is_navigating = True
    
    def on_navigation_end(self) -> None:
        """Called when camera navigation ends."""
        self._is_navigating = False
    
    # =========================================================================
    # Performance-Based Degradation
    # =========================================================================
    
    def record_hover_timing(self, elapsed_ms: float) -> None:
        """
        Record hover timing and potentially disable hover if too slow.
        """
        should_disable = self._metrics.record_hover(elapsed_ms)
        
        if should_disable and self._hover_enabled:
            self._hover_enabled = False
            logger.warning(
                f"PickingController: Hover disabled due to performance "
                f"({elapsed_ms:.1f}ms > {PickingPerformanceThresholds.HOVER_DISABLE_MS}ms)"
            )
    
    def record_click_timing(self, elapsed_ms: float) -> None:
        """
        Record click timing and potentially degrade LOD if too slow.
        """
        should_warn, should_degrade = self._metrics.record_click(elapsed_ms)
        
        if should_warn:
            logger.warning(
                f"PickingController: Click slow ({elapsed_ms:.1f}ms > "
                f"{PickingPerformanceThresholds.CLICK_WARN_MS}ms)"
            )
        
        if should_degrade and self._lod > PickingLOD.P1_ACTOR:
            old_lod = self._lod
            self._lod = PickingLOD.P1_ACTOR
            self._hover_enabled = False
            self._notify_lod_change(old_lod, self._lod)
            logger.warning(
                f"PickingController: Degraded to LOD-P1 due to performance "
                f"({elapsed_ms:.1f}ms > {PickingPerformanceThresholds.CLICK_DEGRADE_MS}ms)"
            )
    
    # =========================================================================
    # Timing Context Managers
    # =========================================================================
    
    class _TimedOperation:
        """Context manager for timing picking operations."""
        
        def __init__(self, controller: 'PickingController', operation: str):
            self._controller = controller
            self._operation = operation
            self._start_time: float = 0
            self.elapsed_ms: float = 0
        
        def __enter__(self):
            self._start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000
            
            if self._operation == 'hover':
                self._controller.record_hover_timing(self.elapsed_ms)
            elif self._operation == 'click':
                self._controller.record_click_timing(self.elapsed_ms)
            
            return False  # Don't suppress exceptions
    
    def timed_hover(self) -> _TimedOperation:
        """Context manager for timing hover operations."""
        return self._TimedOperation(self, 'hover')
    
    def timed_click(self) -> _TimedOperation:
        """Context manager for timing click operations."""
        return self._TimedOperation(self, 'click')
    
    # =========================================================================
    # VTK Picker Management - SEPARATED BY RESPONSIBILITY
    # =========================================================================
    # 
    # CRITICAL: Two separate pickers for two separate purposes:
    # - _prop_picker: Actor-level picking (cheap, O(actors) not O(cells))
    # - _cell_picker: Cell-level picking (expensive, O(cells))
    #
    # RULE: Never use _cell_picker unless LOD >= P2_CELL AND event is a click.
    # =========================================================================
    
    def get_prop_picker(self):
        """
        Get the actor-level picker (vtkPropPicker).
        
        This is CHEAP - only tests against actor bounding boxes, not cells.
        Use for: ALL hover operations, LOD-P1 clicks.
        
        Returns:
            vtkPropPicker: Actor-level picker (fast)
        """
        if self._prop_picker is None:
            try:
                import vtk
                self._prop_picker = vtk.vtkPropPicker()
                logger.debug("PickingController: Created vtkPropPicker (actor-level)")
            except ImportError:
                logger.warning("VTK not available for prop picker")
                return None
        return self._prop_picker
    
    def get_cell_picker(self):
        """
        Get the cell-level picker (vtkCellPicker).
        
        This is EXPENSIVE - tests every cell in the mesh.
        Use ONLY for: LOD >= P2_CELL clicks. NEVER for hover.
        
        Returns:
            vtkCellPicker: Cell-level picker (slow)
        """
        if self._cell_picker is None:
            try:
                import vtk
                self._cell_picker = vtk.vtkCellPicker()
                self._cell_picker.SetTolerance(0.001)
                logger.debug("PickingController: Created vtkCellPicker (cell-level, P2+ only)")
            except ImportError:
                logger.warning("VTK not available for cell picker")
                return None
        return self._cell_picker
    
    # =========================================================================
    # Debug/Diagnostic
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status for debugging."""
        return {
            'lod': self._lod.name,
            'data_loaded': self._data_loaded,
            'cell_count': self._cell_count,
            'has_block_model': self._has_block_model,
            'has_drillholes': self._has_drillholes,
            'hover_allowed': self.hover_allowed,
            'click_allowed': self.click_allowed,
            'is_navigating': self._is_navigating,
            'avg_hover_ms': self._metrics.avg_hover_ms,
            'avg_click_ms': self._metrics.avg_click_ms,
        }
    
    def __repr__(self) -> str:
        return (
            f"PickingController(lod={self._lod.name}, "
            f"data={self._data_loaded}, cells={self._cell_count})"
        )


# Module-level singleton for shared access
_picking_controller: Optional[PickingController] = None


def get_picking_controller() -> PickingController:
    """Get the singleton PickingController instance."""
    global _picking_controller
    if _picking_controller is None:
        _picking_controller = PickingController()
    return _picking_controller


def reset_picking_controller() -> PickingController:
    """Reset the singleton PickingController (for testing)."""
    global _picking_controller
    _picking_controller = PickingController()
    return _picking_controller

