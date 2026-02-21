"""
Axis manager - DEPRECATED.

.. deprecated::
    AxisManager is deprecated. Use OverlayManager instead.
    This module exists only for backward compatibility.
    
The AxisManager functionality has been consolidated into the unified
OverlayManager class in overlay_manager.py.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal


class AxisManager(QObject):
    """
    DEPRECATED: Use OverlayManager instead.
    
    This class is a thin wrapper that delegates to OverlayManager
    for backward compatibility. All axis/coordinate/scale bar functionality
    has been consolidated into the unified OverlayManager.
    
    .. deprecated::
        Use OverlayManager from overlay_manager.py instead.
    """

    bounds_changed = pyqtSignal(tuple)  # (xmin, xmax, ymin, ymax, zmin, zmax)
    camera_changed = pyqtSignal(dict)  # Camera metadata dict

    def __init__(self, overlay_manager: Optional[Any] = None):
        """
        Initialize the deprecated AxisManager.
        
        Args:
            overlay_manager: Optional OverlayManager to delegate to.
        """
        super().__init__()
        
        # Issue deprecation warning
        warnings.warn(
            "AxisManager is deprecated; use OverlayManager instead. "
            "Import from overlay_manager.py: from .overlay_manager import OverlayManager",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Store reference to OverlayManager for delegation
        self._overlay_manager = overlay_manager
        
        # Fallback state if no overlay manager provided
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._camera_metadata: Optional[Dict[str, Any]] = None
        self._elevation_widget: Optional[Any] = None
        self._coordinate_widget: Optional[Any] = None
        self._scale_bar_widget: Optional[Any] = None
        
        # Connect signals from overlay manager if provided
        if overlay_manager is not None:
            try:
                overlay_manager.bounds_changed.connect(self.bounds_changed.emit)
                overlay_manager.camera_changed.connect(self.camera_changed.emit)
            except Exception:
                pass

    def attach_overlay_manager(self, overlay_manager: Any) -> None:
        """
        Attach an OverlayManager for delegation.
        
        Args:
            overlay_manager: OverlayManager instance to delegate to.
        """
        self._overlay_manager = overlay_manager
        try:
            overlay_manager.bounds_changed.connect(self.bounds_changed.emit)
            overlay_manager.camera_changed.connect(self.camera_changed.emit)
        except Exception:
            pass

    def bind_elevation_widget(self, widget: Any) -> None:
        """
        Bind elevation axis widget.
        
        .. deprecated:: Use OverlayManager.bind_elevation_widget() instead.
        """
        if self._overlay_manager is not None:
            self._overlay_manager.bind_elevation_widget(widget)
        else:
            self._elevation_widget = widget
            self._update_widgets()

    def bind_coordinate_widget(self, widget: Any) -> None:
        """
        Bind coordinate display widget.
        
        .. deprecated:: Use OverlayManager.bind_coordinate_widget() instead.
        """
        if self._overlay_manager is not None:
            self._overlay_manager.bind_coordinate_widget(widget)
        else:
            self._coordinate_widget = widget
            self._update_widgets()

    def bind_scale_bar_widget(self, widget: Any) -> None:
        """
        Bind scale bar widget.
        
        .. deprecated:: Use OverlayManager.bind_scale_bar_widget() instead.
        """
        if self._overlay_manager is not None:
            self._overlay_manager.bind_scale_bar_widget(widget)
        else:
            self._scale_bar_widget = widget
            self._update_widgets()

    def set_bounds(self, bounds: Optional[Tuple[float, float, float, float, float, float]]) -> None:
        """
        Update scene bounds and notify widgets.
        
        .. deprecated:: Use OverlayManager.set_bounds() instead.
        """
        if self._overlay_manager is not None:
            self._overlay_manager.set_bounds(bounds)
        else:
            if bounds == self._bounds:
                return
            self._bounds = bounds
            self.bounds_changed.emit(bounds if bounds else ())
            self._update_widgets()

    def set_camera_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update camera metadata.
        
        .. deprecated:: Use OverlayManager.set_camera_metadata() instead.
        """
        if self._overlay_manager is not None:
            self._overlay_manager.set_camera_metadata(metadata)
        else:
            if metadata == self._camera_metadata:
                return
            self._camera_metadata = metadata
            self.camera_changed.emit(metadata)
            self._update_widgets()

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get current scene bounds."""
        if self._overlay_manager is not None:
            return self._overlay_manager.bounds
        return self._bounds

    @property
    def camera_metadata(self) -> Optional[Dict[str, Any]]:
        """Get current camera metadata."""
        if self._overlay_manager is not None:
            return self._overlay_manager.camera_metadata
        return self._camera_metadata

    def _update_widgets(self) -> None:
        """Update all bound widgets with current state (fallback implementation)."""
        bounds = self._bounds
        
        if self._elevation_widget is not None:
            try:
                if bounds:
                    self._elevation_widget.show_for_bounds(bounds)
                else:
                    self._elevation_widget.hide()
            except Exception:
                pass
        
        if self._coordinate_widget is not None:
            try:
                self._coordinate_widget.set_visible(bool(bounds))
                if bounds and hasattr(self._coordinate_widget, 'update_bounds'):
                    self._coordinate_widget.update_bounds(bounds)
            except Exception:
                pass
        
        if self._scale_bar_widget is not None:
            try:
                if bounds:
                    if hasattr(self._scale_bar_widget, 'update_bounds'):
                        self._scale_bar_widget.update_bounds(bounds)
                    if hasattr(self._scale_bar_widget, 'set_visible'):
                        self._scale_bar_widget.set_visible(True)
                else:
                    if hasattr(self._scale_bar_widget, 'set_visible'):
                        self._scale_bar_widget.set_visible(False)
            except Exception:
                pass
