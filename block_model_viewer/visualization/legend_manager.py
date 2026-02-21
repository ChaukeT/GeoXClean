"""
Lightweight legend state manager that bridges renderer updates and the LegendWidget.

This wrapper keeps the existing, feature-rich UI legend manager working while exposing
simple metadata signals for the refactored architecture.

Phase 2.3: Integrated with UnifiedStateManager for centralized state management.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from ..core.state_manager import UnifiedStateManager

try:
    from ..ui.legend_manager import LegendManager as UILegendManager
except ImportError:  # pragma: no cover - optional dependency
    UILegendManager = None

try:
    from ..ui.legend_widget import LegendWidget
except ImportError:  # pragma: no cover
    LegendWidget = None  # type: ignore

logger = logging.getLogger(__name__)


class LegendManager(QObject):
    """
    Store legend metadata and forward renderer updates to any bound widgets.
    
    This class intentionally avoids any PyVista calls so panels/widgets can stay
    decoupled from the rendering backend.
    
    Phase 2.3: Now integrates with UnifiedStateManager for centralized state.
    The state manager is the single source of truth, and this manager subscribes
    to state changes for automatic synchronization.
    """

    legend_changed = pyqtSignal(dict)
    visibility_changed = pyqtSignal(bool)

    def __init__(self, renderer=None, state_manager: Optional["UnifiedStateManager"] = None):
        super().__init__()
        self._current_property: Optional[str] = None
        self._categories: Optional[list] = None
        self._category_colors: Optional[Dict[str, Any]] = None
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None
        self._colormap: Optional[str] = None
        self._visibility: bool = True
        self._widget: Optional[LegendWidget] = None
        
        # State manager integration (Phase 2.3)
        self._state_manager: Optional["UnifiedStateManager"] = state_manager
        if state_manager is not None:
            # Subscribe to state changes
            state_manager.legend_state_changed.connect(self._on_state_changed)
            logger.debug("LegendManager subscribed to UnifiedStateManager")

        self._legacy: Optional[UILegendManager] = None
        if UILegendManager is not None and renderer is not None:
            try:
                self._legacy = UILegendManager(renderer)
            except Exception as e:
                logger.debug(f"Could not initialize legacy legend manager: {e}")
                self._legacy = None
    
    def set_state_manager(self, state_manager: "UnifiedStateManager") -> None:
        """
        Connect to a state manager for centralized state.
        
        This allows late binding if the state manager isn't available at construction.
        """
        if self._state_manager is not None:
            # Disconnect from old state manager
            try:
                self._state_manager.legend_state_changed.disconnect(self._on_state_changed)
            except (TypeError, RuntimeError):
                pass
        
        self._state_manager = state_manager
        if state_manager is not None:
            state_manager.legend_state_changed.connect(self._on_state_changed)
            logger.debug("LegendManager connected to new UnifiedStateManager")
    
    def _on_state_changed(self, state: Dict[str, Any]) -> None:
        """
        Handle state changes from UnifiedStateManager.
        
        Automatically syncs local state when the central state changes.
        """
        self._current_property = state.get("property")
        self._vmin = state.get("vmin")
        self._vmax = state.get("vmax")
        self._colormap = state.get("colormap")
        self._categories = state.get("categories")
        self._category_colors = state.get("category_colors")
        self._visibility = state.get("visible", True)
        
        # Emit our own signal for widgets
        self.legend_changed.emit(state)
        logger.debug(f"LegendManager synced from state: {self._current_property}")

    # ------------------------------------------------------------------ #
    # Modern metadata API
    # ------------------------------------------------------------------ #
    def bind_widget(self, widget: LegendWidget) -> None:
        """Attach the LegendWidget and synchronise state."""
        self._widget = widget
        if hasattr(widget, "bind_manager"):
            try:
                widget.bind_manager(self)
            except Exception:
                pass
        if self._legacy is not None:
            try:
                self._legacy.bind_widget(widget)
            except Exception:
                pass

    def update_from_property(self, name: str, metadata: Dict[str, Any]) -> None:
        """
        Persist the latest legend metadata and notify consumers.
        
        Phase 2.3: Also updates the UnifiedStateManager if connected.
        """
        self._current_property = name
        self._vmin = metadata.get("vmin")
        self._vmax = metadata.get("vmax")
        self._colormap = metadata.get("colormap")
        categories = metadata.get("categories")
        self._categories = list(categories) if categories else None
        category_colors = metadata.get("category_colors")
        self._category_colors = category_colors

        payload = {
            "property": name,
            "title": metadata.get("title", name),
            "vmin": self._vmin,
            "vmax": self._vmax,
            "colormap": self._colormap,
            "categories": self._categories,
            "category_colors": category_colors,
            "mode": "discrete" if self._categories else "continuous",
        }
        
        # Update state manager if connected (Phase 2.3)
        if self._state_manager is not None:
            self._state_manager.update_legend(
                property_name=name,
                vmin=self._vmin,
                vmax=self._vmax,
                colormap=self._colormap or "viridis",
                categories=self._categories,
                category_colors=category_colors,
            )
        else:
            # Only emit if not using state manager (avoid double emission)
            self.legend_changed.emit(payload)

    def set_visibility(self, visible: bool) -> None:
        """Toggle overall legend visibility."""
        visible = bool(visible)
        if self._visibility == visible:
            if self._legacy is not None:
                try:
                    self._legacy.set_visibility(visible)
                except Exception:
                    pass
            return
        self._visibility = visible
        self.visibility_changed.emit(visible)
        if self._legacy is not None:
            try:
                self._legacy.set_visibility(visible)
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the legend state."""
        return {
            "property": self._current_property,
            "vmin": self._vmin,
            "vmax": self._vmax,
            "colormap": self._colormap,
            "categories": list(self._categories) if self._categories else None,
            "category_colors": self._category_colors,
            "visible": self._visibility,
        }

    def clear(self) -> None:
        """
        FIX CS-004: Reset legend state on scene clear.
        
        Clears local state and delegates to the legacy UI manager.
        """
        # Clear local state
        self._current_property = None
        self._categories = None
        self._category_colors = None
        self._vmin = None
        self._vmax = None
        self._colormap = None
        self._visibility = False
        
        # Delegate to legacy manager
        if self._legacy is not None:
            try:
                self._legacy.clear()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Legacy compatibility
    # ------------------------------------------------------------------ #
    @property
    def widget(self) -> Optional[LegendWidget]:
        if self._legacy is not None and getattr(self._legacy, "widget", None):
            return self._legacy.widget
        return self._widget

    def __getattr__(self, item):
        """
        Delegate unknown attributes/methods to the legacy manager so existing
        renderer code keeps working during the migration.
        """
        if item.startswith("_"):
            raise AttributeError(item)
        if self._legacy is not None and hasattr(self._legacy, item):
            return getattr(self._legacy, item)
        raise AttributeError(item)
