"""
Base display/control panel scaffolding.

Extends BasePanel with renderer awareness and scene-layer helpers so that
view/overlay panels share a consistent integration path with the controller.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, TYPE_CHECKING, Any

from .base_panel import BaseDockPanel

if TYPE_CHECKING:
    from ..controllers.app_controller import AppController
    from ..visualization.scene_layer import SceneLayer

logger = logging.getLogger(__name__)


class BaseDisplayPanel(BaseDockPanel):
    """
    Base class for panels that interact with scene layers / renderer state.
    
    Provides:
    - Renderer tracking (sourced from controller)
    - Scene layer refresh hook
    - Helper for overlay configuration requests
    """
    
    def __init__(self, parent: Optional[Any] = None, panel_id: Optional[str] = None):
        self._scene_layers: Optional[Sequence["SceneLayer"]] = None
        self._renderer = None
        super().__init__(parent=parent, panel_id=panel_id)
    
    def setup_ui(self):
        """Default implementation - subclasses should override."""
        pass
    
    # ------------------------------------------------------------------
    # Controller / renderer binding
    # ------------------------------------------------------------------
    def bind_controller(self, controller: Optional["AppController"]) -> None:
        super().bind_controller(controller)
        if controller and hasattr(controller, "r"):
            self.set_renderer(getattr(controller, "r"))
        self.connect_layer_events()
    
    def set_renderer(self, renderer) -> None:
        """
        Assign the renderer instance used by this panel.
        
        Subclasses can override to perform additional hookup while still
        calling super().set_renderer(renderer).
        """
        self._renderer = renderer
    
    @property
    def renderer(self):
        """Return the current renderer instance (if available)."""
        return self._renderer
    
    # ------------------------------------------------------------------
    # Scene layer change helpers
    # ------------------------------------------------------------------
    def connect_layer_events(self) -> None:
        """
        Subscribe to scene layer changes.
        
        Panels can override this to attach to controller-layer events once
        a dedicated signal exists. The default implementation is a no-op.
        """
        return
    
    def update_from_scene(self, layers: Sequence["SceneLayer"]) -> None:
        """
        Called when renderer layers change.
        
        Args:
            layers: Iterable of SceneLayer metadata
        """
        self._scene_layers = layers
        self.refresh()
    
    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    def request_overlay_change(self, **kwargs) -> None:
        """
        Ask the controller to update overlay/legend configuration.
        
        Args:
            **kwargs: Overlay configuration payload
        """
        if self.controller and hasattr(self.controller, "configure_overlays"):
            try:
                self.controller.configure_overlays(**kwargs)
            except Exception as exc:
                try:
                    exc_msg = str(exc)
                    logger.debug(
                        "Overlay change request from %s failed: %s",
                        self.panel_id,
                        exc_msg,
                    )
                except Exception:
                    logger.debug(
                        "Overlay change request from %s failed: <unprintable error>",
                        self.panel_id,
                    )
        else:
            logger.debug(
                "Overlay change requested from %s but controller is missing",
                self.panel_id,
            )
