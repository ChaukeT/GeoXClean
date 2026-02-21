"""
InteractionController - Manages mouse modes, VTK interactor styles, and camera controls.

This controller extracts all interaction logic from MainWindow, providing a clean
separation between window management and user interaction handling.

Responsibilities:
- Own mouse mode state
- Talk to ViewerWidget for mode changes
- Manage VTK interactor style fallbacks
- Update QAction checked states
- Emit signals for UI updates
- Handle zoom in/out operations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QStatusBar

from .mouse_modes import MouseMode, MOUSE_MODE_DESCRIPTIONS, MOUSE_MODE_SHORT_DESCRIPTIONS

if TYPE_CHECKING:
    from ..viewer_widget import ViewerWidget
    from ..signals import UISignals

logger = logging.getLogger(__name__)


class InteractionController(QObject):
    """
    Controller for mouse interaction modes and camera operations.
    
    This controller owns:
    - Current mouse mode state
    - VTK interactor style management
    - QAction state synchronization
    - Camera zoom operations
    
    Signals:
        mode_changed: Emitted when mouse mode changes (str mode_name)
    """
    
    # Emitted when the interaction mode changes
    mode_changed = pyqtSignal(str)
    
    def __init__(
        self,
        viewer: Optional['ViewerWidget'] = None,
        signals: Optional['UISignals'] = None,
        status_bar: Optional[QStatusBar] = None,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the interaction controller.
        
        Args:
            viewer: The ViewerWidget to control
            signals: UISignals hub for centralized signaling
            status_bar: Status bar for showing mode messages
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self._viewer = viewer
        self._signals = signals
        self._status_bar = status_bar
        self._current_mode = MouseMode.ORIGINAL
        
        # Action references (set via bind_actions)
        self._select_action: Optional[QAction] = None
        self._pan_action: Optional[QAction] = None
        self._zoom_box_action: Optional[QAction] = None
        self._action_group: Optional[QActionGroup] = None
        
        # Connect viewer signal if available
        if self._viewer is not None:
            try:
                self._viewer.mouse_mode_changed.connect(self._on_viewer_mode_changed)
            except Exception as e:
                logger.debug(f"Could not connect viewer signal: {e}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def bind_viewer(self, viewer: 'ViewerWidget') -> None:
        """Bind or rebind the viewer widget."""
        if self._viewer is not None:
            try:
                self._viewer.mouse_mode_changed.disconnect(self._on_viewer_mode_changed)
            except Exception:
                pass
        
        self._viewer = viewer
        if viewer is not None:
            try:
                viewer.mouse_mode_changed.connect(self._on_viewer_mode_changed)
            except Exception as e:
                logger.debug(f"Could not connect viewer signal: {e}")
    
    def bind_status_bar(self, status_bar: QStatusBar) -> None:
        """Bind the status bar for messages."""
        self._status_bar = status_bar
    
    def bind_actions(
        self,
        select_action: Optional[QAction] = None,
        pan_action: Optional[QAction] = None,
        zoom_box_action: Optional[QAction] = None,
        action_group: Optional[QActionGroup] = None
    ) -> None:
        """
        Bind QAction references for state synchronization.
        
        Args:
            select_action: Action for select mode
            pan_action: Action for pan mode
            zoom_box_action: Action for zoom box mode
            action_group: The exclusive action group
        """
        self._select_action = select_action
        self._pan_action = pan_action
        self._zoom_box_action = zoom_box_action
        self._action_group = action_group
    
    @property
    def current_mode(self) -> MouseMode:
        """Get the current mouse mode."""
        return self._current_mode
    
    @property
    def current_mode_string(self) -> str:
        """Get the current mode as a string."""
        return self._current_mode.to_string()
    
    # =========================================================================
    # MODE SETTERS
    # =========================================================================
    
    def set_mode_select(self) -> None:
        """Enable selection/clicking; use standard trackball camera style."""
        self._set_mode(MouseMode.SELECT)
    
    def set_mode_pan(self) -> None:
        """Set interactor to a style suited for panning."""
        self._set_mode(MouseMode.PAN)
    
    def set_mode_zoom_box(self) -> None:
        """Use rubber-band (drag box) camera zoom style."""
        self._set_mode(MouseMode.ZOOM_BOX)
    
    def set_mode_reset(self) -> None:
        """Restore the original/default mouse interaction style."""
        self._set_mode(MouseMode.ORIGINAL, show_toast=True)
    
    def set_mode(self, mode: MouseMode) -> None:
        """Set the interaction mode directly."""
        self._set_mode(mode)
    
    def set_mode_from_string(self, mode_str: str) -> None:
        """Set mode from a string identifier."""
        mode = MouseMode.from_string(mode_str)
        self._set_mode(mode)
    
    # =========================================================================
    # ZOOM OPERATIONS
    # =========================================================================
    
    def zoom_in(self) -> None:
        """Zoom in by a fixed step."""
        if self._viewer is not None:
            try:
                if hasattr(self._viewer, 'zoom_in'):
                    self._viewer.zoom_in()
                    self._show_status("Zoomed in", 1500)
                    return
            except Exception:
                logger.debug("viewer.zoom_in failed", exc_info=True)
        
        # Fallback to direct camera zoom
        self._camera_zoom(1.2)
        self._show_status("Zoomed in", 1500)
    
    def zoom_out(self) -> None:
        """Zoom out by a fixed step."""
        if self._viewer is not None:
            try:
                if hasattr(self._viewer, 'zoom_out'):
                    self._viewer.zoom_out()
                    self._show_status("Zoomed out", 1500)
                    return
            except Exception:
                logger.debug("viewer.zoom_out failed", exc_info=True)
        
        # Fallback to direct camera zoom
        self._camera_zoom(1.0 / 1.2)
        self._show_status("Zoomed out", 1500)
    
    def zoom(self, factor: float) -> None:
        """Zoom by a custom factor."""
        if self._viewer is not None:
            try:
                if hasattr(self._viewer, 'zoom'):
                    self._viewer.zoom(factor)
                    return
            except Exception:
                pass
        self._camera_zoom(factor)
    
    # =========================================================================
    # ACTION STATE MANAGEMENT
    # =========================================================================
    
    def update_action_checks(self, mode: Optional[str] = None, show_message: bool = True) -> None:
        """
        Update the checked state of mouse menu actions.
        
        Args:
            mode: Mode string to sync to (uses current mode if None)
            show_message: Whether to show a status bar message
        """
        if mode is not None:
            target_mode = MouseMode.from_string(mode)
        else:
            target_mode = self._current_mode
        
        try:
            # Reset all checks first
            self._safe_set_checked(self._select_action, False)
            self._safe_set_checked(self._pan_action, False)
            self._safe_set_checked(self._zoom_box_action, False)
            
            # Set the appropriate check
            if target_mode == MouseMode.SELECT:
                self._safe_set_checked(self._select_action, True)
            elif target_mode == MouseMode.PAN:
                self._safe_set_checked(self._pan_action, True)
            elif target_mode == MouseMode.ZOOM_BOX:
                self._safe_set_checked(self._zoom_box_action, True)
            # ORIGINAL/ROTATE leave all unchecked
            
            if show_message:
                desc = MOUSE_MODE_SHORT_DESCRIPTIONS.get(target_mode, "Mouse Mode: Unknown")
                self._show_status(desc, 2000)
                
        except Exception:
            logger.debug("Failed to update action checks", exc_info=True)
    
    def sync_from_viewer(self) -> None:
        """
        Synchronize controller state from viewer's current mode.
        
        Useful during initialization when the viewer may already have a mode set.
        """
        if self._viewer is None:
            return
        
        try:
            renderer = getattr(self._viewer, 'renderer', None)
            if renderer is not None:
                current = getattr(renderer, '_current_mouse_mode', None)
                if current:
                    self._current_mode = MouseMode.from_string(current)
                    self.update_action_checks(show_message=False)
        except Exception as e:
            logger.debug(f"Could not sync from viewer: {e}")
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _set_mode(self, mode: MouseMode, show_toast: bool = False) -> None:
        """
        Internal method to set interaction mode.
        
        Args:
            mode: The MouseMode to set
            show_toast: Whether to show a toast notification (for reset)
        """
        mode_str = mode.to_string()
        
        # Try viewer's high-level API first
        if self._viewer is not None:
            try:
                if hasattr(self._viewer, 'set_interaction_mode'):
                    self._viewer.set_interaction_mode(mode_str)
                    self._current_mode = mode
                    self._update_ui_for_mode(mode, show_toast)
                    return
            except Exception:
                logger.debug(f"viewer.set_interaction_mode({mode_str}) failed", exc_info=True)
        
        # Fallback: Set VTK interactor style directly
        interactor = self._get_vtk_interactor()
        if interactor is not None:
            self._apply_vtk_style(interactor, mode)
        
        self._current_mode = mode
        self._update_ui_for_mode(mode, show_toast)
    
    def _update_ui_for_mode(self, mode: MouseMode, show_toast: bool = False) -> None:
        """Update UI elements after mode change."""
        # Update action checks
        if mode == MouseMode.ORIGINAL:
            self._clear_all_action_checks()
        else:
            self._update_action_check_for_mode(mode)
        
        # Show status message
        desc = MOUSE_MODE_DESCRIPTIONS.get(mode, "Mouse Mode: Unknown")
        self._show_status(desc, 3000)
        
        # Show toast for reset
        if show_toast and mode == MouseMode.ORIGINAL:
            self._show_toast("Original mouse interaction restored")
        
        # Emit signal
        try:
            self.mode_changed.emit(mode.to_string())
        except Exception:
            pass
        
        # Also emit to UISignals if available
        if self._signals is not None:
            try:
                self._signals.mouseModeChanged.emit(mode.to_string())
            except Exception:
                pass
    
    def _update_action_check_for_mode(self, mode: MouseMode) -> None:
        """Set the appropriate action as checked."""
        try:
            if mode == MouseMode.SELECT:
                self._safe_set_checked(self._select_action, True)
            elif mode == MouseMode.PAN:
                self._safe_set_checked(self._pan_action, True)
            elif mode == MouseMode.ZOOM_BOX:
                self._safe_set_checked(self._zoom_box_action, True)
        except Exception as e:
            logger.debug(f"Failed to update action check: {e}")
    
    def _clear_all_action_checks(self) -> None:
        """Clear all action checks (for reset mode)."""
        try:
            # Temporarily disable exclusivity to clear all
            if self._action_group is not None:
                try:
                    self._action_group.setExclusive(False)
                except Exception:
                    pass
            
            self._safe_set_checked(self._select_action, False)
            self._safe_set_checked(self._pan_action, False)
            self._safe_set_checked(self._zoom_box_action, False)
            
            # Re-enable exclusivity
            if self._action_group is not None:
                try:
                    self._action_group.setExclusive(True)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to clear action checks: {e}")
    
    def _safe_set_checked(self, action: Optional[QAction], checked: bool) -> None:
        """Safely set action checked state."""
        if action is not None:
            try:
                action.setChecked(checked)
            except Exception as e:
                logger.debug(f"Failed to set action checked: {e}")
    
    def _on_viewer_mode_changed(self, mode_str: str) -> None:
        """Handle mode change signal from viewer."""
        mode = MouseMode.from_string(mode_str)
        if mode != self._current_mode:
            self._current_mode = mode
            self.update_action_checks(mode_str, show_message=False)
    
    # =========================================================================
    # VTK INTERACTOR HELPERS
    # =========================================================================
    
    def _get_vtk_interactor(self) -> Optional[Any]:
        """
        Robust lookup for a VTK interactor object.
        
        Tries several fallbacks to find a working interactor.
        """
        if self._viewer is None:
            return None
        
        try:
            # 1) Preferred: renderer may expose an interactor-like object
            renderer = getattr(self._viewer, 'renderer', None)
            if renderer is not None:
                inter = getattr(renderer, 'interactor', None)
                if inter is not None:
                    return inter
            
            # 2) PyVista QtInteractor path
            plotter = getattr(self._viewer, 'plotter', None)
            if plotter is None and renderer is not None:
                plotter = getattr(renderer, 'plotter', None)
            
            if plotter is not None:
                candidate = (
                    getattr(plotter, 'iren', None) or
                    getattr(plotter, 'interactor', None) or
                    getattr(plotter, 'render_window_interactor', None)
                )
                if candidate is not None:
                    return candidate
            
            # 3) Last resort: raw attribute on viewer
            return getattr(self._viewer, 'interactor', None)
            
        except Exception:
            return None
    
    def _get_vtk_renderer(self) -> Optional[Any]:
        """Get the underlying VTK renderer."""
        if self._viewer is None:
            return None
        try:
            renderer = getattr(self._viewer, 'renderer', None)
            if renderer is not None:
                return getattr(renderer, 'renderer', None)
        except Exception:
            pass
        return None
    
    def _apply_vtk_style(self, interactor: Any, mode: MouseMode) -> None:
        """Apply VTK interactor style for the given mode."""
        try:
            import vtk
            
            if mode == MouseMode.SELECT or mode == MouseMode.PAN or mode == MouseMode.ROTATE:
                style = vtk.vtkInteractorStyleTrackballCamera()
            elif mode == MouseMode.ZOOM_BOX:
                style = vtk.vtkInteractorStyleRubberBandCamera()
            else:
                style = vtk.vtkInteractorStyleTrackballCamera()
            
            interactor.SetInteractorStyle(style)
            
            # Update renderer's mode tracking
            if self._viewer is not None:
                renderer = getattr(self._viewer, 'renderer', None)
                if renderer is not None:
                    renderer._current_mouse_mode = mode.to_string()
                    
        except Exception as e:
            logger.warning(f"Failed to apply VTK style: {e}")
    
    def _camera_zoom(self, factor: float) -> None:
        """Direct camera zoom fallback."""
        ren = self._get_vtk_renderer()
        if ren is None:
            return
        
        try:
            cam = ren.GetActiveCamera()
            cam.Zoom(factor)
            ren.ResetCameraClippingRange()
            
            # Try to trigger render
            if self._viewer is not None:
                renderer = getattr(self._viewer, 'renderer', None)
                if renderer is not None:
                    rw = getattr(renderer, 'render_window', None)
                    if rw is not None:
                        rw.Render()
        except Exception:
            pass
    
    # =========================================================================
    # UI FEEDBACK HELPERS
    # =========================================================================
    
    def _show_status(self, message: str, timeout_ms: int = 3000) -> None:
        """Show a status bar message."""
        if self._status_bar is not None:
            try:
                self._status_bar.showMessage(message, timeout_ms)
            except Exception:
                pass
    
    def _show_toast(self, message: str) -> None:
        """Show a toast notification."""
        try:
            # Import here to avoid circular imports
            from ..toast import ToastWidget
            parent = self.parent()
            if parent is not None:
                ToastWidget.show_message(parent, message, 3000)
        except Exception as e:
            logger.debug(f"Toast failed, using status bar: {e}")
            self._show_status(message, 3000)

