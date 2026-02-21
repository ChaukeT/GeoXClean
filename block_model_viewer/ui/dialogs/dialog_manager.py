"""
DialogManager - Centralized dialog lifecycle management for GeoX.

Extracts dialog management from MainWindow for cleaner architecture.

Responsibilities:
- Track open dialogs
- Save/restore dialog geometries via QSettings
- Handle unsaved changes protection
- Provide show_or_create helpers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Any

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QDialog, QWidget, QMessageBox

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)

# Settings keys
SETTINGS_ORG = "GeoX"
SETTINGS_APP = "DialogGeometries"


class DialogManager(QObject):
    """
    Manages dialog lifecycle: tracking, geometry persistence, and unsaved changes.
    
    Usage:
        self.dialogs = DialogManager(self)
        
        # Setup a dialog with persistence
        self.dialogs.setup_persistence(dialog, 'my_dialog', 'My Dialog')
        
        # Show or create a dialog
        dialog = self.dialogs.show_or_create('my_dialog', create_callback)
        
        # Save all geometries before app close
        self.dialogs.save_all_geometries()
    """
    
    def __init__(self, parent: Optional[QMainWindow] = None):
        """
        Initialize the dialog manager.
        
        Args:
            parent: The main window (for QSettings and dialog parenting)
        """
        super().__init__(parent)
        self._parent = parent
        
        # Track all open dialogs
        self._open_dialogs: List[QDialog] = []
        
        # Map of dialog_name -> dialog instance (for show_or_create)
        self._dialog_registry: Dict[str, QDialog] = {}
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    @property
    def open_dialogs(self) -> List[QDialog]:
        """Get list of currently tracked open dialogs."""
        # Filter out destroyed dialogs
        self._open_dialogs = [d for d in self._open_dialogs if self._is_valid(d)]
        return self._open_dialogs.copy()
    
    def setup_persistence(
        self,
        dialog: QDialog,
        dialog_name: str,
        display_name: Optional[str] = None
    ) -> None:
        """
        Setup persistence for a dialog (save/restore geometry, close protection).
        
        Args:
            dialog: The QDialog to setup
            dialog_name: Unique name for QSettings key
            display_name: Human-readable name for close confirmation dialog
        """
        if dialog is None:
            return
        
        # Setup window flags for taskbar persistence
        self._setup_window_flags(dialog)
        
        # Restore saved geometry
        self._restore_geometry(dialog, dialog_name)
        
        # Track dialog
        if dialog not in self._open_dialogs:
            self._open_dialogs.append(dialog)
        
        # Register by name
        self._dialog_registry[dialog_name] = dialog
        
        # Setup enhanced close event
        self._setup_close_event(dialog, dialog_name, display_name)
        
        logger.debug(f"Setup persistence for dialog: {dialog_name}")
    
    def show_or_create(
        self,
        dialog_name: str,
        create_callback: Callable[[], QDialog],
        attr_holder: Optional[Any] = None,
        attr_name: Optional[str] = None
    ) -> QDialog:
        """
        Show an existing dialog or create a new one.
        
        Args:
            dialog_name: Unique identifier for this dialog
            create_callback: Function to create the dialog if needed
            attr_holder: Optional object holding dialog as attribute
            attr_name: Optional attribute name on attr_holder
            
        Returns:
            The dialog instance (existing or newly created)
        """
        # Check registered dialogs first
        dialog = self._dialog_registry.get(dialog_name)
        
        # If not in registry, check attr_holder
        if dialog is None and attr_holder is not None and attr_name is not None:
            dialog = getattr(attr_holder, attr_name, None)
        
        # Check if dialog is still valid
        if dialog is not None and self._is_valid(dialog):
            # Restore and show
            if dialog.isMinimized():
                dialog.showNormal()
            else:
                dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            logger.debug(f"Restored existing dialog: {dialog_name}")
            return dialog
        
        # Dialog doesn't exist or is invalid - create new
        logger.debug(f"Creating new dialog: {dialog_name}")
        dialog = create_callback()
        
        # Update registry
        self._dialog_registry[dialog_name] = dialog
        
        # Update attr_holder if specified
        if attr_holder is not None and attr_name is not None:
            setattr(attr_holder, attr_name, dialog)
        
        return dialog
    
    def is_valid(self, dialog: Optional[QDialog]) -> bool:
        """
        Check if a dialog is still valid (not destroyed).
        
        Args:
            dialog: The dialog to check
            
        Returns:
            True if dialog is valid, False otherwise
        """
        return self._is_valid(dialog)
    
    def save_all_geometries(self) -> None:
        """Save geometry for all tracked dialogs."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        
        for dialog_name, dialog in self._dialog_registry.items():
            if dialog is not None and self._is_valid(dialog):
                try:
                    geometry = dialog.saveGeometry()
                    if geometry:
                        settings.setValue(dialog_name, geometry)
                        logger.debug(f"Saved geometry for {dialog_name}")
                except Exception as e:
                    logger.warning(f"Failed to save geometry for {dialog_name}: {e}")
        
        logger.debug(f"Saved geometries for {len(self._dialog_registry)} dialogs")
    
    def save_geometry(self, dialog: QDialog, dialog_name: str) -> None:
        """
        Save geometry for a specific dialog.
        
        Args:
            dialog: The dialog to save
            dialog_name: The QSettings key
        """
        if dialog is None or not self._is_valid(dialog):
            return
        
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            geometry = dialog.saveGeometry()
            if geometry:
                settings.setValue(dialog_name, geometry)
                logger.debug(f"Saved geometry for {dialog_name}")
        except Exception as e:
            logger.warning(f"Failed to save geometry for {dialog_name}: {e}")
    
    def close_all(self) -> None:
        """Close all tracked dialogs."""
        for dialog in self._open_dialogs.copy():
            if self._is_valid(dialog):
                try:
                    dialog.close()
                except Exception:
                    pass
        
        self._open_dialogs.clear()
        self._dialog_registry.clear()
    
    def untrack(self, dialog: QDialog) -> None:
        """Remove a dialog from tracking."""
        if dialog in self._open_dialogs:
            self._open_dialogs.remove(dialog)
        
        # Remove from registry
        to_remove = [k for k, v in self._dialog_registry.items() if v is dialog]
        for key in to_remove:
            del self._dialog_registry[key]
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _is_valid(self, dialog: Optional[QDialog]) -> bool:
        """Check if a dialog widget is still valid."""
        if dialog is None:
            return False
        try:
            # Try to access widget properties - raises if destroyed
            _ = dialog.isVisible()
            _ = dialog.windowTitle()
            return True
        except (RuntimeError, AttributeError):
            return False
    
    def _setup_window_flags(self, dialog: QDialog) -> None:
        """Setup window flags for taskbar persistence."""
        if hasattr(dialog, '_panel_flags_set') and dialog._panel_flags_set:
            return
        
        try:
            from ..panel_utils import setup_panel_window_flags
            setup_panel_window_flags(dialog)
            dialog._panel_flags_set = True
        except ImportError:
            # Fallback if panel_utils not available
            pass
    
    def _restore_geometry(self, dialog: QDialog, dialog_name: str) -> None:
        """Restore saved geometry for a dialog."""
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            geometry = settings.value(dialog_name)
            if geometry:
                dialog.restoreGeometry(geometry)
                logger.debug(f"Restored geometry for {dialog_name}")
        except Exception as e:
            logger.warning(f"Failed to restore geometry for {dialog_name}: {e}")
    
    def _setup_close_event(
        self,
        dialog: QDialog,
        dialog_name: str,
        display_name: Optional[str]
    ) -> None:
        """Setup enhanced close event with geometry save and unsaved changes check."""
        
        manager = self  # Capture reference for closure
        
        def enhanced_close_event(event):
            # Remove from tracked dialogs
            if dialog in manager._open_dialogs:
                manager._open_dialogs.remove(dialog)
            
            # Save geometry before closing
            manager.save_geometry(dialog, dialog_name)
            
            # Check for unsaved changes
            has_changes = False
            change_summary = ""
            save_callback = None
            
            if hasattr(dialog, 'has_unsaved_changes'):
                try:
                    has_changes = dialog.has_unsaved_changes()
                except Exception:
                    pass
            
            if hasattr(dialog, 'get_change_summary'):
                try:
                    change_summary = dialog.get_change_summary()
                except Exception:
                    pass
            
            if hasattr(dialog, 'save_changes'):
                save_callback = dialog.save_changes
            
            # Show confirmation if there are unsaved changes
            if has_changes:
                result = manager._show_unsaved_changes_dialog(
                    dialog, 
                    display_name or dialog_name,
                    change_summary,
                    save_callback
                )
                if result == 'cancel':
                    event.ignore()
                    return
                elif result == 'hide':
                    dialog.hide()
                    event.ignore()
                    return
            
            # Default: hide instead of close (preserves dialog state)
            dialog.hide()
            event.ignore()
        
        dialog.closeEvent = enhanced_close_event
    
    def _show_unsaved_changes_dialog(
        self,
        parent: QWidget,
        panel_name: str,
        change_summary: str,
        save_callback: Optional[Callable[[], bool]]
    ) -> str:
        """
        Show unsaved changes confirmation dialog.
        
        Returns:
            'save' - User chose to save
            'hide' - User chose to hide without saving
            'discard' - User chose to discard changes
            'cancel' - User cancelled
        """
        msg = QMessageBox(parent)
        msg.setWindowTitle(f"Close {panel_name}?")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(f"You have unsaved changes in the {panel_name}.")
        
        if change_summary:
            msg.setInformativeText(
                f"Changes made:\n{change_summary}\n\n"
                "What would you like to do?"
            )
        else:
            msg.setInformativeText("What would you like to do?")
        
        # Add buttons
        save_btn = None
        if save_callback:
            save_btn = msg.addButton("Save Changes", QMessageBox.ButtonRole.AcceptRole)
        hide_btn = msg.addButton("Hide Window", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg.addButton("Discard Changes", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        
        clicked = msg.clickedButton()
        
        if clicked == cancel_btn:
            return 'cancel'
        elif save_btn and clicked == save_btn:
            # Try to save
            try:
                if save_callback():
                    return 'hide'
                else:
                    # Save failed - ask user
                    retry = QMessageBox.question(
                        parent, "Save Failed",
                        "Failed to save changes.\n\nDo you want to close anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    return 'hide' if retry == QMessageBox.StandardButton.Yes else 'cancel'
            except Exception as e:
                retry = QMessageBox.question(
                    parent, "Save Failed",
                    f"Failed to save changes:\n{str(e)}\n\nDo you want to close anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                return 'hide' if retry == QMessageBox.StandardButton.Yes else 'cancel'
        elif clicked == hide_btn:
            return 'hide'
        elif clicked == discard_btn:
            return 'discard'
        
        return 'cancel'


# ============================================================================
# KNOWN DIALOG NAMES (for documentation and auto-save)
# ============================================================================
# These are the dialog names used throughout GeoX. The list is maintained
# here for reference and can be used for bulk operations.

KNOWN_DIALOGS = [
    'variogram_dialog',
    'kriging_dialog',
    'sgsim_dialog',
    'simple_kriging_dialog',
    'kmeans_dialog',
    'block_resource_dialog',
    'irr_dialog',
    'resource_classification_dialog',
    'statistics_dialog',
    'charts_dialog',
    'swath_dialog',
    'data_viewer_dialog',
    'domain_compositing_dialog',
    'compositing_window',
    'cross_section_dialog',
    'grade_transformation_dialog',
    'block_model_builder_dialog',
    'pit_optimisation_dialog',
    'universal_kriging_dialog',
    'cokriging_dialog',
    'indicator_kriging_dialog',
    'geotech_dialog',
    'variogram_assistant_dialog',
    'soft_kriging_dialog',
    'uncertainty_propagation_dialog',
    'research_dashboard_dialog',
    'loopstructural_dialog',
    'block_model_import_dialog',
    'qc_window',
    'drillhole_plotting_dialog',
    'preferences_dialog',
]

