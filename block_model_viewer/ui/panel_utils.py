"""
Panel Utilities - Common functionality for all panels.

Provides:
- Close protection (confirmation dialog)
- Taskbar persistence (stay in taskbar when minimized)
- Window state management
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)


def setup_panel_window_flags(widget: QWidget):
    """
    Setup window flags for a panel to behave like a software panel.
    
    - Stays in taskbar when minimized
    - Can be minimized/maximized
    - Non-modal behavior
    - Doesn't get deleted when closed
    
    Args:
        widget: QWidget, QDialog, or QMainWindow to configure
    """
    # Set window flags for proper minimize behavior (stay in taskbar)
    widget.setWindowFlags(
        Qt.WindowType.Window |
        Qt.WindowType.WindowMinimizeButtonHint |
        Qt.WindowType.WindowMaximizeButtonHint |
        Qt.WindowType.WindowCloseButtonHint
    )
    
    # Ensure non-modal behavior
    widget.setWindowModality(Qt.WindowModality.NonModal)
    
    # Prevent window from being deleted when closed or minimized
    widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)


def add_close_protection(
    widget: QWidget,
    has_unsaved_changes: Optional[Callable[[], bool]] = None,
    get_change_summary: Optional[Callable[[], str]] = None,
    save_callback: Optional[Callable[[], bool]] = None,
    panel_name: str = "Panel"
):
    """
    Add close protection to a panel widget.
    
    Shows a confirmation dialog if there are unsaved changes when the user
    tries to close the window.
    
    Args:
        widget: The widget to protect
        has_unsaved_changes: Optional function that returns True if there are unsaved changes
        get_change_summary: Optional function that returns a summary of changes
        save_callback: Optional function to save changes (returns True on success)
        panel_name: Name of the panel for dialog messages
    """
    original_close = getattr(widget, 'closeEvent', None)
    
    def protected_close_event(event):
        """Protected close event that checks for unsaved changes."""
        # Check if there are unsaved changes
        has_changes = False
        change_summary = ""
        
        if has_unsaved_changes:
            try:
                has_changes = has_unsaved_changes()
            except Exception as e:
                logger.warning(f"Error checking for unsaved changes: {e}")
                has_changes = False
        
        if get_change_summary:
            try:
                change_summary = get_change_summary()
            except Exception as e:
                logger.warning(f"Error getting change summary: {e}")
                change_summary = ""
        
        if has_changes:
            # Show confirmation dialog
            msg = QMessageBox(widget)
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
            if save_callback:
                save_btn = msg.addButton("Save Changes", QMessageBox.ButtonRole.AcceptRole)
            hide_btn = msg.addButton("Hide Window", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = msg.addButton("Discard Changes", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.setDefaultButton(cancel_btn)
            msg.exec()
            
            clicked = msg.clickedButton()
            
            if clicked == cancel_btn:
                # Cancel close
                event.ignore()
                return
            elif save_callback and clicked == save_btn:
                # Save changes first
                try:
                    if save_callback():
                        # After saving, hide the window (don't destroy it)
                        widget.hide()
                        event.ignore()
                    else:
                        # Save failed - ask again
                        retry = QMessageBox.question(
                            widget, "Save Failed",
                            f"Failed to save changes.\n\n"
                            "Do you want to close anyway?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if retry == QMessageBox.StandardButton.Yes:
                            widget.hide()
                            event.ignore()
                        else:
                            event.ignore()
                except Exception as e:
                    # If save fails, ask again
                    retry = QMessageBox.question(
                        widget, "Save Failed",
                        f"Failed to save changes:\n{str(e)}\n\n"
                        "Do you want to close anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if retry == QMessageBox.StandardButton.Yes:
                        widget.hide()
                        event.ignore()
                    else:
                        event.ignore()
                return
            elif clicked == hide_btn:
                # Hide window but keep state
                widget.hide()
                event.ignore()
                return
            elif clicked == discard_btn:
                # User confirmed they want to discard changes
                # Hide window (state is preserved in memory)
                widget.hide()
                event.ignore()
                return
        else:
            # No changes - just hide the window
            widget.hide()
            event.ignore()
    
    # Override closeEvent
    widget.closeEvent = protected_close_event

