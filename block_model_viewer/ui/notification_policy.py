"""
Notification Policy
====================

Defines when panels should use toasts, dialogs, or status bar messages.

Import and reference this module in code reviews and panel development
to ensure consistent notification behavior across the application.

Quick Reference:
    - Toast:      Non-blocking, auto-dismiss, informational
    - Dialog:     Blocking, requires user action, critical
    - Status bar: Passive, ephemeral, context updates

Usage:
    from .notification_policy import notify, NotifyLevel

    # Instead of QMessageBox.information():
    notify(self, NotifyLevel.SUCCESS, "Kriging complete", "12,500 blocks estimated")

    # Instead of QMessageBox.warning():
    notify(self, NotifyLevel.WARNING, "Low sample count", "Only 15 composites in domain")

    # Still use dialog for destructive/irreversible:
    notify(self, NotifyLevel.ERROR, "File corrupt", "Cannot recover data", blocking=True)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QMessageBox

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# NOTIFICATION LEVELS
# ═══════════════════════════════════════════════════════════════════

class NotifyLevel(Enum):
    """Notification severity levels."""
    INFO = auto()       # Neutral information
    SUCCESS = auto()    # Operation completed successfully
    WARNING = auto()    # Non-critical issue, operation continues
    ERROR = auto()      # Operation failed, needs attention


# ═══════════════════════════════════════════════════════════════════
# POLICY TABLE
# ═══════════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────┬──────────┬─────────────────────────┐
│ Scenario                            │ Channel  │ Example                 │
├─────────────────────────────────────┼──────────┼─────────────────────────┤
│ Analysis completed successfully     │ Toast    │ "Kriging done: 12.5k    │
│                                     │          │  blocks in 3.2s"        │
│ Data loaded                         │ Toast    │ "Loaded 450 composites" │
│ Export completed                    │ Toast    │ "Saved to results.csv"  │
│ Settings saved                      │ Toast    │ "Preferences updated"   │
│ Theme changed                       │ Toast    │ "Dark theme applied"    │
├─────────────────────────────────────┼──────────┼─────────────────────────┤
│ Non-critical warning                │ Toast    │ "15 blocks had NaN,     │
│                                     │ (yellow) │  excluded from stats"   │
│ Validation error (inline exists)    │ Inline   │ Red border + error text │
│ Low sample count                    │ Toast    │ "Only 15 composites in  │
│                                     │ (yellow) │  search neighborhood"   │
├─────────────────────────────────────┼──────────┼─────────────────────────┤
│ File not found / corrupt            │ Dialog   │ Modal error dialog      │
│ Destructive action confirmation     │ Dialog   │ "Delete 3 variograms?"  │
│ Unsaved changes on close            │ Dialog   │ "Save before closing?"  │
│ License/auth failure                │ Dialog   │ Modal error dialog      │
│ Unrecoverable crash                 │ Dialog   │ Modal with stack trace  │
├─────────────────────────────────────┼──────────┼─────────────────────────┤
│ Background progress                 │ Status   │ "Estimating... 45%"     │
│ Mouse coordinate update             │ Status   │ "X: 1234  Y: 5678"     │
│ Selection count                     │ Status   │ "3 blocks selected"     │
│ Memory usage                        │ Status   │ "RAM: 2.1 GB"          │
└─────────────────────────────────────┴──────────┴─────────────────────────┘

DECISION TREE:

    Does the user need to make a decision?
    ├── YES → Dialog (Save/Don't Save/Cancel, Delete/Keep, etc.)
    └── NO
        Is it an unrecoverable error?
        ├── YES → Dialog (with details + copy-to-clipboard)
        └── NO
            Is there an inline validation widget?
            ├── YES → Inline (red border + error text, no toast needed)
            └── NO
                Is it transient context info?
                ├── YES → Status bar (coordinates, memory, selection)
                └── NO → Toast (success, info, warning)

TOAST DURATION:
    - SUCCESS: 3 seconds
    - INFO:    4 seconds
    - WARNING: 6 seconds (longer so user can read)
    - ERROR:   Sticky (manual dismiss) — but prefer Dialog for errors

TOAST STACKING:
    - Max 3 visible toasts at once
    - New toast pushes oldest off if at capacity
    - Toasts stack from bottom-right, moving upward
"""


# ═══════════════════════════════════════════════════════════════════
# DURATION MAP
# ═══════════════════════════════════════════════════════════════════

TOAST_DURATION_MS = {
    NotifyLevel.SUCCESS: 3000,
    NotifyLevel.INFO: 4000,
    NotifyLevel.WARNING: 6000,
    NotifyLevel.ERROR: 0,  # Sticky (manual dismiss)
}


# ═══════════════════════════════════════════════════════════════════
# UNIFIED NOTIFY FUNCTION
# ═══════════════════════════════════════════════════════════════════

def notify(
    parent: QWidget,
    level: NotifyLevel,
    title: str,
    message: str = "",
    blocking: bool = False,
    duration_ms: Optional[int] = None,
    action_text: str = "",
    action_callback: Optional[Callable] = None,
) -> None:
    """
    Show a notification using the appropriate channel.

    This is the single entry point for all panel notifications.
    It routes to toast or dialog based on the policy.

    Args:
        parent: Parent widget (panel)
        level: Notification severity
        title: Short title (shown in bold for toasts, as dialog title)
        message: Detail message
        blocking: Force dialog instead of toast (for destructive/critical)
        duration_ms: Override auto-dismiss duration (None = use policy default)
        action_text: Optional action button text (e.g., "Undo", "Retry")
        action_callback: Callback for action button

    Usage:
        notify(self, NotifyLevel.SUCCESS, "Kriging complete", "12,500 blocks")
        notify(self, NotifyLevel.ERROR, "File corrupt", blocking=True)
    """
    if blocking or level == NotifyLevel.ERROR:
        # Use dialog for blocking/error notifications
        _show_dialog(parent, level, title, message)
    else:
        # Use toast for non-blocking notifications
        _show_toast(parent, level, title, message, duration_ms, action_text, action_callback)


def notify_confirm(
    parent: QWidget,
    title: str,
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
) -> bool:
    """
    Show a confirmation dialog and return True if confirmed.

    Always blocking. Use for destructive actions.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Question text
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button

    Returns:
        True if user clicked confirm, False otherwise

    Usage:
        if notify_confirm(self, "Delete?", "Delete 3 variograms?"):
            self.delete_variograms()
    """
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(message)
    box.setIcon(QMessageBox.Icon.Question)

    confirm_btn = box.addButton(confirm_text, QMessageBox.ButtonRole.AcceptRole)
    box.addButton(cancel_text, QMessageBox.ButtonRole.RejectRole)

    box.exec()
    return box.clickedButton() == confirm_btn


# ═══════════════════════════════════════════════════════════════════
# INTERNAL — Toast implementation
# ═══════════════════════════════════════════════════════════════════

def _show_toast(
    parent: QWidget,
    level: NotifyLevel,
    title: str,
    message: str,
    duration_ms: Optional[int],
    action_text: str,
    action_callback: Optional[Callable],
) -> None:
    """Route to toast system. Falls back to status bar if toast unavailable."""
    # Try to find the toast manager on the main window
    main_window = _find_main_window(parent)

    if main_window and hasattr(main_window, 'toast_manager'):
        # Use toast system
        duration = duration_ms or TOAST_DURATION_MS.get(level, 4000)
        toast_type = _level_to_toast_type(level)

        main_window.toast_manager.show_toast(
            title=title,
            message=message,
            toast_type=toast_type,
            duration=duration,
            action_text=action_text,
            action_callback=action_callback,
        )
    elif main_window and hasattr(main_window, 'statusBar'):
        # Fallback: status bar
        display = f"{title}: {message}" if message else title
        main_window.statusBar().showMessage(display, TOAST_DURATION_MS.get(level, 4000))
    else:
        # Last resort: log it
        log_func = {
            NotifyLevel.INFO: logger.info,
            NotifyLevel.SUCCESS: logger.info,
            NotifyLevel.WARNING: logger.warning,
            NotifyLevel.ERROR: logger.error,
        }.get(level, logger.info)
        log_func(f"[{level.name}] {title}: {message}")


def _show_dialog(
    parent: QWidget,
    level: NotifyLevel,
    title: str,
    message: str,
) -> None:
    """Show a blocking QMessageBox."""
    icon = {
        NotifyLevel.INFO: QMessageBox.Icon.Information,
        NotifyLevel.SUCCESS: QMessageBox.Icon.Information,
        NotifyLevel.WARNING: QMessageBox.Icon.Warning,
        NotifyLevel.ERROR: QMessageBox.Icon.Critical,
    }.get(level, QMessageBox.Icon.Information)

    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(message)
    box.setIcon(icon)
    box.exec()


def _find_main_window(widget: QWidget):
    """Walk up the parent chain to find MainWindow."""
    current = widget
    while current is not None:
        if current.__class__.__name__ == 'MainWindow':
            return current
        current = current.parent() if hasattr(current, 'parent') else None
    return None


def _level_to_toast_type(level: NotifyLevel) -> str:
    """Map NotifyLevel to toast type string."""
    return {
        NotifyLevel.INFO: "info",
        NotifyLevel.SUCCESS: "success",
        NotifyLevel.WARNING: "warning",
        NotifyLevel.ERROR: "error",
    }.get(level, "info")


# ═══════════════════════════════════════════════════════════════════
# MIGRATION HELPERS — for converting panels from QMessageBox
# ═══════════════════════════════════════════════════════════════════

"""
MIGRATION GUIDE — Replace QMessageBox calls in panels:

BEFORE:
    QMessageBox.information(self, "Success", "Kriging completed")
AFTER:
    notify(self, NotifyLevel.SUCCESS, "Kriging completed")

BEFORE:
    QMessageBox.warning(self, "Warning", "Low sample count: 15")
AFTER:
    notify(self, NotifyLevel.WARNING, "Low sample count", "15 composites in domain")

BEFORE:
    QMessageBox.critical(self, "Error", "File not found")
AFTER:
    notify(self, NotifyLevel.ERROR, "File not found", blocking=True)

BEFORE:
    reply = QMessageBox.question(self, "Delete?", "Delete 3 variograms?")
    if reply == QMessageBox.StandardButton.Yes:
        ...
AFTER:
    if notify_confirm(self, "Delete?", "Delete 3 variograms?"):
        ...

GREP PATTERN to find all QMessageBox calls:
    grep -rn "QMessageBox\\.\\(information\\|warning\\|critical\\|question\\)" ui/*.py
"""
