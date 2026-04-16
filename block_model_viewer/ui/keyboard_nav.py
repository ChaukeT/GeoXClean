"""
Keyboard Navigation Helpers
============================

Provides tab-order utilities, focus indicators, and keyboard navigation
support for panels. Replaces ad-hoc tab ordering with a consistent system.

Usage in BasePanel:
    from .keyboard_nav import KeyboardNavMixin

    class BasePanel(QWidget, KeyboardNavMixin):
        def setup_ui(self):
            ...
            # After all widgets are created:
            self.setup_tab_order([
                self.method_combo,
                self.cutoff_spin,
                self.power_spin,
                self.run_button,
            ])

Focus indicators are handled via QSS (see FOCUS_QSS at bottom of file).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QWidget,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# TAB ORDER UTILITY
# ═══════════════════════════════════════════════════════════════════

def setup_tab_order(widgets: Sequence[QWidget]) -> None:
    """
    Set tab order for a sequence of widgets.

    Chains setTabOrder() calls so pressing Tab moves focus through
    the list in order, and Shift+Tab moves backward.

    Args:
        widgets: Ordered list of focusable widgets. Non-focusable or
                 None entries are silently skipped.

    Example:
        setup_tab_order([
            self.variable_combo,
            self.method_combo,
            self.cutoff_spin,
            self.run_button,
        ])
    """
    focusable = [w for w in widgets if w is not None and _is_focusable(w)]

    for i in range(len(focusable) - 1):
        QWidget.setTabOrder(focusable[i], focusable[i + 1])

    if focusable:
        logger.debug(f"Tab order set for {len(focusable)} widgets")


def collect_focusable_children(parent: QWidget) -> List[QWidget]:
    """
    Recursively collect all focusable children in layout order.

    Useful for auto-generating tab order for a panel without
    manually listing every widget.

    Args:
        parent: Parent widget to scan.

    Returns:
        List of focusable child widgets in top-to-bottom, left-to-right order.
    """
    result = []
    _collect_recursive(parent, result)
    return result


def auto_tab_order(parent: QWidget) -> None:
    """
    Automatically set tab order for all focusable children of a widget.

    Convenience wrapper around collect_focusable_children + setup_tab_order.
    """
    widgets = collect_focusable_children(parent)
    setup_tab_order(widgets)


# ═══════════════════════════════════════════════════════════════════
# KEYBOARD NAVIGATION MIXIN
# ═══════════════════════════════════════════════════════════════════

class KeyboardNavMixin:
    """
    Mixin for BasePanel to add keyboard navigation support.

    Add to BasePanel's inheritance:
        class BasePanel(QWidget, KeyboardNavMixin):
            ...

    Then in setup_ui():
        self.setup_tab_order([widget1, widget2, ...])

    Or for automatic ordering:
        self.auto_tab_order()

    Provides:
        - setup_tab_order(): Explicit tab chain
        - auto_tab_order(): Automatic from layout
        - add_panel_shortcut(): Panel-local keyboard shortcuts
        - Escape key closes/hides panel
        - Enter/Return triggers run button (if present)
    """

    _panel_shortcuts: Optional[List[QShortcut]] = None

    def setup_tab_order(self, widgets: Sequence[QWidget]) -> None:
        """Set explicit tab order for this panel's widgets."""
        setup_tab_order(widgets)

    def auto_tab_order(self) -> None:
        """Automatically set tab order from layout structure."""
        auto_tab_order(self)

    def add_panel_shortcut(
        self,
        key_sequence: str,
        callback,
        context: Qt.ShortcutContext = Qt.ShortcutContext.WidgetWithChildrenShortcut,
    ) -> QShortcut:
        """
        Add a keyboard shortcut scoped to this panel.

        Args:
            key_sequence: Key combo string, e.g. "Ctrl+R", "F5"
            callback: Function to call when shortcut is triggered
            context: Qt shortcut context (default: widget + children)

        Returns:
            The created QShortcut (for later removal if needed)

        Example:
            self.add_panel_shortcut("Ctrl+R", self.on_run_clicked)
            self.add_panel_shortcut("F5", self.refresh)
        """
        if self._panel_shortcuts is None:
            self._panel_shortcuts = []

        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.setContext(context)
        shortcut.activated.connect(callback)
        self._panel_shortcuts.append(shortcut)
        return shortcut

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle panel-level key events.

        - Escape: Hide/close panel
        - Enter/Return: Trigger run button if present
        """
        key = event.key()

        # Escape → close/hide panel
        if key == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return

        # Enter/Return → trigger run button (if not in a text edit)
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            focused = self.focusWidget() if hasattr(self, 'focusWidget') else None
            # Don't steal Enter from text edits or combos
            if not isinstance(focused, (QTextEdit, QComboBox)):
                run_btn = self._find_run_button()
                if run_btn and run_btn.isEnabled():
                    run_btn.click()
                    event.accept()
                    return

        # Pass to parent
        super().keyPressEvent(event)

    def _find_run_button(self) -> Optional[QPushButton]:
        """Find the Run/Execute button in this panel."""
        # Check common attribute names
        for attr in ('run_button', 'run_btn', 'execute_btn', 'analyze_btn'):
            btn = getattr(self, attr, None)
            if isinstance(btn, QPushButton):
                return btn

        # Search by text
        for btn in self.findChildren(QPushButton):
            text = btn.text().lower()
            if text in ('run', 'execute', 'analyze', 'compute', 'calculate'):
                return btn

        return None


# ═══════════════════════════════════════════════════════════════════
# FOCUS GROUP — logical grouping of related widgets
# ═══════════════════════════════════════════════════════════════════

class FocusGroup:
    """
    Groups related widgets for logical focus navigation.

    When a group is "entered" (any widget gains focus), Tab stays within
    the group until the user presses Escape or Tab out of the last widget.

    Useful for form sections (e.g., "Variogram Parameters" group where
    Tab cycles through nugget/sill/range before moving to next section).

    Usage:
        group = FocusGroup([self.nugget_spin, self.sill_spin, self.range_spin])
        group.install()  # Wraps tab order within group
    """

    def __init__(self, widgets: Sequence[QWidget]):
        self._widgets = [w for w in widgets if w is not None and _is_focusable(w)]

    def install(self) -> None:
        """Set tab order within this group."""
        setup_tab_order(self._widgets)

    @property
    def first(self) -> Optional[QWidget]:
        return self._widgets[0] if self._widgets else None

    @property
    def last(self) -> Optional[QWidget]:
        return self._widgets[-1] if self._widgets else None

    def focus_first(self) -> None:
        """Set focus to the first widget in this group."""
        if self.first:
            self.first.setFocus(Qt.FocusReason.TabFocusReason)


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

_FOCUSABLE_TYPES = (
    QLineEdit,
    QAbstractSpinBox,
    QComboBox,
    QPushButton,
    QTextEdit,
    QTabWidget,
)


def _is_focusable(widget: QWidget) -> bool:
    """Check if a widget can receive keyboard focus."""
    if not widget.isEnabled() or not widget.isVisible():
        return False
    if isinstance(widget, _FOCUSABLE_TYPES):
        return True
    policy = widget.focusPolicy()
    return policy in (
        Qt.FocusPolicy.TabFocus,
        Qt.FocusPolicy.StrongFocus,
        Qt.FocusPolicy.WheelFocus,
    )


def _collect_recursive(widget: QWidget, result: List[QWidget]) -> None:
    """Recursively collect focusable widgets in layout order."""
    layout = widget.layout()
    if layout is None:
        if _is_focusable(widget):
            result.append(widget)
        return

    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue

        child_widget = item.widget()
        if child_widget is not None:
            if _is_focusable(child_widget):
                result.append(child_widget)
            else:
                # Recurse into containers (QGroupBox, QFrame, etc.)
                _collect_recursive(child_widget, result)

        child_layout = item.layout()
        if child_layout is not None:
            # Create temporary container to recurse
            for j in range(child_layout.count()):
                sub_item = child_layout.itemAt(j)
                if sub_item and sub_item.widget():
                    sub_w = sub_item.widget()
                    if _is_focusable(sub_w):
                        result.append(sub_w)
                    else:
                        _collect_recursive(sub_w, result)


# ═══════════════════════════════════════════════════════════════════
# FOCUS QSS — append to app-level stylesheet
# ═══════════════════════════════════════════════════════════════════

def get_focus_qss() -> str:
    """Return focus indicator QSS using current theme tokens."""
    from .modern_styles import ModernColors
    accent = ModernColors.ACCENT_PRIMARY
    accent_dark = ModernColors.ACCENT_HOVER
    return f"""
/* ── Focus Indicators ──────────────────────────────────────────── */
/* Visible focus ring on all interactive widgets */

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus,
QTextEdit:focus,
QPlainTextEdit:focus {{
    border: 2px solid {accent};
    outline: none;
}}

QPushButton:focus {{
    border: 2px solid {accent};
    outline: none;
}}

/* Tab widget focus indicator */
QTabBar::tab:selected:focus {{
    border-bottom: 3px solid {accent};
}}

/* Checkbox/Radio focus */
QCheckBox:focus,
QRadioButton:focus {{
    outline: 2px solid {accent};
    outline-offset: 2px;
}}

/* Slider focus */
QSlider:focus {{
    border: 1px solid {accent};
    border-radius: 2px;
}}

/* Table focus */
QTableWidget:focus,
QTreeWidget:focus,
QListWidget:focus {{
    border: 2px solid {accent};
}}

/* Skip link style for keyboard users (optional, add to top of panel) */
QLabel#skipLink {{
    color: {accent};
    text-decoration: underline;
    padding: 4px;
}}
QLabel#skipLink:hover {{
    color: {accent_dark};
}}
"""

# Backward compatibility alias
FOCUS_QSS = get_focus_qss()
