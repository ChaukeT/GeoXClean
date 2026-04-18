"""
Panel Toolkit — Shared UI Primitives
=====================================

All docked panels import from here instead of building UI inline.
This guarantees visual consistency: same margins, spacing, fonts,
colors, section headers, buttons, sliders, and form rows everywhere.

Usage:
    from .panel_toolkit import (
        PANEL_MARGINS, PANEL_SPACING,
        section, form_row, slider_row, action_button,
        info_display, separator, PANEL_QSS,
    )

Standards enforced:
    - Margins:  12px uniform
    - Spacing:  12px between sections, 8px within
    - Sections: CollapsibleGroup, plain text (no emoji)
    - Colors:   ModernColors only (from modern_styles.py)
    - Combos:   28px min-height
    - Buttons:  32px primary, 28px secondary
    - Labels:   11px TEXT_SECONDARY for form labels
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable, List

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QCheckBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .collapsible_group import CollapsibleGroup
from .modern_styles import ModernColors

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS — the single source of truth
# ═══════════════════════════════════════════════════════════════════
#
# Redesigned for professional mining-software aesthetics (Leapfrog /
# Micromine reference).  Previous values caused a "squeezed" look.
# Increased across the board for breathing room and legibility.

PANEL_MARGINS = (12, 12, 12, 12)          # Pictures tight spacing
PANEL_SPACING = 12                         # between sections
SECTION_SPACING = 8                        # within a section
FORM_V_SPACING = 8                         # form vertical
FORM_H_SPACING = 12                        # form horizontal
COMBO_MIN_HEIGHT = 28                      # combobox height
BUTTON_HEIGHT_PRIMARY = 32                 # primary button
BUTTON_HEIGHT_SECONDARY = 28               # secondary button
LABEL_FONT_SIZE = "11px"                   # form labels
PANEL_MIN_WIDTH = 280
PANEL_MAX_WIDTH = 380


# ═══════════════════════════════════════════════════════════════════
# SECTION BUILDER
# ═══════════════════════════════════════════════════════════════════

def section(
    title: str,
    collapsed: bool = False,
) -> CollapsibleGroup:
    """
    Create a standard collapsible section.

    Args:
        title: Section title (plain text, NO emoji).
        collapsed: Initial collapsed state.

    Returns:
        CollapsibleGroup ready for add_layout().
    """
    return CollapsibleGroup(title, collapsed=collapsed)


# ═══════════════════════════════════════════════════════════════════
# FORM HELPERS
# ═══════════════════════════════════════════════════════════════════

def make_form() -> QFormLayout:
    """Create a standard form layout with consistent spacing."""
    form = QFormLayout()
    form.setVerticalSpacing(FORM_V_SPACING)
    form.setHorizontalSpacing(FORM_H_SPACING)
    return form


def form_label(text: str, tooltip: str = "") -> QLabel:
    """Create a standard form field label."""
    lbl = QLabel(text)
    lbl.setObjectName("formLabel")
    if tooltip:
        lbl.setToolTip(tooltip)
    return lbl


def form_row(
    form: QFormLayout,
    label_text: str,
    widget: QWidget,
    tooltip: str = "",
) -> None:
    """
    Add a labeled row to a form layout with consistent styling.

    Args:
        form: Target QFormLayout.
        label_text: Label text (e.g., "Active Layer:").
        widget: The input widget (combo, spin, etc.).
        tooltip: Tooltip for both label and widget.
    """
    lbl = form_label(label_text, tooltip)
    if tooltip:
        widget.setToolTip(tooltip)
    form.addRow(lbl, widget)


def make_combo(
    items: Optional[List[str]] = None,
    tooltip: str = "",
) -> QComboBox:
    """Create a standard combobox with consistent height."""
    combo = QComboBox()
    combo.setMinimumHeight(COMBO_MIN_HEIGHT)
    combo.setObjectName("panelCombo")
    if items:
        combo.addItems(items)
    if tooltip:
        combo.setToolTip(tooltip)
    return combo


def make_spin(
    min_val: float = 0.0,
    max_val: float = 100.0,
    value: float = 0.0,
    decimals: int = 2,
    suffix: str = "",
    tooltip: str = "",
) -> QDoubleSpinBox:
    """Create a standard double spin box."""
    spin = QDoubleSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(value)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    spin.setMinimumHeight(COMBO_MIN_HEIGHT)
    spin.setObjectName("panelSpin")
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


def make_int_spin(
    min_val: int = 0,
    max_val: int = 100,
    value: int = 0,
    suffix: str = "",
    tooltip: str = "",
) -> QSpinBox:
    """Create a standard integer spin box."""
    spin = QSpinBox()
    spin.setRange(min_val, max_val)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    spin.setMinimumHeight(COMBO_MIN_HEIGHT)
    spin.setObjectName("panelSpin")
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


# ═══════════════════════════════════════════════════════════════════
# SLIDER ROW
# ═══════════════════════════════════════════════════════════════════

class SliderRow(QWidget):
    """
    Standard slider row: [Label] [====o====] [Value].

    Signals:
        valueChanged(float): Emitted when slider value changes.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str = "",
        min_val: float = 0.0,
        max_val: float = 1.0,
        value: float = 0.5,
        suffix: str = "",
        decimals: int = 2,
        label_width: int = 60,
        debounce_ms: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._suffix = suffix
        self._steps = max(1, int((max_val - min_val) * (10 ** decimals)))

        # Debounce support
        self._debounce_ms = debounce_ms
        self._pending_value: Optional[float] = None
        if debounce_ms > 0:
            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.setInterval(debounce_ms)
            self._debounce_timer.timeout.connect(self._emit_debounced)
        else:
            self._debounce_timer = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(FORM_H_SPACING)

        # Label
        if label:
            self._label = QLabel(label)
            self._label.setObjectName("formLabel")
            self._label.setFixedWidth(label_width)
            layout.addWidget(self._label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setValue(self._float_to_int(value))
        self._slider.setMinimumHeight(24)
        self._slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self._slider, stretch=1)

        # Value display
        self._value_label = QLabel()
        self._value_label.setObjectName("sliderValue")
        self._value_label.setFixedWidth(50)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._update_text(value)
        layout.addWidget(self._value_label)

    def _float_to_int(self, val: float) -> int:
        ratio = (val - self._min) / max(1e-9, self._max - self._min)
        return int(ratio * self._steps)

    def _int_to_float(self, ival: int) -> float:
        return self._min + (ival / max(1, self._steps)) * (self._max - self._min)

    def _update_text(self, val: float) -> None:
        self._value_label.setText(f"{val:.{self._decimals}f}{self._suffix}")

    def _on_changed(self, ival: int) -> None:
        val = self._int_to_float(ival)
        self._update_text(val)
        if self._debounce_timer is not None:
            self._pending_value = val
            self._debounce_timer.start()
        else:
            self.valueChanged.emit(val)

    def _emit_debounced(self) -> None:
        if self._pending_value is not None:
            self.valueChanged.emit(self._pending_value)
            self._pending_value = None

    def value(self) -> float:
        return self._int_to_float(self._slider.value())

    def setValue(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._float_to_int(val))
        self._update_text(val)
        self._slider.blockSignals(False)

    def setRange(self, min_val: float, max_val: float) -> None:
        """Update the slider range at runtime."""
        self._min = min_val
        self._max = max_val
        self._steps = max(1, int((max_val - min_val) * (10 ** self._decimals)))
        cur = self.value()
        self._slider.blockSignals(True)
        self._slider.setRange(0, self._steps)
        clamped = max(min_val, min(max_val, cur))
        self._slider.setValue(self._float_to_int(clamped))
        self._update_text(clamped)
        self._slider.blockSignals(False)


# ═══════════════════════════════════════════════════════════════════
# BUTTONS
# ═══════════════════════════════════════════════════════════════════

def action_button(
    text: str,
    style: str = "secondary",
    tooltip: str = "",
    checkable: bool = False,
) -> QPushButton:
    """
    Create a standard button.

    Args:
        text: Button text (plain text, no emoji).
        style: "primary" | "secondary" | "toggle" | "danger".
        tooltip: Tooltip text.
        checkable: Whether the button is toggle-able.

    Returns:
        QPushButton with objectName set for QSS targeting.
    """
    btn = QPushButton(text)
    btn.setObjectName(f"btn_{style}")
    if style == "primary":
        btn.setMinimumHeight(BUTTON_HEIGHT_PRIMARY)
    else:
        btn.setMinimumHeight(BUTTON_HEIGHT_SECONDARY)
    if tooltip:
        btn.setToolTip(tooltip)
    if checkable:
        btn.setCheckable(True)
    return btn


def button_row(*buttons: QPushButton, spacing: int = 8) -> QHBoxLayout:
    """Create a horizontal row of buttons."""
    row = QHBoxLayout()
    row.setSpacing(spacing)
    for btn in buttons:
        row.addWidget(btn)
    return row


# ═══════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════

def info_display(text: str = "") -> QLabel:
    """Create a monospace info display label (for coordinates, stats, etc.)."""
    lbl = QLabel(text)
    lbl.setObjectName("infoDisplay")
    lbl.setWordWrap(True)
    return lbl


def separator() -> QFrame:
    """Create a horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setObjectName("panelSeparator")
    return line


def hint_label(text: str) -> QLabel:
    """Create a hint/description label."""
    lbl = QLabel(text)
    lbl.setObjectName("hintLabel")
    lbl.setWordWrap(True)
    return lbl


def status_label(text: str = "Ready") -> QLabel:
    """Create a status display label."""
    lbl = QLabel(text)
    lbl.setObjectName("statusLabel")
    return lbl


# ═══════════════════════════════════════════════════════════════════
# SCROLL AREA SETUP
# ═══════════════════════════════════════════════════════════════════

def scrollable_panel_layout(parent: QWidget) -> Tuple[QVBoxLayout, QVBoxLayout]:
    """
    Create a scrollable panel layout.

    Returns:
        (root_layout, content_layout)
        - root_layout: Set on the parent widget, contains the scroll area.
        - content_layout: Inside the scroll area, add your sections here.
    """
    root = QVBoxLayout(parent)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    content_widget = QWidget()
    content_widget.setObjectName("panelContent")
    content = QVBoxLayout(content_widget)
    content.setContentsMargins(*PANEL_MARGINS)
    content.setSpacing(PANEL_SPACING)

    scroll.setWidget(content_widget)
    root.addWidget(scroll)

    return root, content


# ═══════════════════════════════════════════════════════════════════
# PANEL QSS — Append to global stylesheet
# ═══════════════════════════════════════════════════════════════════

PANEL_QSS = f"""
/* ── Panel Toolkit – Professional Mining Software Panel Styling ── */
/* Redesigned for spacious, breathable layout (Leapfrog/Micromine   */
/* reference).  Increased padding, taller controls, clearer labels. */

/* ── Form labels: stacked above controls, clear hierarchy ────── */
QLabel#formLabel {{
    color: {ModernColors.TEXT_SECONDARY};
    font-size: {LABEL_FONT_SIZE};
    font-weight: 500;
    padding-bottom: 2px;
}}

/* ── Hint / caption labels ──────────────────────────────────── */
QLabel#hintLabel {{
    color: {ModernColors.TEXT_HINT};
    font-size: 11px;
    font-style: italic;
    padding: 4px 0;
}}

/* ── Status labels ───────────────────────────────────────────── */
QLabel#statusLabel {{
    color: {ModernColors.TEXT_SECONDARY};
    font-size: 12px;
    padding: 8px 0;
}}

/* ── Info / monospace readouts ──────────────────────────────── */
QLabel#infoDisplay {{
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    color: {ModernColors.TEXT_SECONDARY};
    background-color: {ModernColors.ELEVATED_BG};
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid {ModernColors.DIVIDER};
    line-height: 1.5;
}}

/* ── Slider value badges ────────────────────────────────────── */
QLabel#sliderValue {{
    color: {ModernColors.ACCENT_PRIMARY};
    font-weight: 700;
    font-size: 12px;
    padding: 4px 12px;
    background-color: {ModernColors.ELEVATED_BG};
    border-radius: 6px;
    border: 1px solid {ModernColors.DIVIDER};
    min-width: 52px;
}}

/* ── Separator line ─────────────────────────────────────────── */
QFrame#panelSeparator {{
    background-color: {ModernColors.DIVIDER};
    border: none;
}}

/* ── Standard combos ────────────────────────────────────────── */
QComboBox#panelCombo {{
    min-height: {COMBO_MIN_HEIGHT}px;
    padding: 0 10px;
    font-size: 12px;
}}

/* ── Primary button — bold accent fill ──────────────────────── */
QPushButton#btn_primary {{
    background-color: {ModernColors.ACCENT_PRIMARY};
    color: #ffffff;
    font-weight: 600;
    font-size: 13px;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    min-height: {BUTTON_HEIGHT_PRIMARY}px;
    letter-spacing: 0.2px;
}}
QPushButton#btn_primary:hover {{
    background-color: {ModernColors.ACCENT_HOVER};
}}
QPushButton#btn_primary:pressed {{
    background-color: {ModernColors.ACCENT_PRESSED};
}}
QPushButton#btn_primary:disabled {{
    background-color: {ModernColors.BORDER};
    color: {ModernColors.TEXT_DISABLED};
}}

/* ── Secondary button — outline style ───────────────────────── */
QPushButton#btn_secondary {{
    background-color: {ModernColors.ELEVATED_BG};
    color: {ModernColors.TEXT_PRIMARY};
    border: 1px solid {ModernColors.BORDER};
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 12px;
    min-height: {BUTTON_HEIGHT_SECONDARY}px;
}}
QPushButton#btn_secondary:hover {{
    border-color: {ModernColors.ACCENT_PRIMARY};
    color: {ModernColors.ACCENT_PRIMARY};
    background-color: {ModernColors.CARD_BG};
}}
QPushButton#btn_secondary:disabled {{
    color: {ModernColors.TEXT_DISABLED};
    border-color: {ModernColors.DIVIDER};
}}

/* ── Toggle button — accent when checked ───────────────────── */
QPushButton#btn_toggle {{
    background-color: {ModernColors.ELEVATED_BG};
    color: {ModernColors.TEXT_PRIMARY};
    border: 1px solid {ModernColors.BORDER};
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 12px;
    min-height: {BUTTON_HEIGHT_SECONDARY}px;
}}
QPushButton#btn_toggle:checked {{
    background-color: {ModernColors.ACCENT_PRIMARY};
    color: #ffffff;
    border-color: {ModernColors.ACCENT_PRIMARY};
}}

/* ── Danger button ──────────────────────────────────────────── */
QPushButton#btn_danger {{
    background-color: {ModernColors.ERROR};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 12px;
    min-height: {BUTTON_HEIGHT_SECONDARY}px;
}}
QPushButton#btn_danger:hover {{
    background-color: #f87171;
}}

/* ── Panel content area ─────────────────────────────────────── */
QWidget#panelContent {{
    background-color: transparent;
}}

/* ── Checkboxes: more vertical spacing, bigger indicator ─────── */
QCheckBox {{
    spacing: 10px;
    font-size: 12px;
    min-height: 26px;
    padding: 2px 0;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
}}

/* ── Sliders: taller groove, larger handle ───────────────────── */
QSlider::groove:horizontal {{
    background: {ModernColors.ELEVATED_BG};
    height: 6px;
    border-radius: 3px;
    border: 1px solid {ModernColors.DIVIDER};
}}
QSlider::handle:horizontal {{
    background: {ModernColors.ACCENT_PRIMARY};
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
    border: 2px solid {ModernColors.CARD_BG};
}}
QSlider::handle:horizontal:hover {{
    background: {ModernColors.ACCENT_HOVER};
}}
QSlider::sub-page:horizontal {{
    background: {ModernColors.ACCENT_PRIMARY};
    border-radius: 3px;
}}
"""
