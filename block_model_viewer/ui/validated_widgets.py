"""
Inline Validation System — replaces QMessageBox.warning() for input validation.

Provides validated widget wrappers that show red borders + inline error text,
real-time validation as user types/selects, a summary widget, and integration
hooks for BaseAnalysisPanel.

Usage in a panel:
    from .validated_widgets import ValidatedSpinBox, ValidatedComboBox, ValidationGroup

    class MyPanel(BaseAnalysisPanel):
        def setup_ui(self):
            self.validation = ValidationGroup()

            self.cutoff = ValidatedSpinBox(
                label="Cutoff Grade",
                validator=lambda v: (v > 0, "Must be positive") if v <= 0 else (True, ""),
            )
            self.validation.add(self.cutoff)

            self.method = ValidatedComboBox(
                label="Method",
                validator=lambda v: (bool(v), "Select a method") if not v else (True, ""),
            )
            self.validation.add(self.method)

            # Summary at top of panel
            self.validation_summary = self.validation.summary_widget()
            layout.insertWidget(0, self.validation_summary)

            # Run button auto-disabled when invalid
            self.validation.bind_run_button(self.run_button)

Replaces the old pattern:
    def validate_inputs(self):
        if self.cutoff.value() <= 0:
            QMessageBox.warning(self, "Error", "Cutoff must be positive")
            return False
        ...
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from .modern_styles import ModernColors
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Type alias for validator functions
# ═══════════════════════════════════════════════════════════════════
# Validator returns (is_valid: bool, error_message: str)
ValidatorFunc = Callable[[Any], Tuple[bool, str]]


# ═══════════════════════════════════════════════════════════════════
# ERROR LABEL (shared by all validated widgets)
# ═══════════════════════════════════════════════════════════════════

class _InlineErrorLabel(QLabel):
    """Small red error label shown below a widget."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("validationError")
        self.setWordWrap(True)
        self._apply_style()
        self.hide()

    def _apply_style(self):
        self.setStyleSheet(
            f"QLabel#validationError {{"
            f"  color: {ModernColors.ERROR};"
            f"  font-size: 11px;"
            f"  padding: 2px 0 0 2px;"
            f"  margin: 0;"
            f"}}"
        )

    def show_error(self, message: str):
        self.setText(message)
        self.show()

    def clear_error(self):
        self.setText("")
        self.hide()


# ═══════════════════════════════════════════════════════════════════
# VALIDATED WRAPPER — base for all validated widgets
# ═══════════════════════════════════════════════════════════════════

class _ValidatedWrapper(QWidget):
    """
    Base wrapper that adds validation display to any input widget.

    Subclasses set self._inner_widget and call _connect_change_signal().
    """

    # Emitted when validation state changes: (field_name, is_valid, error_msg)
    validation_changed = pyqtSignal(str, bool, str)

    @staticmethod
    def _border_normal():
        return f"1px solid {ModernColors.BORDER}"

    @staticmethod
    def _border_error():
        return f"2px solid {ModernColors.ERROR}"

    @staticmethod
    def _bg_error():
        return ModernColors.ERROR + "18"  # 10% opacity tint of error color

    def __init__(
        self,
        label: str = "",
        validator: Optional[ValidatorFunc] = None,
        required: bool = False,
        field_name: str = "",
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self._label_text = label
        self._validator = validator
        self._required = required
        self._field_name = field_name or label
        self._is_valid = True
        self._error_message = ""

        self._inner_widget: Optional[QWidget] = None  # set by subclass
        self._label_widget: Optional[QLabel] = None
        self._error_label = _InlineErrorLabel()

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        if label:
            self._label_widget = QLabel(label)
            self._label_widget.setObjectName("validationFieldLabel")
            self._layout.addWidget(self._label_widget)

    def _finish_setup(self):
        """Called by subclass after self._inner_widget is set."""
        if self._inner_widget:
            self._layout.addWidget(self._inner_widget)
        self._layout.addWidget(self._error_label)
        self._connect_change_signal()

    def _connect_change_signal(self):
        """Override to connect the inner widget's change signal to validate()."""
        pass

    # ── Public API ────────────────────────────────────────────────

    def validate(self) -> bool:
        """Run validation and update display. Returns True if valid."""
        value = self.value()

        if self._required and self._is_empty(value):
            self._set_error(f"{self._field_name} is required")
            return False

        if self._validator:
            is_valid, msg = self._validator(value)
            if not is_valid:
                self._set_error(msg)
                return False

        self._clear_error()
        return True

    def value(self) -> Any:
        """Get current value. Override in subclass."""
        return None

    def is_valid(self) -> bool:
        return self._is_valid

    def error_message(self) -> str:
        return self._error_message

    def field_name(self) -> str:
        return self._field_name

    def set_validator(self, validator: ValidatorFunc):
        self._validator = validator

    def set_required(self, required: bool):
        self._required = required

    # ── Internal ──────────────────────────────────────────────────

    def _is_empty(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    def _set_error(self, message: str):
        self._is_valid = False
        self._error_message = message
        self._error_label.show_error(message)
        if self._inner_widget:
            self._inner_widget.setStyleSheet(
                f"border: {self._border_error()}; background-color: {self._bg_error()};"
            )
        self.validation_changed.emit(self._field_name, False, message)

    def _clear_error(self):
        was_invalid = not self._is_valid
        self._is_valid = True
        self._error_message = ""
        self._error_label.clear_error()
        if self._inner_widget:
            self._inner_widget.setStyleSheet("")
        if was_invalid:
            self.validation_changed.emit(self._field_name, True, "")


# ═══════════════════════════════════════════════════════════════════
# CONCRETE VALIDATED WIDGETS
# ═══════════════════════════════════════════════════════════════════

class ValidatedLineEdit(_ValidatedWrapper):
    """QLineEdit with inline validation."""

    def __init__(
        self,
        label: str = "",
        placeholder: str = "",
        validator: Optional[ValidatorFunc] = None,
        required: bool = False,
        field_name: str = "",
        parent: QWidget = None,
    ):
        super().__init__(label, validator, required, field_name, parent)
        self._inner_widget = QLineEdit()
        self._inner_widget.setObjectName("validatedLineEdit")
        if placeholder:
            self._inner_widget.setPlaceholderText(placeholder)
        self._finish_setup()

    def _connect_change_signal(self):
        self._inner_widget.textChanged.connect(lambda _: self.validate())

    def value(self) -> str:
        return self._inner_widget.text()

    def setText(self, text: str):
        self._inner_widget.setText(text)

    @property
    def line_edit(self) -> QLineEdit:
        return self._inner_widget


class ValidatedSpinBox(_ValidatedWrapper):
    """QSpinBox or QDoubleSpinBox with inline validation."""

    def __init__(
        self,
        label: str = "",
        validator: Optional[ValidatorFunc] = None,
        required: bool = False,
        field_name: str = "",
        minimum: float = 0,
        maximum: float = 999999,
        decimals: int = 0,
        suffix: str = "",
        parent: QWidget = None,
    ):
        super().__init__(label, validator, required, field_name, parent)

        if decimals > 0:
            self._inner_widget = QDoubleSpinBox()
            self._inner_widget.setDecimals(decimals)
        else:
            self._inner_widget = QSpinBox()

        self._inner_widget.setObjectName("validatedSpinBox")
        self._inner_widget.setMinimum(int(minimum) if decimals == 0 else minimum)
        self._inner_widget.setMaximum(int(maximum) if decimals == 0 else maximum)
        if suffix:
            self._inner_widget.setSuffix(f" {suffix}")
        self._finish_setup()

    def _connect_change_signal(self):
        self._inner_widget.valueChanged.connect(lambda _: self.validate())

    def value(self) -> float:
        return self._inner_widget.value()

    def setValue(self, val):
        self._inner_widget.setValue(val)

    @property
    def spin_box(self) -> QAbstractSpinBox:
        return self._inner_widget


class ValidatedComboBox(_ValidatedWrapper):
    """QComboBox with inline validation."""

    def __init__(
        self,
        label: str = "",
        items: Optional[List[str]] = None,
        validator: Optional[ValidatorFunc] = None,
        required: bool = False,
        field_name: str = "",
        parent: QWidget = None,
    ):
        super().__init__(label, validator, required, field_name, parent)
        self._inner_widget = QComboBox()
        self._inner_widget.setObjectName("validatedComboBox")
        if items:
            self._inner_widget.addItems(items)
        self._finish_setup()

    def _connect_change_signal(self):
        self._inner_widget.currentTextChanged.connect(lambda _: self.validate())

    def value(self) -> str:
        return self._inner_widget.currentText()

    def currentText(self) -> str:
        return self._inner_widget.currentText()

    def currentIndex(self) -> int:
        return self._inner_widget.currentIndex()

    def addItems(self, items: List[str]):
        self._inner_widget.addItems(items)

    def clear(self):
        self._inner_widget.clear()

    @property
    def combo_box(self) -> QComboBox:
        return self._inner_widget


# ═══════════════════════════════════════════════════════════════════
# VALIDATION SUMMARY WIDGET
# ═══════════════════════════════════════════════════════════════════

class ValidationSummary(QFrame):
    """
    Shows a compact summary of all current validation errors.

    Displayed at the top of a panel. Hidden when all fields are valid.
    Turns into a green "All inputs valid" bar briefly on successful validation.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("validationSummary")
        self._errors: Dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._icon_label = QLabel()
        self._title_label = QLabel()
        self._title_label.setObjectName("validationSummaryTitle")

        header = QHBoxLayout()
        header.addWidget(self._icon_label)
        header.addWidget(self._title_label, 1)
        layout.addLayout(header)

        self._detail_label = QLabel()
        self._detail_label.setObjectName("validationSummaryDetail")
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

        self.hide()
        self._apply_error_style()

    def update_errors(self, errors: Dict[str, str]):
        """Update displayed errors. errors = {field_name: error_message}."""
        self._errors = {k: v for k, v in errors.items() if v}

        if not self._errors:
            self.hide()
            return

        count = len(self._errors)
        self._icon_label.setText("⚠")
        self._title_label.setText(
            f"{count} validation {'error' if count == 1 else 'errors'}"
        )

        details = []
        for field, msg in self._errors.items():
            details.append(f"• <b>{field}:</b> {msg}")
        self._detail_label.setText("<br>".join(details))

        self._apply_error_style()
        self.show()

    def show_success(self):
        """Briefly show success state, then hide."""
        self._icon_label.setText("✓")
        self._title_label.setText("All inputs valid")
        self._detail_label.setText("")
        self._apply_success_style()
        self.show()

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self.hide)

    def _apply_error_style(self):
        self.setStyleSheet(
            f"QFrame#validationSummary {{"
            f"  background-color: {ModernColors.WARNING}18;"
            f"  border: 1px solid {ModernColors.WARNING};"
            f"  border-radius: 4px;"
            f"}}"
            f"QLabel#validationSummaryTitle {{"
            f"  color: {ModernColors.WARNING}; font-weight: bold; font-size: 12px;"
            f"}}"
            f"QLabel#validationSummaryDetail {{"
            f"  color: {ModernColors.WARNING}; font-size: 11px;"
            f"}}"
        )

    def _apply_success_style(self):
        self.setStyleSheet(
            f"QFrame#validationSummary {{"
            f"  background-color: {ModernColors.SUCCESS}18;"
            f"  border: 1px solid {ModernColors.SUCCESS};"
            f"  border-radius: 4px;"
            f"}}"
            f"QLabel#validationSummaryTitle {{"
            f"  color: {ModernColors.SUCCESS}; font-weight: bold; font-size: 12px;"
            f"}}"
            f"QLabel#validationSummaryDetail {{ color: {ModernColors.SUCCESS}; }}"
        )


# ═══════════════════════════════════════════════════════════════════
# VALIDATION GROUP — orchestrates multiple validated widgets
# ═══════════════════════════════════════════════════════════════════

class ValidationGroup:
    """
    Groups multiple validated widgets and provides:
    - validate_all() that checks everything
    - Auto-enables/disables Run button
    - Updates ValidationSummary widget
    - Replaces the old validate_inputs() → QMessageBox pattern

    Usage:
        self.vg = ValidationGroup()
        self.vg.add(self.cutoff_input)
        self.vg.add(self.method_combo)
        self.vg.bind_run_button(self.run_button)

        # In run handler:
        if not self.vg.validate_all():
            return  # errors shown inline, no QMessageBox needed
    """

    def __init__(self):
        self._widgets: List[_ValidatedWrapper] = []
        self._summary: Optional[ValidationSummary] = None
        self._run_buttons: List[QPushButton] = []

    def add(self, widget: _ValidatedWrapper):
        """Register a validated widget."""
        self._widgets.append(widget)
        widget.validation_changed.connect(self._on_field_changed)

    def remove(self, widget: _ValidatedWrapper):
        """Unregister a validated widget."""
        if widget in self._widgets:
            self._widgets.remove(widget)
            try:
                widget.validation_changed.disconnect(self._on_field_changed)
            except Exception:
                pass

    def bind_run_button(self, button: QPushButton):
        """Auto-disable button when any field is invalid."""
        self._run_buttons.append(button)

    def summary_widget(self) -> ValidationSummary:
        """Get or create the summary widget."""
        if self._summary is None:
            self._summary = ValidationSummary()
        return self._summary

    def validate_all(self) -> bool:
        """
        Validate all fields, update display, return True if all valid.

        This is the replacement for the old validate_inputs() method.
        """
        all_valid = True
        errors: Dict[str, str] = {}

        for widget in self._widgets:
            valid = widget.validate()
            if not valid:
                all_valid = False
                errors[widget.field_name()] = widget.error_message()

        if self._summary:
            if all_valid:
                self._summary.show_success()
            else:
                self._summary.update_errors(errors)

        self._update_run_buttons(all_valid)
        return all_valid

    def is_all_valid(self) -> bool:
        """Check cached validity without re-running validators."""
        return all(w.is_valid() for w in self._widgets)

    def clear_all(self):
        """Clear all validation errors."""
        for widget in self._widgets:
            widget._clear_error()
        if self._summary:
            self._summary.hide()
        self._update_run_buttons(True)

    def errors(self) -> Dict[str, str]:
        """Get all current errors as {field_name: message}."""
        return {
            w.field_name(): w.error_message()
            for w in self._widgets
            if not w.is_valid()
        }

    # ── Internal ──────────────────────────────────────────────────

    def _on_field_changed(self, field_name: str, is_valid: bool, error_msg: str):
        """Callback when any field's validation state changes."""
        all_valid = self.is_all_valid()
        self._update_run_buttons(all_valid)

        if self._summary:
            if all_valid:
                self._summary.hide()
            else:
                self._summary.update_errors(self.errors())

    def _update_run_buttons(self, enabled: bool):
        for btn in self._run_buttons:
            try:
                btn.setEnabled(enabled)
            except RuntimeError:
                pass  # button deleted
