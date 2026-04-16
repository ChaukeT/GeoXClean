"""
Fault Definition Dialog — single fault editor.
================================================

Lets the user define or edit a single fault with:
  - Name, type (normal / reverse / strike-slip)
  - Displacement magnitude and vector
  - Chronological order
  - Optional contact / orientation point overrides

Returns dict compatible with GeologicalModelBuilder.set_fault_definitions().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QLineEdit, QSpinBox,
    QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

_FAULT_TYPES = ["normal", "reverse", "strike_slip"]


class FaultDefinitionDialog(QDialog):
    """Single fault definition editor.

    Parameters
    ----------
    existing : dict | None
        Pre-existing fault definition to edit.
    order_hint : int
        Default chronological order (index in fault list).
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self,
        existing: Optional[Dict[str, Any]] = None,
        order_hint: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Fault Definition")
        self.setMinimumWidth(400)

        self._build_ui(order_hint)

        if existing:
            self._load(existing)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, order_hint: int) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(12)

        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Main_Fault_1")
        form.addRow("Name:", self.name_edit)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(_FAULT_TYPES)
        form.addRow("Fault Type:", self.type_combo)

        # Displacement
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0, 100000)
        self.displacement_spin.setValue(0)
        self.displacement_spin.setDecimals(1)
        self.displacement_spin.setSuffix(" m")
        form.addRow("Displacement:", self.displacement_spin)

        # Displacement vector
        vec_hint = QLabel("Displacement vector (dX, dY, dZ):")
        vec_hint.setStyleSheet("font-size: 10px; color: #808080;")
        form.addRow(vec_hint)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(-100000, 100000)
        self.dx_spin.setDecimals(1)
        form.addRow("dX:", self.dx_spin)

        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(-100000, 100000)
        self.dy_spin.setDecimals(1)
        form.addRow("dY:", self.dy_spin)

        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(-100000, 100000)
        self.dz_spin.setDecimals(1)
        form.addRow("dZ:", self.dz_spin)

        # Chronological order
        self.order_spin = QSpinBox()
        self.order_spin.setRange(0, 100)
        self.order_spin.setValue(order_hint)
        self.order_spin.setToolTip(
            "0 = youngest (most recent fault). "
            "Higher = older. Faults are applied youngest-first."
        )
        form.addRow("Chronological Order:", self.order_spin)

        root.addLayout(form)

        # Dialog buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self, data: Dict[str, Any]) -> None:
        self.name_edit.setText(str(data.get("name", "")))

        ftype = str(data.get("fault_type", "normal"))
        idx = self.type_combo.findText(ftype)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)

        self.displacement_spin.setValue(float(data.get("displacement", 0)))

        vec = data.get("displacement_vector", [0, 0, 0])
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            self.dx_spin.setValue(float(vec[0]))
            self.dy_spin.setValue(float(vec[1]))
            self.dz_spin.setValue(float(vec[2]))

        self.order_spin.setValue(int(data.get("chronological_order", 0)))

    def _on_accept(self) -> None:
        if not self.name_edit.text().strip():
            self.name_edit.setFocus()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fault(self) -> Dict[str, Any]:
        """Return the fault definition dict."""
        return {
            "name": self.name_edit.text().strip(),
            "fault_type": self.type_combo.currentText(),
            "displacement": self.displacement_spin.value(),
            "displacement_vector": [
                self.dx_spin.value(),
                self.dy_spin.value(),
                self.dz_spin.value(),
            ],
            "chronological_order": self.order_spin.value(),
        }
