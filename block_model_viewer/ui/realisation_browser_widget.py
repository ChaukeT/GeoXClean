"""
Reusable realisation browser widget for simulation panels.

Provides a slider + navigation buttons to browse individual simulation
realisations in the 3D viewer. Embeddable in any simulation panel
(SGSIM, CoSGSIM, IK-SGSIM, SIS, Turning Bands, DBS, GRF, MPS).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer

logger = logging.getLogger(__name__)


class RealisationBrowserWidget(QWidget):
    """Browse individual simulation realisations with a slider.

    Parameters
    ----------
    parent : QWidget, optional
    on_realisation_changed : callable(index: int, data_3d: np.ndarray)
        Called when the user selects a different realisation.
        ``data_3d`` is the (nz, ny, nx) array for that realisation.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        on_realisation_changed: Optional[Callable] = None,
    ) -> None:
        super().__init__(parent)
        self._realisations: Optional[np.ndarray] = None  # (nreal, nz, ny, nx)
        self._n_real: int = 0
        self._current: int = 0
        self._on_changed = on_realisation_changed
        self._animate_timer = QTimer(self)
        self._animate_timer.setInterval(500)  # 2 per second
        self._animate_timer.timeout.connect(self._animate_step)
        self._build_ui()
        self.setEnabled(False)

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Realisation Browser")
        gl = QVBoxLayout(group)

        self._label = QLabel("No realisations loaded")
        gl.addWidget(self._label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(1)
        self._slider.setValue(1)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.valueChanged.connect(self._on_slider)
        gl.addWidget(self._slider)

        # Navigation row
        nav = QHBoxLayout()

        self._btn_first = QPushButton("|<")
        self._btn_first.setFixedWidth(30)
        self._btn_first.clicked.connect(lambda: self._go_to(0))
        nav.addWidget(self._btn_first)

        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(30)
        self._btn_prev.clicked.connect(lambda: self._go_to(self._current - 1))
        nav.addWidget(self._btn_prev)

        self._spin = QSpinBox()
        self._spin.setMinimum(1)
        self._spin.setMaximum(1)
        self._spin.setValue(1)
        self._spin.valueChanged.connect(self._on_spin)
        nav.addWidget(self._spin)

        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(30)
        self._btn_next.clicked.connect(lambda: self._go_to(self._current + 1))
        nav.addWidget(self._btn_next)

        self._btn_last = QPushButton(">|")
        self._btn_last.setFixedWidth(30)
        self._btn_last.clicked.connect(lambda: self._go_to(self._n_real - 1))
        nav.addWidget(self._btn_last)

        nav.addSpacing(10)

        self._btn_animate = QPushButton("Animate")
        self._btn_animate.setCheckable(True)
        self._btn_animate.toggled.connect(self._toggle_animate)
        nav.addWidget(self._btn_animate)

        self._btn_summary = QPushButton("Summary")
        self._btn_summary.setToolTip("Return to mean/summary view")
        self._btn_summary.clicked.connect(self._show_summary)
        nav.addWidget(self._btn_summary)

        nav.addStretch()
        gl.addLayout(nav)
        layout.addWidget(group)

    # ── Public API ───────────────────────────────────────────────────

    def set_realisations(self, realisations: np.ndarray) -> None:
        """Load realisations array (nreal, nz, ny, nx)."""
        if realisations is None or realisations.ndim != 4:
            self.setEnabled(False)
            self._label.setText("No realisations loaded")
            return

        self._realisations = realisations
        self._n_real = realisations.shape[0]
        self._current = 0

        self._slider.setMaximum(self._n_real)
        self._spin.setMaximum(self._n_real)
        self._slider.setValue(1)
        self._spin.setValue(1)
        self._update_label()
        self.setEnabled(True)

    def set_callback(self, cb: Callable) -> None:
        self._on_changed = cb

    # ── Internal ─────────────────────────────────────────────────────

    def _go_to(self, index: int) -> None:
        index = max(0, min(index, self._n_real - 1))
        if index == self._current and self._realisations is not None:
            return
        self._current = index
        self._slider.blockSignals(True)
        self._slider.setValue(index + 1)
        self._slider.blockSignals(False)
        self._spin.blockSignals(True)
        self._spin.setValue(index + 1)
        self._spin.blockSignals(False)
        self._update_label()
        self._emit_change()

    def _on_slider(self, value: int) -> None:
        self._go_to(value - 1)

    def _on_spin(self, value: int) -> None:
        self._go_to(value - 1)

    def _update_label(self) -> None:
        self._label.setText(
            f"Realisation {self._current + 1} of {self._n_real}"
        )

    def _emit_change(self) -> None:
        if self._on_changed is not None and self._realisations is not None:
            try:
                data = self._realisations[self._current]
                self._on_changed(self._current, data)
            except Exception as exc:
                logger.warning("Realisation callback failed: %s", exc)

    def _toggle_animate(self, checked: bool) -> None:
        if checked:
            self._animate_timer.start()
            self._btn_animate.setText("Stop")
        else:
            self._animate_timer.stop()
            self._btn_animate.setText("Animate")

    def _animate_step(self) -> None:
        next_idx = (self._current + 1) % self._n_real
        self._go_to(next_idx)

    def _show_summary(self) -> None:
        self._animate_timer.stop()
        self._btn_animate.setChecked(False)
        # Emit with index=-1 to signal "show summary"
        if self._on_changed is not None:
            try:
                self._on_changed(-1, None)
            except Exception:
                pass
