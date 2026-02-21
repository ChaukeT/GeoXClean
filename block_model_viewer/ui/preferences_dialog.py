"""
Preferences Dialog for major UX settings

Allows users to tweak common UI and behavior defaults.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox,
    QComboBox, QSpinBox, QPushButton
)
from PyQt6.QtCore import QSettings

logger = logging.getLogger(__name__)


class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(520, 420)  # Slightly taller for new Panels group

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Session group
        session_group = QGroupBox("Session")
        session_layout = QVBoxLayout()
        self.restore_session_check = QCheckBox("Restore last session on startup")
        session_layout.addWidget(self.restore_session_check)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        # View defaults
        view_group = QGroupBox("View Defaults")
        view_layout = QVBoxLayout()

        self.show_axes_check = QCheckBox("Show axes by default")
        view_layout.addWidget(self.show_axes_check)

        self.show_grid_check = QCheckBox("Show grid by default")
        view_layout.addWidget(self.show_grid_check)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Lighting preset:"))
        self.lighting_combo = QComboBox()
        self.lighting_combo.addItems(["soft", "balanced", "sharp"])
        hl.addWidget(self.lighting_combo)
        hl.addStretch()
        view_layout.addLayout(hl)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Panel settings
        panels_group = QGroupBox("Panel Controls")
        panels_layout = QVBoxLayout()

        self.show_header_controls_check = QCheckBox("Show panel header controls (Collapse/Refresh/Clear/Hide)")
        self.show_header_controls_check.setToolTip(
            "When enabled, docked panels show control buttons in their header bar.\n"
            "Changes apply immediately to all open panels."
        )
        panels_layout.addWidget(self.show_header_controls_check)

        self.confirm_clear_check = QCheckBox("Confirm before clearing panels")
        self.confirm_clear_check.setToolTip("Show confirmation dialog before clearing panel data")
        panels_layout.addWidget(self.confirm_clear_check)

        panels_group.setLayout(panels_layout)
        layout.addWidget(panels_group)

        # Table viewer
        table_group = QGroupBox("Table Viewer")
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Row limit (for performance):"))
        self.row_limit_spin = QSpinBox()
        self.row_limit_spin.setRange(100, 200000)
        self.row_limit_spin.setSingleStep(500)
        table_layout.addWidget(self.row_limit_spin)
        table_layout.addStretch()
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_and_close)
        btns.addWidget(apply_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _load_settings(self):
        try:
            s_session = QSettings("GeoX", "Session")
            self.restore_session_check.setChecked(bool(s_session.value("restore_on_startup", True, type=bool)))
        except Exception:
            self.restore_session_check.setChecked(True)

        s_view = QSettings("GeoX", "View")
        self.show_axes_check.setChecked(bool(s_view.value("show_axes_default", True, type=bool)))
        self.show_grid_check.setChecked(bool(s_view.value("show_grid_default", False, type=bool)))
        self.lighting_combo.setCurrentText(str(s_view.value("default_lighting", "balanced")))

        s_table = QSettings("GeoX", "TableViewer")
        self.row_limit_spin.setValue(int(s_table.value("row_limit", 5000)))

        # Panel settings
        s_panels = QSettings("GeoX", "Panels")
        self.show_header_controls_check.setChecked(s_panels.value("show_header_controls", True, type=bool))
        self.confirm_clear_check.setChecked(s_panels.value("confirm_clear", False, type=bool))

    def apply_and_close(self):
        # Save settings
        try:
            s_session = QSettings("GeoX", "Session")
            s_session.setValue("restore_on_startup", self.restore_session_check.isChecked())
        except Exception:
            pass

        s_view = QSettings("GeoX", "View")
        s_view.setValue("show_axes_default", self.show_axes_check.isChecked())
        s_view.setValue("show_grid_default", self.show_grid_check.isChecked())
        s_view.setValue("default_lighting", self.lighting_combo.currentText())

        s_table = QSettings("GeoX", "TableViewer")
        s_table.setValue("row_limit", self.row_limit_spin.value())

        # Panel settings
        s_panels = QSettings("GeoX", "Panels")
        s_panels.setValue("show_header_controls", self.show_header_controls_check.isChecked())
        s_panels.setValue("confirm_clear", self.confirm_clear_check.isChecked())

        # Apply immediate changes to the main window if available
        try:
            if self.parent():
                mw = self.parent()
                # Axes/grid
                if hasattr(mw, 'axes_action') and mw.axes_action.isChecked() != self.show_axes_check.isChecked():
                    mw.axes_action.setChecked(self.show_axes_check.isChecked())
                    mw.toggle_axes(self.show_axes_check.isChecked())
                if hasattr(mw, 'grid_action') and mw.grid_action.isChecked() != self.show_grid_check.isChecked():
                    mw.grid_action.setChecked(self.show_grid_check.isChecked())
                    mw.toggle_bounds(self.show_grid_check.isChecked())
                # Lighting
                if hasattr(mw, 'apply_lighting_preset'):
                    mw.apply_lighting_preset(self.lighting_combo.currentText())

                # Panel header controls - apply immediately to all PersistentDockWidgets
                from .persistent_dock import PersistentDockWidget
                show_controls = self.show_header_controls_check.isChecked()
                for dock in mw.findChildren(PersistentDockWidget):
                    if hasattr(dock, 'set_header_controls_visible'):
                        dock.set_header_controls_visible(show_controls)
        except Exception as e:
            logger.warning(f"Failed to apply preferences immediately: {e}")

        self.accept()
