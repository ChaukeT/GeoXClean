"""
Selection Panel

UI for multi-block selection, named sets, and export.

Redesigned for usable UX:
- No message-box spam; status label provides feedback
- Controls disabled until a model is loaded
- Click selection respects the active selection mode
- Named-set list stores real names via Qt.UserRole (no brittle text parsing)
- Interactive selection doesn't block the 3D view with dialogs
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog,
    QRadioButton, QButtonGroup, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from ..utils.selection_manager import SelectionManager
from .collapsible_group import CollapsibleGroup

logger = logging.getLogger(__name__)

# How long the status "flash" message stays visible (ms)
STATUS_FLASH_MS = 4000


class SelectionPanel(QWidget):
    """
    Panel for managing block selections.

    Features:
    - Marquee (box) selection with manual ranges or "From Model Bounds"
    - Interactive 3D box selection
    - Click-to-select individual blocks
    - Property-based query builder
    - Named selection sets (save / load / delete)
    - Export to CSV / VTK
    """

    selection_changed = pyqtSignal(set)  # Emits current selection indices

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_manager = SelectionManager()
        self.plotter = None
        self.main_window = None
        self.grid_spec = None

        # Signal-connection guards (replaced later by proper disconnect)
        self._box_signal_connected = False
        self._click_signal_connected = False

        self._build_ui()
        self._set_controls_enabled(False)  # nothing loaded yet
        logger.info("Initialized SelectionPanel")

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        """Build the complete UI."""
        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        # Title
        title = QLabel("Block Selection")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4CAF50;")
        layout.addWidget(title)

        # --- Status bar (replaces MessageBox spam) ---
        self.status_label = QLabel("⚠️ No block model loaded. Please load a block model from Data → Block Model to enable selection.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #2b2b2b; padding: 6px; border-radius: 3px; color: #FFA726;"
        )
        layout.addWidget(self.status_label)

        # Collapsible sections
        layout.addWidget(self._create_marquee_group())
        layout.addWidget(self._create_query_group())
        layout.addWidget(self._create_mode_group())
        layout.addWidget(self._create_named_sets_group())
        layout.addWidget(self._create_export_group())

        layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    # Marquee / box selection
    # ------------------------------------------------------------------

    def _create_marquee_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Marquee Selection (Box)", collapsed=False)

        form = QFormLayout()
        form.setSpacing(5)

        self._range_spins = {}
        for axis in ('X', 'Y', 'Z'):
            row = QHBoxLayout()
            spin_min = QDoubleSpinBox()
            spin_min.setRange(-1e9, 1e9)
            spin_min.setDecimals(2)
            spin_min.setValue(0)
            spin_min.setMaximumWidth(100)
            row.addWidget(spin_min)
            row.addWidget(QLabel("to"))

            spin_max = QDoubleSpinBox()
            spin_max.setRange(-1e9, 1e9)
            spin_max.setDecimals(2)
            spin_max.setValue(0)
            spin_max.setMaximumWidth(100)
            row.addWidget(spin_max)
            row.addStretch()

            form.addRow(f"{axis} Range:", row)
            self._range_spins[axis] = (spin_min, spin_max)

        group.add_layout(form)

        # Row 1 — Select Box / From Model Bounds
        row1 = QHBoxLayout()
        self.btn_select_box = QPushButton("Select Box")
        self.btn_select_box.setToolTip("Select blocks within the specified box ranges")
        self.btn_select_box.clicked.connect(self._on_select_marquee)
        row1.addWidget(self.btn_select_box)

        self.btn_from_bounds = QPushButton("Model Bounds")
        self.btn_from_bounds.setToolTip(
            "Reset ranges to the full spatial extent of the loaded block model"
        )
        self.btn_from_bounds.clicked.connect(self._on_from_model_bounds)
        row1.addWidget(self.btn_from_bounds)
        group.add_layout(row1)

        # Row 2 — Interactive 3D box
        row2 = QHBoxLayout()
        self.btn_interactive = QPushButton("Select in 3D View")
        self.btn_interactive.setToolTip("Draw a selection box interactively in the 3D view")
        self.btn_interactive.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.btn_interactive.clicked.connect(self._on_interactive_box_selection)
        row2.addWidget(self.btn_interactive)
        group.add_layout(row2)

        # Row 3 — Click selection toggle
        sep = QLabel("Or click blocks directly:")
        sep.setStyleSheet("margin-top: 10px; font-style: italic; color: #888;")
        group.add_widget(sep)

        row3 = QHBoxLayout()
        self.btn_click_select = QPushButton("Click to Select Blocks")
        self.btn_click_select.setToolTip(
            "Click individual blocks in 3D view to select them "
            "(respects the current selection mode)"
        )
        self.btn_click_select.setCheckable(True)
        self.btn_click_select.clicked.connect(self._on_toggle_click_selection)
        row3.addWidget(self.btn_click_select)
        group.add_layout(row3)

        return group

    # ------------------------------------------------------------------
    # Property query
    # ------------------------------------------------------------------

    def _create_query_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Property Query", collapsed=True)

        form = QFormLayout()
        form.setSpacing(5)

        self.property_combo = QComboBox()
        self.property_combo.setEditable(False)  # prevent typing invalid names
        self.property_combo.setToolTip("Select a numeric property column to query")
        form.addRow("Property:", self.property_combo)

        self.operator_combo = QComboBox()
        self.operator_combo.addItems(['>', '>=', '<', '<=', '==', '!='])
        self.operator_combo.setCurrentText('>=')
        form.addRow("Operator:", self.operator_combo)

        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1e9, 1e9)
        self.value_spin.setDecimals(4)
        self.value_spin.setValue(0)
        form.addRow("Value:", self.value_spin)

        group.add_layout(form)

        self.btn_select_query = QPushButton("Select by Property")
        self.btn_select_query.setToolTip("Select blocks matching the property criteria")
        self.btn_select_query.clicked.connect(self._on_select_property)
        group.add_widget(self.btn_select_query)

        return group

    # ------------------------------------------------------------------
    # Selection mode (new / add / subtract / intersect)
    # ------------------------------------------------------------------

    def _create_mode_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Selection Mode", collapsed=True)

        row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.mode_new = QRadioButton("New")
        self.mode_new.setToolTip("Replace current selection")
        self.mode_new.setChecked(True)

        self.mode_add = QRadioButton("Add")
        self.mode_add.setToolTip("Union with current selection")

        self.mode_subtract = QRadioButton("Subtract")
        self.mode_subtract.setToolTip("Remove from current selection")

        self.mode_intersect = QRadioButton("Intersect")
        self.mode_intersect.setToolTip("Keep only blocks in both selections")

        for btn in (self.mode_new, self.mode_add, self.mode_subtract, self.mode_intersect):
            self.mode_group.addButton(btn)
            row.addWidget(btn)

        group.add_layout(row)
        return group

    # ------------------------------------------------------------------
    # Named sets
    # ------------------------------------------------------------------

    def _create_named_sets_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Named Selection Sets", collapsed=True)

        self.sets_list = QListWidget()
        self.sets_list.setMaximumHeight(120)
        self.sets_list.setToolTip("Double-click to load a set")
        self.sets_list.itemDoubleClicked.connect(self._on_load_set)
        group.add_widget(self.sets_list)

        # Save row
        save_row = QHBoxLayout()
        self.set_name_edit = QLineEdit()
        self.set_name_edit.setPlaceholderText("Set name…")
        save_row.addWidget(self.set_name_edit)

        self.btn_save_set = QPushButton("Save")
        self.btn_save_set.setToolTip("Save current selection as a named set")
        self.btn_save_set.clicked.connect(self._on_save_set)
        save_row.addWidget(self.btn_save_set)
        group.add_layout(save_row)

        # Load / Delete row
        ops_row = QHBoxLayout()
        self.btn_load_set = QPushButton("Load")
        self.btn_load_set.setToolTip("Load selected set (respects selection mode)")
        self.btn_load_set.clicked.connect(self._on_load_set_button)
        ops_row.addWidget(self.btn_load_set)

        self.btn_delete_set = QPushButton("Delete")
        self.btn_delete_set.setToolTip("Delete selected set")
        self.btn_delete_set.clicked.connect(self._on_delete_set)
        ops_row.addWidget(self.btn_delete_set)
        group.add_layout(ops_row)

        return group

    # ------------------------------------------------------------------
    # Export & clear
    # ------------------------------------------------------------------

    def _create_export_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Export & Actions", collapsed=True)

        col = QVBoxLayout()
        col.setSpacing(5)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setToolTip("Export selected blocks to CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        col.addWidget(self.btn_export_csv)

        self.btn_export_vtk = QPushButton("Export VTK")
        self.btn_export_vtk.setToolTip("Export selected blocks to VTK")
        self.btn_export_vtk.clicked.connect(self._on_export_vtk)
        col.addWidget(self.btn_export_vtk)

        self.btn_clear = QPushButton("Clear Selection")
        self.btn_clear.setToolTip("Clear current selection")
        self.btn_clear.setStyleSheet("background-color: #8B4513;")
        self.btn_clear.clicked.connect(self._on_clear_selection)
        col.addWidget(self.btn_clear)

        group.add_layout(col)
        return group

    # ==================================================================
    # Public API — called by the main window
    # ==================================================================

    def set_plotter(self, plotter):
        """Set the pyvista plotter reference."""
        self.plotter = plotter

    def set_main_window(self, main_window):
        """Set the main window reference (for viewer_widget access)."""
        self.main_window = main_window

    def set_block_model(self, block_df, grid_spec=None):
        """Load a block model into the panel and manager."""
        self.selection_manager.set_block_model(block_df, grid_spec)
        self.grid_spec = grid_spec

        # Populate property combo
        self.property_combo.clear()
        self.property_combo.addItems(self.selection_manager.numeric_columns)

        # Set range spin boxes to actual model bounds
        bounds = self.selection_manager.model_bounds
        if bounds:
            for axis, (lo, hi) in bounds.items():
                spin_min, spin_max = self._range_spins[axis.upper()]
                spin_min.setValue(lo)
                spin_max.setValue(hi)

        self._set_controls_enabled(True)
        self._update_status(f"Loaded {len(block_df):,} blocks")
        logger.info(f"Set block model: {len(block_df)} blocks")

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_selection_mode(self) -> str:
        if self.mode_add.isChecked():
            return 'add'
        if self.mode_subtract.isChecked():
            return 'subtract'
        if self.mode_intersect.isChecked():
            return 'intersect'
        return 'new'

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable all action controls (used before a model is loaded)."""
        for w in (
            self.btn_select_box, self.btn_from_bounds, self.btn_interactive,
            self.btn_click_select, self.btn_select_query,
            self.btn_save_set, self.btn_load_set, self.btn_delete_set,
            self.btn_export_csv, self.btn_export_vtk, self.btn_clear,
        ):
            w.setEnabled(enabled)

    def _update_status(self, message: str, flash: bool = False):
        """
        Update the status label.

        If *flash* is True the message reverts to the standard selection
        summary after STATUS_FLASH_MS milliseconds.
        """
        self.status_label.setText(message)
        if flash:
            QTimer.singleShot(STATUS_FLASH_MS, self._refresh_status_to_selection)

    def _refresh_status_to_selection(self):
        """Reset the status label to the current selection summary."""
        self.status_label.setText(self.selection_manager.get_selection_summary())

    def _notify_selection_changed(self):
        """Common post-selection hook: update status + emit signal."""
        self._refresh_status_to_selection()
        self.selection_changed.emit(self.selection_manager.current_selection)

    def _get_viewer(self):
        """Return the viewer_widget or None (with a status message on failure)."""
        if self.main_window is None:
            self._update_status("Main window not connected", flash=True)
            return None
        if not hasattr(self.main_window, 'viewer_widget'):
            self._update_status("Viewer widget not available", flash=True)
            return None
        return self.main_window.viewer_widget

    # ==================================================================
    # Marquee callbacks
    # ==================================================================

    def _on_select_marquee(self):
        """Select blocks within the specified X/Y/Z ranges."""
        x_lo, x_hi = (s.value() for s in self._range_spins['X'])
        y_lo, y_hi = (s.value() for s in self._range_spins['Y'])
        z_lo, z_hi = (s.value() for s in self._range_spins['Z'])
        mode = self._get_selection_mode()

        count = self.selection_manager.select_by_marquee(
            (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi), mode
        )
        self._update_status(
            f"Box select ({mode}): {count:,} blocks  |  "
            f"X [{x_lo:.1f}–{x_hi:.1f}]  Y [{y_lo:.1f}–{y_hi:.1f}]  "
            f"Z [{z_lo:.1f}–{z_hi:.1f}]",
            flash=True,
        )
        self.selection_changed.emit(self.selection_manager.current_selection)

    def _on_from_model_bounds(self):
        """Reset range spin boxes to the full model extent."""
        bounds = self.selection_manager.model_bounds
        if bounds is None:
            self._update_status("No model bounds available", flash=True)
            return

        for axis, (lo, hi) in bounds.items():
            spin_min, spin_max = self._range_spins[axis.upper()]
            spin_min.setValue(lo)
            spin_max.setValue(hi)

        self._update_status("Ranges reset to full model bounds", flash=True)

    # ==================================================================
    # Interactive 3D box selection
    # ==================================================================

    def _on_interactive_box_selection(self):
        """Start interactive box selection in the 3D viewer."""
        viewer = self._get_viewer()
        if viewer is None:
            return

        try:
            viewer.start_box_selection_mode()

            # Connect once; use UniqueConnection to avoid duplicates
            if not self._box_signal_connected:
                viewer.box_selection_completed.connect(
                    self._on_box_selection_completed
                )
                self._box_signal_connected = True

            # Non-blocking hint — no QMessageBox
            self._update_status(
                "Drag in the 3D view to define a selection box…", flash=False
            )

        except RuntimeError as e:
            # User-friendly error from viewer_widget (persistent, not flash)
            logger.error(f"Interactive box selection error: {e}", exc_info=True)
            self._update_status(f"❌ {str(e)}", flash=False)
        except Exception as e:
            logger.error(f"Interactive box selection error: {e}", exc_info=True)
            self._update_status(f"❌ Unexpected error: {str(e)}", flash=False)

    def _on_box_selection_completed(self, bounds):
        """Callback when the user finishes drawing a box in 3D."""
        try:
            # Update spin boxes
            axes = ('X', 'Y', 'Z')
            for i, axis in enumerate(axes):
                spin_min, spin_max = self._range_spins[axis]
                spin_min.setValue(bounds[i * 2])
                spin_max.setValue(bounds[i * 2 + 1])

            # Immediately apply selection (no confirmation dialog)
            self._on_select_marquee()

        except Exception as e:
            logger.error(f"Box selection completion error: {e}", exc_info=True)
            self._update_status(f"Error: {e}", flash=True)

    # ==================================================================
    # Click-to-select
    # ==================================================================

    def _on_toggle_click_selection(self, checked: bool):
        """Toggle click-to-select mode in the 3D viewer."""
        viewer = self._get_viewer()
        if viewer is None:
            self.btn_click_select.setChecked(False)
            return

        try:
            if checked:
                viewer.start_click_selection_mode()

                if not self._click_signal_connected:
                    viewer.blocks_selected.connect(self._on_blocks_clicked)
                    self._click_signal_connected = True

                self.btn_click_select.setText("End Click Selection")
                self.btn_click_select.setStyleSheet(
                    "background-color: #f44336; color: white;"
                )
                self._update_status(
                    "Click-select active — click blocks in the 3D view. "
                    "Press the button again to finish.",
                    flash=False,
                )
            else:
                selected = viewer.end_click_selection_mode()

                self.btn_click_select.setText("Click to Select Blocks")
                self.btn_click_select.setStyleSheet("")

                if selected and len(selected) > 0:
                    mode = self._get_selection_mode()
                    self.selection_manager.select_by_indices(selected, mode)
                    self._notify_selection_changed()
                    self._update_status(
                        f"Click select ({mode}): {len(self.selection_manager.current_selection):,} blocks",
                        flash=True,
                    )
                else:
                    self._update_status("Click selection ended — no blocks selected", flash=True)

        except RuntimeError as e:
            # User-friendly error from viewer_widget (persistent, not flash)
            logger.error(f"Click selection toggle error: {e}", exc_info=True)
            self.btn_click_select.setChecked(False)
            self.btn_click_select.setText("Click to Select Blocks")
            self.btn_click_select.setStyleSheet("")
            self._update_status(f"❌ {str(e)}", flash=False)
        except Exception as e:
            logger.error(f"Click selection toggle error: {e}", exc_info=True)
            self.btn_click_select.setChecked(False)
            self.btn_click_select.setText("Click to Select Blocks")
            self.btn_click_select.setStyleSheet("")
            self._update_status(f"❌ Unexpected error: {str(e)}", flash=False)

    def _on_blocks_clicked(self, block_indices):
        """Real-time feedback as blocks are clicked."""
        count = len(block_indices) if block_indices else 0
        self._update_status(f"Click-selecting… {count} block(s) picked so far")

    # ==================================================================
    # Property query
    # ==================================================================

    def _on_select_property(self):
        property_name = self.property_combo.currentText()
        if not property_name:
            self._update_status("No property selected", flash=True)
            return

        operator = self.operator_combo.currentText()
        value = self.value_spin.value()
        mode = self._get_selection_mode()

        count = self.selection_manager.select_by_property(
            property_name, operator, value, mode
        )
        self._update_status(
            f"Query ({mode}): {property_name} {operator} {value} → {count:,} blocks",
            flash=True,
        )
        self.selection_changed.emit(self.selection_manager.current_selection)

    # ==================================================================
    # Named sets
    # ==================================================================

    def _on_save_set(self):
        name = self.set_name_edit.text().strip()
        if not name:
            self._update_status("Enter a name for the selection set", flash=True)
            return
        if not self.selection_manager.current_selection:
            self._update_status("Nothing selected to save", flash=True)
            return

        self.selection_manager.save_selection_as_set(name)
        self._update_sets_list()
        self.set_name_edit.clear()
        self._update_status(
            f"Saved set '{name}' ({len(self.selection_manager.current_selection):,} blocks)",
            flash=True,
        )

    def _on_load_set(self, item: Optional[QListWidgetItem] = None):
        if item is None:
            item = self.sets_list.currentItem()
        if item is None:
            return

        # Retrieve the real name stored via UserRole (not parsed from display text)
        name = item.data(Qt.ItemDataRole.UserRole)
        if name is None:
            return

        mode = self._get_selection_mode()
        if self.selection_manager.load_selection_set(name, mode):
            self._notify_selection_changed()
            self._update_status(
                f"Loaded set '{name}' ({mode}) → "
                f"{len(self.selection_manager.current_selection):,} blocks",
                flash=True,
            )

    def _on_load_set_button(self):
        self._on_load_set()

    def _on_delete_set(self):
        item = self.sets_list.currentItem()
        if item is None:
            self._update_status("Select a set to delete", flash=True)
            return

        name = item.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete selection set '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.selection_manager.delete_selection_set(name)
            self._update_sets_list()
            self._update_status(f"Deleted set '{name}'", flash=True)

    def _update_sets_list(self):
        """Rebuild the named-sets QListWidget from the manager."""
        self.sets_list.clear()
        for name, sel_set in self.selection_manager.named_sets.items():
            display = f"{name}  ({len(sel_set):,} blocks)"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, name)  # store real name
            self.sets_list.addItem(item)

    # ==================================================================
    # Export
    # ==================================================================

    def _on_export_csv(self):
        if not self.selection_manager.current_selection:
            self._update_status("Nothing selected to export", flash=True)
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Selection to CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return

        if self.selection_manager.export_selection_csv(Path(filepath)):
            self._update_status(
                f"Exported {len(self.selection_manager.current_selection):,} blocks → {Path(filepath).name}",
                flash=True,
            )
        else:
            self._update_status("CSV export failed — check the log", flash=True)

    def _on_export_vtk(self):
        if not self.selection_manager.current_selection:
            self._update_status("Nothing selected to export", flash=True)
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Selection to VTK", "",
            "VTK Files (*.vtk *.vtp);;All Files (*)",
        )
        if not filepath:
            return

        if self.selection_manager.export_selection_vtk(Path(filepath), self.grid_spec):
            self._update_status(
                f"Exported {len(self.selection_manager.current_selection):,} blocks → {Path(filepath).name}",
                flash=True,
            )
        else:
            self._update_status("VTK export failed — check the log", flash=True)

    # ==================================================================
    # Clear
    # ==================================================================

    def _on_clear_selection(self):
        self.selection_manager.clear_selection()
        self._notify_selection_changed()
        self._update_status("Selection cleared", flash=True)

    # ==================================================================
    # Theme
    # ==================================================================

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet
        self.setStyleSheet(get_analysis_panel_stylesheet())
