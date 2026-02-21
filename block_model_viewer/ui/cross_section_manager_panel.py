"""
Cross-Section Manager Panel

UI for managing named cross-sections, quick rendering, and exports.

Redesigned:
- Status label replaces MessageBox spam (except destructive confirm)
- Named-set items store real names via Qt.UserRole
- Single handler for loaded/generated/classified model signals
- Property combo non-editable
- Controls disabled until a model is loaded
- No recursive combo-box update loops
"""

import logging
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog,
    QScrollArea, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from ..utils.cross_section_manager import CrossSectionManager, CrossSectionSpec
from .collapsible_group import CollapsibleGroup

try:
    from .base_analysis_panel import BaseAnalysisPanel
    from .signals import UISignals
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    UISignals = None

logger = logging.getLogger(__name__)

STATUS_FLASH_MS = 4000


def _safe_panel_category():
    try:
        from block_model_viewer.ui.panel_manager import PanelCategory  # noqa: F401
        return "Visualization"
    except Exception:
        return None


def _safe_dock_area():
    try:
        from block_model_viewer.ui.panel_manager import DockArea  # noqa: F401
        return "Right"
    except Exception:
        return None


class CrossSectionManagerPanel(BaseAnalysisPanel if BASE_AVAILABLE else QWidget):
    """
    Panel for managing named cross-sections.

    Features:
    - Create named axis-aligned sections
    - Save/load section library (JSON)
    - Quick render sections in 3D viewer
    - Export section data (CSV) and images (PNG/PDF)
    - Interactive plane positioning in 3D view
    - Auto-detect available block models via DataRegistry
    """

    section_render_requested = pyqtSignal(str, str)  # section_name, property_name

    # PanelManager metadata
    PANEL_ID = "CrossSectionManagerPanel"
    PANEL_NAME = "Cross-Section Manager Panel"
    PANEL_CATEGORY = _safe_panel_category()
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = _safe_dock_area()

    def __init__(self, parent=None, signals: Optional['UISignals'] = None):
        # Data attributes — set BEFORE super().__init__
        self.section_manager = CrossSectionManager()
        self.main_window = None
        self._block_model = None
        self.available_properties: List[str] = []
        self.available_models: dict = {}
        self.signals = signals
        self._plane_signal_connected = False
        self._updating_combo = False  # recursion guard

        if BASE_AVAILABLE:
            super().__init__(parent=parent, panel_id="cross_section_manager")
        else:
            super().__init__(parent)

        self._connect_registry()
        self._build_ui()
        # BUG FIX: Only disable controls if no block model is loaded
        # If a block model was loaded during _connect_registry(), keep controls enabled
        self._set_controls_enabled(self._block_model is not None)
        logger.info("Initialized CrossSectionManagerPanel")

    # ==================================================================
    # Registry integration
    # ==================================================================

    def _connect_registry(self):
        self.registry = None
        if not BASE_AVAILABLE:
            return
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.blockModelLoaded.connect(self._on_block_model_signal)
                self.registry.blockModelGenerated.connect(self._on_block_model_signal)
                self.registry.blockModelClassified.connect(self._on_block_model_signal)
                self._refresh_registry_models()
                logger.info("Cross-section manager connected to DataRegistry")
        except Exception as e:
            logger.warning(f"Registry connection failed: {e}")
            self.registry = None

    def _on_block_model_signal(self, block_model):
        """Single handler for any block-model registry signal."""
        self._update_block_model(block_model)
        self._refresh_registry_models()
        if hasattr(self, 'block_model_combo'):
            self._rebuild_model_combo()
        logger.info("Cross-section manager updated with block model")

    def _refresh_registry_models(self):
        """Populate self.available_models from the registry."""
        if not self.registry:
            return
        try:
            m = self.registry.get_block_model()
            if m is not None:
                self.available_models["Current"] = m
                if self._block_model is None:
                    self._update_block_model(m)
            try:
                cm = self.registry.get_classified_block_model()
                if cm is not None:
                    self.available_models["Classified"] = cm
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed to refresh models: {e}")

    def _update_block_model(self, block_model):
        """Update internal reference + property lists. Does NOT touch combos."""
        self._block_model = block_model
        if hasattr(self, 'on_block_model_changed'):
            self.on_block_model_changed()

        if block_model is not None:
            if hasattr(block_model, 'columns'):
                self.available_properties = [
                    c for c in block_model.columns
                    if c not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ')
                ]
            elif hasattr(block_model, 'properties'):
                self.available_properties = list(block_model.properties.keys())
            else:
                self.available_properties = []

            self.available_models["Current"] = block_model

            if hasattr(self, 'render_property_combo'):
                self._populate_property_combo()

            self._set_controls_enabled(True)
        else:
            self.available_properties = []
            self._set_controls_enabled(False)

        # BUG FIX: Update status label to reflect the block model load state
        # Without this, the status label stays at "No block model loaded" even though
        # self._block_model was successfully set during panel initialization
        if hasattr(self, 'status_label'):
            self._restore_status()

        # BUG FIX: Rebuild combo box to populate it with available block models
        # During initialization, combo boxes don't exist when _connect_registry() runs,
        # so we need to rebuild them here after _build_ui() has created them
        if hasattr(self, 'block_model_combo'):
            self._rebuild_model_combo()

    def _populate_property_combo(self):
        """Populate the render property combo from available_properties."""
        if not hasattr(self, 'render_property_combo'):
            return
        self.render_property_combo.clear()
        self.render_property_combo.addItems(self.available_properties)

        # Auto-select a grade-like property
        for col in self.available_properties:
            if any(g in col.upper() for g in ['ZN', 'CU', 'AU', 'AG', 'PB', 'FE', 'GRADE']):
                self.render_property_combo.setCurrentText(col)
                break

    def _rebuild_model_combo(self):
        """Rebuild the block-model combo with all known models, guarded against recursion."""
        if self._updating_combo or not hasattr(self, 'block_model_combo'):
            return
        self._updating_combo = True
        try:
            self.block_model_combo.clear()
            for label, model in self.available_models.items():
                if model is not None:
                    display = f"Block Model: {label}" if label not in ("Current",) else "Current Block Model"
                    self.block_model_combo.addItem(display, model)
        finally:
            self._updating_combo = False

    def _on_block_model_selected(self, index: int):
        """Handle block-model combo selection."""
        if self._updating_combo or index < 0:
            return
        model = self.block_model_combo.itemData(index)
        if model is not None:
            self._update_block_model(model)

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        # Use the main_layout from BaseAnalysisPanel (already inside a scroll area)
        layout = self.main_layout
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        title = QLabel("Cross-Section Manager")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2196F3;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("No block model loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #2b2b2b; padding: 6px; border-radius: 3px;"
        )
        layout.addWidget(self.status_label)

        layout.addWidget(self._create_section_group())
        layout.addWidget(self._create_sections_list_group())
        layout.addWidget(self._create_render_group())
        layout.addWidget(self._create_library_group())

        layout.addStretch()

    # ------------------------------------------------------------------
    # Create section
    # ------------------------------------------------------------------

    def _create_section_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Create New Section", collapsed=False)

        form = QFormLayout()
        form.setSpacing(5)

        self.section_name_edit = QLineEdit()
        self.section_name_edit.setPlaceholderText("e.g., Section_4500mN")
        form.addRow("Name:", self.section_name_edit)

        self.plane_type_combo = QComboBox()
        self.plane_type_combo.addItems(['X', 'Y', 'Z'])
        self.plane_type_combo.setToolTip("Axis-aligned cutting-plane direction")
        self.plane_type_combo.currentTextChanged.connect(self._on_plane_type_changed)
        form.addRow("Plane Type:", self.plane_type_combo)

        self.position_spin = QDoubleSpinBox()
        self.position_spin.setRange(-1e9, 1e9)
        self.position_spin.setDecimals(2)
        self.position_spin.setToolTip("Position along the selected axis")
        form.addRow("Position:", self.position_spin)

        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 1000)
        self.thickness_spin.setDecimals(1)
        self.thickness_spin.setValue(10)
        self.thickness_spin.setToolTip("Slice thickness (± half from plane)")
        form.addRow("Thickness:", self.thickness_spin)

        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Optional description…")
        form.addRow("Description:", self.description_edit)

        group.add_layout(form)

        # Interactive + clear row
        btn_row = QHBoxLayout()
        self.btn_interactive = QPushButton("Position Plane in 3D View")
        self.btn_interactive.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_interactive.setToolTip("Interactively position the cutting plane in the 3D view")
        self.btn_interactive.clicked.connect(self._on_interactive_plane_positioning)
        btn_row.addWidget(self.btn_interactive)

        self.btn_clear_plane = QPushButton("Clear Plane")
        self.btn_clear_plane.setToolTip("Remove the positioning plane from the 3D view")
        self.btn_clear_plane.clicked.connect(self._on_clear_plane_overlay)
        btn_row.addWidget(self.btn_clear_plane)
        group.add_layout(btn_row)

        self.btn_create_section = QPushButton("Create Section")
        self.btn_create_section.setToolTip("Add section to library")
        self.btn_create_section.clicked.connect(self._on_create_section)
        group.add_widget(self.btn_create_section)

        return group

    # ------------------------------------------------------------------
    # Saved sections list
    # ------------------------------------------------------------------

    def _create_sections_list_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Saved Sections", collapsed=False)

        self.sections_list = QListWidget()
        self.sections_list.setMaximumHeight(120)
        self.sections_list.setToolTip("Click to inspect, double-click to load into form")
        self.sections_list.itemClicked.connect(self._on_section_selected)
        self.sections_list.itemDoubleClicked.connect(self._on_load_to_form_item)
        group.add_widget(self.sections_list)

        self.section_info_label = QLabel("No section selected")
        self.section_info_label.setWordWrap(True)
        self.section_info_label.setStyleSheet(
            "background-color: #2b2b2b; padding: 5px; border-radius: 3px; font-size: 9pt;"
        )
        group.add_widget(self.section_info_label)

        btn_row = QHBoxLayout()
        self.btn_load_section = QPushButton("Load to Form")
        self.btn_load_section.setToolTip("Load selected section into form for editing")
        self.btn_load_section.clicked.connect(self._on_load_to_form)
        btn_row.addWidget(self.btn_load_section)

        self.btn_delete_section = QPushButton("Delete")
        self.btn_delete_section.setToolTip("Delete selected section")
        self.btn_delete_section.clicked.connect(self._on_delete_section)
        btn_row.addWidget(self.btn_delete_section)

        group.add_layout(btn_row)
        return group

    # ------------------------------------------------------------------
    # Render & export
    # ------------------------------------------------------------------

    def _create_render_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Render & Export", collapsed=True)

        # Block model combo
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Block Model:"))
        self.block_model_combo = QComboBox()
        self.block_model_combo.setToolTip("Select block model for cross-sectioning")
        self.block_model_combo.currentIndexChanged.connect(self._on_block_model_selected)
        model_row.addWidget(self.block_model_combo)
        group.add_layout(model_row)

        # Property combo — non-editable
        prop_row = QHBoxLayout()
        prop_row.addWidget(QLabel("Property:"))
        self.render_property_combo = QComboBox()
        self.render_property_combo.setEditable(False)
        self.render_property_combo.setToolTip("Property to visualise in the section")
        prop_row.addWidget(self.render_property_combo)
        group.add_layout(prop_row)

        self.render_as_cubes = QCheckBox("Render as cubes (match block model)")
        self.render_as_cubes.setChecked(True)
        group.add_widget(self.render_as_cubes)

        self.btn_render = QPushButton("Render in 3D View")
        self.btn_render.clicked.connect(self._on_render_section)
        group.add_widget(self.btn_render)

        export_row = QHBoxLayout()
        self.btn_export_data = QPushButton("Export Data (CSV)")
        self.btn_export_data.clicked.connect(self._on_export_data)
        export_row.addWidget(self.btn_export_data)

        self.btn_export_image = QPushButton("Export Image")
        self.btn_export_image.clicked.connect(self._on_export_image)
        export_row.addWidget(self.btn_export_image)
        group.add_layout(export_row)

        return group

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def _create_library_group(self) -> CollapsibleGroup:
        group = CollapsibleGroup("Section Library", collapsed=True)

        row = QHBoxLayout()
        self.btn_save_library = QPushButton("Save Library")
        self.btn_save_library.clicked.connect(self._on_save_library)
        row.addWidget(self.btn_save_library)

        self.btn_load_library = QPushButton("Load Library")
        self.btn_load_library.clicked.connect(self._on_load_library)
        row.addWidget(self.btn_load_library)
        group.add_layout(row)

        return group

    # ==================================================================
    # Enable / disable
    # ==================================================================

    def _set_controls_enabled(self, enabled: bool):
        for w in (
            self.btn_create_section, self.btn_interactive, self.btn_clear_plane,
            self.btn_load_section, self.btn_delete_section,
            self.btn_render, self.btn_export_data, self.btn_export_image,
            self.btn_save_library, self.btn_load_library,
        ):
            w.setEnabled(enabled)

    # ==================================================================
    # Status helpers
    # ==================================================================

    def _flash(self, msg: str):
        self.status_label.setText(msg)
        QTimer.singleShot(STATUS_FLASH_MS, self._restore_status)

    def _restore_status(self):
        n = len(self.section_manager.sections) if hasattr(self.section_manager, 'sections') else 0
        self.status_label.setText(
            f"{n} section(s) in library" if self._block_model is not None
            else "No block model loaded"
        )

    # ==================================================================
    # List helpers
    # ==================================================================

    def _selected_section_name(self) -> Optional[str]:
        """Return the real name stored in the currently selected list item."""
        item = self.sections_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _update_sections_list(self):
        self.sections_list.clear()
        for name, spec in self.section_manager.sections.items():
            display = f"{name} ({spec.plane_type}-plane @ {spec.position:.2f})"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.sections_list.addItem(item)

    # ==================================================================
    # Public API
    # ==================================================================

    def set_main_window(self, main_window):
        self.main_window = main_window

    def set_block_model(self, block_df, grid_spec=None):
        """Set block model data (called externally by main window)."""
        self.section_manager.set_block_model(block_df, grid_spec)

        # Build a filtered property list
        numeric_cols = block_df.select_dtypes(include=['number']).columns.tolist()
        exclude = {
            'X', 'Y', 'Z', 'x', 'y', 'z',
            'DX', 'DY', 'DZ', 'dx', 'dy', 'dz',
            'XINC', 'YINC', 'ZINC', 'xinc', 'yinc', 'zinc',
            'GLOBAL_INTERVAL_ID', 'global_interval_id',
            'IJK', 'ID', 'ijk', 'id', 'BlockID', 'blockid',
        }
        props = [c for c in numeric_cols if c not in exclude]
        self.available_properties = props

        self._populate_property_combo()

        # Default position to midpoint of current axis
        axis = self.plane_type_combo.currentText()
        if axis in block_df.columns:
            mid = (block_df[axis].min() + block_df[axis].max()) / 2
            self.position_spin.setValue(mid)

        self._block_model = block_df
        self._set_controls_enabled(True)
        self._flash(f"Block model loaded: {len(block_df):,} blocks, {len(props)} properties")

    def get_section_mesh(self, section_name: str, property_name: str):
        """Get PyVista mesh for a section (called by main viewer for rendering)."""
        as_cubes = self.render_as_cubes.isChecked() if hasattr(self, 'render_as_cubes') else True
        return self.section_manager.get_section_mesh(section_name, property_name, as_cubes=as_cubes)

    # ==================================================================
    # Plane type / position helpers
    # ==================================================================

    def _on_plane_type_changed(self, plane_type: str):
        if self.section_manager.block_df is None:
            return
        df = self.section_manager.block_df
        if plane_type in df.columns:
            mid = (df[plane_type].min() + df[plane_type].max()) / 2
            self.position_spin.setValue(mid)

    def _get_viewer(self):
        """Return the viewer_widget or None."""
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            return self.main_window.viewer_widget
        self._flash("Viewer not available")
        return None

    # ==================================================================
    # Interactive plane positioning
    # ==================================================================

    def _on_interactive_plane_positioning(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.start_plane_positioning_mode(
                axis=self.plane_type_combo.currentText(),
                position=self.position_spin.value(),
            )
            if not self._plane_signal_connected:
                viewer.plane_position_changed.connect(self._on_plane_moved)
                self._plane_signal_connected = True

            self._flash(
                "Drag the plane in the 3D view to position it. "
                "The position spinner will update as you drag."
            )
        except Exception as e:
            logger.error(f"Interactive positioning error: {e}", exc_info=True)
            self._flash(f"Error: {e}")

    def _on_clear_plane_overlay(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            if hasattr(viewer, 'clear_plane_overlay'):
                viewer.clear_plane_overlay()
            else:
                viewer.end_plane_positioning_mode()
            self._flash("Plane overlay cleared")
        except Exception as e:
            logger.error(f"Clear plane error: {e}")
            self._flash(f"Error: {e}")

    def _on_plane_moved(self, axis: str, position: float):
        if self.plane_type_combo.currentText() != axis:
            self.plane_type_combo.setCurrentText(axis)
        self.position_spin.setValue(position)

    # ==================================================================
    # Create section
    # ==================================================================

    def _on_create_section(self):
        name = self.section_name_edit.text().strip()
        if not name:
            self._flash("Enter a name for the section")
            return

        if name in self.section_manager.sections:
            reply = QMessageBox.question(
                self, "Section Exists",
                f"Section '{name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        success = self.section_manager.create_section(
            name=name,
            plane_type=self.plane_type_combo.currentText(),
            position=self.position_spin.value(),
            thickness=self.thickness_spin.value(),
            description=self.description_edit.text().strip(),
        )
        if success:
            self._update_sections_list()
            self.section_name_edit.clear()
            self.description_edit.clear()
            self._flash(
                f"Created '{name}' — "
                f"{self.plane_type_combo.currentText()}-plane @ {self.position_spin.value():.2f}"
            )

    # ==================================================================
    # Section list callbacks
    # ==================================================================

    def _on_section_selected(self, item: QListWidgetItem):
        name = item.data(Qt.ItemDataRole.UserRole)
        spec = self.section_manager.get_section(name)
        if spec:
            self.section_info_label.setText(
                f"Name: {spec.name}\n"
                f"Type: {spec.plane_type}-plane\n"
                f"Position: {spec.position:.2f}\n"
                f"Thickness: {spec.thickness:.2f}\n"
                f"Description: {spec.description or '–'}"
            )

    def _on_load_to_form_item(self, item: QListWidgetItem):
        """Double-click handler for the list."""
        self._load_section_to_form(item.data(Qt.ItemDataRole.UserRole))

    def _on_load_to_form(self):
        name = self._selected_section_name()
        if name:
            self._load_section_to_form(name)
        else:
            self._flash("Select a section first")

    def _load_section_to_form(self, name: str):
        spec = self.section_manager.get_section(name)
        if spec is None:
            return
        self.section_name_edit.setText(spec.name)
        self.plane_type_combo.setCurrentText(spec.plane_type)
        self.position_spin.setValue(spec.position)
        self.thickness_spin.setValue(spec.thickness)
        self.description_edit.setText(spec.description)
        self._flash(f"Loaded '{name}' into form")

    def _on_delete_section(self):
        name = self._selected_section_name()
        if name is None:
            self._flash("Select a section to delete")
            return
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete section '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.section_manager.delete_section(name)
            self._update_sections_list()
            self.section_info_label.setText("No section selected")
            self._flash(f"Deleted '{name}'")

    # ==================================================================
    # Render
    # ==================================================================

    def _on_render_section(self):
        name = self._selected_section_name()
        if name is None:
            self._flash("Select a section to render")
            return
        prop = self.render_property_combo.currentText()
        if not prop:
            self._flash("Select a property to visualise")
            return

        if self.signals:
            self.signals.crossSectionManagerRenderRequested.emit(name, prop)
        self.section_render_requested.emit(name, prop)
        self._flash(f"Render requested: '{name}' → {prop}")

    # ==================================================================
    # Export
    # ==================================================================

    def _on_export_data(self):
        name = self._selected_section_name()
        if name is None:
            self._flash("Select a section to export")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Section Data", f"{name}_data.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return
        if self.section_manager.export_section_csv(name, Path(filepath)):
            self._flash(f"Exported CSV → {Path(filepath).name}")
        else:
            self._flash("CSV export failed — section may be empty")

    def _on_export_image(self):
        name = self._selected_section_name()
        if name is None:
            self._flash("Select a section to export")
            return
        prop = self.render_property_combo.currentText()
        if not prop:
            self._flash("Select a property to visualise")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Section Image", f"{name}_{prop}.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;All Files (*)",
        )
        if not filepath:
            return
        if self.section_manager.export_section_image(name, Path(filepath), prop, dpi=150, figsize=(10, 8)):
            self._flash(f"Exported image → {Path(filepath).name}")
        else:
            self._flash("Image export failed — section may be empty")

    # ==================================================================
    # Library save/load
    # ==================================================================

    def _on_save_library(self):
        if not self.section_manager.sections:
            self._flash("No sections to save")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Section Library", "sections_library.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath and self.section_manager.save_sections_to_file(Path(filepath)):
            self._flash(f"Saved {len(self.section_manager.sections)} section(s) → {Path(filepath).name}")

    def _on_load_library(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Section Library", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath and self.section_manager.load_sections_from_file(Path(filepath)):
            self._update_sections_list()
            self._flash(f"Loaded {len(self.section_manager.sections)} section(s)")

    # ==================================================================
    # Theme
    # ==================================================================

    def refresh_theme(self) -> None:
        try:
            from .modern_styles import get_analysis_panel_stylesheet
            self.setStyleSheet(get_analysis_panel_stylesheet())
        except Exception:
            pass
