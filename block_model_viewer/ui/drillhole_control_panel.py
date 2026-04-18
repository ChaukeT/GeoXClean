"""
Drillhole Control Panel

Control panel for drillhole visualization settings.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Set

import logging

import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QScrollArea, QLineEdit, QFrame,
)

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea
from .signals import UISignals
from .modern_widgets import StatusBadge
from ..controllers.app_state import AppState, get_empty_state_message
from .panel_toolkit import (
    section, make_form, form_row, make_combo, action_button, button_row,
    SliderRow, hint_label,
    PANEL_MARGINS, PANEL_SPACING, SECTION_SPACING,
)

logger = logging.getLogger(__name__)


class DrillholeControlPanel(BaseDockPanel):
    # Legacy signals - DEPRECATED (DR-005 fix)
    # All new code should use UISignals bus via self.signals
    # These are kept only for backward compatibility with external code
    plot_drillholes = pyqtSignal(str)
    clear_drillholes = pyqtSignal()
    radius_changed = pyqtSignal(float)
    color_mode_changed = pyqtSignal(str)
    assay_field_changed = pyqtSignal(str)
    show_ids_toggled = pyqtSignal(bool)
    hole_visibility_changed = pyqtSignal(str, bool)
    focus_selected_requested = pyqtSignal()

    # PanelManager metadata
    PANEL_ID = "DrillholeControlPanel"
    PANEL_NAME = "Drillhole Control Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    def __init__(self, parent: Optional[QWidget] = None, signals: Optional[UISignals] = None):
        QWidget.__init__(self, parent)

        self._hole_ids: List[str] = []
        self._hole_checkboxes: Dict[str, QCheckBox] = {}

        # Lithology filter state
        self._unique_liths: List[str] = []
        self._lith_checkboxes: Dict[str, QCheckBox] = {}
        self._lith_filter_debounce_timer: Optional[QTimer] = None

        # DataFrame references for UI access (DR-006 note)
        self._collars_df: Optional[pd.DataFrame] = None
        self._assays_df: Optional[pd.DataFrame] = None
        self._composites_df: Optional[pd.DataFrame] = None

        self.signals: Optional[UISignals] = signals
        self.renderer = None

        # Application state tracking
        self._app_state: AppState = AppState.EMPTY

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        BaseDockPanel.__init__(self, parent)

        # Connect to registry
        registry = self.get_registry()
        if registry:
            try:
                registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                if registry.get_drillhole_data():
                    self._on_drillhole_data_loaded(registry.get_drillhole_data())
            except Exception as e:
                logger.error(f"Failed to connect drillhole data signal: {e}", exc_info=True)

    def setup_ui(self):
        self._init_ui()
        self._apply_empty_state()

    def _init_ui(self):
        # Create main scroll area for the entire panel content
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Content widget inside scroll area
        scroll_content = QWidget()
        scroll_content.setObjectName("panelContent")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(*PANEL_MARGINS)
        content_layout.setSpacing(PANEL_SPACING)

        # Header
        header = QHBoxLayout()
        header.addStretch()
        self.status_badge = StatusBadge("No Data", StatusBadge.State.NEUTRAL)
        header.addWidget(self.status_badge)
        content_layout.addLayout(header)

        # --- VISUALIZATION SETTINGS ---
        sec_viz = section("Visualization")
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(SECTION_SPACING)

        # Radius slider
        self.radius_slider = SliderRow(
            label="Radius", min_val=0.1, max_val=50.0, value=10.0,
            suffix="m", decimals=1, debounce_ms=200,
        )
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        viz_layout.addWidget(self.radius_slider)

        # Color Mode
        viz_form = make_form()
        self.color_mode_combo = make_combo(items=["Lithology", "Assay"])
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_changed)
        form_row(viz_form, "Color By:", self.color_mode_combo)

        # Assay Field
        self.assay_field_combo = make_combo()
        self.assay_field_combo.currentTextChanged.connect(self._on_assay_field_changed)
        form_row(viz_form, "Element:", self.assay_field_combo)

        # Dataset source
        self.dataset_combo = make_combo(items=["Raw Assays"])
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        form_row(viz_form, "Source:", self.dataset_combo)

        viz_layout.addLayout(viz_form)

        # Rendering toggles
        self.collar_check = QCheckBox("Collars")
        self.collar_check.setChecked(True)
        self.collar_check.setToolTip("Show white sphere markers at collar positions")
        self.collar_check.toggled.connect(self._on_collar_toggled)
        viz_layout.addWidget(self.collar_check)

        self.pbr_check = QCheckBox("Smooth shading")
        self.pbr_check.setChecked(True)
        self.pbr_check.setToolTip("PBR physically-based rendering for realistic lighting")
        self.pbr_check.toggled.connect(self._on_pbr_toggled)
        viz_layout.addWidget(self.pbr_check)

        self.ssao_check = QCheckBox("SSAO")
        self.ssao_check.setChecked(False)
        self.ssao_check.setToolTip("Screen-space ambient occlusion — soft shadows between holes")
        self.ssao_check.toggled.connect(self._on_ssao_toggled)
        viz_layout.addWidget(self.ssao_check)

        self.edl_check = QCheckBox("Edge glow (EDL)")
        self.edl_check.setChecked(False)
        self.edl_check.setToolTip("Eye-dome lighting — subtle edge halos for depth perception")
        self.edl_check.toggled.connect(self._on_edl_toggled)
        viz_layout.addWidget(self.edl_check)

        self.hide_barren_check = QCheckBox("Hide barren intervals")
        self.hide_barren_check.setChecked(True)
        self.hide_barren_check.setToolTip("Fade barren intervals (grade ≤ 0 or NaN) to 20% opacity and 40% radius")
        self.hide_barren_check.toggled.connect(self._on_hide_barren_toggled)
        viz_layout.addWidget(self.hide_barren_check)

        sec_viz.add_layout(viz_layout)
        content_layout.addWidget(sec_viz)

        # --- LITHOLOGY FILTER ---
        sec_lith = section("Lithology Filter")
        lith_layout = QVBoxLayout()
        lith_layout.setSpacing(SECTION_SPACING)

        # Lith filter buttons row
        btn_lith_all = action_button("All", style="secondary")
        btn_lith_all.clicked.connect(self._select_all_liths)

        btn_lith_none = action_button("None", style="secondary")
        btn_lith_none.clicked.connect(self._select_no_liths)

        lith_btn_row = QHBoxLayout()
        lith_btn_row.setSpacing(8)
        lith_btn_row.addWidget(btn_lith_all)
        lith_btn_row.addWidget(btn_lith_none)
        lith_btn_row.addStretch()
        lith_layout.addLayout(lith_btn_row)

        # Scrollable lithology checkbox list
        lith_scroll = QScrollArea()
        lith_scroll.setWidgetResizable(True)
        lith_scroll.setMinimumHeight(80)
        lith_scroll.setMaximumHeight(150)
        lith_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.lith_container = QWidget()
        self.lith_layout = QVBoxLayout(self.lith_container)
        self.lith_layout.setSpacing(4)
        self.lith_layout.setContentsMargins(8, 8, 8, 8)
        self.lith_layout.addStretch()
        lith_scroll.setWidget(self.lith_container)

        lith_layout.addWidget(lith_scroll)

        # Status label for lithology filter
        self.lith_status_label = hint_label("Load data to see lithologies")
        self.lith_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lith_layout.addWidget(self.lith_status_label)

        sec_lith.add_layout(lith_layout)
        content_layout.addWidget(sec_lith)

        # --- SELECTION ---
        sec_sel = section("Hole Selection")
        sel_layout = QVBoxLayout()
        sel_layout.setSpacing(SECTION_SPACING)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter holes...")
        self.search_bar.textChanged.connect(self._filter_holes)
        sel_layout.addWidget(self.search_bar)

        # Select buttons row
        btn_all = action_button("Select All", style="secondary")
        btn_all.clicked.connect(self._select_all_holes)

        btn_none = action_button("Deselect All", style="secondary")
        btn_none.clicked.connect(self._select_no_holes)

        sel_layout.addLayout(button_row(btn_all, btn_none))

        # Show labels checkbox
        self.show_ids_checkbox = QCheckBox("Show Hole Labels")
        self.show_ids_checkbox.toggled.connect(self._on_show_ids_toggled)
        sel_layout.addWidget(self.show_ids_checkbox)

        # Scrollable hole list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(200)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.holes_container = QWidget()
        self.holes_layout = QVBoxLayout(self.holes_container)
        self.holes_layout.setSpacing(4)
        self.holes_layout.setContentsMargins(8, 8, 8, 8)
        self.holes_layout.addStretch()
        scroll.setWidget(self.holes_container)

        sel_layout.addWidget(scroll)

        sec_sel.add_layout(sel_layout)
        content_layout.addWidget(sec_sel)

        # --- ACTION BUTTONS ---
        content_layout.addStretch()

        self.plot_button = action_button("Plot 3D", style="primary")
        self.plot_button.clicked.connect(self._on_plot_clicked)

        self.clear_button = action_button("Clear", style="secondary")
        self.clear_button.clicked.connect(self._on_clear_clicked)

        content_layout.addLayout(button_row(self.plot_button, self.clear_button))

        # Status label
        self.status_label = hint_label("Load drillhole data to begin")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.status_label)

        # Set content widget in scroll area
        main_scroll.setWidget(scroll_content)

        # Add scroll area to main layout
        self.main_layout.addWidget(main_scroll)

    # --- LOGIC ---

    def _on_drillhole_data_loaded(self, data: Dict):
        """Fast population using Pandas - deferred to prevent UI freeze."""
        logger.info(f"DrillholeControlPanel._on_drillhole_data_loaded called with data type: {type(data)}")
        logger.info(f"Data keys: {data.keys() if isinstance(data, dict) else 'NOT A DICT'}")

        if not isinstance(data, dict):
            logger.error(f"Data is not a dict! Type: {type(data)}")
            return

        def _do_load():
            try:
                logger.info("_do_load() started")
                self._collars_df = data.get('collars')
                self._assays_df = data.get('assays')
                # AUDIT FIX: Can't use 'or' with DataFrames - use proper None check
                composites = data.get('composites')
                if composites is None:
                    composites = data.get('composites_df')
                self._composites_df = composites

                logger.info(f"Loaded DataFrames:")
                logger.info(f"   - Collars: {len(self._collars_df) if self._collars_df is not None else 'None'}")
                logger.info(f"   - Assays: {len(self._assays_df) if self._assays_df is not None else 'None'}")
                logger.info(f"   - Composites: {len(self._composites_df) if self._composites_df is not None else 'None'}")

                # Update Dataset Combo
                current_selection = self.dataset_combo.currentText() if self.dataset_combo.count() > 0 else None
                self.dataset_combo.blockSignals(True)
                self.dataset_combo.clear()
                available_datasets = []

                if self._assays_df is not None and not self._assays_df.empty:
                    self.dataset_combo.addItem("Raw Assays")
                    available_datasets.append("Raw Assays")
                if self._composites_df is not None and not self._composites_df.empty:
                    self.dataset_combo.addItem("Composites")
                    available_datasets.append("Composites")

                if current_selection in available_datasets:
                    self.dataset_combo.setCurrentText(current_selection)
                elif available_datasets:
                    self.dataset_combo.setCurrentIndex(0)
                else:
                    self.dataset_combo.addItem("No Data")

                self.dataset_combo.blockSignals(False)

                # Populate Holes
                if self._collars_df is not None and not self._collars_df.empty:
                    hole_col = next((c for c in self._collars_df.columns if c.lower() in ['holeid','hole_id','bhid']), None)
                    if hole_col:
                        self._hole_ids = sorted(self._collars_df[hole_col].astype(str).unique().tolist())
                        self._rebuild_checklist()
                        self.status_label.setText(f"{len(self._hole_ids)} holes available")
                        self.status_badge.setText(f"{len(self._hole_ids)} Holes")
                        self.status_badge.set_state(StatusBadge.State.SUCCESS)

                self._populate_assay_fields()

                # Extract unique lithology codes for filtering
                self._unique_liths = []
                unique_liths_set = set()

                # Source 1: Dedicated lithology DataFrame
                lith_df = data.get('lithology')
                if lith_df is not None and not lith_df.empty:
                    lith_col = None
                    for col in lith_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            lith_col = col
                            break

                    if lith_col:
                        unique_liths_set.update(
                            lith_df[lith_col].dropna().astype(str).unique().tolist()
                        )

                # Source 2: Lithology column in assays DataFrame
                assays_df = data.get('assays')
                if assays_df is not None and not assays_df.empty:
                    for col in assays_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            unique_liths_set.update(
                                assays_df[col].dropna().astype(str).unique().tolist()
                            )
                            break

                # Source 3: Lithology column in composites DataFrame
                composites_df = data.get('composites')
                if composites_df is None:
                    composites_df = data.get('composites_df')
                if composites_df is not None and not composites_df.empty:
                    for col in composites_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            unique_liths_set.update(
                                composites_df[col].dropna().astype(str).unique().tolist()
                            )
                            break

                # Convert to sorted list and filter out empty strings
                self._unique_liths = sorted([lith for lith in unique_liths_set if lith and str(lith).strip()])

                self._rebuild_lith_checklist()

                # Auto-select color mode based on available data
                has_lith_data = len(self._unique_liths) > 0
                has_assay_data = self.assay_field_combo.count() > 0

                if not has_lith_data and has_assay_data:
                    self.color_mode_combo.blockSignals(True)
                    self.color_mode_combo.setCurrentText("Assay")
                    self.color_mode_combo.blockSignals(False)
                    logger.info("Auto-selected Assay color mode (no lithology data found)")
                elif has_lith_data:
                    self.color_mode_combo.blockSignals(True)
                    self.color_mode_combo.setCurrentText("Lithology")
                    self.color_mode_combo.blockSignals(False)
                    logger.info(f"Auto-selected Lithology color mode ({len(self._unique_liths)} codes found)")

                logger.info("_do_load() COMPLETED SUCCESSFULLY")

            except Exception as e:
                logger.error(f"Error loading drillhole data: {e}", exc_info=True)

        QTimer.singleShot(10, _do_load)

    def _rebuild_checklist(self):
        """Create checkboxes for holes (limited to 100)."""
        for i in reversed(range(self.holes_layout.count())):
            w = self.holes_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        self._hole_checkboxes.clear()
        self.holes_layout.addStretch()

        display_limit = 100

        for hid in self._hole_ids[:display_limit]:
            cb = QCheckBox(hid)
            cb.setChecked(True)
            cb.toggled.connect(lambda c, h=hid: self._on_hole_visibility_changed(h, c))
            self.holes_layout.insertWidget(self.holes_layout.count()-1, cb)
            self._hole_checkboxes[hid] = cb

        if len(self._hole_ids) > display_limit:
            lbl = hint_label(f"... +{len(self._hole_ids) - display_limit} more (use search)")
            self.holes_layout.insertWidget(self.holes_layout.count()-1, lbl)

    def _populate_assay_fields(self):
        """Populate assay fields based on selected dataset."""
        dataset_name = self.dataset_combo.currentText()
        df = self._composites_df if dataset_name == "Composites" else self._assays_df

        if df is None or df.empty:
            self.assay_field_combo.clear()
            return

        ignore = {
            'holeid','hole_id','from','to','x','y','z','length','depth_from','depth_to','global_interval_id',
            'sample_count','total_mass','total_length','support','is_partial',
            'method','weighting','element_weights','merged_partial','merged_partial_auto'
        }
        cols = [c for c in df.columns if c.lower() not in ignore and pd.api.types.is_numeric_dtype(df[c])]

        self.assay_field_combo.blockSignals(True)
        self.assay_field_combo.clear()
        self.assay_field_combo.addItems(sorted(cols))
        self.assay_field_combo.blockSignals(False)

    def _filter_holes(self, text):
        text = text.lower()
        for hid, cb in self._hole_checkboxes.items():
            cb.setVisible(text in hid.lower())

    # --- Lithology Filter Methods ---

    def _select_all_liths(self):
        """Select all lithology types."""
        for cb in self._lith_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._emit_lith_filter_signal()

    def _select_no_liths(self):
        """Deselect all lithology types."""
        for cb in self._lith_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._emit_lith_filter_signal()

    def _on_lith_visibility_changed(self, lith_code: str, visible: bool):
        """Handle lithology checkbox toggle - debounced signal emission."""
        if self._lith_filter_debounce_timer is None:
            self._lith_filter_debounce_timer = QTimer(self)
            self._lith_filter_debounce_timer.setSingleShot(True)
            self._lith_filter_debounce_timer.setInterval(500)
            self._lith_filter_debounce_timer.timeout.connect(self._emit_lith_filter_signal)

        self._lith_filter_debounce_timer.start()

    def _emit_lith_filter_signal(self):
        """Emit the lithology filter signal after debounce delay."""
        selected = self.get_selected_lithologies()
        if self.signals:
            self.signals.drillholeLithFilterChanged.emit(selected)

    def _rebuild_lith_checklist(self):
        """Create checkboxes for unique lithology codes."""
        for i in reversed(range(self.lith_layout.count())):
            w = self.lith_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        self._lith_checkboxes.clear()
        self.lith_layout.addStretch()

        if not self._unique_liths:
            self.lith_status_label.setText("No lithology data available")
            return

        for lith in self._unique_liths:
            cb = QCheckBox(lith)
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            cb.toggled.connect(lambda c, l=lith: self._on_lith_visibility_changed(l, c))
            self.lith_layout.insertWidget(self.lith_layout.count() - 1, cb)
            self._lith_checkboxes[lith] = cb

        self.lith_status_label.setText(f"{len(self._unique_liths)} lithology types")

    def _select_all_holes(self):
        for cb in self._hole_checkboxes.values():
            if cb.isVisible():
                cb.setChecked(True)

    def _select_no_holes(self):
        for cb in self._hole_checkboxes.values():
            if cb.isVisible():
                cb.setChecked(False)

    def _on_plot_clicked(self):
        ds = self.dataset_combo.currentText()
        if self.signals:
            self.signals.drillholePlotRequested.emit(ds)
        else:
            self.plot_drillholes.emit(ds)

    def _on_clear_clicked(self):
        if self.signals:
            self.signals.drillholeClearRequested.emit()
        else:
            self.clear_drillholes.emit()

    def _on_radius_changed(self, radius: float):
        if self.signals:
            self.signals.drillholeRadiusChanged.emit(radius)
        else:
            self.radius_changed.emit(radius)

    def _on_color_mode_changed(self, mode):
        if self.signals:
            self.signals.drillholeColorModeChanged.emit(mode)
        else:
            self.color_mode_changed.emit(mode)

    def _on_dataset_changed(self, dataset_name: str):
        self._populate_assay_fields()
        if self.signals:
            self.signals.drillholePlotRequested.emit(dataset_name)
        else:
            self.plot_drillholes.emit(dataset_name)

    def _on_assay_field_changed(self, field):
        if field:
            # Auto-switch to Assay mode when an element is selected
            if hasattr(self, 'color_mode_combo') and self.color_mode_combo.currentText() != "Assay":
                self.color_mode_combo.blockSignals(True)
                self.color_mode_combo.setCurrentText("Assay")
                self.color_mode_combo.blockSignals(False)
            if self.signals:
                self.signals.drillholeAssayFieldChanged.emit(field)
            else:
                self.assay_field_changed.emit(field)

    def _on_show_ids_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholeShowIdsToggled.emit(checked)
        else:
            self.show_ids_toggled.emit(checked)

    def _on_collar_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholeCollarToggled.emit(checked)

    def _on_pbr_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholePbrToggled.emit(checked)

    def _on_ssao_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholeSsaoToggled.emit(checked)

    def _on_edl_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholeEdlToggled.emit(checked)

    def _on_hide_barren_toggled(self, checked: bool):
        if self.signals:
            self.signals.drillholeHideBarrenToggled.emit(checked)

    def _on_hole_visibility_changed(self, hole_id: str, visible: bool):
        if self.signals:
            self.signals.drillholeVisibilityChanged.emit(hole_id, visible)
        else:
            self.hole_visibility_changed.emit(hole_id, visible)

    def _on_focus_requested(self):
        if self.signals:
            self.signals.drillholeFocusRequested.emit()
        else:
            self.focus_selected_requested.emit()

    # Public getters
    def get_visible_holes(self) -> Set[str]:
        """Get set of hole IDs that should be visible."""
        checked_holes = {h for h, cb in self._hole_checkboxes.items() if cb.isChecked()}

        display_limit = 100
        if len(self._hole_ids) > display_limit:
            holes_without_checkboxes = set(self._hole_ids[display_limit:])
            checked_holes = checked_holes.union(holes_without_checkboxes)

        return checked_holes

    def get_color_mode(self):
        return self.color_mode_combo.currentText()

    def get_assay_field(self):
        return self.assay_field_combo.currentText()

    def get_radius(self) -> float:
        return self.radius_slider.value()

    def get_dataset(self) -> str:
        return self.dataset_combo.currentText() if self.dataset_combo.count() > 0 else "Raw Assays"

    def get_selected_lithologies(self) -> List[str]:
        """Get list of selected lithology codes for filtering."""
        if not self._lith_checkboxes:
            return []

        selected = [lith for lith, cb in self._lith_checkboxes.items() if cb.isChecked()]

        if len(selected) == len(self._lith_checkboxes):
            return []

        if len(selected) == 0:
            return ["__NONE__"]

        return selected

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_renderer(self, renderer):
        self.renderer = renderer

    # =========================================================================
    # Application State Handling
    # =========================================================================

    def on_app_state_changed(self, state: int) -> None:
        """Handle application state changes."""
        try:
            new_state = AppState(state)
        except ValueError:
            logger.warning(f"Invalid app state value: {state}")
            return

        if self._app_state == new_state:
            return

        old_state = self._app_state
        self._app_state = new_state
        logger.debug(f"DrillholeControlPanel: State changed {old_state.name} -> {new_state.name}")

        if new_state == AppState.EMPTY:
            self._apply_empty_state()
        elif new_state == AppState.DATA_LOADED:
            self._apply_data_loaded_state()
        elif new_state == AppState.RENDERED:
            self._apply_rendered_state()
        elif new_state == AppState.BUSY:
            self._apply_busy_state()

    def _apply_empty_state(self) -> None:
        """Apply EMPTY state: Disable all controls, show helpful message."""
        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(False)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(False)
        if hasattr(self, 'assay_field_combo'):
            self.assay_field_combo.setEnabled(False)
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(False)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(False)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(False)
        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(False)
        if hasattr(self, 'focus_button'):
            self.focus_button.setEnabled(False)
        for attr in ('collar_check', 'pbr_check', 'ssao_check', 'edl_check', 'hide_barren_check'):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(False)

        for cb in self._lith_checkboxes.values():
            cb.setEnabled(False)

        if hasattr(self, 'status_badge'):
            self.status_badge.setText("No Data")
            self.status_badge.set_state(StatusBadge.State.NEUTRAL)

        if hasattr(self, 'status_label'):
            self.status_label.setText(get_empty_state_message("drillhole_controls"))

    def _apply_data_loaded_state(self) -> None:
        """Apply DATA_LOADED state: Enable controls for configuring before plotting."""
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(True)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(True)

        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(True)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(True)
        if hasattr(self, 'assay_field_combo'):
            self.assay_field_combo.setEnabled(True)

        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(True)
        for attr in ('collar_check', 'pbr_check', 'ssao_check', 'edl_check', 'hide_barren_check'):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(True)

        for cb in self._lith_checkboxes.values():
            cb.setEnabled(True)

        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(True)

    def _apply_rendered_state(self) -> None:
        """Apply RENDERED state: Enable all controls."""
        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(True)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(True)
        if hasattr(self, 'assay_field_combo'):
            self.assay_field_combo.setEnabled(True)
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(True)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(True)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(True)
        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(True)
        if hasattr(self, 'focus_button'):
            self.focus_button.setEnabled(True)
        for attr in ('collar_check', 'pbr_check', 'ssao_check', 'edl_check', 'hide_barren_check'):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(True)

        for cb in self._lith_checkboxes.values():
            cb.setEnabled(True)

    def _apply_busy_state(self) -> None:
        """Apply BUSY state: Disable interactive controls during processing."""
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(False)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(False)

    # =========================================================================
    # Project Save/Restore
    # =========================================================================

    def get_panel_settings(self):
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            settings = {}
            settings['radius'] = self.radius_slider.value() if hasattr(self, 'radius_slider') else None
            settings['color_mode'] = get_safe_widget_value(self, 'color_mode_combo')
            settings['assay_field'] = get_safe_widget_value(self, 'assay_field_combo')
            settings['dataset'] = get_safe_widget_value(self, 'dataset_combo')
            settings['show_ids'] = self.show_ids_checkbox.isChecked() if hasattr(self, 'show_ids_checkbox') else None
            settings['collar_visible'] = self.collar_check.isChecked() if hasattr(self, 'collar_check') else None
            settings['pbr_enabled'] = self.pbr_check.isChecked() if hasattr(self, 'pbr_check') else None
            settings['ssao_enabled'] = self.ssao_check.isChecked() if hasattr(self, 'ssao_check') else None
            settings['edl_enabled'] = self.edl_check.isChecked() if hasattr(self, 'edl_check') else None
            settings['hide_barren'] = self.hide_barren_check.isChecked() if hasattr(self, 'hide_barren_check') else None
            settings = {k: v for k, v in settings.items() if v is not None}
            return settings if settings else None
        except Exception as e:
            logger.warning(f"Could not save drillhole control panel settings: {e}")
            return None

    def apply_panel_settings(self, settings):
        """Apply panel settings from project load."""
        if not settings:
            return
        try:
            from .panel_settings_utils import set_safe_widget_value
            if 'radius' in settings and hasattr(self, 'radius_slider'):
                self.radius_slider.setValue(settings['radius'])
            set_safe_widget_value(self, 'color_mode_combo', settings.get('color_mode'))
            set_safe_widget_value(self, 'dataset_combo', settings.get('dataset'))
            set_safe_widget_value(self, 'assay_field_combo', settings.get('assay_field'))
            if 'show_ids' in settings and hasattr(self, 'show_ids_checkbox'):
                self.show_ids_checkbox.setChecked(settings['show_ids'])
            if 'collar_visible' in settings and hasattr(self, 'collar_check'):
                self.collar_check.setChecked(settings['collar_visible'])
            if 'pbr_enabled' in settings and hasattr(self, 'pbr_check'):
                self.pbr_check.setChecked(settings['pbr_enabled'])
            if 'ssao_enabled' in settings and hasattr(self, 'ssao_check'):
                self.ssao_check.setChecked(settings['ssao_enabled'])
            if 'edl_enabled' in settings and hasattr(self, 'edl_check'):
                self.edl_check.setChecked(settings['edl_enabled'])
            if 'hide_barren' in settings and hasattr(self, 'hide_barren_check'):
                self.hide_barren_check.setChecked(settings['hide_barren'])
            logger.info("Restored drillhole control panel settings from project")
        except Exception as e:
            logger.warning(f"Could not restore drillhole control panel settings: {e}")
