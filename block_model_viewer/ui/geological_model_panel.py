"""
Geological Model Panel — Implicit geological modelling interface.
==================================================================

6-tab panel for building implicit geological models from drillhole
contacts, structural measurements, and fault definitions.

Tabs:
  Data & Grouping  — auto-populated from registry, popup grouping dialog
  Setup & Interp   — model type, grid, continuity, advanced kernel
  Build            — resolution presets, run, progress, QC summary
  Faults           — fault list with popup editor dialog
  Domains          — domain assignment to block model
  Export           — surfaces, domains, audit record

Design principles:
  1. NO FILE LOADING — all data comes from the DataRegistry
  2. NO REDUNDANT DATA LOADING — contacts/structural already in registry
  3. POPUP DIALOGS — for lithology grouping, strat column, fault editing

Uses the same CollapsibleGroup / panel_toolkit / design_tokens house
style as ARBF, Variogram, and FastRBF panels.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QScrollArea,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from .base_analysis_panel import BaseAnalysisPanel
from .modern_styles import ModernColors
from .panel_manager import PanelCategory, DockArea
from .panel_toolkit import (
    section, make_form, form_row, make_combo, make_spin,
    action_button, hint_label,
    PANEL_MARGINS, PANEL_SPACING,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Tab scroll helper
# ═══════════════════════════════════════════════════════════════════

def _make_tab(tab_widget: QTabWidget, name: str) -> Tuple[QVBoxLayout, QWidget]:
    """Create a scrollable tab with consistent margins."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    content = QWidget()
    lay = QVBoxLayout(content)
    lay.setContentsMargins(*PANEL_MARGINS)
    lay.setSpacing(PANEL_SPACING)
    scroll.setWidget(content)
    tab_widget.addTab(scroll, name)
    return lay, content


# ═══════════════════════════════════════════════════════════════════
# Model type descriptions
# ═══════════════════════════════════════════════════════════════════

_MODEL_DESCRIPTIONS = {
    "stratiform": "Layered deposits (BIF, coal, manganese). Parallel surfaces, fold support.",
    "vein": "Narrow tabular bodies (gold veins, shear zones). Separate HW/FW surfaces.",
    "intrusive": "Closed bodies (porphyry, VMS, kimberlite). Single contact surface.",
    "structural": "Fold-controlled deposits. Requires structural measurements.",
}


# ═══════════════════════════════════════════════════════════════════
# Worker thread for model building
# ═══════════════════════════════════════════════════════════════════

class _ModelBuildWorker(QThread):
    """Run GeologicalModelBuilder.build() off the UI thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self._builder = builder

    def run(self):
        try:
            self._builder.set_progress_callback(
                lambda pct, msg: self.progress.emit(pct, msg),
            )
            result = self._builder.build()
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Model build failed")
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════
# Geological Model Panel
# ═══════════════════════════════════════════════════════════════════

class GeologicalModelPanel(BaseAnalysisPanel):
    """Implicit geological modelling panel.

    Integrates Phases 1-5 (scalar field, stratigraphy, veins, faults,
    fold frames) into a single workflow panel.

    All data comes from the DataRegistry — no file loading.
    Complex editing via popup QDialogs.
    """

    PANEL_ID = "GeologicalModelPanel"
    PANEL_NAME = "Geological Model"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    task_name = "geological_model"
    request_visualization = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._contacts_df: Optional[pd.DataFrame] = None
        self._orientations_df: Optional[pd.DataFrame] = None
        self._surveys_df: Optional[pd.DataFrame] = None
        self._build_result: Optional[Dict[str, Any]] = None
        self._worker: Optional[_ModelBuildWorker] = None
        self._lithology_grouping: Optional[Dict[str, List[str]]] = None
        self._strat_column: Optional[List[Dict[str, str]]] = None
        self._unit_order: List[str] = []
        self._fault_definitions: List[Dict[str, Any]] = []
        self._drillhole_data: Optional[Dict[str, Any]] = None
        self._lith_df: Optional[pd.DataFrame] = None   # full interval log for consistency check
        self._lith_col: Optional[str] = None
        self._registry_signals_connected = False

        super().__init__(parent=parent, panel_id="geological_model")

        # Deferred registry signal connection
        QTimer.singleShot(0, self._connect_registry_signals)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_base_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        root.addWidget(self.tab_widget, stretch=1)

        self._build_tab_data()
        self._build_tab_setup()
        self._build_tab_build()
        self._build_tab_faults()
        self._build_tab_domains()
        self._build_tab_export()

        self._is_initialized = True

    # ------------------------------------------------------------------
    # Header bar
    # ------------------------------------------------------------------

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Card")
        frame.setStyleSheet(
            f"QFrame#Card {{"
            f"  background-color: {ModernColors.ELEVATED_BG};"
            f"  border-bottom: 1px solid {ModernColors.DIVIDER};"
            f"}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        title = QLabel("Geological Model")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        lay.addWidget(title)
        lay.addStretch()

        sub = QLabel("Implicit  |  Gradient RBF  |  JORC 2012")
        sub.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        lay.addWidget(sub)
        lay.addStretch()

        self.header_status = QLabel("Ready")
        self.header_status.setObjectName("statusLabel")
        lay.addWidget(self.header_status)

        return frame

    def _set_status(self, text: str) -> None:
        self.header_status.setText(text)

    # ------------------------------------------------------------------
    # Registry integration (validation panel pattern)
    # ------------------------------------------------------------------

    def _connect_registry_signals(self) -> None:
        """Connect to DataRegistry drillhole signals (deferred via QTimer)."""
        if self._registry_signals_connected:
            return
        try:
            registry = self.get_registry()
        except Exception:
            return
        if registry is None:
            return

        if hasattr(registry, 'drillholeDataLoaded'):
            try:
                registry.drillholeDataLoaded.connect(self._on_registry_drillhole_loaded)
            except Exception:
                pass
        if hasattr(registry, 'compositesLoaded'):
            try:
                registry.compositesLoaded.connect(self._on_registry_composites_loaded)
            except Exception:
                pass

        self._registry_signals_connected = True

        # If data is already in the registry, load it NOW
        try:
            dh_data = registry.get_drillhole_data()
            if dh_data is not None:
                self._on_registry_drillhole_loaded(dh_data)
        except Exception:
            pass

    def _on_registry_drillhole_loaded(self, data) -> None:
        """Called automatically when drillhole data is loaded/updated in registry."""
        logger.info("GeologicalModelPanel: drillhole data received from registry")
        self._drillhole_data = data
        QTimer.singleShot(10, lambda: self._do_populate(data))

    def _on_registry_composites_loaded(self, composites_df) -> None:
        """Called when composites become available in the registry."""
        logger.info("GeologicalModelPanel: composites received (%d rows)",
                     len(composites_df) if hasattr(composites_df, '__len__') else 0)
        if self._contacts_df is None and isinstance(composites_df, pd.DataFrame):
            if any(c.lower() in ('lithology', 'lith_code', 'rock_type', 'geology', 'unit')
                   for c in composites_df.columns):
                self._drillhole_data = {'composites': composites_df}
                QTimer.singleShot(10, lambda: self._do_populate(self._drillhole_data))

    def _do_populate(self, dh_data) -> None:
        """Heavy data extraction — runs on deferred tick to keep UI responsive.

        1. blockSignals on combos during population
        2. Case-insensitive column detection
        3. Auto-build lithology grouping
        4. Auto-extract contacts
        5. Auto-extract structural orientations from registry
        6. Update status labels
        """
        if dh_data is None:
            return

        try:
            # ── Resolve data tables from dict ──────────────────────
            collars_df = None
            lith_df = None
            assays_df = None
            composites_df = None
            structural_df = None
            surveys_df = None
            df_for_xyz = None

            if isinstance(dh_data, dict):
                logger.info("Registry drillhole keys: %s", list(dh_data.keys()))

                collars_df = dh_data.get('collars')
                surveys_df = dh_data.get('surveys')
                assays_df = dh_data.get('assays')
                _comp = dh_data.get('composites')
                composites_df = _comp if _comp is not None else dh_data.get('composites_df')
                _struct = dh_data.get('structures')
                structural_df = _struct if _struct is not None else dh_data.get('structural')

                # Find lithology data
                for key in ('lithology', 'geology', 'lithology_df'):
                    candidate = dh_data.get(key)
                    if isinstance(candidate, pd.DataFrame) and len(candidate) > 0:
                        lith_df = candidate
                        break

                # Find best table with X, Y, Z for contacts
                for key in ('composites', 'assays', 'intervals', 'lithology',
                            'survey', 'collar', 'collars'):
                    candidate = dh_data.get(key)
                    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
                        continue
                    col_lower = {c.lower() for c in candidate.columns}
                    if col_lower & {'x', 'easting'} and col_lower & {'y', 'northing'} and col_lower & {'z', 'elevation'}:
                        df_for_xyz = candidate
                        break

            elif isinstance(dh_data, pd.DataFrame):
                df_for_xyz = dh_data
                lith_df = dh_data

            # ── Count holes (case-insensitive) ─────────────────────
            n_holes = '?'
            check_df = collars_df if collars_df is not None else df_for_xyz
            if check_df is not None:
                for col in check_df.columns:
                    if col.lower() in ('holeid', 'hole_id', 'bhid', 'drill_hole_id'):
                        n_holes = check_df[col].nunique()
                        break

            # ── Auto-detect lithology column (case-insensitive) ────
            lith_col = None
            search_df = lith_df if lith_df is not None else df_for_xyz
            if search_df is None and composites_df is not None:
                search_df = composites_df
            if search_df is None and assays_df is not None:
                search_df = assays_df

            if search_df is not None:
                col_lower_map = {c.lower(): c for c in search_df.columns}
                for candidate in ('lithology', 'lith_code', 'lithcode', 'lith',
                                  'formation', 'rock_type', 'rocktype', 'geology',
                                  'unit', 'geo_unit', 'rock', 'litho', 'rock_code',
                                  'geo_code', 'code'):
                    if candidate in col_lower_map:
                        lith_col = col_lower_map[candidate]
                        break

            # Store lithology table for interval consistency check (run after build)
            if search_df is not None and lith_col:
                self._lith_df = search_df
                self._lith_col = lith_col

            # ── Build lithology grouping from unique values ────────
            if lith_col and search_df is not None:
                unique_liths = search_df[lith_col].dropna().astype(str).unique()
                unique_liths = [v for v in unique_liths if v.strip()]
                if len(unique_liths) > 0:
                    grouping = {v: [v] for v in sorted(unique_liths)}
                    self._lithology_grouping = grouping
                    self._unit_order = list(grouping.keys())
                    self._update_grouping_display()
                    self._set_status(f"{len(unique_liths)} units | {n_holes} holes")
                    logger.info("Auto-built lithology grouping: %d units from '%s'",
                                len(unique_liths), lith_col)

            # ── Also scan assays/composites for lithology codes ────
            if lith_col is None:
                for extra_df in [assays_df, composites_df]:
                    if extra_df is None or extra_df.empty:
                        continue
                    for col in extra_df.columns:
                        if col.lower() in ('lith_code', 'lithology', 'code', 'lith'):
                            lith_col = col
                            search_df = extra_df
                            unique_liths = extra_df[col].dropna().astype(str).unique()
                            unique_liths = [v for v in unique_liths if v.strip()]
                            if unique_liths:
                                grouping = {v: [v] for v in sorted(unique_liths)}
                                self._lithology_grouping = grouping
                                self._unit_order = list(grouping.keys())
                                self._update_grouping_display()
                            break
                    if lith_col:
                        break

            # ── Store survey data for trajectory-based normal estimation ──
            if surveys_df is not None and isinstance(surveys_df, pd.DataFrame):
                if not surveys_df.empty:
                    self._surveys_df = surveys_df
                    logger.info(
                        "Survey data loaded: %d survey stations from %d holes",
                        len(surveys_df),
                        surveys_df.iloc[:, 0].nunique(),
                    )

            # ── Extract contacts from lithology logs ───────────────
            if search_df is not None and lith_col:
                try:
                    self._extract_contacts_from_dataframe(search_df, lith_col)
                except Exception as exc:
                    logger.warning("Auto-contact extraction failed: %s", exc)

            # ── Auto-load structural orientations from registry ────
            if structural_df is not None and isinstance(structural_df, pd.DataFrame):
                if not structural_df.empty:
                    self._orientations_df = structural_df
                    self._update_orientations_display()

            # ── Update status ──────────────────────────────────────
            if self._contacts_df is not None and not self._contacts_df.empty:
                self.data_status.setText(
                    f"{len(self._contacts_df)} contacts from {n_holes} holes"
                )
            elif df_for_xyz is not None:
                self.data_status.setText(
                    f"Data loaded: {len(df_for_xyz)} rows, {n_holes} holes (no lithology contacts)"
                )
            elif collars_df is not None:
                self.data_status.setText(
                    f"{n_holes} holes loaded (no lithology intervals)"
                )

            logger.info("_do_populate() completed: contacts=%s, grouping=%s, orientations=%s",
                         len(self._contacts_df) if self._contacts_df is not None else 0,
                         len(self._lithology_grouping) if self._lithology_grouping else 0,
                         len(self._orientations_df) if self._orientations_df is not None else 0)

        except Exception as exc:
            logger.error("_do_populate failed: %s", exc, exc_info=True)
            self.data_status.setText(f"Auto-load error: {exc}")

    def _extract_contacts_from_dataframe(self, lith_df: pd.DataFrame,
                                         lith_col: str) -> None:
        """Extract geological contacts from lithology intervals."""
        try:
            from geology.implicit.signed_distance import extract_contacts_from_lithology

            # Case-insensitive column mapping for the geology module
            col_map = {c.lower(): c for c in lith_df.columns}
            hole_col = None
            for name in ('holeid', 'hole_id', 'bhid', 'drill_hole_id'):
                if name in col_map:
                    hole_col = col_map[name]
                    break
            from_col = col_map.get('from') or col_map.get('from_depth') or col_map.get('depth_from')
            to_col = col_map.get('to') or col_map.get('to_depth') or col_map.get('depth_to')
            x_col = col_map.get('x') or col_map.get('easting')
            y_col = col_map.get('y') or col_map.get('northing')
            z_col = col_map.get('z') or col_map.get('elevation')

            kwargs = {"lithology_column": lith_col}
            if hole_col:
                kwargs["hole_id_col"] = hole_col
            if from_col:
                kwargs["from_col"] = from_col
            if to_col:
                kwargs["to_col"] = to_col
            if x_col:
                kwargs["x_col"] = x_col
            if y_col:
                kwargs["y_col"] = y_col
            if z_col:
                kwargs["z_col"] = z_col
            if self._lithology_grouping:
                kwargs["grouping"] = self._lithology_grouping
            if self._surveys_df is not None and not self._surveys_df.empty:
                kwargs["surveys_df"] = self._surveys_df

            contacts = extract_contacts_from_lithology(lith_df, **kwargs)
            self._contacts_df = contacts
            self._update_contact_display()
            logger.info("Extracted %d contacts from lithology data", len(contacts))
        except ImportError:
            self._extract_contacts_simple(lith_df, lith_col)
        except Exception as exc:
            logger.warning("Contact extraction failed: %s", exc)
            self._extract_contacts_simple(lith_df, lith_col)

    def _extract_contacts_simple(self, lith_df: pd.DataFrame,
                                 lith_col: str) -> None:
        """Simple contact extraction — detect lithology transitions per hole."""
        col_map = {c.lower(): c for c in lith_df.columns}

        holeid_col = None
        for name in ('holeid', 'hole_id', 'bhid', 'drill_hole_id'):
            if name in col_map:
                holeid_col = col_map[name]
                break
        if holeid_col is None:
            logger.warning("Cannot extract contacts: no hole ID column found")
            return

        from_col = col_map.get('from') or col_map.get('from_depth') or col_map.get('depth_from')
        to_col = col_map.get('to') or col_map.get('to_depth') or col_map.get('depth_to')
        x_col = col_map.get('x') or col_map.get('easting')
        y_col = col_map.get('y') or col_map.get('northing')
        z_col = col_map.get('z') or col_map.get('elevation')

        if not all([from_col, to_col, x_col, y_col, z_col]):
            logger.warning("Cannot extract contacts: missing required columns "
                           "(from/to/x/y/z). Found: %s", list(lith_df.columns))
            return

        contacts = []
        for hole_id, group in lith_df.groupby(holeid_col):
            group = group.sort_values(from_col)
            prev_lith = None
            for _, row_data in group.iterrows():
                curr_lith = str(row_data[lith_col]).strip()
                if not curr_lith:
                    continue
                if prev_lith is not None and curr_lith != prev_lith:
                    contacts.append({
                        'hole_id': hole_id,
                        'X': row_data[x_col],
                        'Y': row_data[y_col],
                        'Z': row_data[z_col],
                        'unit_above': prev_lith,
                        'unit_below': curr_lith,
                        'surface_name': f"{prev_lith}_{curr_lith}",
                    })
                prev_lith = curr_lith

        if contacts:
            self._contacts_df = pd.DataFrame(contacts)
            self._update_contact_display()
            logger.info("Simple contact extraction: %d contacts", len(contacts))

    def showEvent(self, event) -> None:
        """On first show, connect signals and auto-load if not yet populated."""
        super().showEvent(event)
        if not self._registry_signals_connected:
            self._connect_registry_signals()
        elif self._contacts_df is None:
            try:
                registry = self.get_registry()
                if registry:
                    dh_data = registry.get_drillhole_data()
                    if dh_data is not None:
                        self._on_registry_drillhole_loaded(dh_data)
            except Exception:
                pass

    # ==================================================================
    # Tab 1: Data & Grouping
    # ==================================================================

    def _build_tab_data(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Data & Grouping")

        # ── Data Status ───────────────────────────────────────────
        grp_data = section("Drillhole Data (from Registry)")
        df = make_form()

        self.data_status = QLabel("Waiting for drillhole data...")
        self.data_status.setObjectName("statusLabel")
        df.addRow("Status:", self.data_status)

        data_hint = hint_label(
            "Data is loaded automatically from the Data Registry when "
            "drillholes are imported. No manual loading required."
        )
        df.addRow(data_hint)

        btn_reload = action_button("Reload from Registry", style="secondary",
                             tooltip="Manually re-fetch data from the registry")
        btn_reload.clicked.connect(self._on_reload_from_registry)
        df.addRow("", btn_reload)

        grp_data.add_layout(df)
        lay.addWidget(grp_data)

        # ── Lithology Grouping ────────────────────────────────────
        grp_lith = section("Lithology Grouping")
        lf = make_form()

        self.grouping_summary = QLabel("No grouping defined")
        self.grouping_summary.setObjectName("statusLabel")
        lf.addRow("Grouping:", self.grouping_summary)

        self.grouping_table = QTableWidget(0, 2)
        self.grouping_table.setHorizontalHeaderLabels(["Modelling Unit", "Raw Codes"])
        self.grouping_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.grouping_table.setMaximumHeight(160)
        self.grouping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lf.addRow(self.grouping_table)

        btn_edit_grouping = action_button("Edit Grouping...", style="primary",
                                    tooltip="Open the lithology grouping dialog")
        btn_edit_grouping.clicked.connect(self._on_edit_grouping)
        lf.addRow("", btn_edit_grouping)

        grp_lith.add_layout(lf)
        lay.addWidget(grp_lith)

        # ── Stratigraphic Column ──────────────────────────────────
        grp_strat = section("Stratigraphic Column", collapsed=True)
        sf = make_form()

        self.strat_summary = QLabel("Not defined")
        self.strat_summary.setObjectName("statusLabel")
        sf.addRow("Column:", self.strat_summary)

        strat_hint = hint_label(
            "Required for stratiform models. Defines unit order "
            "(youngest on top) and contact types between units."
        )
        sf.addRow(strat_hint)

        btn_edit_strat = action_button("Edit Stratigraphic Column...",
                                 tooltip="Open the stratigraphic column dialog")
        btn_edit_strat.clicked.connect(self._on_edit_strat_column)
        sf.addRow("", btn_edit_strat)

        grp_strat.add_layout(sf)
        lay.addWidget(grp_strat)

        # ── Contact Summary ───────────────────────────────────────
        grp_contact = section("Extracted Contacts")
        cf = make_form()

        self.contact_status = QLabel("No contacts extracted")
        self.contact_status.setObjectName("statusLabel")
        cf.addRow("Contacts:", self.contact_status)

        self.surface_summary_table = QTableWidget(0, 2)
        self.surface_summary_table.setHorizontalHeaderLabels(["Surface", "Count"])
        self.surface_summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.surface_summary_table.setMaximumHeight(120)
        self.surface_summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        cf.addRow(self.surface_summary_table)

        grp_contact.add_layout(cf)
        lay.addWidget(grp_contact)

        # ── Structural Orientations ───────────────────────────────
        grp_orient = section("Structural Orientations", collapsed=True)
        of = make_form()

        self.orient_status = QLabel("No orientations loaded")
        self.orient_status.setObjectName("statusLabel")
        of.addRow("Status:", self.orient_status)

        orient_hint = hint_label(
            "Structural measurements (dip/azimuth) are loaded automatically "
            "from the registry if present in drillhole data."
        )
        of.addRow(orient_hint)

        grp_orient.add_layout(of)
        lay.addWidget(grp_orient)

        lay.addStretch()

    # ── Data tab handlers ─────────────────────────────────────────

    def _on_reload_from_registry(self) -> None:
        """Manually re-fetch drillhole data from the DataRegistry."""
        try:
            registry = self.get_registry()
            if registry is None:
                self.data_status.setText("No data registry available")
                return
            dh_data = registry.get_drillhole_data()
            if dh_data is None:
                self.data_status.setText("No drillhole data — import drillholes first")
                return
            self._on_registry_drillhole_loaded(dh_data)
        except Exception as exc:
            logger.exception("Registry reload failed")
            self.data_status.setText(f"Error: {exc}")

    def _on_edit_grouping(self) -> None:
        """Open the LithologyGroupingDialog popup."""
        raw_codes = []
        if self._lithology_grouping:
            for codes in self._lithology_grouping.values():
                raw_codes.extend(codes)
        elif self._drillhole_data:
            raw_codes = self._get_raw_lith_codes()

        if not raw_codes:
            QMessageBox.information(
                self, "No Data",
                "Load drillhole data first. Lithology codes will be "
                "extracted automatically from the registry.",
            )
            return

        from .dialogs.lithology_grouping_dialog import LithologyGroupingDialog
        dlg = LithologyGroupingDialog(
            raw_codes=raw_codes,
            existing_grouping=self._lithology_grouping,
            parent=self,
        )
        if dlg.exec() == LithologyGroupingDialog.DialogCode.Accepted:
            self._lithology_grouping = dlg.get_grouping()
            self._unit_order = list(self._lithology_grouping.keys())
            self._update_grouping_display()
            # Re-extract contacts with new grouping
            if self._drillhole_data is not None:
                try:
                    search_df, lith_col = self._find_lithology_df_and_col()
                    if search_df is not None and lith_col:
                        self._extract_contacts_from_dataframe(search_df, lith_col)
                except Exception as exc:
                    logger.warning("Contact re-extraction failed: %s", exc)

    def _on_edit_strat_column(self) -> None:
        """Open the StratigraphicColumnDialog popup."""
        if not self._unit_order:
            QMessageBox.information(
                self, "No Units",
                "Define lithology grouping first.",
            )
            return

        from .dialogs.stratigraphic_column_dialog import StratigraphicColumnDialog
        dlg = StratigraphicColumnDialog(
            unit_names=self._unit_order,
            existing_column=self._strat_column,
            parent=self,
        )
        if dlg.exec() == StratigraphicColumnDialog.DialogCode.Accepted:
            self._strat_column = dlg.get_column()
            self._unit_order = dlg.get_unit_order()
            n = len(self._strat_column)
            self.strat_summary.setText(f"{n} units defined (top → bottom)")

    def _get_raw_lith_codes(self) -> List[str]:
        """Extract raw lithology codes from current drillhole data."""
        codes = []
        if not self._drillhole_data:
            return codes

        search_dfs = []
        if isinstance(self._drillhole_data, dict):
            for key in ('lithology', 'geology', 'composites', 'assays'):
                candidate = self._drillhole_data.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    search_dfs.append(candidate)
        elif isinstance(self._drillhole_data, pd.DataFrame):
            search_dfs.append(self._drillhole_data)

        for df in search_dfs:
            col_lower_map = {c.lower(): c for c in df.columns}
            for candidate in ('lithology', 'lith_code', 'lithcode', 'lith',
                              'formation', 'rock_type', 'geology', 'unit'):
                if candidate in col_lower_map:
                    lith_col = col_lower_map[candidate]
                    vals = df[lith_col].dropna().astype(str).unique()
                    codes.extend(v for v in vals if v.strip())
                    return sorted(set(codes))

        return codes

    def _find_lithology_df_and_col(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Find the DataFrame and column name containing lithology codes."""
        if not self._drillhole_data:
            return None, None

        search_dfs = []
        if isinstance(self._drillhole_data, dict):
            for key in ('lithology', 'geology', 'composites', 'assays'):
                candidate = self._drillhole_data.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    search_dfs.append(candidate)
        elif isinstance(self._drillhole_data, pd.DataFrame):
            search_dfs.append(self._drillhole_data)

        for df in search_dfs:
            col_lower_map = {c.lower(): c for c in df.columns}
            for candidate in ('lithology', 'lith_code', 'lithcode', 'lith',
                              'formation', 'rock_type', 'geology', 'unit'):
                if candidate in col_lower_map:
                    return df, col_lower_map[candidate]

        return None, None

    def _update_grouping_display(self) -> None:
        """Update the grouping summary table and label."""
        if not self._lithology_grouping:
            return
        grouping = self._lithology_grouping
        n_units = len(grouping)
        total_codes = sum(len(v) for v in grouping.values())

        self.grouping_table.setRowCount(n_units)
        for i, (unit_name, codes) in enumerate(grouping.items()):
            self.grouping_table.setItem(i, 0, QTableWidgetItem(unit_name))
            display = ", ".join(str(c) for c in codes[:5])
            if len(codes) > 5:
                display += f"... (+{len(codes) - 5})"
            self.grouping_table.setItem(i, 1, QTableWidgetItem(display))

        self.grouping_summary.setText(
            f"{n_units} modelling units, {total_codes} raw codes mapped"
        )

    def _update_contact_display(self) -> None:
        """Update contact summary table."""
        if self._contacts_df is None:
            return

        self.contact_status.setText(f"{len(self._contacts_df)} contacts extracted")

        if "surface_name" in self._contacts_df.columns:
            surface_counts = self._contacts_df["surface_name"].value_counts()
            self.surface_summary_table.setRowCount(len(surface_counts))
            for i, (sname, count) in enumerate(surface_counts.items()):
                self.surface_summary_table.setItem(i, 0, QTableWidgetItem(str(sname)))
                item = QTableWidgetItem(str(count))
                if count < 5:
                    item.setForeground(Qt.GlobalColor.red)
                self.surface_summary_table.setItem(i, 1, item)

    def _update_orientations_display(self) -> None:
        """Update structural orientations status."""
        if self._orientations_df is None or self._orientations_df.empty:
            return
        n = len(self._orientations_df)
        col_lower = {c.lower() for c in self._orientations_df.columns}
        if col_lower & {'feature_type', 'type'}:
            type_col = next(c for c in self._orientations_df.columns
                           if c.lower() in ('feature_type', 'type'))
            counts = self._orientations_df[type_col].value_counts()
            detail = ", ".join(f"{k}: {v}" for k, v in counts.items())
            self.orient_status.setText(f"{n} measurements ({detail})")
        else:
            self.orient_status.setText(f"{n} orientation measurements")

    # ==================================================================
    # Tab 2: Setup & Interpolation
    # ==================================================================

    def _build_tab_setup(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Setup & Interp")

        # ── Model Type ────────────────────────────────────────────
        grp_type = section("Model Type")
        tf = make_form()

        self.model_type_combo = make_combo(
            ["stratiform", "vein", "intrusive", "structural"],
            tooltip="Type of geological model to build",
        )
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        form_row(tf, "Model Type:", self.model_type_combo)

        self.model_type_desc = hint_label(_MODEL_DESCRIPTIONS["stratiform"])
        tf.addRow(self.model_type_desc)

        grp_type.add_layout(tf)
        lay.addWidget(grp_type)

        # ── Grid Settings ─────────────────────────────────────────
        grp_grid = section("Grid Settings")
        gf = make_form()

        self.grid_resolution = make_spin(0.5, 500.0, 10.0, 1,
                                     tooltip="Grid cell size for surface extraction",
                                     suffix=" m")
        form_row(gf, "Grid Resolution:", self.grid_resolution)

        self.grid_extent_combo = make_combo(
            ["Auto (from data)", "Custom"],
            tooltip="Auto computes extent from contact data with 20% padding",
        )
        form_row(gf, "Grid Extent:", self.grid_extent_combo)

        self.tolerance = make_spin(0.1, 500.0, 5.0, 1,
                               tooltip="Maximum acceptable spatial distance from surface to contact point (metres). Misfit is converted from isovalue units using the field gradient.",
                               suffix=" m")
        form_row(gf, "Contact Tolerance:", self.tolerance)

        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("Operator name (for JORC audit)")
        form_row(gf, "Operator:", self.operator_edit)

        grp_grid.add_layout(gf)
        lay.addWidget(grp_grid)

        # ── Surface Continuity ────────────────────────────────────
        grp_cont = section("Surface Continuity")
        cf = make_form()

        self.range_max = make_spin(1, 10000, 250, 1,
                               tooltip="Maximum range of influence along the major axis (along-strike)",
                               suffix=" m")
        form_row(cf, "Along-Strike:", self.range_max)

        self.range_mid = make_spin(1, 10000, 200, 1,
                               tooltip="Range of influence perpendicular to strike (across-strike)",
                               suffix=" m")
        form_row(cf, "Across-Strike:", self.range_mid)

        self.range_min = make_spin(1, 10000, 150, 1,
                               tooltip="Range of influence in the vertical/down-dip direction",
                               suffix=" m")
        form_row(cf, "Down-Dip:", self.range_min)

        self.azimuth_spin = make_spin(0, 360, 0, 1,
                                  tooltip="Azimuth of the major axis (clockwise from North)")
        form_row(cf, "Strike Azimuth:", self.azimuth_spin)

        self.dip_spin = make_spin(-90, 90, 0, 1,
                              tooltip="Dip angle of the anisotropy ellipsoid")
        form_row(cf, "Dip:", self.dip_spin)

        cont_hint = hint_label(
            "Set anisotropy ratios to match the deposit geometry. "
            "Isotropic: all three equal. Tabular: reduce Down-Dip."
        )
        cf.addRow(cont_hint)

        btn_auto_params = action_button(
            "Auto-Detect from Data",
            style="secondary",
            tooltip="Compute range and anisotropy automatically from contact geometry",
        )
        btn_auto_params.clicked.connect(self._on_auto_detect_params)
        cf.addRow("", btn_auto_params)

        grp_cont.add_layout(cf)
        lay.addWidget(grp_cont)

        # ── Surface Quality ───────────────────────────────────────
        grp_qual = section("Surface Quality")
        qf = make_form()

        self.alpha_spin = make_spin(0.1, 10.0, 1.0, 2,
                                tooltip="Controls surface smoothness. Higher = smoother.")
        form_row(qf, "Surface Smoothness:", self.alpha_spin)

        self.nugget_spin = make_spin(0, 100, 0, 4,
                                 tooltip="Contact tolerance: 0 = exact honouring.")
        form_row(qf, "Contact Tolerance:", self.nugget_spin)

        self.accuracy_spin = make_spin(1e-8, 1.0, 1e-8, 8,
                                   tooltip="Numerical precision for matrix solver.")
        form_row(qf, "Numerical Precision:", self.accuracy_spin)

        grp_qual.add_layout(qf)
        lay.addWidget(grp_qual)

        # ── Advanced ──────────────────────────────────────────────
        grp_adv = section("Advanced", collapsed=True)
        af = make_form()

        self.kernel_combo = make_combo(
            ["spheroidal", "cubic", "gaussian", "multiquadric"],
            tooltip="Radial basis function kernel type.",
        )
        form_row(af, "Kernel:", self.kernel_combo)

        self.drift_combo = make_combo(
            ["constant", "linear"],
            tooltip="Polynomial drift: constant (recommended) or linear.",
        )
        form_row(af, "Drift:", self.drift_combo)

        self.constraint_combo = make_combo(
            ["gradient", "offset"],
            tooltip="Gradient (Hillier 2014, recommended) or offset (Cowan 2003).",
        )
        self.constraint_combo.currentTextChanged.connect(self._on_constraint_changed)
        form_row(af, "Surface Construction:", self.constraint_combo)

        self.offset_distance = make_spin(0.1, 100.0, 2.0, 1,
                                     tooltip="Distance for offset constraint points",
                                     suffix=" m")
        self.offset_distance.setVisible(False)
        self.offset_label = QLabel("Offset Distance:")
        self.offset_label.setVisible(False)
        af.addRow(self.offset_label, self.offset_distance)

        grp_adv.add_layout(af)
        lay.addWidget(grp_adv)

        # ── Fold Frame ────────────────────────────────────────────
        self.grp_fold = section("Fold Frame", collapsed=True)
        ff = make_form()

        self.fold_enabled = QCheckBox("Enable fold frame construction")
        self.fold_enabled.setToolTip(
            "Build S1/S2/S0 orthogonal scalar fields aligned with fold geometry"
        )
        ff.addRow(self.fold_enabled)

        self.fold_auto_detect = QCheckBox("Auto-detect fold axis from bedding poles")
        self.fold_auto_detect.setChecked(True)
        ff.addRow(self.fold_auto_detect)

        self.fold_azimuth = make_spin(0, 360, 0, 1,
                                  tooltip="Fold axis azimuth (ignored if auto-detect)")
        form_row(ff, "Fold Axis Azimuth:", self.fold_azimuth)

        self.fold_plunge = make_spin(-90, 90, 0, 1,
                                 tooltip="Fold axis plunge (ignored if auto-detect)")
        form_row(ff, "Fold Axis Plunge:", self.fold_plunge)

        self.fold_reg_weight = make_spin(0, 10, 1.0, 2,
                                     tooltip="Higher = smoother surfaces along fold axis")
        form_row(ff, "Smoothing Weight:", self.fold_reg_weight)

        fold_hint = hint_label(
            "Fold frame requires structural orientation data. "
            "Model type will be set to 'structural' when fold frame is built."
        )
        ff.addRow(fold_hint)

        self.grp_fold.add_layout(ff)
        lay.addWidget(self.grp_fold)

        lay.addStretch()

    def _on_model_type_changed(self, text: str) -> None:
        desc = _MODEL_DESCRIPTIONS.get(text, "")
        self.model_type_desc.setText(desc)
        if hasattr(self.grp_fold, "set_collapsed"):
            self.grp_fold.set_collapsed(text != "structural")

    def _on_auto_detect_params(self) -> None:
        """Run auto_compute_parameters() and fill in the continuity spin boxes."""
        if self._contacts_df is None or self._contacts_df.empty:
            QMessageBox.information(
                self, "No Data",
                "Load drillhole data first — contacts must be extracted before "
                "auto-detection.",
            )
            return
        try:
            from geology.implicit.auto_parameters import auto_compute_parameters
            params = auto_compute_parameters(
                self._contacts_df,
                self._orientations_df,
            )
            self.range_max.setValue(params["range_max"])
            self.range_mid.setValue(params["range_mid"])
            self.range_min.setValue(params["range_min"])
            self.azimuth_spin.setValue(params["azimuth"])
            self.dip_spin.setValue(params["dip"])
            logger.info("Auto-detected parameters: %s", params)
        except Exception as exc:
            QMessageBox.warning(self, "Auto-Detect Failed", str(exc))

    def _on_constraint_changed(self, text: str) -> None:
        show_offset = (text == "offset")
        self.offset_distance.setVisible(show_offset)
        self.offset_label.setVisible(show_offset)

    # ==================================================================
    # Tab 3: Build
    # ==================================================================

    def _build_tab_build(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Build")

        # ── Resolution Presets ────────────────────────────────────
        grp_res = section("Resolution")
        rf = make_form()

        res_hint = hint_label("Quick presets override the grid resolution on the Setup tab.")
        rf.addRow(res_hint)

        preset_row = QHBoxLayout()
        for label, res in [("Preview (50m)", 50.0), ("Standard (10m)", 10.0),
                           ("Fine (5m)", 5.0), ("Custom", None)]:
            btn = action_button(label, tooltip=f"Set grid resolution to {res}m" if res else "Use Setup tab value")
            btn.clicked.connect(lambda checked, r=res: self._set_resolution(r))
            preset_row.addWidget(btn)
        rf.addRow(preset_row)

        self.active_resolution_label = QLabel("Current: 10.0 m")
        self.active_resolution_label.setObjectName("statusLabel")
        rf.addRow(self.active_resolution_label)

        grp_res.add_layout(rf)
        lay.addWidget(grp_res)

        # ── Build Controls ────────────────────────────────────────
        self.btn_build = action_button("Build Geological Model", style="primary",
                                 tooltip="Build the implicit geological model")
        self.btn_build.clicked.connect(self._on_build)
        lay.addWidget(self.btn_build)

        btn_cancel_row = QHBoxLayout()
        self.btn_cancel = action_button("Cancel", tooltip="Terminate the running build")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_cancel_row.addWidget(self.btn_cancel)
        btn_cancel_row.addStretch()
        lay.addLayout(btn_cancel_row)

        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 100)
        self.build_progress.setVisible(False)
        lay.addWidget(self.build_progress)

        self.build_status = QLabel("")
        self.build_status.setObjectName("hintLabel")
        lay.addWidget(self.build_status)

        # ── Contact Honouring QC ──────────────────────────────────
        grp_qc = section("Contact Honouring QC")
        qf = make_form()

        self.qc_pct_label = QLabel("-")
        form_row(qf, "Honoured:", self.qc_pct_label)

        self.qc_mean_label = QLabel("-")
        form_row(qf, "Mean Misfit:", self.qc_mean_label)

        self.qc_max_label = QLabel("-")
        form_row(qf, "Max Misfit:", self.qc_max_label)

        self.qc_grad_label = QLabel("-")
        self.qc_grad_label.setToolTip(
            "Mean gradient magnitude |∇f| at contact points (isovalue units/m).\n"
            "Spatial misfit = raw isovalue misfit / |∇f|.\n"
            "Values near 0 suggest the scalar field is nearly flat at contacts."
        )
        form_row(qf, "Mean |∇f|:", self.qc_grad_label)

        self.qc_iso_label = QLabel("-")
        self.qc_iso_label.setToolTip(
            "Mean misfit in raw isovalue units (before gradient conversion).\n"
            "Spatial misfit [m] = isovalue misfit / |∇f|."
        )
        form_row(qf, "Mean Misfit (iso):", self.qc_iso_label)

        self.qc_surfaces_label = QLabel("-")
        form_row(qf, "Surfaces:", self.qc_surfaces_label)

        self.qc_interval_label = QLabel("-")
        self.qc_interval_label.setToolTip(
            "Interval consistency: fraction of logged lithology intervals whose\n"
            "predicted domain (from the scalar field) matches the logged unit.\n"
            "Low values indicate surface oscillation between drillholes."
        )
        form_row(qf, "Interval Match:", self.qc_interval_label)

        grp_qc.add_layout(qf)
        lay.addWidget(grp_qc)

        # ── Build Log ─────────────────────────────────────────────
        grp_log = section("Build Log", collapsed=True)
        self.build_log = QTextEdit()
        self.build_log.setReadOnly(True)
        self.build_log.setMaximumHeight(150)
        grp_log.add_widget(self.build_log)
        lay.addWidget(grp_log)

        lay.addStretch()

    def _set_resolution(self, res: Optional[float]) -> None:
        if res is not None:
            self.grid_resolution.setValue(res)
        self.active_resolution_label.setText(
            f"Current: {self.grid_resolution.value():.1f} m"
        )

    # ── Build config & orchestration ──────────────────────────────

    def _gather_config(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type_combo.currentText(),
            "kernel_type": self.kernel_combo.currentText(),
            "alpha": self.alpha_spin.value(),
            "range_max": self.range_max.value(),
            "range_mid": self.range_mid.value(),
            "range_min": self.range_min.value(),
            "azimuth": self.azimuth_spin.value(),
            "dip": self.dip_spin.value(),
            "pitch": 0.0,
            "nugget": self.nugget_spin.value(),
            "accuracy": self.accuracy_spin.value(),
            "drift_type": self.drift_combo.currentText(),
            "constraint_method": self.constraint_combo.currentText(),
            "offset_distance": self.offset_distance.value(),
            "grid_resolution": self.grid_resolution.value(),
            "grid_extent": "auto",
            "tolerance": self.tolerance.value(),
            "operator": self.operator_edit.text(),
        }

    def _on_build(self) -> None:
        if self._contacts_df is None or self._contacts_df.empty:
            QMessageBox.warning(self, "No Data",
                                "Load contact data first (Data & Grouping tab).")
            return

        from geology.implicit import GeologicalModelBuilder

        config = self._gather_config()
        builder = GeologicalModelBuilder(config)
        builder.set_contacts(self._contacts_df)

        if self._orientations_df is not None:
            builder.set_orientations(self._orientations_df)

        if self._surveys_df is not None:
            builder.set_surveys(self._surveys_df)

        # Lithology grouping
        if self._lithology_grouping:
            builder.set_lithology_grouping(self._lithology_grouping)

        # Stratigraphic column — build StratigraphicColumn dataclass from
        # the dialog's list-of-dicts or from _unit_order strings
        if self._unit_order and len(self._unit_order) >= 2:
            from geology.implicit.contact_data import StratigraphicColumn
            try:
                if self._strat_column:
                    # Dialog result: [{name, contact_below}, ...]
                    units = [e["name"] for e in self._strat_column if isinstance(e, dict)]
                    contacts = [e.get("contact_below", "conformable")
                                for e in self._strat_column[:-1] if isinstance(e, dict)]
                else:
                    # Auto-built from lithology grouping (no dialog)
                    units = list(self._unit_order)
                    contacts = ["conformable"] * (len(units) - 1)

                if len(units) >= 2:
                    strat_col = StratigraphicColumn(
                        units=units,
                        contact_types=contacts,
                    )
                    builder.set_stratigraphic_column(strat_col)
                    logger.info("Set stratigraphic column: %d units", len(units))
            except Exception as exc:
                logger.warning("Could not set stratigraphic column: %s", exc)

        # Fold frame
        if self.fold_enabled.isChecked() and self._orientations_df is not None:
            builder.set_fold_config({
                "auto_detect": self.fold_auto_detect.isChecked(),
                "fold_axis_azimuth": self.fold_azimuth.value(),
                "fold_axis_plunge": self.fold_plunge.value(),
                "regularisation_weight": self.fold_reg_weight.value(),
            })

        # Faults
        if self._fault_definitions:
            builder.set_fault_definitions(self._fault_definitions)

        self.btn_build.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.build_progress.setVisible(True)
        self.build_progress.setValue(0)
        self._set_status("Building...")
        self.build_status.setText("Building...")
        self.build_log.clear()

        self._worker = _ModelBuildWorker(builder, self)
        self._worker.progress.connect(self._on_build_progress)
        self._worker.finished.connect(self._on_build_finished)
        self._worker.error.connect(self._on_build_error)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._set_status("Cancelled")
            self.build_status.setText("Cancelled")
            self.btn_build.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.build_progress.setVisible(False)

    def _on_build_progress(self, pct: int, msg: str) -> None:
        self.build_progress.setValue(pct)
        self.build_status.setText(msg)
        self._set_status(f"{pct}%")
        self.build_log.append(f"[{pct:3d}%] {msg}")

    def _on_build_finished(self, result: Dict[str, Any]) -> None:
        self._build_result = result
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.build_progress.setVisible(False)

        elapsed = result.get("elapsed_seconds", 0)
        self._set_status(f"Complete ({elapsed:.1f}s)")
        self.build_status.setText(f"Complete ({elapsed:.1f}s)")

        # Update QC
        summary = result.get("contact_honouring_summary", {})
        pct = summary.get("pct_honoured", 0)
        self.qc_pct_label.setText(f"{pct:.1f}%")
        self.qc_mean_label.setText(f"{summary.get('mean_misfit', 0):.3f} m")
        self.qc_max_label.setText(f"{summary.get('max_misfit', 0):.3f} m")
        grad = summary.get("mean_grad_mag", 0.0)
        self.qc_grad_label.setText(f"{grad:.5f} /m" if grad else "-")
        iso = summary.get("mean_misfit_isovalue", 0.0)
        self.qc_iso_label.setText(f"{iso:.4f}" if iso else "-")

        surfaces = result.get("surfaces", {})
        n_surfaces = len(surfaces)
        if isinstance(surfaces, dict):
            self.qc_surfaces_label.setText(
                f"{n_surfaces} ({', '.join(surfaces.keys())})"
            )
        else:
            self.qc_surfaces_label.setText(f"{n_surfaces} surfaces")

        # ── Interval Consistency QC ─────────────────────────────────
        evaluate_fn = result.get("evaluate_fn")
        iso_raw     = result.get("isovalues", {})
        u_names     = result.get("unit_names", [])
        if (evaluate_fn is not None
                and self._lith_df is not None
                and self._lith_col is not None
                and u_names):
            try:
                from geology.implicit.validation import (
                    check_interval_consistency,
                    interval_consistency_summary,
                )
                iso_list = sorted(iso_raw.values()) if isinstance(iso_raw, dict) else sorted(iso_raw)
                consistency_df = check_interval_consistency(
                    self._lith_df, evaluate_fn, iso_list, u_names,
                    grouping=self._lithology_grouping,
                    lith_col=self._lith_col,
                )
                ic_summary = interval_consistency_summary(consistency_df)
                n_match  = ic_summary.get("n_match", 0)
                n_total  = ic_summary.get("n_intervals", 0)
                pct_ok   = 100.0 - ic_summary.get("pct_mismatch", 0.0)
                self.qc_interval_label.setText(
                    f"{n_match}/{n_total} ({pct_ok:.0f}%)"
                )
                if ic_summary.get("pct_mismatch", 0) > 10.0:
                    self.qc_interval_label.setStyleSheet("color: #e74c3c;")
                else:
                    self.qc_interval_label.setStyleSheet("")
            except Exception as _exc:
                logger.debug("Interval consistency check failed: %s", _exc)
                self.qc_interval_label.setText("–")
        else:
            self.qc_interval_label.setText("– (no drillhole log)")

        # Enable domain + export tabs
        self.btn_assign_domains.setEnabled(True)
        self._update_export_section(result)

        self.build_log.append(f"\nBuild complete: {n_surfaces} surfaces")

        # ── Build geology package and trigger 3D rendering ────────
        self._register_to_viewer(result)

        # ── Register in data registry ─────────────────────────────
        try:
            registry = self.get_registry()
            if registry and hasattr(registry, 'register_geological_surfaces'):
                registry.register_geological_surfaces(
                    result,
                    source_panel="GeologicalModelPanel",
                    metadata={"config": self._gather_config()},
                )
        except Exception as exc:
            logger.warning("Could not register surfaces: %s", exc)

    def _register_to_viewer(self, result: Dict[str, Any]) -> None:
        """Build a geology package from the build result and send to the 3D viewer.

        Signal chain:
        1. Convert builder result → renderer package format
        2. Emit request_visualization signal (caught by MainWindow)
        3. MainWindow calls renderer.load_geology_package()
        4. Surfaces + solids appear in 3D viewer
        5. GeologicalExplorerPanel tree is populated
        6. PropertyPanel discovers new layers
        """
        try:
            from geology.implicit.visualisation import build_geology_package

            package = build_geology_package(
                build_result=result,
                contacts_df=self._contacts_df,
                orientations_df=self._orientations_df,
                strat_column=self._strat_column,
                lithology_grouping=self._lithology_grouping,
            )

            n_surfaces = len(package.get("surfaces", []))
            n_solids = len(package.get("solids", []))
            self.build_log.append(
                f"Visualization package: {n_surfaces} surfaces, {n_solids} solids"
            )

            # Emit signal for MainWindow to render in 3D viewer
            self.request_visualization.emit(package)
            logger.info(
                "Emitted request_visualization: %d surfaces, %d solids",
                n_surfaces, n_solids,
            )

        except ImportError:
            logger.warning("geology.implicit.visualisation not available — "
                          "3D rendering skipped")
            self.build_log.append("Warning: visualisation module not found")
        except Exception as exc:
            logger.error("Failed to build visualization package: %s", exc,
                        exc_info=True)
            self.build_log.append(f"Visualization error: {exc}")

    def _on_build_error(self, msg: str) -> None:
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.build_progress.setVisible(False)
        self._set_status("Error")
        self.build_status.setText(f"Error: {msg}")
        self.build_log.append(f"\nERROR: {msg}")

    # ==================================================================
    # Tab 4: Faults
    # ==================================================================

    def _build_tab_faults(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Faults")

        grp_faults = section("Fault Definitions")
        ff = make_form()

        fault_hint = hint_label(
            "Define faults in reverse chronological order (youngest first). "
            "Each fault splits the model domain and restores displacement."
        )
        ff.addRow(fault_hint)

        grp_faults.add_layout(ff)

        self.fault_table = QTableWidget(0, 5)
        self.fault_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Displacement (m)", "Vector (dX,dY,dZ)", "Order"],
        )
        self.fault_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.fault_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        grp_faults.add_widget(self.fault_table)

        lay.addWidget(grp_faults)

        # Buttons
        btn_row = QHBoxLayout()
        btn_add = action_button("Add Fault...", style="primary",
                          tooltip="Open fault definition dialog")
        btn_add.clicked.connect(self._on_add_fault)
        btn_row.addWidget(btn_add)

        btn_edit = action_button("Edit...", tooltip="Edit selected fault")
        btn_edit.clicked.connect(self._on_edit_fault)
        btn_row.addWidget(btn_edit)

        btn_del = action_button("Remove", style="danger",
                          tooltip="Remove selected fault")
        btn_del.clicked.connect(self._on_remove_fault)
        btn_row.addWidget(btn_del)

        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.fault_status = QLabel("")
        self.fault_status.setObjectName("statusLabel")
        lay.addWidget(self.fault_status)

        lay.addStretch()

    def _on_add_fault(self) -> None:
        """Open FaultDefinitionDialog to add a new fault."""
        from .dialogs.fault_definition_dialog import FaultDefinitionDialog
        dlg = FaultDefinitionDialog(
            order_hint=len(self._fault_definitions),
            parent=self,
        )
        if dlg.exec() == FaultDefinitionDialog.DialogCode.Accepted:
            fault = dlg.get_fault()
            self._fault_definitions.append(fault)
            self._refresh_fault_table()

    def _on_edit_fault(self) -> None:
        """Edit the selected fault."""
        row = self.fault_table.currentRow()
        if row < 0 or row >= len(self._fault_definitions):
            return
        from .dialogs.fault_definition_dialog import FaultDefinitionDialog
        dlg = FaultDefinitionDialog(
            existing=self._fault_definitions[row],
            parent=self,
        )
        if dlg.exec() == FaultDefinitionDialog.DialogCode.Accepted:
            self._fault_definitions[row] = dlg.get_fault()
            self._refresh_fault_table()

    def _on_remove_fault(self) -> None:
        rows = sorted(set(idx.row() for idx in self.fault_table.selectedIndexes()),
                       reverse=True)
        for r in rows:
            if 0 <= r < len(self._fault_definitions):
                self._fault_definitions.pop(r)
        self._refresh_fault_table()

    def _refresh_fault_table(self) -> None:
        """Rebuild fault table from internal list."""
        self.fault_table.setRowCount(len(self._fault_definitions))
        for i, fault in enumerate(self._fault_definitions):
            self.fault_table.setItem(i, 0, QTableWidgetItem(fault.get("name", "")))
            self.fault_table.setItem(i, 1, QTableWidgetItem(fault.get("fault_type", "")))
            self.fault_table.setItem(i, 2, QTableWidgetItem(
                f"{fault.get('displacement', 0):.1f}"
            ))
            vec = fault.get("displacement_vector", [0, 0, 0])
            self.fault_table.setItem(i, 3, QTableWidgetItem(
                f"({vec[0]:.1f}, {vec[1]:.1f}, {vec[2]:.1f})"
            ))
            self.fault_table.setItem(i, 4, QTableWidgetItem(
                str(fault.get("chronological_order", i))
            ))

        n = len(self._fault_definitions)
        self.fault_status.setText(f"{n} fault{'s' if n != 1 else ''} defined" if n else "")

    # ==================================================================
    # Tab 5: Domains
    # ==================================================================

    def _build_tab_domains(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Domains")

        grp_dom = section("Domain Assignment")
        df = make_form()

        dom_hint = hint_label(
            "Assign geological domain codes to block model centroids. "
            "Domains drive per-domain ARBF estimation."
        )
        df.addRow(dom_hint)

        self.btn_assign_domains = action_button(
            "Assign Domains to Block Model", style="primary",
            tooltip="Classify block centroids into geological units",
        )
        self.btn_assign_domains.clicked.connect(self._on_assign_domains)
        self.btn_assign_domains.setEnabled(False)
        df.addRow(self.btn_assign_domains)

        self.domain_stats_label = QLabel("No model built yet")
        self.domain_stats_label.setObjectName("statusLabel")
        df.addRow(self.domain_stats_label)

        self.domain_table = QTableWidget(0, 3)
        self.domain_table.setHorizontalHeaderLabels(["Domain", "Blocks", "Percentage"])
        self.domain_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.domain_table.setMaximumHeight(200)
        self.domain_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        df.addRow(self.domain_table)

        grp_dom.add_layout(df)
        lay.addWidget(grp_dom)

        lay.addStretch()

    def _on_assign_domains(self) -> None:
        if self._build_result is None:
            return

        registry = self.get_registry()
        if registry is None:
            self.domain_stats_label.setText("No data registry")
            return

        try:
            bm = None
            if hasattr(registry, 'get_block_model'):
                bm = registry.get_block_model()
            elif hasattr(registry, 'get'):
                bm = registry.get("block_model")
            if bm is None:
                self.domain_stats_label.setText("No block model loaded")
                return

            evaluate_fn = self._build_result.get("evaluate_fn")
            if evaluate_fn is None:
                self.domain_stats_label.setText("No evaluate function in build result")
                return

            # Get centroids from block model
            if hasattr(bm, "cell_centers"):
                centroids = np.array(bm.cell_centers().points)
            elif hasattr(bm, "points"):
                centroids = np.array(bm.points)
            else:
                self.domain_stats_label.setText("Cannot extract block centroids")
                return

            from geology.implicit.domain_model import assign_domains_potential_field

            isovalues = self._build_result.get("isovalues", [0.0])
            unit_names = self._build_result.get(
                "unit_names",
                [f"unit_{i}" for i in range(len(isovalues) + 1)],
            )

            codes, names = assign_domains_potential_field(
                centroids, evaluate_fn, isovalues, unit_names,
            )

            # Store result
            self._build_result["domain_codes"] = codes
            self._build_result["domain_names"] = names
            self._update_domain_table(codes, names)

            # Publish to registry
            if hasattr(registry, 'register_domain_model'):
                registry.register_domain_model(
                    {"codes": codes, "names": names},
                    source_panel="GeologicalModelPanel",
                    metadata={"method": "potential_field"},
                )
            if hasattr(registry, 'geologicalModelUpdated'):
                try:
                    registry.geologicalModelUpdated.emit({
                        "domain_codes": codes,
                        "domain_names": names,
                        "source": "GeologicalModelPanel",
                    })
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Domain assignment failed")
            self.domain_stats_label.setText(f"Error: {exc}")

    def _update_domain_table(self, codes, names) -> None:
        unique, counts = np.unique(codes, return_counts=True)
        total = len(codes)

        self.domain_table.setRowCount(len(unique))
        for i, (u, c) in enumerate(zip(unique, counts)):
            self.domain_table.setItem(i, 0, QTableWidgetItem(str(u)))
            self.domain_table.setItem(i, 1, QTableWidgetItem(str(c)))
            self.domain_table.setItem(i, 2, QTableWidgetItem(f"{100*c/total:.1f}%"))

        self.domain_stats_label.setText(
            f"{len(unique)} domains, {total} blocks classified"
        )

    # ==================================================================
    # Tab 6: Export
    # ==================================================================

    def _build_tab_export(self) -> None:
        lay, _ = _make_tab(self.tab_widget, "Export")

        grp_export = section("Export Results")
        ef = make_form()

        export_row = QHBoxLayout()
        self.btn_export_surfaces = action_button("Surfaces (VTK)",
                                           tooltip="Export geological surfaces as VTK meshes")
        self.btn_export_surfaces.clicked.connect(self._export_surfaces)
        self.btn_export_surfaces.setEnabled(False)
        export_row.addWidget(self.btn_export_surfaces)

        self.btn_export_domains = action_button("Domains (CSV)",
                                          tooltip="Export domain codes for every block centroid")
        self.btn_export_domains.clicked.connect(self._export_domains)
        self.btn_export_domains.setEnabled(False)
        export_row.addWidget(self.btn_export_domains)

        self.btn_export_audit = action_button("Audit (JSON)",
                                        tooltip="Export JORC-compliant audit record")
        self.btn_export_audit.clicked.connect(self._export_audit)
        self.btn_export_audit.setEnabled(False)
        export_row.addWidget(self.btn_export_audit)
        ef.addRow(export_row)

        self.export_status = QLabel("")
        self.export_status.setObjectName("statusLabel")
        ef.addRow(self.export_status)

        grp_export.add_layout(ef)
        lay.addWidget(grp_export)

        # ── Audit Record ──────────────────────────────────────────
        grp_audit = section("Audit Record", collapsed=True)
        self.audit_text = QTextEdit()
        self.audit_text.setReadOnly(True)
        self.audit_text.setMaximumHeight(200)
        grp_audit.add_widget(self.audit_text)
        lay.addWidget(grp_audit)

        lay.addStretch()

    def _update_export_section(self, result: Dict[str, Any]) -> None:
        self.btn_export_surfaces.setEnabled(bool(result.get("surfaces")))
        self.btn_export_audit.setEnabled(bool(result.get("audit_record")))
        self.btn_export_domains.setEnabled("domain_codes" in result)

        audit = result.get("audit_record", {})
        if audit:
            self.audit_text.setText(json.dumps(audit, indent=2, default=str))

    def _export_surfaces(self) -> None:
        if not self._build_result:
            return
        surfaces = self._build_result.get("surfaces", {})
        if not surfaces:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Export Surfaces To")
        if not dir_path:
            return

        count = 0
        for name, mesh in surfaces.items():
            try:
                if hasattr(mesh, "save"):
                    mesh.save(os.path.join(dir_path, f"{name}.vtk"))
                    count += 1
            except Exception as exc:
                logger.error("Surface export %s failed: %s", name, exc)

        self.export_status.setText(f"Exported {count} surfaces to {dir_path}")

    def _export_domains(self) -> None:
        if not self._build_result or "domain_codes" not in self._build_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Domain Codes", "domain_codes.csv", "CSV (*.csv)",
        )
        if not path:
            return
        try:
            codes = self._build_result["domain_codes"]
            names = self._build_result.get("domain_names", codes)
            pd.DataFrame({"domain_code": codes, "domain_name": names}).to_csv(
                path, index=False,
            )
            self.export_status.setText(f"Domains saved to {os.path.basename(path)}")
        except Exception as exc:
            self.export_status.setText(f"Export failed: {exc}")

    def _export_audit(self) -> None:
        if not self._build_result:
            return
        audit = self._build_result.get("audit_record", {})
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Record", "geological_model_audit.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(audit, f, indent=2, default=str)
            self.export_status.setText(f"Audit saved to {os.path.basename(path)}")
        except Exception as exc:
            self.export_status.setText(f"Export failed: {exc}")

    # ------------------------------------------------------------------
    # BaseAnalysisPanel interface
    # ------------------------------------------------------------------

    def gather_parameters(self) -> Dict[str, Any]:
        return self._gather_config()

    def validate_inputs(self) -> bool:
        if self._contacts_df is None or self._contacts_df.empty:
            return False
        return True

    def on_results(self, payload: Dict[str, Any]) -> None:
        self._on_build_finished(payload)
