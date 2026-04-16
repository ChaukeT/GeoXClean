"""
Bayesian Kriging Panel (PyQt6)

Provides a unified UI for all Bayesian/soft kriging methods:
- Bayesian Ordinary Kriging (OK + soft prior)
- Bayesian Universal Kriging (UK + soft prior)
- Bayesian Indicator Kriging (IK + soft probabilities)
- Bayesian Co-Kriging (CoK + soft secondary)

Soft data can come from:
- External CSV (mean/variance per location)
- Previous IK results (converted to soft moments)
- Block model properties

The Bayesian update uses precision-weighted combination:
  Z*_bayes = (Z*_krig / σ²_krig + Z_soft / σ²_soft) / (1/σ²_krig + 1/σ²_soft)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QFileDialog, QLineEdit, QWidget, QScrollArea,
    QFrame, QMessageBox, QProgressBar, QTabWidget, QCheckBox,
    QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QSignalBlocker

from .panel_manager import PanelCategory, DockArea
from .modern_styles import ModernColors
from ..utils.variable_utils import populate_variable_combo
from .base_analysis_panel import BaseAnalysisPanel
from .panel_utils import resolve_variogram_for_variable
from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin

logger = logging.getLogger(__name__)


from .mixins.domain_mask_mixin import DomainMaskMixin


class BayesianKrigingPanel(CodedDomainFilterMixin, DomainMaskMixin, BaseAnalysisPanel):
    """
    Panel for configuring and running Bayesian Kriging with soft data integration.

    Supports OK, UK, IK, and CoK as base methods with precision-weighted
    Bayesian updates from soft (uncertain) data sources.
    """

    # PanelManager metadata
    PANEL_ID = "BayesianKrigingPanel"
    PANEL_NAME = "Bayesian Kriging"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "bayesian_kriging"
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.kriging_results: Optional[Dict[str, Any]] = None
        self.registry = None

        super().__init__(parent=parent, panel_id="bayesian_kriging")
        self.setWindowTitle("Bayesian Kriging")
        self.resize(1100, 750)

        self._build_ui()
        self._init_registry()
        self.progress_updated.connect(self._update_progress)

    # =========================================================
    # UI CONSTRUCTION
    # =========================================================

    def _build_ui(self):
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # --- TAB 1: CONFIGURATION ---
        config_tab = QWidget()
        c_layout = QVBoxLayout(config_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)

        self._build_method_group(config_layout)
        self._build_soft_data_group(config_layout)
        self._build_bayesian_group(config_layout)
        self._build_variogram_group(config_layout)
        self._build_search_group(config_layout)
        self._build_grid_group(config_layout)

        config_layout.addWidget(self._build_domain_mask_group(default_enabled=True))
        config_layout.addStretch()
        scroll.setWidget(config_container)
        c_layout.addWidget(scroll)

        # Action buttons
        btn_row = QHBoxLayout()

        self.btn_validate = QPushButton("Validate Soft Data")
        self.btn_validate.setStyleSheet(
            f"background-color: {ModernColors.WARNING}; color: {ModernColors.TEXT_PRIMARY};"
            f" font-weight: bold; padding: 6px;"
        )
        self.btn_validate.clicked.connect(self._validate_soft_data)

        self.run_btn = QPushButton("Run Bayesian Kriging")
        self.run_btn.setStyleSheet(
            f"background-color: {ModernColors.SUCCESS}; color: {ModernColors.TEXT_PRIMARY};"
            f" font-weight: bold; padding: 6px;"
        )
        self.run_btn.clicked.connect(self.run_analysis)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Reload latest data from registry")
        self.refresh_btn.clicked.connect(self._manual_refresh)
        self.view_table_btn = QPushButton("View Results Table")
        self.view_table_btn.setStyleSheet(
            f"background-color: {ModernColors.ACCENT_PRIMARY}; color: {ModernColors.TEXT_PRIMARY};"
            f" padding: 6px;"
        )
        self.view_table_btn.clicked.connect(self._open_results_table)
        self.view_table_btn.setEnabled(False)

        btn_row.addWidget(self.btn_validate)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.view_table_btn)
        c_layout.addLayout(btn_row)

        self.tabs.addTab(config_tab, "Configuration")

        # --- TAB 2: LOG / RESULTS ---
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            f"background-color: {ModernColors.PANEL_BG}; color: {ModernColors.SUCCESS};"
            f" font-family: Consolas, Monospace; font-size: 11px;"
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.progress_bar)
        log_layout.addWidget(self.lbl_status)
        self.tabs.addTab(log_tab, "Log / Results")

        main_layout.addWidget(self.tabs)

    # --- Group builders ---

    def _build_method_group(self, layout):
        grp = QGroupBox("1. Base Method && Variable")
        grp.setStyleSheet(_group_style(ModernColors.INFO))
        form = QFormLayout(grp)

        self.base_method_combo = QComboBox()
        self.base_method_combo.addItems([
            "Ordinary Kriging (OK)",
            "Universal Kriging (UK)",
            "Indicator Kriging (IK)",
            "Co-Kriging (CoK)",
        ])
        self.base_method_combo.currentTextChanged.connect(self._on_method_changed)

        self.variable_combo = QComboBox()
        self.domain_combo = QComboBox()
        self.domain_combo.addItem("All Data")
        self.domain_combo.currentTextChanged.connect(self._on_domain_filter_selection_changed)
        self.domain_combo.currentTextChanged.connect(
            lambda _: self._reload_variogram_for_domain()
        )

        # Secondary variable (for CoK)
        self.secondary_combo = QComboBox()
        self.secondary_label = QLabel("Secondary Var:")
        self.secondary_combo.setVisible(False)
        self.secondary_label.setVisible(False)

        # Drift type (for UK)
        self.drift_combo = QComboBox()
        self.drift_combo.addItems(["Linear", "Quadratic", "Constant"])
        self.drift_label = QLabel("Drift Type:")
        self.drift_combo.setVisible(False)
        self.drift_label.setVisible(False)

        form.addRow("Base Method:", self.base_method_combo)
        form.addRow("Primary Var:", self.variable_combo)
        form.addRow("Domain:", self.domain_combo)
        form.addRow(self.secondary_label, self.secondary_combo)
        form.addRow(self.drift_label, self.drift_combo)
        layout.addWidget(grp)

    def _build_soft_data_group(self, layout):
        grp = QGroupBox("2. Soft Data Source")
        grp.setStyleSheet(_group_style(ModernColors.WARNING))
        v = QVBoxLayout(grp)

        self.soft_source_combo = QComboBox()
        self.soft_source_combo.addItems([
            "From External CSV",
            "From IK Results (auto-convert)",
            "From Block Model Property",
        ])
        self.soft_source_combo.currentTextChanged.connect(self._on_soft_source_changed)
        v.addWidget(self.soft_source_combo)

        # CSV path
        self.csv_frame = QWidget()
        h = QHBoxLayout(self.csv_frame)
        h.setContentsMargins(0, 0, 0, 0)
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("Select CSV with X, Y, Z, Mean, Variance columns...")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_csv)
        h.addWidget(self.csv_path_edit)
        h.addWidget(self.browse_btn)
        v.addWidget(self.csv_frame)

        # IK / Block model note
        self.soft_note = QLabel(
            "CSV must have columns: X, Y, Z, and one of (mean/grade/value) + optional (var/variance/std)"
        )
        self.soft_note.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        self.soft_note.setWordWrap(True)
        v.addWidget(self.soft_note)

        layout.addWidget(grp)

    def _build_bayesian_group(self, layout):
        grp = QGroupBox("3. Bayesian Update Parameters")
        grp.setStyleSheet(_group_style(ModernColors.TEXT_SECONDARY))
        form = QFormLayout(grp)

        self.prior_mode_combo = QComboBox()
        self.prior_mode_combo.addItems(["Mean & Variance", "Mean Only"])

        self.soft_weight_spin = QDoubleSpinBox()
        self.soft_weight_spin.setRange(0.0, 1.0)
        self.soft_weight_spin.setSingleStep(0.05)
        self.soft_weight_spin.setValue(0.5)
        self.soft_weight_spin.setToolTip(
            "Weight for soft data in precision-weighted update.\n"
            "0.0 = ignore soft data (pure kriging)\n"
            "0.5 = equal precision weighting\n"
            "1.0 = full soft data precision"
        )

        form.addRow("Prior Mode:", self.prior_mode_combo)
        form.addRow("Soft Data Confidence:", self.soft_weight_spin)
        layout.addWidget(grp)

    def _build_variogram_group(self, layout):
        grp = QGroupBox("4. Variogram Model")
        grp.setStyleSheet(_group_style(ModernColors.SUCCESS))
        v = QVBoxLayout(grp)

        # Auto-load buttons
        btn_row = QHBoxLayout()
        self.auto_vario_btn = QPushButton("Load from Variogram Panel")
        self.auto_vario_btn.clicked.connect(self._resolve_and_load_variogram)
        self.auto_vario_btn.setEnabled(False)
        btn_row.addWidget(self.auto_vario_btn)
        v.addLayout(btn_row)

        # Model type
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Spherical", "Exponential", "Gaussian"])
        form.addRow("Model:", self.model_combo)

        # Parameters
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1, 100000)
        self.range_spin.setValue(100)
        self.range_spin.setDecimals(1)

        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.001, 100000)
        self.sill_spin.setValue(1.0)
        self.sill_spin.setDecimals(4)

        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 100000)
        self.nugget_spin.setValue(0.0)
        self.nugget_spin.setDecimals(4)

        form.addRow("Range (major):", self.range_spin)
        form.addRow("Sill (partial C₁):", self.sill_spin)
        form.addRow("Nugget (C₀):", self.nugget_spin)
        v.addLayout(form)
        layout.addWidget(grp)

    def _build_search_group(self, layout):
        grp = QGroupBox("5. Search Parameters")
        grp.setStyleSheet(_group_style(ModernColors.ACCENT_PRIMARY))
        form = QFormLayout(grp)

        self.ndmax_spin = QSpinBox()
        self.ndmax_spin.setRange(1, 64)
        self.ndmax_spin.setValue(12)

        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(0, 1e6)
        self.max_dist_spin.setValue(500)
        self.max_dist_spin.setDecimals(1)

        self.nmin_spin = QSpinBox()
        self.nmin_spin.setRange(1, 32)
        self.nmin_spin.setValue(3)

        form.addRow("Max Neighbours:", self.ndmax_spin)
        form.addRow("Max Distance:", self.max_dist_spin)
        form.addRow("Min Neighbours:", self.nmin_spin)
        layout.addWidget(grp)

    def _build_grid_group(self, layout):
        grp = QGroupBox("6. Estimation Grid")
        grp.setStyleSheet(_group_style(ModernColors.WARNING))
        v = QVBoxLayout(grp)

        # Origin
        origin_row = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e7, 1e7)
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e7, 1e7)
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e7, 1e7)
        for lbl, spin in [("Xmin:", self.xmin_spin), ("Ymin:", self.ymin_spin), ("Zmin:", self.zmin_spin)]:
            origin_row.addWidget(QLabel(lbl))
            origin_row.addWidget(spin)
        v.addLayout(origin_row)

        # Block size
        bs_row = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 10000)
        self.dx_spin.setValue(10)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 10000)
        self.dy_spin.setValue(10)
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 10000)
        self.dz_spin.setValue(5)
        self.dx_spin.valueChanged.connect(self._on_block_size_changed)
        self.dy_spin.valueChanged.connect(self._on_block_size_changed)
        self.dz_spin.valueChanged.connect(self._on_block_size_changed)
        for lbl, spin in [("dX:", self.dx_spin), ("dY:", self.dy_spin), ("dZ:", self.dz_spin)]:
            bs_row.addWidget(QLabel(lbl))
            bs_row.addWidget(spin)
        v.addLayout(bs_row)

        # Counts
        cnt_row = QHBoxLayout()
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 1000)
        self.nx_spin.setValue(20)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 1000)
        self.ny_spin.setValue(20)
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(1, 500)
        self.nz_spin.setValue(10)
        for lbl, spin in [("nX:", self.nx_spin), ("nY:", self.ny_spin), ("nZ:", self.nz_spin)]:
            cnt_row.addWidget(QLabel(lbl))
            cnt_row.addWidget(spin)
        v.addLayout(cnt_row)

        auto_btn = QPushButton("Auto-detect from data")
        auto_btn.clicked.connect(self._auto_detect_grid)
        v.addWidget(auto_btn)

        layout.addWidget(grp)

    # =========================================================
    # REGISTRY / DATA
    # =========================================================

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
                if hasattr(self.registry, "compositesLoaded"):
                    self.registry.compositesLoaded.connect(self._on_composites_refreshed)
                if hasattr(self.registry, "indicatorRBFDomainLoaded"):
                    self.registry.indicatorRBFDomainLoaded.connect(self._on_indicator_rbf_domain_loaded)
                self.registry.variogramResultsLoaded.connect(self._on_variogram_loaded)
                if hasattr(self.registry, "blockModelDefinitionChanged"):
                    self.registry.blockModelDefinitionChanged.connect(
                        lambda defn: self._apply_shared_block_model_definition()
                    )

                # Try to load existing data
                data = None
                try:
                    data = self.registry.get_estimation_ready_data(
                        prefer_declustered=True, require_validation=False
                    )
                except (ValueError, AttributeError):
                    data = self.registry.get_drillhole_data()
                if data is not None:
                    self._on_data_loaded(data)

                vario = self.registry.get_variogram_results()
                if vario is not None:
                    self._on_variogram_loaded(vario)

                # Seed grid spinboxes from shared block model definition (if any)
                self._apply_shared_block_model_definition()
        except ImportError:
            logger.warning("DataRegistry not found. Running standalone.")
            self.registry = None

    def _apply_shared_block_model_definition(self):
        """Populate grid spinboxes from the shared BlockModelDefinition (if any)."""
        try:
            reg = getattr(self, "registry", None)
            if reg is None and hasattr(self, "_get_any_registry"):
                reg = self._get_any_registry()
            if reg is None and hasattr(self, "get_registry"):
                reg = self.get_registry()
            if reg is None or not hasattr(reg, "get_block_model_definition"):
                return
            defn = reg.get_block_model_definition()
            if defn is None:
                return
            from PyQt6.QtCore import QSignalBlocker
            ox, oy, oz = defn.origin
            nx, ny, nz = defn.dims
            dx, dy, dz = defn.block_size
            self._suspend_block_size_refit = True
            try:
                with QSignalBlocker(self.dx_spin), QSignalBlocker(self.dy_spin), QSignalBlocker(self.dz_spin):
                    self.xmin_spin.setValue(float(ox))
                    self.ymin_spin.setValue(float(oy))
                    self.zmin_spin.setValue(float(oz))
                    self.nx_spin.setValue(int(nx))
                    self.ny_spin.setValue(int(ny))
                    self.nz_spin.setValue(int(nz))
                    self.dx_spin.setValue(float(dx))
                    self.dy_spin.setValue(float(dy))
                    self.dz_spin.setValue(float(dz))
            finally:
                self._suspend_block_size_refit = False
            logger.info(
                "%s: applied shared BlockModelDefinition '%s' (%dx%dx%d, %gx%gx%g)",
                type(self).__name__, defn.name, nx, ny, nz, dx, dy, dz,
            )
        except Exception as exc:
            logger.debug(
                "%s: failed to apply shared BlockModelDefinition: %s",
                type(self).__name__, exc,
            )

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_shared_block_model_definition()

    def _manual_refresh(self):
        """Reload latest drillhole data from registry."""
        try:
            registry = self.get_registry()
            if registry:
                data = registry.get_drillhole_data()
                if data is not None:
                    self._on_data_loaded(data)
                    logger.info("BayesianKrigingPanel: Manual refresh completed")
        except Exception as exc:
            logger.error("BayesianKrigingPanel: Refresh failed: %s", exc, exc_info=True)

    def _on_data_loaded(self, data):
        if data is None:
            return
        self._registry_data = data
        df = data
        if isinstance(data, dict):
            composites = data.get("composites")
            assays = data.get("assays")
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                df = composites
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
            else:
                return
        self.drillhole_data = self._prepare_domain_filter_dataframe(
            df,
            registry_payload=data,
            populate_combo=True,
            all_label="All Data",
        )
        populate_variable_combo(self.variable_combo, self.drillhole_data)
        populate_variable_combo(self.secondary_combo, self.drillhole_data)
        self._log("Data loaded: {} samples".format(len(self.drillhole_data)))

    def _get_filtered_data(self) -> Optional[pd.DataFrame]:
        filtered_df, metadata = self._get_current_domain_filtered_data(
            df=self.drillhole_data,
            registry_payload=getattr(self, "_registry_data", None),
            all_label="All Data",
        )
        self._active_domain_filter_metadata = metadata
        return filtered_df

    def _on_variogram_loaded(self, vario_results):
        self.variogram_results = vario_results
        self.auto_vario_btn.setEnabled(True)
        self._log("Variogram results available")

    def _resolve_and_load_variogram(self):
        """Button handler: resolve the correct variogram for the selected variable, then load."""
        selected_var = self.variable_combo.currentText() if hasattr(self, 'variable_combo') else None
        resolved = resolve_variogram_for_variable(self.registry, selected_var, self)
        if resolved:
            self.variogram_results = resolved
            if hasattr(self, 'auto_vario_btn'):
                self.auto_vario_btn.setEnabled(True)
        self.load_variogram_parameters()

    def load_variogram_parameters(self):
        """Load variogram parameters from variogram results (same pattern as other panels)."""
        if not self.variogram_results:
            QMessageBox.warning(self, "Error", "No variogram results available.")
            return

        try:
            model_type = self.model_combo.currentText().lower()

            # Priority 1: combined_3d_model
            combined = self.variogram_results.get('combined_3d_model', {})
            if combined and combined.get('model_type', '').lower() == model_type:
                nugget = combined.get('nugget', 0.0)
                # FIX: 'sill' in combined_3d_model is PARTIAL sill (C₁). Use 'total_sill' first.
                total_sill = combined.get('total_sill', None)
                if total_sill is None:
                    total_sill = nugget + combined.get('sill', 1.0)
                partial_sill = max(total_sill - nugget, 0.001)
                self.range_spin.setValue(combined.get('major_range', 100.0))
                self.sill_spin.setValue(partial_sill)
                self.nugget_spin.setValue(nugget)
                self._log(f"Loaded from combined model: range={self.range_spin.value():.1f}, "
                         f"C₁={partial_sill:.4f}, C₀={nugget:.4f}")
                return

            # Priority 2: fitted_models — use max of major/minor range (swap protection)
            fitted = self.variogram_results.get('fitted_models', {})
            for direction in ['major', 'omni']:
                model = fitted.get(direction, {}).get(model_type)
                if model:
                    nugget = model.get('nugget', 0.0)
                    ts = model.get('total_sill')
                    if ts is None:
                        s = model.get('sill', 0.0)
                        ts = (nugget + s) if s > nugget else s
                    partial_sill = max(ts - nugget, 0.001)
                    range_val = model.get('range', 100.0)
                    # If loading from 'major', check if 'minor' has a longer range (label swap)
                    if direction == 'major':
                        minor_model = fitted.get('minor', {}).get(model_type)
                        if minor_model and minor_model.get('range', 0) > range_val * 1.01:
                            range_val = minor_model['range']
                            logger.warning(f"Bayesian Kriging: minor range > major range — using minor range {range_val:.1f}")
                    self.range_spin.setValue(range_val)
                    self.sill_spin.setValue(partial_sill)
                    self.nugget_spin.setValue(nugget)
                    self._log(f"Loaded from {direction}: range={self.range_spin.value():.1f}, "
                             f"C₁={partial_sill:.4f}, C₀={nugget:.4f}")
                    return

            self._log("No matching variogram found for current model type", "warning")
        except Exception as e:
            logger.error(f"Error loading variogram: {e}", exc_info=True)
            self._log(f"Error loading variogram: {e}", "error")

    # =========================================================
    # UI EVENT HANDLERS
    # =========================================================

    def _on_method_changed(self, text):
        is_cok = "CoK" in text
        is_uk = "UK" in text
        self.secondary_combo.setVisible(is_cok)
        self.secondary_label.setVisible(is_cok)
        self.drift_combo.setVisible(is_uk)
        self.drift_label.setVisible(is_uk)

    def _on_soft_source_changed(self, text):
        is_csv = "CSV" in text
        self.csv_frame.setVisible(is_csv)
        if "IK" in text:
            self.soft_note.setText("Will use IK results from DataRegistry (must run IK first).")
        elif "Block" in text:
            self.soft_note.setText("Will use existing block model property as soft prior.")
        else:
            self.soft_note.setText(
                "CSV must have columns: X, Y, Z, and one of (mean/grade/value) + optional (var/variance/std)"
            )

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Soft Data CSV", "", "CSV (*.csv)")
        if path:
            self.csv_path_edit.setText(path)

    def _on_block_size_changed(self, *_):
        """Re-fit grid origin and NX/NY/NZ when DX/DY/DZ change.

        Without this, origin stays snapped to the old spacing and NX/NY/NZ
        reflect the old size, so the grid no longer matches drillhole coverage.
        """
        if getattr(self, "_suspend_block_size_refit", False):
            return
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return
        self._suspend_block_size_refit = True
        try:
            self._auto_detect_grid()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("refit after block-size change failed: %s", exc)
        finally:
            self._suspend_block_size_refit = False

    def _auto_fix_grid_origin_if_needed(self):
        """Auto-run Auto-Detect if origin is (0,0,0) but drillhole centroid is far away.

        Catches the UTM vs local coordinate mismatch case where user loaded
        UTM drillholes but never clicked Auto-Detect, leaving the grid at
        default origin ~500 km from the data.
        """
        try:
            x0 = self.xmin_spin.value()
            y0 = self.ymin_spin.value()
            z0 = self.zmin_spin.value()
        except AttributeError:
            return
        if x0 != 0.0 or y0 != 0.0 or z0 != 0.0:
            return
        df = self._get_filtered_data()
        if df is None or getattr(df, "empty", True):
            return
        coord_cols = None
        for cx, cy, cz in [("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL")]:
            if cx in df.columns and cy in df.columns and cz in df.columns:
                coord_cols = (cx, cy, cz)
                break
        if coord_cols is None:
            return
        import numpy as _np
        x = df[coord_cols[0]].dropna().to_numpy()
        y = df[coord_cols[1]].dropna().to_numpy()
        z = df[coord_cols[2]].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            return
        cx = float(_np.mean(x))
        cy = float(_np.mean(y))
        cz = float(_np.mean(z)) if len(z) else 0.0
        if abs(cx) < 1000 and abs(cy) < 1000:
            return
        import logging as _logging
        _logging.getLogger(__name__).info(
            "%s: grid origin is (0,0,0) but drillhole centroid is (%.0f, %.0f, %.0f). "
            "Auto-detecting grid from drillhole extents.",
            type(self).__name__, cx, cy, cz,
        )
        try:
            self._auto_detect_grid()
        except Exception as _exc:
            _logging.getLogger(__name__).debug("auto-detect from UTM fix failed: %s", _exc)

    def _auto_detect_grid(self):
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
        try:
            from ..utils.coordinate_utils import ensure_xyz_columns
            df = ensure_xyz_columns(filtered_df)
            if not all(c in df.columns for c in ('X', 'Y', 'Z')):
                QMessageBox.warning(self, "Error", "Missing X, Y, Z columns.")
                return

            pad = 0.05
            for axis, min_spin, n_spin, d_spin in [
                ('X', self.xmin_spin, self.nx_spin, self.dx_spin),
                ('Y', self.ymin_spin, self.ny_spin, self.dy_spin),
                ('Z', self.zmin_spin, self.nz_spin, self.dz_spin),
            ]:
                mn, mx = df[axis].min(), df[axis].max()
                rng = mx - mn
                bs = d_spin.value()
                start = np.floor((mn - rng * pad) / bs) * bs
                n = int(np.ceil(((mx + rng * pad) - start) / bs))
                min_spin.setValue(start)
                n_spin.setValue(max(1, n))

            self._log(f"Grid auto-fitted: {self.nx_spin.value()}x{self.ny_spin.value()}x{self.nz_spin.value()}")
        except Exception as e:
            self._log(f"Grid auto-detect failed: {e}", "error")

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_soft_data(self):
        self.tabs.setCurrentIndex(1)
        self.log_text.clear()
        self._log("--- Soft Data Validation ---")

        try:
            source = self.soft_source_combo.currentText()

            if "CSV" in source:
                path = self.csv_path_edit.text()
                if not path:
                    self._log("No CSV path specified.", "error")
                    return
                from ..geostats.soft_data import soft_from_csv
                t0 = time.time()
                sd = soft_from_csv(path)
                dt = time.time() - t0
                self._log(f"Loaded {sd.n_points:,} soft data points in {dt:.2f}s")
                self._log(f"  Mean grade:    {np.mean(sd.means):.4f}")
                self._log(f"  Mean variance: {np.mean(sd.variances):.4f}")
                self._log(f"  Grade range:   [{np.min(sd.means):.4f}, {np.max(sd.means):.4f}]")
                neg = np.sum(sd.variances < 0)
                if neg > 0:
                    self._log(f"  WARNING: {neg} negative variances detected — will be clipped to 0", "warning")
                else:
                    self._log("  Variance integrity: OK")
            elif "IK" in source:
                if self.registry:
                    ik = self.registry.get_indicator_kriging_results()
                    if ik:
                        self._log(f"IK results found with {len(ik.get('thresholds', []))} thresholds")
                    else:
                        self._log("No IK results in registry. Run Indicator Kriging first.", "warning")
                else:
                    self._log("DataRegistry not available.", "error")
            else:
                self._log("Block model source validation not yet implemented.", "warning")

            self._log("--- Validation Complete ---")
        except Exception as e:
            self._log(f"Validation error: {e}", "error")

    # =========================================================
    # RUN ANALYSIS
    # =========================================================

    def _resolve_domain_mask(self):
        """Resample the IRBF domain onto this panel's estimation grid.

        IRBF builds its mask on an auto-resolution grid that generally has
        different origin/spacing/counts than this panel's grid. We resample
        via nearest-neighbour so the mask lines up in space.
        """
        for _cb_name in ("domain_check", "use_domain_check", "mask_checkbox", "domain_enabled_check"):
            _cb = getattr(self, _cb_name, None)
            if _cb is not None:
                try:
                    if not _cb.isChecked():
                        return None
                except Exception:
                    pass
                break
        reg = getattr(self, "registry", None)
        if reg is None and hasattr(self, "get_registry"):
            try:
                reg = self.get_registry()
            except Exception:
                reg = None
        if reg is None:
            return None
        irbf_domain = None
        try:
            if hasattr(reg, "get_indicator_rbf_domain"):
                irbf_domain = reg.get_indicator_rbf_domain()
            if not irbf_domain and hasattr(reg, "get_data"):
                irbf_domain = reg.get_data("indicator_rbf_domain", copy_data=False)
        except Exception:
            irbf_domain = None
        if not irbf_domain:
            return None
        try:
            from ..geostats.domain_mask import resample_irbf_mask_to_grid
            target_origin = (
                float(self.xmin_spin.value()),
                float(self.ymin_spin.value()),
                float(self.zmin_spin.value()),
            )
            target_spacing = (
                float(self.dx_spin.value()),
                float(self.dy_spin.value()),
                float(self.dz_spin.value()),
            )
            target_dims = (
                int(self.nx_spin.value()),
                int(self.ny_spin.value()),
                int(self.nz_spin.value()),
            )
            return resample_irbf_mask_to_grid(
                irbf_domain,
                target_origin=target_origin,
                target_spacing=target_spacing,
                target_dims=target_dims,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "%s: resample_irbf_mask_to_grid failed: %s",
                type(self).__name__, exc,
            )
            return None

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all UI parameters into a dict for the controller."""
        self._auto_fix_grid_origin_if_needed()
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            raise ValueError("No samples remain after applying the selected domain filter.")
        sill_partial = self.sill_spin.value()
        nugget = self.nugget_spin.value()
        sill_total = sill_partial + nugget  # AUDIT: Always send total sill to engines

        return {
            "data_df": filtered_df,
            "variable": self.variable_combo.currentText(),
            "base_method": self.base_method_combo.currentText(),
            "secondary_variable": self.secondary_combo.currentText(),
            "drift_type": self.drift_combo.currentText().lower(),
            "soft_source": self.soft_source_combo.currentText(),
            "soft_path": self.csv_path_edit.text(),
            "config": {
                "prior_type": "mean_var" if "Variance" in self.prior_mode_combo.currentText() else "mean_only",
                "soft_weighting": self.soft_weight_spin.value(),
            },
            "variogram_params": {
                "range": self.range_spin.value(),
                "sill": sill_total,  # TOTAL sill (C₀ + C₁) — canonical convention
                "nugget": nugget,
                "model_type": self.model_combo.currentText().lower(),
            },
            "search_params": {
                "n_neighbors": self.ndmax_spin.value(),
                "max_distance": self.max_dist_spin.value(),
                "min_neighbors": self.nmin_spin.value(),
            },
            "grid_origin": (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()),
            "grid_spacing": (self.dx_spin.value(), self.dy_spin.value(), self.dz_spin.value()),
            "grid_counts": (self.nx_spin.value(), self.ny_spin.value(), self.nz_spin.value()),
            "domain_column": getattr(self, "_active_domain_filter_metadata", {}).get("domain_filter_column"),
            "domain_value": getattr(self, "_active_domain_filter_metadata", {}).get("domain_filter_value"),
            "domain_mask": self._resolve_domain_mask(),
        }

    def validate_inputs(self) -> bool:
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return False
        if not self.variable_combo.currentText():
            QMessageBox.warning(self, "Error", "No variable selected.")
            return False
        method = self.base_method_combo.currentText()
        if "CoK" in method and self.variable_combo.currentText() == self.secondary_combo.currentText():
            QMessageBox.warning(self, "Error", "Primary and secondary variables must differ.")
            return False
        # Check total grid size
        total = self.nx_spin.value() * self.ny_spin.value() * self.nz_spin.value()
        if total > 5_000_000:
            reply = QMessageBox.question(
                self, "Large Grid",
                f"Grid has {total:,} blocks. This may take a while. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
        # Check drillhole coverage within estimation grid
        from .base_analysis_panel import check_drillhole_grid_coverage
        if not check_drillhole_grid_coverage(
            self, filtered_df,
            xmin=self.xmin_spin.value(), ymin=self.ymin_spin.value(), zmin=self.zmin_spin.value(),
            nx=self.nx_spin.value(), ny=self.ny_spin.value(), nz=self.nz_spin.value(),
            dx=self.dx_spin.value(), dy=self.dy_spin.value(), dz=self.dz_spin.value(),
            panel_name="Bayesian Kriging",
        ):
            return False
        return True

    def run_analysis(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not connected.")
            return
        if not self.validate_inputs():
            return

        try:
            params = self.gather_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "Parameter Error", str(exc))
            return
        self.show_progress("Starting Bayesian Kriging...")
        self.tabs.setCurrentIndex(1)
        self.log_text.clear()
        self._log(f"Running Bayesian {params['base_method']} on '{params['variable']}'")
        self._log(f"  Soft weight: {params['config']['soft_weighting']:.2f}")
        self._log(f"  Grid: {params['grid_counts']}")

        def progress_cb(pct, msg):
            self.progress_updated.emit(int(pct), msg)

        params['progress_callback'] = progress_cb

        self.controller.run_task(
            self.task_name,
            params,
            callback=self._on_complete,
            progress_callback=progress_cb,
        )

    def _on_complete(self, result: Dict[str, Any]):
        self.hide_progress()

        if result is None:
            self._log("Kriging returned no result.", "error")
            QMessageBox.critical(self, "Error", "No result returned.")
            return

        if result.get("error"):
            self._log(f"Error: {result['error']}", "error")
            QMessageBox.critical(self, "Error", str(result['error']))
            return

        result = self._merge_domain_filter_metadata(result)
        self.kriging_results = result
        self.view_table_btn.setEnabled(True)

        # --- Domain masking ---
        if isinstance(self.kriging_results, dict):
            self.kriging_results = self._apply_domain_masking(
                self.kriging_results,
                grade_keys=['estimates', 'estimate', 'grade', 'mean'],
                variance_keys=['variances', 'kriging_variance', 'variance', 'posterior_variance'],
            )

        # Register results
        if self.registry:
            try:
                self.registry.register_soft_kriging_results(self.kriging_results, source_panel="Bayesian Kriging")
                self._log("Results registered to DataRegistry")
            except Exception as e:
                self._log(f"Warning: Registry error: {e}", "warning")

        n_valid = np.sum(~np.isnan(np.asarray(result.get('estimates', [])).ravel()))
        self._log(f"Bayesian Kriging complete: {n_valid:,} valid estimates")
        QMessageBox.information(self, "Complete", "Bayesian Kriging finished successfully.")

    # =========================================================
    # PROGRESS / LOGGING
    # =========================================================

    def _log(self, msg, level="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": ModernColors.TEXT_PRIMARY,
            "success": ModernColors.SUCCESS,
            "warning": ModernColors.WARNING,
            "error": ModernColors.ERROR,
        }
        c = colors.get(level, ModernColors.TEXT_PRIMARY)
        self.log_text.append(
            f'<span style="color:{ModernColors.TEXT_HINT};">[{ts}]</span>'
            f' <span style="color:{c};">{msg}</span>'
        )

    def show_progress(self, message: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText(message)
        self.run_btn.setEnabled(False)

    def hide_progress(self):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("")
        self.run_btn.setEnabled(True)

    def _update_progress(self, pct, msg):
        self.progress_bar.setValue(max(0, min(100, pct)))
        if msg:
            self.progress_bar.setFormat(f"{pct}% — {msg}")
            self.lbl_status.setText(msg)

    # =========================================================
    # RESULTS TABLE
    # =========================================================

    def _open_results_table(self):
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Run kriging first.")
            return
        try:
            grid_x = self.kriging_results.get('grid_x')
            estimates = self.kriging_results.get('estimates')
            variances = self.kriging_results.get('variances')
            var_name = self.kriging_results.get('variable', 'estimate')

            if grid_x is None or estimates is None:
                QMessageBox.warning(self, "Error", "Incomplete result data.")
                return

            grid_y = self.kriging_results['grid_y']
            grid_z = self.kriging_results['grid_z']
            coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
            df = pd.DataFrame({
                'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2],
                f'{var_name}_est': np.asarray(estimates).ravel(),
            })
            if variances is not None:
                df[f'{var_name}_var'] = np.asarray(variances).ravel()
            df = df.dropna()

            title = f"Bayesian Kriging — {var_name}"
            parent = self.parent()
            main_window = None
            while parent:
                if hasattr(parent, 'open_table_viewer_window_from_df'):
                    main_window = parent
                    break
                parent = parent.parent()

            if main_window:
                main_window.open_table_viewer_window_from_df(df, title=title)
            else:
                from .table_viewer_panel import TableViewerPanel
                dlg = QDialog(self)
                dlg.setWindowTitle(title)
                dlg.resize(900, 700)
                layout = QVBoxLayout(dlg)
                tv = TableViewerPanel()
                tv.set_dataframe(df)
                layout.addWidget(tv)
                dlg.show()
        except Exception as e:
            logger.error(f"Results table error: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", str(e))

    # =========================================================
    # PROJECT SAVE / RESTORE
    # =========================================================

    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        try:
            from .panel_settings_utils import get_safe_widget_value
            s = {}
            for attr in [
                'base_method_combo', 'variable_combo', 'secondary_combo', 'drift_combo',
                'soft_source_combo', 'prior_mode_combo', 'model_combo',
            ]:
                s[attr] = get_safe_widget_value(self, attr)
            for attr in [
                'range_spin', 'sill_spin', 'nugget_spin', 'soft_weight_spin',
                'ndmax_spin', 'max_dist_spin', 'nmin_spin',
                'xmin_spin', 'ymin_spin', 'zmin_spin',
                'dx_spin', 'dy_spin', 'dz_spin',
                'nx_spin', 'ny_spin', 'nz_spin',
            ]:
                s[attr] = get_safe_widget_value(self, attr)
            s['csv_path'] = self.csv_path_edit.text() if hasattr(self, 'csv_path_edit') else None
            return {k: v for k, v in s.items() if v is not None} or None
        except Exception as e:
            logger.warning(f"Save settings failed: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        if not settings:
            return
        try:
            from .panel_settings_utils import set_safe_widget_value
            for attr, val in settings.items():
                if attr == 'csv_path' and hasattr(self, 'csv_path_edit'):
                    self.csv_path_edit.setText(val or "")
                else:
                    set_safe_widget_value(self, attr, val)
            logger.info("Restored Bayesian Kriging panel settings")
        except Exception as e:
            logger.warning(f"Restore settings failed: {e}")


# =========================================================
# UTILITY
# =========================================================

def _group_style(color: str) -> str:
    return (
        f"QGroupBox {{ font-weight: bold; color: {color}; "
        f"border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}"
    )
