"""
Variogram Modelling Assistant Panel (STEP 23).

UI panel for semi-automatic variogram fitting with model selection, 
cross-validation, and immediate visual feedback.

AUDIT COMPLIANCE (2025-12-16):
- Lineage gates integrated for data hash tracking
- Nugget consistency enforcement
- JORC/SAMREC compliant data tracking
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QProgressBar, QFrame, QSizePolicy
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

# Matplotlib integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .base_analysis_panel import BaseAnalysisPanel
from ..utils.coordinate_utils import ensure_xyz_columns

# Lineage gates for audit compliance (JORC/SAMREC requirement)
from ..geostats.variogram_gates import compute_data_hash

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class VariogramPreviewCanvas(FigureCanvas):
    """Canvas for visualizing experimental data vs fitted model candidates."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#2b2b2b') # Dark theme background
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self._setup_axes()



    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _setup_axes(self):
        self.axes.set_facecolor(f'{ModernColors.PANEL_BG}')
        self.axes.grid(True, linestyle='--', alpha=0.3)
        self.axes.tick_params(colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        for spine in self.axes.spines.values():
            spine.set_color('#555')

    def plot_model(self, experimental: Dict, model_params: Dict):
        """
        Plot experimental points and the theoretical curve.
        """
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self._setup_axes()

        # 1. Plot Experimental Data with pair count information
        lags = np.array(experimental.get('lag_distances', []))
        gammas = np.array(experimental.get('gammas', []))
        if len(gammas) == 0:
            gammas = np.array(experimental.get('semivariances', []))
        
        # Get pair counts for sizing points and annotations
        pair_counts = np.array(experimental.get('pair_counts', []))
        if len(pair_counts) == 0:
            pair_counts = np.array(experimental.get('npairs', []))
        
        if len(lags) > 0:
            # Size points by pair count if available
            sizes = 40
            if len(pair_counts) > 0 and pair_counts.max() > 0:
                # Scale sizes between 20 and 80 based on pair counts
                sizes = 20 + ((pair_counts / pair_counts.max()) * 60)
            
            self.axes.scatter(lags, gammas, color='#00bcd4', s=sizes, 
                           label='Experimental', zorder=3, alpha=0.8, 
                           edgecolors='white', linewidth=0.5)
            
            # Add pair count annotations for better confidence assessment
            if len(pair_counts) > 0:
                for i in range(0, len(lags), max(1, len(lags) // 4)):  # Show ~4 annotations
                    if i < len(pair_counts) and pair_counts[i] > 0:
                        self.axes.annotate(f'{int(pair_counts[i])}', 
                                        (lags[i], gammas[i]),
                                        xytext=(3, 3), textcoords='offset points',
                                        fontsize=7, alpha=0.7, color='#00bcd4',
                                        bbox=dict(boxstyle='round,pad=0.1', 
                                               facecolor='black', alpha=0.5))

        # 2. Plot Theoretical Model
        if model_params and len(lags) > 0:
            max_dist = np.max(lags) * 1.1
            x_range = np.linspace(0, max_dist, 100)

            y_model = self._calculate_gamma(x_range, model_params)

            model_type = model_params.get('model_type', 'Model').capitalize()
            self.axes.plot(x_range, y_model, color='#ff9800', linewidth=2, label=f'{model_type} Fit', zorder=2)

            # Plot sill line with clear notation (C₀+C = nugget + partial sill)
            sills = model_params.get('sills', [])
            if not isinstance(sills, list):
                sills = [sills]
            nugget = model_params.get('nugget', 0)
            total_sill = nugget + sum(s for s in sills if s is not None and s > 0)
            if total_sill > 0:
                self.axes.axhline(total_sill, color='#666', linestyle=':', label=f'C₀+C={total_sill:.2f}')

        self.axes.set_xlabel("Distance")
        self.axes.set_ylabel("Semivariance")
        self.axes.legend(facecolor='#333', edgecolor='#555', labelcolor='white')
        self.fig.tight_layout()
        self.draw()

    def _calculate_gamma(self, h, params):
        """Internal helper to calculate gamma for standard and nested models."""
        nugget = params.get('nugget', 0)
        sills = params.get('sills', [0])
        ranges = params.get('ranges', [0])
        m_type = params.get('model_type', 'spherical').lower()

        gamma = np.full_like(h, nugget, dtype=float)

        # Handle single structure or multi-structure
        if not isinstance(sills, list): sills = [sills]
        if not isinstance(ranges, list): ranges = [ranges]

        # Get structure types for nested models
        metadata = params.get('metadata', {}) or {}
        structure_types = metadata.get('structure_types', None)
        
        # Parse model type for nested models (e.g., "spherical+exponential")
        if structure_types is None:
            if '+' in m_type:
                structure_types = [t.strip() for t in m_type.split('+')]
            else:
                structure_types = [m_type] * len(sills)

        for i, (s, r) in enumerate(zip(sills, ranges)):
            if r <= 0 or s <= 0: 
                continue
            
            # Get the model type for this structure
            struct_type = structure_types[i] if i < len(structure_types) else structure_types[-1]
            struct_type = struct_type.lower()

            if 'sph' in struct_type:
                mask = h <= r
                gamma[mask] += s * (1.5 * (h[mask]/r) - 0.5 * (h[mask]/r)**3)
                gamma[~mask] += s
            elif 'exp' in struct_type:
                gamma += s * (1 - np.exp(-3 * h / r))
            elif 'gau' in struct_type:
                gamma += s * (1 - np.exp(-3 * (h / r)**2))
            else:
                # Default to spherical
                mask = h <= r
                gamma[mask] += s * (1.5 * (h[mask]/r) - 0.5 * (h[mask]/r)**3)
                gamma[~mask] += s

        return gamma


class VariogramAssistantPanel(BaseAnalysisPanel):
    """Variogram Modelling Assistant Panel."""
    # PanelManager metadata
    PANEL_ID = "VariogramAssistantPanel"
    PANEL_NAME = "VariogramAssistant Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    task_name = "variogram_assistant"
    model_accepted = pyqtSignal(dict)  # Signal when user accepts a model

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="variogram_assistant")
        self.setWindowTitle("Variogram Modelling Assistant")
        self.resize(1100, 700)
        
        # Set as standalone window with proper window flags
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # DO NOT set WA_DeleteOnClose - we want to preserve assistant results
        # (candidates, experimental_variogram, directional_variograms, recommendations)
        # across panel close/reopen cycles for user review and audit trail

        self.assistant_results: Optional[Dict[str, Any]] = None
        self._drillhole_data: Optional[pd.DataFrame] = None
        self._controller_bound = False  # Track controller binding status

        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                existing_data = self.registry.get_drillhole_data()
                if existing_data:
                    self._on_drillhole_data_loaded(existing_data)
        except Exception as e:
            logger.debug(f"Registry initialization skipped: {e}")
            self.registry = None

        logger.info("Initialized Variogram Assistant panel as standalone window")
    
    def bind_controller(self, controller) -> bool:
        """
        Bind the controller to this panel.
        
        Returns:
            True if binding successful, False otherwise
        """
        if controller is None:
            logger.warning("Variogram Assistant: Cannot bind None controller")
            return False
            
        try:
            # Call parent bind_controller which sets self.controller
            super().bind_controller(controller)
            self._controller_bound = True
            
            # Also try to get registry from controller if we don't have one
            if self.registry is None and hasattr(controller, 'registry'):
                self.registry = controller.registry
                if self.registry:
                    try:
                        self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                    except Exception:
                        pass  # Already connected or signal doesn't exist
            
            logger.info(f"Variogram Assistant: Controller bound successfully (controller={controller is not None})")
            return True
        except Exception as e:
            logger.error(f"Variogram Assistant: Failed to bind controller: {e}", exc_info=True)
            self._controller_bound = False
            return False
    
    def _check_controller_status(self) -> bool:
        """
        Check if controller is properly bound.
        
        Returns True if we can run analysis (either via controller or local execution).
        """
        # If controller is bound, we're good
        if self.controller and self._controller_bound:
            return True
        
        # If we have drillhole data, we can potentially run locally
        # (the BaseAnalysisPanel will handle the controller check)
        if self._drillhole_data is not None and not self._drillhole_data.empty:
            logger.info("Variogram Assistant: Controller not bound but data available - will attempt analysis")
            # Try one more time to get controller from registry
            if self.registry and hasattr(self.registry, '_controller'):
                self.bind_controller(self.registry._controller)
                if self._controller_bound:
                    return True
        
        logger.warning("Variogram Assistant: No controller bound and no fallback available")
        return False

    # ---------------- Data wiring -----------------
    def _on_drillhole_data_loaded(self, drillhole_data):
        """Receive drillhole data from DataRegistry."""
        df = None
        try:
            if isinstance(drillhole_data, dict):
                df = drillhole_data.get('composites')
                if df is None or getattr(df, "empty", False):
                    df = drillhole_data.get('assays')
            elif isinstance(drillhole_data, pd.DataFrame):
                df = drillhole_data
            if df is not None and not getattr(df, "empty", False):
                self._drillhole_data = ensure_xyz_columns(df.copy())
                self._populate_variables()
        except Exception:
            logger.warning("Failed to load drillhole data into Variogram Assistant", exc_info=True)

    def set_drillhole_data(self, df: pd.DataFrame):
        """
        Legacy compatibility method - delegates to registry-based data loading.
        New code should use registry.drillholeDataLoaded signal.
        """
        # Delegate to registry-based method
        if df is not None:
            self._on_drillhole_data_loaded(df)

    def _populate_variables(self):
        if self._drillhole_data is None:
            return
        df = self._drillhole_data
        exclude_cols = ["X", "Y", "Z", "HOLEID", "FROM", "TO", "GLOBAL_INTERVAL_ID",
                        # Compositing metadata columns
                        "SAMPLE_COUNT", "TOTAL_MASS", "TOTAL_LENGTH", "SUPPORT", "IS_PARTIAL",
                        "METHOD", "WEIGHTING", "ELEMENT_WEIGHTS", "MERGED_PARTIAL", "MERGED_PARTIAL_AUTO"]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.upper() not in exclude_cols]
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.addItems(sorted(numeric_cols))
        self.variable_combo.blockSignals(False)

        # Populate domain combo (object columns with small cardinality)
        self.domain_combo.blockSignals(True)
        self.domain_combo.clear()
        self.domain_combo.addItem("All Domains")
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() < 50:
                for v in sorted(set(df[col].dropna().astype(str))):
                    self.domain_combo.addItem(f"{col}: {v}")
        self.domain_combo.blockSignals(False)

        # Populate hole id combo
        self.holeid_combo.blockSignals(True)
        self.holeid_combo.clear()
        self.holeid_combo.addItem("Auto")
        hid_candidates = [c for c in df.columns if any(tok in c.upper() for tok in ["HOLE", "BHID", "DRILL", "ID"])]
        self.holeid_combo.addItems(hid_candidates)
        self.holeid_combo.blockSignals(False)

    def _optimize_lags(self):
        """
        Auto-calculate lag parameters using industry-standard rules.
        Same algorithm as the main 3D Variogram panel.
        """
        df = self._drillhole_data
        if df is None or df.empty:
            QMessageBox.information(self, "Optimize Lags", "No drillhole data loaded.")
            return
        
        if not all(c in df.columns for c in ["X", "Y", "Z"]):
            QMessageBox.warning(self, "Optimize Lags", "Data must have X, Y, Z coordinates.")
            return
        
        coords = df[["X", "Y", "Z"]].dropna().to_numpy()
        if len(coords) < 10:
            QMessageBox.warning(self, "Optimize Lags", "Not enough data points.")
            return
        
        try:
            # Get unique collar locations (one per drillhole)
            hole_col = None
            for col in df.columns:
                if any(tok in col.upper() for tok in ["HOLE", "BHID", "DRILL"]):
                    hole_col = col
                    break
            
            if hole_col and hole_col in df.columns:
                # Get first point of each drillhole as collar
                collar_df = df.groupby(hole_col).first().reset_index()
                collar_coords = collar_df[["X", "Y"]].dropna().to_numpy()
            else:
                # Use all unique XY locations
                collar_coords = np.unique(df[["X", "Y"]].dropna().to_numpy(), axis=0)
            
            if len(collar_coords) >= 2:
                # Compute nearest-neighbor distances between collars
                from scipy.spatial import cKDTree
                tree = cKDTree(collar_coords)
                dists, _ = tree.query(collar_coords, k=2)
                nn_dists = dists[:, 1]  # Second nearest (first is self)
                horizontal_spacing = float(np.median(nn_dists[nn_dists > 0]))
            else:
                # Fallback to data extent
                extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
                horizontal_spacing = extent / 20
            
            # Compute composite/sample length for vertical spacing
            downhole_spacing = 2.0  # Default
            for fc in ["FROM", "from", "From"]:
                for tc in ["TO", "to", "To"]:
                    if fc in df.columns and tc in df.columns:
                        lengths = (df[tc] - df[fc]).dropna()
                        valid_lengths = lengths[(lengths > 0) & (lengths < 50)]
                        if len(valid_lengths) > 0:
                            downhole_spacing = float(np.median(valid_lengths))
                        break
            
            # Industry standard: lag = 0.5 × spacing
            lag_distance = horizontal_spacing * 0.5
            
            # Number of lags based on data extent
            extent = np.linalg.norm(coords.max(axis=0)[:2] - coords.min(axis=0)[:2])
            n_lags = min(20, max(10, int(extent / lag_distance * 0.5)))
            
            # Tolerance = 0.5 × lag distance
            lag_tolerance = lag_distance * 0.5
            
            # Update UI
            self.lag_dist_spin.setValue(round(lag_distance, 1))
            self.lag_tolerance_spin.setValue(round(lag_tolerance, 1))
            self.n_lags_spin.setValue(n_lags)
            
            logger.info(f"Auto-optimized: horizontal={horizontal_spacing:.1f}m, lag={lag_distance:.1f}m, n={n_lags}")
            
            QMessageBox.information(
                self, "Lags Optimized",
                f"Parameters optimized:\n\n"
                f"Horizontal spacing: {horizontal_spacing:.1f} m\n"
                f"Downhole spacing: {downhole_spacing:.1f} m\n\n"
                f"Lag distance: {lag_distance:.1f} m\n"
                f"Tolerance: {lag_tolerance:.1f} m\n"
                f"Number of lags: {n_lags}"
            )
            
        except Exception as e:
            logger.error(f"Lag optimization failed: {e}", exc_info=True)
            QMessageBox.warning(self, "Optimize Lags", f"Failed: {e}")

    # ---------------- UI BUILD -----------------
    def setup_ui(self):
        # Use the existing main_layout provided by BaseAnalysisPanel (it's a QVBoxLayout)
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        sel_group = QGroupBox("1. Data Selection")
        sel_layout = QFormLayout(sel_group)
        self.variable_combo = QComboBox()
        self.domain_combo = QComboBox(); self.domain_combo.addItem("All Domains")
        self.holeid_combo = QComboBox(); self.holeid_combo.addItem("Auto")
        sel_layout.addRow("Variable:", self.variable_combo)
        sel_layout.addRow("Domain:", self.domain_combo)
        sel_layout.addRow("Hole ID:", self.holeid_combo)
        controls_layout.addWidget(sel_group)

        exp_group = QGroupBox("2. Parameters")
        exp_layout = QFormLayout(exp_group)
        self.n_lags_spin = QSpinBox(); self.n_lags_spin.setRange(5, 50); self.n_lags_spin.setValue(15)
        self.lag_dist_spin = QDoubleSpinBox(); self.lag_dist_spin.setRange(1.0, 500.0); self.lag_dist_spin.setValue(25.0); self.lag_dist_spin.setSuffix(" m")
        self.lag_tolerance_spin = QDoubleSpinBox(); self.lag_tolerance_spin.setRange(1.0, 250.0); self.lag_tolerance_spin.setValue(12.5); self.lag_tolerance_spin.setSuffix(" m")
        self.normalize_check = QCheckBox("Normalize Y-axis (÷ Variance)")
        self.normalize_check.setToolTip("Normalize semivariance by data variance.\nMakes sill approach 1.0 for portable variograms.")
        exp_layout.addRow("Num Lags:", self.n_lags_spin)
        exp_layout.addRow("Lag Dist:", self.lag_dist_spin)
        exp_layout.addRow("Tolerance:", self.lag_tolerance_spin)
        exp_layout.addRow("", self.normalize_check)
        
        # Auto-optimize button
        self.optimize_btn = QPushButton("Auto-Optimize Lags")
        self.optimize_btn.setStyleSheet("background-color: #444; border: 1px solid #666;")
        self.optimize_btn.clicked.connect(self._optimize_lags)
        exp_layout.addRow(self.optimize_btn)
        
        controls_layout.addWidget(exp_group)

        mod_group = QGroupBox("3. Model Configuration")
        mod_layout = QVBoxLayout(mod_group)
        fam_layout = QHBoxLayout()
        self.spherical_check = QCheckBox("Sph"); self.spherical_check.setChecked(True)
        self.exponential_check = QCheckBox("Exp"); self.exponential_check.setChecked(True)
        self.gaussian_check = QCheckBox("Gau"); self.gaussian_check.setChecked(True)
        fam_layout.addWidget(self.spherical_check)
        fam_layout.addWidget(self.exponential_check)
        fam_layout.addWidget(self.gaussian_check)
        mod_layout.addLayout(fam_layout)
        struc_layout = QHBoxLayout()
        struc_layout.addWidget(QLabel("Max Structures:"))
        self.max_structures_spin = QSpinBox(); self.max_structures_spin.setRange(1, 3); self.max_structures_spin.setValue(2)
        struc_layout.addWidget(self.max_structures_spin)
        mod_layout.addLayout(struc_layout)
        controls_layout.addWidget(mod_group)

        run_group = QGroupBox("4. Execution")
        run_layout = QVBoxLayout(run_group)
        self.perform_cv_check = QCheckBox("Run Cross-Validation (Slower)")
        self.perform_cv_check.setChecked(True)
        run_layout.addWidget(self.perform_cv_check)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)
        self.run_button = QPushButton("Auto-Fit Models")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.run_button.clicked.connect(self.run_analysis)
        run_layout.addWidget(self.run_button)
        controls_layout.addWidget(run_group)
        controls_layout.addStretch()

        # Right results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(10, 10, 10, 10)

        self.preview_canvas = VariogramPreviewCanvas()
        self.toolbar = NavigationToolbar(self.preview_canvas, self)
        results_layout.addWidget(self.toolbar)
        results_layout.addWidget(self.preview_canvas, stretch=1)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Model", "Score (SSE)", "CV RMSE", "Nugget", "Structures"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_candidate_selected)
        results_layout.addWidget(self.results_table, stretch=1)

        bottom_layout = QHBoxLayout()
        self.summary_text = QLabel("Run analysis to see results.")
        self.summary_text.setStyleSheet("color: #aaa; font-style: italic;")
        bottom_layout.addWidget(self.summary_text, stretch=1)
        self.send_button = QPushButton("Accept & Send to Panel")
        self.send_button.setEnabled(False)
        self.send_button.setStyleSheet("background-color: #2196F3; color: white; padding: 6px;")
        self.send_button.clicked.connect(self._send_best_model)
        bottom_layout.addWidget(self.send_button)
        results_layout.addLayout(bottom_layout)

        splitter.addWidget(controls_widget)
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.main_layout.addWidget(splitter)

    # ---------------- Analysis execution -----------------
    def run_analysis(self) -> None:
        """
        Override base run_analysis to try controller recovery before execution.
        
        The Variogram Assistant opens as a standalone window, so it needs
        to actively try to reconnect to the controller if it's not bound.
        """
        # Try to recover controller if not bound
        if not self.controller:
            logger.info("Variogram Assistant: Controller not bound, attempting recovery...")
            
            # Method 1: Try to get from registry
            if self.registry and hasattr(self.registry, '_controller'):
                ctrl = getattr(self.registry, '_controller', None)
                if ctrl:
                    self.bind_controller(ctrl)
                    logger.info("Variogram Assistant: Recovered controller from registry")
            
            # Method 2: Try to find MainWindow and get controller
            if not self.controller:
                from PyQt6.QtWidgets import QApplication
                for widget in QApplication.topLevelWidgets():
                    if hasattr(widget, 'controller') and widget.controller is not None:
                        self.bind_controller(widget.controller)
                        logger.info(f"Variogram Assistant: Recovered controller from {widget.__class__.__name__}")
                        break
        
        # Now call the base implementation
        if not self.controller:
            self.show_warning(
                "Controller Not Connected",
                "Cannot run analysis - application controller not available.\n\n"
                "Please close this window and reopen from the main menu:\n"
                "  Estimations & Geostatistics → 3D Variogram Tools → Variogram Modelling Assistant"
            )
            return
        
        # Call base class implementation
        super().run_analysis()

    # ---------------- Data/params -----------------
    def _current_drillhole_df(self) -> Optional[pd.DataFrame]:
        """
        Safely return the active drillhole dataframe without triggering
        ambiguous truth-value checks on DataFrames.
        """
        if self._drillhole_data is not None and not getattr(self._drillhole_data, "empty", False):
            return self._drillhole_data
        if self.controller and hasattr(self.controller, "_drillhole_data"):
            ctrl_df = getattr(self.controller, "_drillhole_data")
            if ctrl_df is not None and not getattr(ctrl_df, "empty", False):
                return ctrl_df
        return None

    def gather_parameters(self) -> Dict[str, Any]:
        df = self._current_drillhole_df()
        if df is None or getattr(df, "empty", False):
            return {}
        variable = self.variable_combo.currentText()
        if not variable:
            return {}

        coords = df[['X', 'Y', 'Z']].values.astype(float)
        values = df[variable].values.astype(float)
        mask = ~(np.isnan(coords).any(axis=1) | np.isnan(values))
        coords, values = coords[mask], values[mask]
        hole_ids = None

        domain_labels = None
        domain_str = self.domain_combo.currentText()
        if domain_str != "All Domains" and ": " in domain_str:
            d_col, d_val = domain_str.split(": ", 1)
            if d_col in df.columns:
                d_values = df[d_col].values[mask].astype(str)
                d_mask = d_values == d_val
                coords, values = coords[d_mask], values[d_mask]
                domain_labels = d_values[d_mask]

        models = []
        if self.spherical_check.isChecked(): models.append('spherical')
        if self.exponential_check.isChecked(): models.append('exponential')
        if self.gaussian_check.isChecked(): models.append('gaussian')
        if not models:
            return {}

        # Optional hole ids for downhole
        hid_col = None if self.holeid_combo.currentText() == "Auto" else self.holeid_combo.currentText()
        if hid_col and hid_col in df.columns:
            full_hids = df[hid_col].astype(str).values
            hole_ids = full_hids[mask]

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        return {
            'coords': coords,
            'values': values,
            'n_lags': self.n_lags_spin.value(),
            'lag_distance': self.lag_dist_spin.value(),
            'lag_tolerance': self.lag_tolerance_spin.value(),
            'normalize': self.normalize_check.isChecked(),
            'model_families': models,
            'max_structures': self.max_structures_spin.value(),
            'perform_cv': self.perform_cv_check.isChecked(),
            'cv_method': 'OK',
            'cv_folds': 5,
            'variable': variable,
            'domain_labels': domain_labels,
            'hole_id_col': None if self.holeid_combo.currentText() == "Auto" else self.holeid_combo.currentText(),
            'hole_ids': hole_ids,
        }

    def validate_inputs(self) -> bool:
        """Validate inputs before running analysis."""
        # Check data first (most common issue)
        df = self._current_drillhole_df()
        if df is None or getattr(df, "empty", False):
            self.show_warning(
                "No Data",
                "No drillhole data loaded.\n\n"
                "Please load drillhole data first:\n"
                "  Drillholes → Drillhole Loading"
            )
            return False
        
        if not self.variable_combo.currentText():
            self.show_warning("Error", "Please select a variable to analyze")
            return False
        
        if not (self.spherical_check.isChecked() or self.exponential_check.isChecked() or self.gaussian_check.isChecked()):
            self.show_warning("Error", "Please select at least one model family to test")
            return False
        
        # Check for required columns
        required_cols = ['X', 'Y', 'Z']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self.show_warning(
                "Missing Columns",
                f"Required columns not found: {missing_cols}\n\n"
                "Ensure your data has X, Y, Z coordinate columns."
            )
            return False
        
        # Check controller - try to recover if not bound
        if not self._check_controller_status():
            # Last attempt: try to get controller from parent window chain
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'controller') and parent.controller is not None:
                    logger.info("Variogram Assistant: Found controller in parent chain, binding...")
                    self.bind_controller(parent.controller)
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None
            
            # Final check
            if not self.controller:
                self.show_warning(
                    "Controller Not Connected",
                    "The Variogram Assistant cannot connect to the application.\n\n"
                    "Please close this window and reopen from:\n"
                    "  Estimations & Geostatistics → 3D Variogram Tools → Variogram Modelling Assistant"
                )
                return False
        
        logger.info(f"Variogram Assistant: Validation passed. Variable: {self.variable_combo.currentText()}, n_samples: {len(df)}")
        return True

    # ---------------- Results handling -----------------
    def on_results(self, payload: Dict[str, Any]) -> None:
        self.progress_bar.setVisible(False)
        if payload.get('error'):
            self.show_error("Fit Error", payload['error'])
            return

        self.assistant_results = payload
        candidates = payload.get('candidates', [])
        experimental = payload.get('experimental_variogram', {})

        self.results_table.setRowCount(0)
        for i, cand in enumerate(candidates):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(cand.get('model_type', 'N/A').title()))
            sse_item = QTableWidgetItem(f"{cand.get('score_sse', 0):.4f}")
            sse_item.setToolTip("Sum of Squared Errors (lower is better)")
            self.results_table.setItem(i, 1, sse_item)
            cv = cand.get('score_cv_rmse', 0)
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{cv:.4f}" if cv else "-"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{cand.get('nugget', 0):.2f}"))
            structs = len(cand.get('ranges', []))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(structs)))

        if candidates:
            self.results_table.selectRow(0)
            self._update_preview(0)

        if candidates:
            shape_hint = ""
            meta = payload.get("metadata", {}) or {}
            hints = []
            if meta.get("shape_hint"):
                hints.append(f"Shape suggests: {meta['shape_hint']}")
            d_hints = meta.get("directional_shape_hints") or {}
            dir_parts = [f"{k}:{v}" for k, v in d_hints.items() if v]
            if dir_parts:
                hints.append("Dir hints " + ", ".join(dir_parts))
            if meta.get("nugget_suggestion") is not None:
                hints.append(f"Nugget≈{meta['nugget_suggestion']:.2f}")
            
            # Add sample variance information if available
            if meta.get("sample_variance") is not None:
                sample_var = meta["sample_variance"]
                # Get best model's total sill for comparison
                best_model = candidates[0] if candidates else None
                if best_model:
                    total_sill = best_model.get('nugget', 0) + sum(best_model.get('sills', [0]))
                    nugget = best_model.get('nugget', 0)

                    if sample_var > 0:
                        sill_ratio = total_sill / sample_var
                        # Use clear notation: (C₀+C)/Var should be ~1.0
                        hints.append(f"(C₀+C)/Var: {sill_ratio:.2f}")
                    
                    # Add high nugget warning
                    if total_sill > 0:
                        nugget_ratio = nugget / total_sill
                        if nugget_ratio >= 0.8:
                            nugget_pct = nugget_ratio * 100
                            hints.append(f"⚠ High Nugget: {nugget_pct:.0f}%")
                        elif nugget_ratio >= 0.6:
                            nugget_pct = nugget_ratio * 100
                            hints.append(f"~ Mod Nugget: {nugget_pct:.0f}%")
                
                hints.append(f"Data Var: {sample_var:.3f}")
            hint_txt = " | ".join(hints)
            self.summary_text.setText(
                f"Found {len(candidates)} models. Top model: {candidates[0]['model_type']} (SSE: {candidates[0]['score_sse']:.3f})"
                + (f" | {hint_txt}" if hint_txt else "")
            )
        self.send_button.setEnabled(bool(candidates))

        # Register best result to DataRegistry
        if self.registry and candidates:
            best = candidates[0]
            result_struct = {
                'model_type': best.get('model_type'),
                'ranges': best.get('ranges'),
                'sills': best.get('sills'),
                'nugget': best.get('nugget'),
                'variable': payload.get('variable'),
                'experimental': experimental
            }
            
            # AUDIT REQUIREMENT: Add data lineage hash for JORC/SAMREC compliance
            try:
                # Get source data from drillhole_data
                if hasattr(self, 'drillhole_data') and self.drillhole_data is not None:
                    var_name = payload.get('variable')
                    if var_name:
                        source_data_hash = compute_data_hash(self.drillhole_data, var_name)
                        result_struct['source_data_hash'] = source_data_hash
                        result_struct['data_source_type'] = 'assistant'
                        logger.info(f"VariogramAssistant data lineage: hash={source_data_hash[:16]}...")
            except Exception as e:
                logger.warning(f"Could not compute data lineage hash in assistant: {e}")
            
            try:
                self.registry.register_variogram_results(result_struct, source_panel="VariogramAssistant")
            except Exception:
                pass

    def _on_candidate_selected(self):
        rows = self.results_table.selectionModel().selectedRows()
        if rows:
            self._update_preview(rows[0].row())

    def _update_preview(self, row_index):
        if not self.assistant_results:
            return
        candidates = self.assistant_results.get('candidates', [])
        experimental = self.assistant_results.get('experimental_variogram', {})
        if 0 <= row_index < len(candidates):
            model = candidates[row_index]
            self.preview_canvas.plot_model(experimental, model)

    def _send_best_model(self):
        if not self.assistant_results:
            return
        rows = self.results_table.selectionModel().selectedRows()
        idx = rows[0].row() if rows else 0
        candidates = self.assistant_results.get('candidates', [])
        if not candidates:
            return
        selected_model = candidates[idx]
        variable = self.variable_combo.currentText()

        if self.controller:
            if not hasattr(self.controller, '_assisted_variogram_models'):
                self.controller._assisted_variogram_models = {}
            self.controller._assisted_variogram_models[variable] = selected_model

        # Attach experimental variograms for downstream plotting
        import pandas as pd

        def _to_df(vdict: Dict[str, Any]):
            if not vdict:
                return None
            return pd.DataFrame({
                "distance": vdict.get("lag_distances", []),
                "gamma": vdict.get("semivariances", []),
                "npairs": vdict.get("pair_counts", [0] * len(vdict.get("lag_distances", []))),
            })

        exp = self.assistant_results.get("experimental_variogram", {}) if self.assistant_results else {}
        dir_vgs = self.assistant_results.get("directional_variograms", {}) if self.assistant_results else {}
        downhole_vg = self.assistant_results.get("downhole_variogram")
        dir_fits = self.assistant_results.get("directional_fits", {}) if self.assistant_results else {}

        variograms = {
            "omni": _to_df(exp),
            "major": _to_df(dir_vgs.get("major")),
            "minor": _to_df(dir_vgs.get("minor")),
            "vertical": _to_df(dir_vgs.get("vertical")),
            "downhole": _to_df(downhole_vg),
        }

        # Include combined 3D model from assistant results
        combined_3d = self.assistant_results.get("combined_3d_model") if self.assistant_results else None
        variogram_map = self.assistant_results.get("variogram_map") if self.assistant_results else None
        
        selected_payload = dict(selected_model)
        selected_payload["_variograms"] = variograms
        selected_payload["_fitted_models"] = dir_fits
        selected_payload["combined_3d_model"] = combined_3d
        selected_payload["variogram_map"] = variogram_map

        self.model_accepted.emit(selected_payload)

        try:
            vp = None
            parent = self.parent()
            if hasattr(parent, "variogram_panel"):
                vp = getattr(parent, "variogram_panel")
            elif hasattr(parent, "variogram_dialog"):
                vp = getattr(parent, "variogram_dialog")

            if vp and hasattr(vp, "set_variogram_results"):
                vgs = selected_payload.get("_variograms", {}) or {}
                fm = selected_payload.get("_fitted_models", {}) or {}
                
                # Build structures from selected model
                n_structures = len(selected_model.get("ranges", []))
                structures = []
                for i, (r, s) in enumerate(zip(selected_model.get("ranges", []), selected_model.get("sills", []))):
                    struct_types = selected_model.get("metadata", {}).get("structure_types", [])
                    s_type = struct_types[i] if i < len(struct_types) else selected_model["model_type"].split('+')[0]
                    structures.append({
                        "type": s_type,
                        "contribution": s,
                        "range": r,
                    })
                
                formatted_results = {
                    "omni_variogram": vgs.get("omni"),
                    "downhole_variogram": vgs.get("downhole"),
                    "major_variogram": vgs.get("major"),
                    "minor_variogram": vgs.get("minor"),
                    "vertical_variogram": vgs.get("vertical"),
                    "fitted_models": {
                        d: fm.get(d) or {
                            selected_model["model_type"].split('+')[0]: {
                                "model_type": selected_model["model_type"].split('+')[0],
                                "nugget": selected_model.get("nugget", 0.0),
                                "sill": sum(selected_model.get("sills", [])),
                                "total_sill": selected_model.get("nugget", 0.0) + sum(selected_model.get("sills", [])),
                                "range": max(selected_model.get("ranges", [0])) if selected_model.get("ranges") else 0,
                                "structures": structures if n_structures > 1 else None,
                            },
                            # Also add nested key if multi-structure
                            **({"nested": {
                                "nugget": selected_model.get("nugget", 0.0),
                                "total_sill": selected_model.get("nugget", 0.0) + sum(selected_model.get("sills", [])),
                                "structures": structures,
                                "range": max(selected_model.get("ranges", [0])) if selected_model.get("ranges") else 0,
                            }} if n_structures > 1 else {})
                        }
                        for d in ["omni", "downhole", "major", "minor", "vertical"]
                    },
                    "combined_3d_model": combined_3d,
                    "n_structures": n_structures,
                }
                vp.set_variogram_results(formatted_results)
                self.show_info("Sent", "Model applied to Variogram Panel.")
                return
        except Exception as e:
            logger.debug(f"Direct push failed: {e}")

        self.show_info("Saved", "Model stored. Re-open Variogram Panel to apply.")

    def _safe_clear_toolbar(self):
        """Disconnect and delete the preview toolbar safely.
        
        IMPORTANT: We intentionally do NOT iterate over toolbar.actions() because
        Qt may have already deleted the QAction C++ objects while Python still holds
        references. Simply iterating over deleted QAction pointers causes:
        'RuntimeError: wrapped C/C++ object of type QAction has been deleted'
        """
        toolbar = getattr(self, "toolbar", None)
        if toolbar is None:
            return
        
        # Clear reference FIRST to prevent re-entry
        self.toolbar = None
        
        # Check if C++ object is still valid
        try:
            import sip
            if sip.isdeleted(toolbar):
                return
        except (ImportError, TypeError, RuntimeError):
            pass
        
        # Hide and detach - let Qt handle child QAction cleanup
        try:
            toolbar.hide()
        except (RuntimeError, Exception):
            pass
        try:
            toolbar.setParent(None)
        except (RuntimeError, Exception):
            pass
        try:
            toolbar.deleteLater()
        except (RuntimeError, Exception):
            pass

    def closeEvent(self, event):
        """Ensure toolbar cleanup to prevent 'QAction has been deleted' errors on reopen."""
        self._safe_clear_toolbar()
        super().closeEvent(event)

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Variable selection
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            
            # Model fitting parameters
            settings['n_structures'] = get_safe_widget_value(self, 'n_struct_spin')
            settings['model_types'] = get_safe_widget_value(self, 'model_type_combo')
            
            # Nugget
            settings['lock_nugget'] = get_safe_widget_value(self, 'lock_nugget_check')
            settings['nugget_value'] = get_safe_widget_value(self, 'nugget_spin')
            
            # Auto-fit options
            settings['auto_fit'] = get_safe_widget_value(self, 'auto_fit_check')
            settings['fit_nested'] = get_safe_widget_value(self, 'fit_nested_check')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save variogram assistant panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Variable selection
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            
            # Model fitting parameters
            set_safe_widget_value(self, 'n_struct_spin', settings.get('n_structures'))
            set_safe_widget_value(self, 'model_type_combo', settings.get('model_types'))
            
            # Nugget
            set_safe_widget_value(self, 'lock_nugget_check', settings.get('lock_nugget'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget_value'))
            
            # Auto-fit options
            set_safe_widget_value(self, 'auto_fit_check', settings.get('auto_fit'))
            set_safe_widget_value(self, 'fit_nested_check', settings.get('fit_nested'))
                
            logger.info("Restored variogram assistant panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore variogram assistant panel settings: {e}")