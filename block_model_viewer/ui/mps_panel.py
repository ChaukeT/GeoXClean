"""
Multiple-Point Simulation (MPS) Panel
=====================================

Refactored for Modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False
    logger.warning("tifffile not available - TIFF training image support disabled")
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QComboBox,
    QPushButton, QLabel, QMessageBox, QWidget, QSplitter, QFileDialog, QScrollArea, QFrame, QCheckBox,
    QProgressBar, QTextEdit
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QDateTime
from .base_analysis_panel import BaseAnalysisPanel
from ..utils.coordinate_utils import ensure_xyz_columns

logger = logging.getLogger(__name__)

class MPSPanel(BaseAnalysisPanel):
    task_name = "mps"
    request_visualization = pyqtSignal(object, str)
    
    def __init__(self, parent=None):
        self.training_image = None
        self.drillhole_data = None
        super().__init__(parent=parent, panel_id="mps")
        self.setWindowTitle("Multiple-Point Simulation (MPS)")
        self.resize(900, 700)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()
        
        self._init_registry()
    


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
    def _build_ui(self):
        """Build custom UI. Called by base class."""
        self._setup_ui()

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
                d = self.registry.get_drillhole_data()
                if d is not None:
                    self._on_data_loaded(d)
        except Exception:
            pass

    def _setup_ui(self):
        # Clear any existing layout from base class (BaseAnalysisPanel creates a scroll area)
        old_layout = self.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                    del item
            QWidget().setLayout(old_layout)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_ti_group(s_lay)
        self._create_pattern_group(s_lay)
        self._create_grid_group(s_lay)
        self._create_sim_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # RIGHT
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress section
        progress_box = QGroupBox("Progress")
        progress_box.setStyleSheet("QGroupBox { font-weight: bold; color: #26c6da; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        progress_layout = QVBoxLayout(progress_box)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid #555; border-radius: 3px; background: {ModernColors.CARD_BG}; height: 20px; text-align: center; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #26c6da, stop:1 #4CAF50); border-radius: 3px; }}
        """)
        progress_layout.addWidget(self.progress_bar)
        r_lay.addWidget(progress_box)
        
        # Status box
        r_box = QGroupBox("Status")
        r_box.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        r_l = QVBoxLayout(r_box)
        self.res_lbl = QLabel("Load Training Image to start.")
        self.res_lbl.setWordWrap(True)
        self.res_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        r_l.addWidget(self.res_lbl)
        r_lay.addWidget(r_box)
        
        # Event log
        log_box = QGroupBox("Event Log")
        log_box.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        log_layout = QVBoxLayout(log_box)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet(f"QTextEdit {{ background: {ModernColors.PANEL_BG}; border: 1px solid #444; font-family: monospace; font-size: 11px; }}")
        log_layout.addWidget(self.results_text)
        r_lay.addWidget(log_box)
        
        self.run_btn = QPushButton("RUN MPS")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        r_lay.addWidget(self.run_btn)
        
        # Visualization button
        self.vis_btn = QPushButton("VISUALIZE RESULT")
        self.vis_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.vis_btn.setEnabled(False)
        self.vis_btn.clicked.connect(self._visualize_results)
        r_lay.addWidget(self.vis_btn)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_ti_group(self, layout):
        g = QGroupBox("1. Training Image")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        self.load_btn = QPushButton("Load Image...")
        self.load_btn.clicked.connect(self._load_ti)
        self.ti_stat = QLabel("No image loaded.")
        self.ti_stat.setStyleSheet("color: #f39c12;")
        l.addWidget(self.load_btn)
        l.addWidget(self.ti_stat)
        layout.addWidget(g)

    def _create_pattern_group(self, layout):
        g = QGroupBox("2. Patterns")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.tx = QSpinBox()
        self.tx.setValue(5)
        self.ty = QSpinBox()
        self.ty.setValue(5)
        self.tz = QSpinBox()
        self.tz.setValue(3)
        self.max_pat = QSpinBox()
        self.max_pat.setRange(100, 100000)
        self.max_pat.setValue(10000)
        
        f.addRow("Template X/Y:", self.tx)  # simplified viz for form
        f.addRow("Template Z:", self.tz)
        f.addRow("Max Patterns:", self.max_pat)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        g = QGroupBox("3. Grid")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ba68c8; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QHBoxLayout(g)
        self.nx = QSpinBox()
        self.nx.setRange(1, 1000)
        self.nx.setValue(100)
        self.ny = QSpinBox()
        self.ny.setRange(1, 1000)
        self.ny.setValue(100)
        self.nz = QSpinBox()
        self.nz.setRange(1, 1000)
        self.nz.setValue(20)
        l.addWidget(QLabel("NX:"))
        l.addWidget(self.nx)
        l.addWidget(QLabel("NY:"))
        l.addWidget(self.ny)
        l.addWidget(QLabel("NZ:"))
        l.addWidget(self.nz)
        layout.addWidget(g)

    def _create_sim_group(self, layout):
        g = QGroupBox("4. Config")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.reals = QSpinBox()
        self.reals.setValue(10)
        self.servo = QCheckBox("Servo System")
        self.servo.setChecked(True)
        f.addRow("Reals:", self.reals)
        f.addRow("", self.servo)
        layout.addWidget(g)

    def _on_data_loaded(self, data):
        df = None
        if isinstance(data, dict):
            # Prefer composites if available and non-empty, otherwise use assays
            # Fix: Explicitly check for non-empty DataFrames to avoid ValueError
            composites = data.get('composites')
            composites_df = data.get('composites_df')
            assays_data = data.get('assays')
            assays_df = data.get('assays_df')
            
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                comp = composites
            elif isinstance(composites_df, pd.DataFrame) and not composites_df.empty:
                comp = composites_df
            else:
                comp = None
            
            if isinstance(assays_data, pd.DataFrame) and not assays_data.empty:
                assays = assays_data
            elif isinstance(assays_df, pd.DataFrame) and not assays_df.empty:
                assays = assays_df
            else:
                assays = None
            if isinstance(comp, pd.DataFrame) and not comp.empty:
                df = comp
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
        elif isinstance(data, pd.DataFrame):
            df = data
        
        if df is not None and not df.empty:
            self.drillhole_data = ensure_xyz_columns(df)

    def _load_gslib_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load GSLIB format training image.

        GSLIB format typically has:
        - First line: nx ny nz (dimensions)
        - Subsequent lines: data values in order

        Args:
            filepath: Path to GSLIB file

        Returns:
            3D numpy array or None if error
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                QMessageBox.warning(self, "GSLIB Format Error",
                                  "GSLIB file must have at least 2 lines (dimensions + data)")
                return None

            # Parse dimensions from first line
            dim_line = lines[0].strip().split()
            if len(dim_line) < 2:
                QMessageBox.warning(self, "GSLIB Format Error",
                                  "First line must contain at least nx ny dimensions")
                return None

            try:
                nx = int(dim_line[0])
                ny = int(dim_line[1])
                nz = int(dim_line[2]) if len(dim_line) > 2 else 1  # Default to 1 if 2D
            except ValueError as e:
                QMessageBox.warning(self, "GSLIB Format Error",
                                  f"Could not parse dimensions: {e}")
                return None

            # Parse data values
            data_lines = lines[1:]
            data_values = []

            for line in data_lines:
                line = line.strip()
                if line:  # Skip empty lines
                    values = line.split()
                    for val in values:
                        try:
                            data_values.append(float(val))
                        except ValueError:
                            QMessageBox.warning(self, "GSLIB Format Error",
                                              f"Non-numeric value found: {val}")
                            return None

            expected_size = nx * ny * nz
            if len(data_values) != expected_size:
                QMessageBox.warning(self, "GSLIB Format Error",
                                  f"Expected {expected_size} values, got {len(data_values)}")
                return None

            # Reshape to 3D array (nz, ny, nx) - always 3D for MPS algorithm
            data_array = np.array(data_values, dtype=np.int32)
            if nz == 1:
                # 2D case - expand to 3D with nz=1
                training_image = data_array.reshape(1, ny, nx)
            else:
                # 3D case - GSLIB is typically stored as (nx, ny, nz) but we want (nz, ny, nx)
                temp_array = data_array.reshape(nz, ny, nx)
                training_image = np.transpose(temp_array, (0, 1, 2))  # Already in correct order

            return training_image

        except Exception as e:
            QMessageBox.warning(self, "GSLIB Load Error", f"Failed to load GSLIB file: {e}")
            return None

    def _load_ti(self):
        filter_parts = ["Numpy (*.npy)", "CSV (*.csv)", "GSLIB (*.dat *.out)", "ASCII (*.txt *.asc)"]
        if TIFF_AVAILABLE:
            filter_parts.extend(["TIFF (*.tif *.tiff)", "GeoTIFF (*.tif *.tiff)"])
        filter_str = ";;".join(filter_parts)
        f, _ = QFileDialog.getOpenFileName(self, "Load TI", "", filter_str)
        if f:
            try:
                if f.endswith('.npy'):
                    loaded = np.load(f)
                    # Ensure 3D array for MPS algorithm
                    if loaded.ndim == 2:
                        self.training_image = loaded[np.newaxis, :, :]
                    else:
                        self.training_image = loaded
                elif f.endswith('.csv'):
                    df = pd.read_csv(f)
                    loaded = df.values
                    # Ensure 3D array for MPS algorithm
                    if loaded.ndim == 2:
                        self.training_image = loaded[np.newaxis, :, :].astype(np.int32)
                    else:
                        self.training_image = loaded.astype(np.int32)
                elif f.lower().endswith(('.dat', '.out')):
                    # Load GSLIB format
                    self.training_image = self._load_gslib_file(f)
                    if self.training_image is None:
                        return  # Error already shown in _load_gslib_file
                elif f.lower().endswith(('.txt', '.asc')):
                    # Load ASCII format (generic text file)
                    try:
                        # Try to load as space/tab/comma separated values
                        df = pd.read_csv(f, sep=None, engine='python')  # sep=None auto-detects
                        loaded = df.values.astype(np.int32)
                        # Ensure 3D array for MPS algorithm
                        if loaded.ndim == 2:
                            self.training_image = loaded[np.newaxis, :, :]
                        else:
                            self.training_image = loaded
                    except Exception as e:
                        QMessageBox.warning(self, "ASCII Load Error",
                                          f"Failed to load ASCII file: {e}\n"
                                          "Expected space, tab, or comma-separated numerical values.")
                        return
                elif TIFF_AVAILABLE and f.lower().endswith(('.tif', '.tiff')):
                    # Load TIFF file - tifffile handles both regular TIFF and GeoTIFF
                    tiff_data = tifffile.imread(f)
                    # Convert to numpy array and ensure integer type for categorical data
                    if tiff_data.ndim == 2:
                        # 2D TIFF - expand to 3D with nz=1 for MPS algorithm
                        self.training_image = tiff_data.astype(np.int32)[np.newaxis, :, :]
                    elif tiff_data.ndim == 3:
                        # 3D TIFF (multi-page) or RGB image
                        if tiff_data.shape[-1] in [3, 4]:  # RGB/RGBA image
                            # For RGB images, we need to convert to categorical
                            # This is a simplified approach - in practice you'd want facies mapping
                            QMessageBox.warning(self, "RGB Image Detected",
                                              "RGB images detected. For MPS, you need categorical facies data.\n"
                                              "Please convert your image to contain integer facies codes (1, 2, 3, etc.)\n"
                                              "representing different rock types, not RGB colors.")
                            return
                        else:
                            # Multi-page TIFF or 3D array
                            self.training_image = tiff_data.astype(np.int32)
                    else:
                        QMessageBox.warning(self, "Unsupported TIFF Format",
                                          f"TIFF has {tiff_data.ndim} dimensions. MPS expects 2D or 3D arrays.")
                        return
                else:
                    QMessageBox.warning(self, "Unsupported", "Unsupported file format.")
                    return
                
                shape = self.training_image.shape
                unique = len(np.unique(self.training_image[~np.isnan(self.training_image)]))
                self.ti_stat.setText(f"Loaded: {f.split('/')[-1]}\nShape: {shape}, Categories: {unique}")
                self.ti_stat.setStyleSheet("color: #27ae60;")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def gather_parameters(self) -> Dict[str, Any]:
        if self.training_image is None:
            raise ValueError("No TI")
        return {
            'training_image': self.training_image,
            'drillhole_data': self.drillhole_data,
            'template_size': (self.tz.value(), self.ty.value(), self.tx.value()),  # (Z, Y, X) order
            'max_patterns': self.max_pat.value(),
            'grid_shape': (self.nz.value(), self.ny.value(), self.nx.value()),
            'n_realizations': self.reals.value(),
            'use_servo': self.servo.isChecked()
        }

    def validate_inputs(self) -> bool:
        if self.training_image is None:
            QMessageBox.warning(self, "Error", "No TI")
            return False
        return True
    
    def _log_event(self, message: str, level: str = "info"):
        """Add timestamped message to event log."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        color_map = {"info": "#b0bec5", "success": "#81c784", "warning": "#ffb74d", "error": "#ef5350"}
        color = color_map.get(level, "#b0bec5")
        self.results_text.append(f'<span style="color:{color}">[{timestamp}] {message}</span>')
    
    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and optionally log message."""
        self.progress_bar.setValue(percent)
        if message:
            self._log_event(message)
    
    def run_analysis(self):
        """Override to add progress reporting."""
        self._log_event("Starting MPS simulation...", "info")
        self._update_progress(5, "Validating inputs...")
        super().run_analysis()
        self._update_progress(20, "Simulation submitted...")

    def on_results(self, payload):
        self._update_progress(100, "MPS simulation complete!")
        self._log_event("✓ Simulation finished successfully", "success")
        self.res_lbl.setText("MPS Simulation Complete.")
        self.vis_btn.setEnabled(True)
        self._sim_results = payload
    
    def _visualize_results(self):
        """Visualize simulation results in main 3D viewer."""
        if not hasattr(self, '_sim_results') or self._sim_results is None:
            self._log_event("No results to visualize", "warning")
            return
        
        try:
            grid = self._sim_results.get('grid')
            if grid is not None:
                self._log_event("Sending results to 3D viewer...", "info")
                self.request_visualization.emit(grid, "MPS_Result")
                self._log_event("✓ Visualization request sent", "success")
            else:
                self._log_event("No grid data in results", "warning")
        except Exception as e:
            self._log_event(f"Visualization error: {e}", "error")
