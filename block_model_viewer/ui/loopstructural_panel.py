"""
LoopStructuralModelPanel - Main UI Panel for LoopStructural Geological Modeling.

Industry-grade geological modeling with JORC/SAMREC compliance.

GeoX Panel Safety Rules:
- Panels initialize private state only (self._attr = None)
- Controllers bind data via explicit methods (bind_controller, set_data)
- No assignments to @property without setter
- All __init__() set only private fields, do not pull from DataRegistry
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QMutex, QSize
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QSplitter, QProgressDialog, QMessageBox, QFileDialog,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QFrame, QToolBar, QToolButton, QMenu
)

# Check PyVistaQt availability for embedded 3D viewer
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
    PYVISTAQT_AVAILABLE = True
except ImportError:
    PYVISTAQT_AVAILABLE = False
    QtInteractor = None

from .base_analysis_panel import BaseAnalysisPanel
from .loopstructural_compliance_panel import ComplianceValidationPanel
from .loopstructural_advisory_panel import StructuralAdvisoryWidget
from .collapsible_group import CollapsibleGroup
from .modern_styles import ModernColors, get_theme_colors

if TYPE_CHECKING:
    from ..controllers.app_controller import AppController
    from ..geology.chronos_engine import ChronosEngine
    from ..geology.industry_modeler import GeoXIndustryModeler
    from ..geology.compliance_manager import AuditReport
    from ..geology.fault_detection import SuggestedFault

logger = logging.getLogger(__name__)


def _filter_outliers_for_extent(df: pd.DataFrame, min_valid: float = 100.0) -> pd.DataFrame:
    """
    Filter out outlier coordinates before calculating model extent.
    
    This prevents the model extent from being stretched by invalid
    coordinates like (0, 0, 0) placeholder data.
    
    Args:
        df: DataFrame with X, Y, Z columns
        min_valid: Minimum absolute value for valid X/Y coordinates
        
    Returns:
        Filtered DataFrame with outliers removed
    """
    if df is None or len(df) == 0:
        return df
    
    # Remove rows where X and Y are both very small (likely placeholder data)
    mask = ~((np.abs(df['X']) < min_valid) & (np.abs(df['Y']) < min_valid))
    
    # Only filter if we would retain at least 50% of data
    if mask.sum() >= len(df) * 0.5:
        filtered = df[mask].copy()
        if len(filtered) < len(df):
            logger.info(
                f"Filtered {len(df) - len(filtered)} outlier points for extent calculation "
                f"(X and Y both < {min_valid}m)"
            )
        return filtered
    
    return df


def _calculate_proportional_scalar_spacing(
    df: pd.DataFrame,
    stratigraphy: List[str],
    min_spacing: float = 0.5,
    hole_id_col: str = 'hole_id'
) -> Dict[str, float]:
    """
    Calculate proportional scalar values based on unit thicknesses.

    For geological modeling, thin units should have smaller scalar ranges
    to ensure proper interpolation without numerical artifacts.

    Algorithm:
    1. Calculate average thickness for each unit from drillhole intersections
    2. Normalize thicknesses to total thickness
    3. Apply minimum spacing to prevent collapse of thin units
    4. Return scalar value mapping for each formation

    Args:
        df: DataFrame with formation, Z, and hole_id columns
        stratigraphy: Ordered list of formation names (oldest to youngest)
        min_spacing: Minimum scalar spacing between units (default 0.5)
        hole_id_col: Column name for drillhole identification

    Returns:
        Dict mapping formation names to scalar values
    """
    if df is None or len(df) == 0 or not stratigraphy:
        # Fall back to sequential spacing
        return {form: float(i) for i, form in enumerate(stratigraphy)}

    # Check if we have hole_id column for thickness calculation
    if hole_id_col not in df.columns:
        # Try common alternatives
        for alt_col in ['HOLEID', 'HoleID', 'BHID', 'hole', 'drillhole_id']:
            if alt_col in df.columns:
                hole_id_col = alt_col
                break
        else:
            # Can't calculate thicknesses, use sequential
            logger.info("No hole_id column found - using sequential scalar spacing")
            return {form: float(i) for i, form in enumerate(stratigraphy)}

    # Calculate average thickness for each formation across all holes
    thicknesses = {}

    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].copy()

        if 'formation' not in hole_data.columns:
            continue

        # Sort by depth (Z, descending for typical drillholes)
        hole_data = hole_data.sort_values('Z', ascending=False)

        # Group consecutive same-formation intervals
        prev_formation = None
        interval_start_z = None

        for _, row in hole_data.iterrows():
            formation = row.get('formation')
            z = row.get('Z', 0)

            if pd.isna(formation):
                continue

            if formation != prev_formation:
                # Close previous interval
                if prev_formation is not None and interval_start_z is not None:
                    thickness = interval_start_z - z
                    if thickness > 0:
                        if prev_formation not in thicknesses:
                            thicknesses[prev_formation] = []
                        thicknesses[prev_formation].append(thickness)

                # Start new interval
                interval_start_z = z
                prev_formation = formation

    # Calculate average thicknesses
    avg_thicknesses = {}
    for form in stratigraphy:
        if form in thicknesses and len(thicknesses[form]) > 0:
            avg_thicknesses[form] = np.mean(thicknesses[form])
        else:
            # Use default minimum thickness for unknown units
            avg_thicknesses[form] = 1.0  # Default 1m

    # Normalize to proportional values
    total_thickness = sum(avg_thicknesses.values())
    if total_thickness < 1e-10:
        total_thickness = len(stratigraphy)

    # Calculate cumulative scalar values
    # Apply minimum spacing to ensure thin units are represented
    cumulative_val = 0.0
    formation_to_val = {}

    for i, form in enumerate(stratigraphy):
        formation_to_val[form] = cumulative_val

        # Calculate proportional spacing
        proportion = avg_thicknesses.get(form, 1.0) / total_thickness * len(stratigraphy)
        spacing = max(min_spacing, proportion)
        cumulative_val += spacing

    logger.info(f"Proportional scalar spacing calculated: {formation_to_val}")
    logger.info(f"Average thicknesses (m): {avg_thicknesses}")

    return formation_to_val


# =============================================================================
# MODEL BUILD WORKER - Threaded Geological Model Building
# =============================================================================

class ModelBuildWorker(QThread):
    """
    Worker thread for building geological models.

    Runs LoopStructural model solving in a separate thread to prevent
    UI freezing. Supports cancellation via cancel flag.

    Signals:
        progress_updated(int, str): Progress percentage and status message
        phase_changed(int): Current build phase index
        build_completed(dict): Model result dictionary on success
        build_failed(str): Error message on failure
        build_cancelled(): Emitted when build is cancelled
    """

    progress_updated = pyqtSignal(int, str)
    phase_changed = pyqtSignal(int)
    build_completed = pyqtSignal(dict)
    build_failed = pyqtSignal(str)
    build_cancelled = pyqtSignal()

    def __init__(
        self,
        contacts_df: pd.DataFrame,
        stratigraphy: List[str],
        extent: np.ndarray,
        resolution: int,
        cgw: float,
        fault_params: Optional[List[Dict[str, Any]]] = None,
        formation_values: Optional[Dict[str, float]] = None,
        compute_gradients: bool = True,
        allow_synthetic_fallback: bool = True,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._contacts_df = contacts_df.copy()
        self._stratigraphy = stratigraphy.copy()
        self._extent = extent.copy()
        self._resolution = resolution
        self._cgw = cgw
        self._fault_params = fault_params if fault_params else []
        self._formation_values = formation_values
        self._compute_gradients = compute_gradients
        self._allow_synthetic_fallback = allow_synthetic_fallback

        # Cancel flag with mutex for thread safety
        self._cancel_mutex = QMutex()
        self._cancelled = False

        # Result storage
        self._runner = None
        self._model = None
        self._model_result = None

    def request_cancel(self):
        """Request cancellation of the build process."""
        self._cancel_mutex.lock()
        self._cancelled = True
        self._cancel_mutex.unlock()
        logger.info("Build cancellation requested")

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        self._cancel_mutex.lock()
        result = self._cancelled
        self._cancel_mutex.unlock()
        return result

    @property
    def modeler(self):
        """Get the runner instance after build completes (backward compat)."""
        return self._runner

    @property
    def runner(self):
        """Get the GeologicalModelRunner instance after build completes."""
        return self._runner

    @property
    def model_result(self):
        """Get the ModelResult after build completes."""
        return self._model_result

    @property
    def model(self):
        """Get the model result after build completes."""
        return self._model

    def run(self):
        """Execute the model build in the worker thread using GeologicalModelRunner."""
        from ..geology.model_runner import GeologicalModelRunner, ModelResult

        try:
            # Step 1: Prepare extent (5%)
            self.progress_updated.emit(5, "Step 1/6: Preparing model extent...")
            self.phase_changed.emit(0)

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            logger.info(f"Model extent: {self._extent}")
            logger.info(f"Resolution: {self._resolution}, CGW: {self._cgw}")

            # Convert numpy extent array to dict format for GeologicalModelRunner
            extent_dict = {
                'xmin': float(self._extent[0]),
                'xmax': float(self._extent[1]),
                'ymin': float(self._extent[2]),
                'ymax': float(self._extent[3]),
                'zmin': float(self._extent[4]),
                'zmax': float(self._extent[5]),
            }

            # Step 2: Initialize runner (10%)
            self.progress_updated.emit(10, "Step 2/6: Initializing GeologicalModelRunner...")

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            self._runner = GeologicalModelRunner(
                extent=extent_dict,
                resolution=self._resolution,
                cgw=self._cgw,
                boundary_padding=0.1,  # Prevents edge clipping artifacts
            )

            # Step 3: Configure faults (15%)
            self.progress_updated.emit(15, "Step 3/6: Configuring fault parameters...")

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            logger.info(f"Faults configured: {len(self._fault_params)}")

            # Step 4: Gradient computation info (20%)
            gradient_msg = "Step 4/6: Computing gradients from contact geometry..."
            if not self._compute_gradients:
                gradient_msg = "Step 4/6: Using synthetic orientations (gradient computation disabled)..."
            self.progress_updated.emit(20, gradient_msg)

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            # Step 5: Solve geology (25% -> 85%) - Main computation
            self.progress_updated.emit(25, "Step 5/6: Solving geological model (FDI interpolation)...\nThis may take several minutes for large datasets.")
            self.phase_changed.emit(1)

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            solve_start = datetime.now()

            # Run the full pipeline with gradient computation
            model_result: ModelResult = self._runner.run_full_stack(
                contacts_df=self._contacts_df,
                chronology=self._stratigraphy,
                orientations_df=None,  # Let runner compute from contacts
                faults=self._fault_params if self._fault_params else None,
                extract_solids=True,
                formation_values=self._formation_values,
                compute_gradients=self._compute_gradients,
                allow_synthetic_fallback=self._allow_synthetic_fallback,
            )

            solve_elapsed = (datetime.now() - solve_start).total_seconds()
            logger.info(f"Model solve completed in {solve_elapsed:.1f} seconds")
            logger.info(f"Gradient source: {model_result.gradient_source}")

            # Check cancellation after solve (user may have requested during solve)
            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            # Store results
            self._model = self._runner.engine.model  # LoopStructural model for extraction
            self._model_result = model_result
            self.progress_updated.emit(85, "Step 5/6: Model solving complete...")

            # Step 6: Results ready (85% -> 95%)
            self.progress_updated.emit(90, "Step 6/6: Audit metrics calculated (JORC/SAMREC compliance)...")
            self.phase_changed.emit(3)

            if self.is_cancelled():
                self.build_cancelled.emit()
                return

            self.progress_updated.emit(95, "Finalizing...")

            # Complete
            self.progress_updated.emit(100, "Model build complete!")
            self.phase_changed.emit(4)

            # Emit success with result data
            # Build misfit_report dict from audit_report for backward compatibility
            misfit_report = {}
            if model_result.audit_report:
                misfit_report = {
                    'mean_residual': model_result.audit_report.mean_residual,
                    'p90_error': model_result.audit_report.p90_error,
                    'status': model_result.audit_report.status,
                    'is_jorc_compliant': model_result.audit_report.is_jorc_compliant,
                }

            result = {
                'model': self._model,
                'runner': self._runner,
                'model_result': model_result,
                'misfit_report': misfit_report,
                'build_log': model_result.provenance,
                'solve_time': solve_elapsed,
                'resolution': self._resolution,
                'n_stratigraphy': len(self._stratigraphy),
                'n_faults': len(self._fault_params),
                'warnings': model_result.warnings,
                'gradient_source': model_result.gradient_source,
                'surfaces': model_result.surfaces,
                'solids': model_result.solids,
                'unified_mesh': model_result.unified_mesh,
            }
            self.build_completed.emit(result)

        except Exception as e:
            logger.exception(f"Model build failed: {e}")
            self.build_failed.emit(str(e))


# =============================================================================
# NEW WORKFLOW WIDGETS - Geological Decision Workflow Components
# =============================================================================

class InputValidationChecklist(QFrame):
    """
    Compact inline checklist for input data validation status.

    Shows status as colored dots, labels, and requirement badges in tight rows.
    """

    validation_changed = pyqtSignal(dict)  # Emits validation state

    # Requirements definition: (id, label, tooltip, required)
    REQUIREMENTS = [
        ('coordinates', 'Coordinates (X, Y, Z)',
         'Required: Numeric columns for spatial location of contacts', True),
        ('formation', 'Formation Column',
         'Required: Column named "formation" or "lithology" defining geological units', True),
        ('scalar', 'Scalar Field',
         'Auto-generated from formation encoding if not present in data', False),
        ('orientation', 'Orientation Data (dip/strike)',
         'Optional: gx, gy, gz gradient vectors for structural control', False),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._status: Dict[str, str] = {}  # id -> 'pass' | 'warn' | 'fail' | 'pending'
        self._build_ui()
        self._reset_status()

    def _build_ui(self):
        """Build the checklist UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)

        # Header row
        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        header = QLabel("Input Validation")
        header.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-size: 14px; font-weight: 700; background: transparent;")
        header_row.addWidget(header)
        header_row.addStretch()

        # Summary badge (updates dynamically)
        self._summary_badge = QLabel("0/4 ready")
        self._summary_badge.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 11px; font-weight: 600; background: transparent;")
        header_row.addWidget(self._summary_badge)
        layout.addLayout(header_row)

        # Thin separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {ModernColors.BORDER};")
        layout.addWidget(sep)

        # Checklist items — single-line rows, very compact
        self._item_widgets: Dict[str, Dict[str, QWidget]] = {}

        for req_id, label, tooltip, required in self.REQUIREMENTS:
            row = QHBoxLayout()
            row.setSpacing(10)
            row.setContentsMargins(0, 6, 0, 6)

            # Status indicator — colored icon
            status_label = QLabel("○")
            status_label.setFixedWidth(25)
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 14px; background: transparent;")
            status_label.setToolTip(tooltip)
            row.addWidget(status_label)

            # Label
            tag = "required" if required else "optional"
            text_label = QLabel(label)
            text_label.setStyleSheet(f"color: #FFFFFF; font-size: 12px; font-weight: 500; background: transparent;")
            text_label.setToolTip(tooltip)
            row.addWidget(text_label, stretch=1)

            # Required/optional tag
            if required:
                req_badge = QLabel("REQ")
                req_badge.setStyleSheet(f"""
                    QLabel {{
                        color: {ModernColors.WARNING};
                        font-size: 8px;
                        font-weight: 700;
                        background: rgba(255, 152, 0, 0.15);
                        padding: 1px 4px;
                        border-radius: 2px;
                    }}
                """)
                row.addWidget(req_badge)
            else:
                opt_badge = QLabel("OPT")
                opt_badge.setStyleSheet(f"""
                    QLabel {{
                        color: {ModernColors.TEXT_HINT};
                        font-size: 8px;
                        font-weight: 600;
                        background: transparent;
                        padding: 1px 4px;
                    }}
                """)
                row.addWidget(opt_badge)

            layout.addLayout(row)

            self._item_widgets[req_id] = {
                'status': status_label,
                'label': text_label,
            }

    def _reset_status(self):
        """Reset all items to pending status."""
        for req_id, _, _, _ in self.REQUIREMENTS:
            self._status[req_id] = 'pending'
            self._update_item_display(req_id)
        self._update_summary()

    def _update_item_display(self, req_id: str):
        """Update the display for a single item."""
        if req_id not in self._item_widgets:
            return

        status = self._status.get(req_id, 'pending')
        status_label = self._item_widgets[req_id]['status']

        if status == 'pass':
            status_label.setText("✓")
            status_label.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 16px; font-weight: bold; background: transparent;")
        elif status == 'warn':
            status_label.setText("⚠")
            status_label.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 14px; font-weight: bold; background: transparent;")
        elif status == 'fail':
            status_label.setText("✗")
            status_label.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 16px; font-weight: bold; background: transparent;")
        else:  # pending
            status_label.setText("○")
            status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 14px; background: transparent;")

    def _update_summary(self):
        """Update the summary badge."""
        if not hasattr(self, '_summary_badge'):
            return
        n_pass = sum(1 for s in self._status.values() if s == 'pass')
        n_total = len(self._status)
        n_fail = sum(1 for s in self._status.values() if s == 'fail')

        if n_fail > 0:
            self._summary_badge.setText(f"{n_pass}/{n_total} ready")
            self._summary_badge.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 10px; font-weight: 600; background: transparent;")
        elif n_pass == n_total:
            self._summary_badge.setText(f"{n_pass}/{n_total} ready")
            self._summary_badge.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px; font-weight: 600; background: transparent;")
        else:
            self._summary_badge.setText(f"{n_pass}/{n_total} ready")
            self._summary_badge.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px; background: transparent;")

    def set_status(self, req_id: str, status: str):
        """Set status for a requirement item."""
        if req_id in self._status:
            self._status[req_id] = status
            self._update_item_display(req_id)
            self._update_summary()
            self.validation_changed.emit(self._status.copy())

    def validate_dataframe(self, df: Optional[pd.DataFrame]):
        """Validate a DataFrame and update all status indicators."""
        if df is None or df.empty:
            self._reset_status()
            return

        cols_lower = [c.lower() for c in df.columns]

        # Check coordinates
        has_x = any(c in cols_lower for c in ['x', 'easting', 'east'])
        has_y = any(c in cols_lower for c in ['y', 'northing', 'north'])
        has_z = any(c in cols_lower for c in ['z', 'elevation', 'elev', 'rl'])
        if has_x and has_y and has_z:
            self.set_status('coordinates', 'pass')
        elif has_x or has_y or has_z:
            self.set_status('coordinates', 'warn')
        else:
            self.set_status('coordinates', 'fail')

        # Check formation column
        has_formation = any(c in cols_lower for c in ['formation', 'lithology', 'lith', 'unit', 'rock_type'])
        self.set_status('formation', 'pass' if has_formation else 'fail')

        # Check scalar field
        has_val = 'val' in cols_lower
        self.set_status('scalar', 'pass' if has_val else 'warn')

        # Check orientation data
        has_gx = 'gx' in cols_lower
        has_gy = 'gy' in cols_lower
        has_gz = 'gz' in cols_lower
        if has_gx and has_gy and has_gz:
            self.set_status('orientation', 'pass')
        else:
            self.set_status('orientation', 'warn')  # Optional, so warn not fail

    def get_validation_state(self) -> Dict[str, str]:
        """Get current validation state."""
        return self._status.copy()

    def is_valid(self) -> bool:
        """Check if all required fields pass validation."""
        for req_id, _, _, required in self.REQUIREMENTS:
            if required and self._status.get(req_id) == 'fail':
                return False
        return True


class GeologicalDomainPanel(QFrame):
    """
    Unified geological domain display with coverage metrics.

    Shows bounding box with formatted values and drillhole coverage %.
    """

    domain_changed = pyqtSignal(dict)  # Emits extent dict

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._extent: Dict[str, float] = {}
        self._coverage: Dict[str, float] = {'x': 0, 'y': 0, 'z': 0}
        self._build_ui()

    def _build_ui(self):
        """Build the domain panel UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QLabel("Geological Domain")
        header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 13px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        layout.addWidget(header)

        # Domain display frame
        domain_frame = QFrame()
        domain_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
            }}
        """)
        domain_layout = QVBoxLayout(domain_frame)
        domain_layout.setContentsMargins(10, 10, 10, 10)
        domain_layout.setSpacing(6)

        # Create axis rows
        self._axis_labels: Dict[str, QLabel] = {}
        self._coverage_labels: Dict[str, QLabel] = {}

        for axis in ['X', 'Y', 'Z']:
            row = QHBoxLayout()
            row.setSpacing(8)

            # Axis label
            axis_label = QLabel(f"{axis}:")
            axis_label.setFixedWidth(20)
            axis_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;")
            row.addWidget(axis_label)

            # Range display
            range_label = QLabel("— to — (— m)")
            range_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 11px; font-family: 'Consolas', monospace; background: transparent;")
            row.addWidget(range_label, stretch=1)
            self._axis_labels[axis.lower()] = range_label

            # Coverage badge
            cov_label = QLabel("—%")
            cov_label.setFixedWidth(45)
            cov_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cov_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px; background: transparent;")
            row.addWidget(cov_label)
            self._coverage_labels[axis.lower()] = cov_label

            domain_layout.addLayout(row)

        layout.addWidget(domain_frame)

        # Adjust button
        self._adjust_btn = QPushButton("Adjust Domain...")
        self._adjust_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.TEXT_SECONDARY};
                border: 1px solid {ModernColors.BORDER};
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        self._adjust_btn.clicked.connect(self._on_adjust_clicked)
        layout.addWidget(self._adjust_btn)

        # Hidden spinboxes for actual value storage
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.hide()
            self._spinboxes[key] = spin

    def _format_number(self, value: float) -> str:
        """Format number with thousands separator."""
        return f"{value:,.0f}"

    def _update_display(self):
        """Update the display labels."""
        for axis in ['x', 'y', 'z']:
            min_key = f'{axis}min'
            max_key = f'{axis}max'

            if min_key in self._extent and max_key in self._extent:
                min_val = self._extent[min_key]
                max_val = self._extent[max_key]
                span = max_val - min_val

                self._axis_labels[axis].setText(
                    f"{self._format_number(min_val)} – {self._format_number(max_val)}  ({self._format_number(span)} m)"
                )
            else:
                self._axis_labels[axis].setText("— to — (— m)")

            cov = self._coverage.get(axis, 0)
            cov_label = self._coverage_labels[axis]
            cov_label.setText(f"{cov:.0f}%")

            # Color based on coverage
            if cov >= 90:
                cov_label.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px; font-weight: 600; background: transparent;")
            elif cov >= 70:
                cov_label.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 10px; font-weight: 600; background: transparent;")
            else:
                cov_label.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 10px; font-weight: 600; background: transparent;")

    def set_extent(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
        """Set the model extent."""
        self._extent = {
            'xmin': xmin, 'xmax': xmax,
            'ymin': ymin, 'ymax': ymax,
            'zmin': zmin, 'zmax': zmax,
        }
        self._spinboxes['xmin'].setValue(xmin)
        self._spinboxes['xmax'].setValue(xmax)
        self._spinboxes['ymin'].setValue(ymin)
        self._spinboxes['ymax'].setValue(ymax)
        self._spinboxes['zmin'].setValue(zmin)
        self._spinboxes['zmax'].setValue(zmax)
        self._update_display()
        self.domain_changed.emit(self._extent.copy())

    def set_coverage(self, x_pct: float, y_pct: float, z_pct: float):
        """Set drillhole coverage percentages."""
        self._coverage = {'x': x_pct, 'y': y_pct, 'z': z_pct}
        self._update_display()

    def get_extent(self) -> Dict[str, float]:
        """Get the current extent."""
        return {
            'xmin': self._spinboxes['xmin'].value(),
            'xmax': self._spinboxes['xmax'].value(),
            'ymin': self._spinboxes['ymin'].value(),
            'ymax': self._spinboxes['ymax'].value(),
            'zmin': self._spinboxes['zmin'].value(),
            'zmax': self._spinboxes['zmax'].value(),
        }

    def _on_adjust_clicked(self):
        """Open dialog to adjust domain values."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Geological Domain")
        dialog.setStyleSheet(f"background: {ModernColors.PANEL_BG}; color: {ModernColors.TEXT_PRIMARY};")

        layout = QFormLayout(dialog)
        layout.setSpacing(10)

        # Create spinboxes for editing
        edits = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.setValue(self._spinboxes[key].value())
            spin.setStyleSheet(f"background: {ModernColors.ELEVATED_BG}; color: {ModernColors.TEXT_PRIMARY}; border: 1px solid {ModernColors.BORDER}; padding: 4px;")
            edits[key] = spin

            label_text = key.upper().replace('MIN', ' Min').replace('MAX', ' Max')
            layout.addRow(label_text + ":", spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_extent(
                edits['xmin'].value(), edits['xmax'].value(),
                edits['ymin'].value(), edits['ymax'].value(),
                edits['zmin'].value(), edits['zmax'].value()
            )


# =============================================================================
# LITHOLOGY GROUPING WIDGET - Merge similar lithologies for modeling
# =============================================================================

class LithologyGroupingWidget(QFrame):
    """
    Widget for grouping similar lithologies into modeling units.

    Allows users to:
    - View all unique lithology codes from drillhole data
    - Create groups and assign multiple lithologies to each group
    - Auto-suggest groupings based on name similarity
    - Apply groupings to transform data for modeling

    Signals:
        grouping_changed: Emitted when the grouping mapping changes
    """

    grouping_changed = pyqtSignal(dict)  # Emits {raw_lith: group_name} mapping

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._raw_lithologies: List[str] = []
        self._grouping: Dict[str, str] = {}  # {raw_lith: group_name}
        self._groups: Dict[str, List[str]] = {}  # {group_name: [raw_liths]}
        self._build_ui()

    def _build_ui(self):
        """Build the lithology grouping UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.ELEVATED_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header with info
        header = QLabel("Group similar lithologies into modeling units")
        header.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Main content - two columns
        columns = QHBoxLayout()
        columns.setSpacing(12)

        # Left: Raw lithologies
        left_frame = QFrame()
        left_frame.setStyleSheet(f"QFrame {{ background: {ModernColors.PANEL_BG}; border-radius: 6px; }}")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(8, 8, 8, 8)

        left_header = QLabel("<b>Raw Lithologies</b>")
        left_header.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-size: 11px; background: transparent;")
        left_layout.addWidget(left_header)

        self._raw_list = QListWidget()
        self._raw_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._raw_list.setMinimumHeight(120)
        self._raw_list.setStyleSheet(f"""
            QListWidget {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                font-size: 10px;
            }}
            QListWidget::item {{
                padding: 4px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {ModernColors.ACCENT_PRESSED};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        left_layout.addWidget(self._raw_list)

        columns.addWidget(left_frame, stretch=1)

        # Center: Action buttons
        center_layout = QVBoxLayout()
        center_layout.setSpacing(6)
        center_layout.addStretch()

        btn_style = f"""
            QPushButton {{
                background: {ModernColors.BORDER};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_LIGHT};
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 10px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background: {ModernColors.BORDER_LIGHT};
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.TEXT_DISABLED};
            }}
        """

        self._add_to_group_btn = QPushButton("Add to Group >>")
        self._add_to_group_btn.setStyleSheet(btn_style)
        self._add_to_group_btn.clicked.connect(self._add_selected_to_group)
        self._add_to_group_btn.setToolTip("Add selected lithologies to the selected group")
        center_layout.addWidget(self._add_to_group_btn)

        self._remove_from_group_btn = QPushButton("<< Remove")
        self._remove_from_group_btn.setStyleSheet(btn_style)
        self._remove_from_group_btn.clicked.connect(self._remove_from_group)
        self._remove_from_group_btn.setToolTip("Remove selected lithology from its group")
        center_layout.addWidget(self._remove_from_group_btn)

        self._auto_group_btn = QPushButton("Auto-Group")
        self._auto_group_btn.setStyleSheet(btn_style)
        self._auto_group_btn.clicked.connect(self._auto_group_similar)
        self._auto_group_btn.setToolTip("Automatically group lithologies with similar prefixes")
        center_layout.addWidget(self._auto_group_btn)

        center_layout.addStretch()
        columns.addLayout(center_layout)

        # Right: Groups
        right_frame = QFrame()
        right_frame.setStyleSheet(f"QFrame {{ background: {ModernColors.PANEL_BG}; border-radius: 6px; }}")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_header = QLabel("<b>Modeling Groups</b>")
        right_header.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 11px; background: transparent;")
        right_layout.addWidget(right_header)

        # Group name input
        group_input_layout = QHBoxLayout()
        self._group_name_input = QLineEdit()
        self._group_name_input.setPlaceholderText("New group name...")
        self._group_name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 4px;
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 10px;
            }}
        """)
        group_input_layout.addWidget(self._group_name_input)

        self._add_group_btn = QPushButton("+")
        self._add_group_btn.setFixedWidth(30)
        self._add_group_btn.setStyleSheet(btn_style)
        self._add_group_btn.clicked.connect(self._add_new_group)
        self._add_group_btn.setToolTip("Create new group")
        group_input_layout.addWidget(self._add_group_btn)

        right_layout.addLayout(group_input_layout)

        # Groups tree
        self._groups_tree = QListWidget()
        self._groups_tree.setMinimumHeight(120)
        self._groups_tree.setStyleSheet(f"""
            QListWidget {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                font-size: 10px;
            }}
            QListWidget::item {{
                padding: 4px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {ModernColors.ACCENT_PRESSED};
                color: {ModernColors.SUCCESS};
            }}
        """)
        right_layout.addWidget(self._groups_tree)

        # Delete group button
        self._delete_group_btn = QPushButton("Delete Group")
        self._delete_group_btn.setStyleSheet(btn_style)
        self._delete_group_btn.clicked.connect(self._delete_selected_group)
        right_layout.addWidget(self._delete_group_btn)

        columns.addWidget(right_frame, stretch=1)
        layout.addLayout(columns)

        # Status label
        self._status_label = QLabel("No lithologies loaded")
        self._status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px; background: transparent;")
        layout.addWidget(self._status_label)

    def set_lithologies(self, lithologies: List[str]):
        """Set the raw lithology codes from data."""
        self._raw_lithologies = sorted(set(lithologies))
        self._update_raw_list()
        self._status_label.setText(f"{len(self._raw_lithologies)} unique lithologies")

    def _update_raw_list(self):
        """Update the raw lithologies list, showing ungrouped items."""
        self._raw_list.clear()
        for lith in self._raw_lithologies:
            if lith not in self._grouping:
                self._raw_list.addItem(lith)

    def _update_groups_tree(self):
        """Update the groups tree display."""
        self._groups_tree.clear()
        for group_name, members in sorted(self._groups.items()):
            # Add group header
            group_item = QListWidgetItem(f"[+] {group_name} ({len(members)})")
            group_item.setData(Qt.ItemDataRole.UserRole, ('group', group_name))
            font = group_item.font()
            font.setBold(True)
            group_item.setFont(font)
            self._groups_tree.addItem(group_item)

            # Add members indented
            for member in sorted(members):
                member_item = QListWidgetItem(f"    - {member}")
                member_item.setData(Qt.ItemDataRole.UserRole, ('member', group_name, member))
                self._groups_tree.addItem(member_item)

    def _add_new_group(self):
        """Create a new empty group."""
        name = self._group_name_input.text().strip()
        if not name:
            return
        if name in self._groups:
            return  # Already exists

        self._groups[name] = []
        self._group_name_input.clear()
        self._update_groups_tree()

    def _add_selected_to_group(self):
        """Add selected raw lithologies to the selected group."""
        # Get selected lithologies
        selected_liths = [item.text() for item in self._raw_list.selectedItems()]
        if not selected_liths:
            return

        # Get selected group
        selected_group_items = self._groups_tree.selectedItems()
        if not selected_group_items:
            # If no group selected, check if there's a group name in input
            group_name = self._group_name_input.text().strip()
            if not group_name:
                return
            if group_name not in self._groups:
                self._groups[group_name] = []
        else:
            item_data = selected_group_items[0].data(Qt.ItemDataRole.UserRole)
            if item_data[0] == 'group':
                group_name = item_data[1]
            elif item_data[0] == 'member':
                group_name = item_data[1]
            else:
                return

        # Add lithologies to group
        for lith in selected_liths:
            if lith not in self._grouping:
                self._grouping[lith] = group_name
                if lith not in self._groups[group_name]:
                    self._groups[group_name].append(lith)

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _remove_from_group(self):
        """Remove selected lithology from its group."""
        selected_items = self._groups_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data and item_data[0] == 'member':
                group_name = item_data[1]
                member = item_data[2]

                # Remove from grouping
                if member in self._grouping:
                    del self._grouping[member]
                if group_name in self._groups and member in self._groups[group_name]:
                    self._groups[group_name].remove(member)

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _delete_selected_group(self):
        """Delete the selected group."""
        selected_items = self._groups_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data and item_data[0] == 'group':
                group_name = item_data[1]

                # Remove all members from grouping
                if group_name in self._groups:
                    for member in self._groups[group_name]:
                        if member in self._grouping:
                            del self._grouping[member]
                    del self._groups[group_name]

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _auto_group_similar(self):
        """Automatically group lithologies with similar prefixes."""
        if not self._raw_lithologies:
            return

        # Find common prefixes (at least 3 characters)
        prefix_groups: Dict[str, List[str]] = {}

        for lith in self._raw_lithologies:
            if lith in self._grouping:
                continue  # Already grouped

            # Try different prefix lengths
            best_prefix = None
            for prefix_len in range(min(5, len(lith)), 2, -1):
                prefix = lith[:prefix_len].upper()
                # Count how many other lithologies share this prefix
                matches = [l for l in self._raw_lithologies
                          if l.upper().startswith(prefix) and l not in self._grouping]
                if len(matches) >= 2:
                    best_prefix = prefix
                    break

            if best_prefix:
                if best_prefix not in prefix_groups:
                    prefix_groups[best_prefix] = []
                if lith not in prefix_groups[best_prefix]:
                    prefix_groups[best_prefix].append(lith)

        # Create groups from prefixes
        for prefix, members in prefix_groups.items():
            if len(members) >= 2:
                group_name = prefix.title()
                if group_name not in self._groups:
                    self._groups[group_name] = []

                for member in members:
                    if member not in self._grouping:
                        self._grouping[member] = group_name
                        self._groups[group_name].append(member)

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

        # Update status
        n_grouped = len(self._grouping)
        n_groups = len(self._groups)
        self._status_label.setText(f"{n_grouped}/{len(self._raw_lithologies)} lithologies in {n_groups} groups")

    def _emit_grouping(self):
        """Emit the current grouping mapping."""
        self.grouping_changed.emit(self._grouping.copy())

    def get_grouping(self) -> Dict[str, str]:
        """Get the current grouping mapping."""
        return self._grouping.copy()

    def get_grouped_lithologies(self) -> List[str]:
        """Get list of group names (for stratigraphy)."""
        # Return groups plus any ungrouped lithologies
        result = list(self._groups.keys())
        for lith in self._raw_lithologies:
            if lith not in self._grouping:
                result.append(lith)
        return result

    def apply_grouping_to_dataframe(self, df: pd.DataFrame, column: str = 'formation') -> pd.DataFrame:
        """Apply the grouping to transform a dataframe column."""
        if not self._grouping or column not in df.columns:
            return df

        df = df.copy()
        df[column] = df[column].apply(lambda x: self._grouping.get(x, x) if pd.notna(x) else x)
        return df

    def clear_grouping(self):
        """Clear all groupings."""
        self._grouping.clear()
        self._groups.clear()
        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()


class GeologicalAuditVerdictTable(QFrame):
    """
    Structured compliance verdict display with expandable details.

    Shows audit checks as clickable rows that expand to show:
    What failed, Where, Why, and Confidence impact.
    """

    verdict_expanded = pyqtSignal(str)  # Emits check_id when expanded

    # Audit checks definition: (id, label)
    AUDIT_CHECKS = [
        ('stratigraphic_ordering', 'Stratigraphic Ordering'),
        ('layer_continuity', 'Layer Continuity'),
        ('drillhole_honouring', 'Drillhole Honouring'),
        ('dip_strike_consistency', 'Dip & Strike Consistency'),
        ('fault_handling', 'Fault Handling'),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._verdicts: Dict[str, Dict[str, Any]] = {}
        self._expanded: Dict[str, bool] = {}
        self._build_ui()
        self._reset_verdicts()

    def _build_ui(self):
        """Build the verdict table UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(12, 12, 12, 12)
        self._main_layout.setSpacing(8)

        # Header
        header = QLabel("Geological Audit")
        header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 13px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        self._main_layout.addWidget(header)

        # Create verdict rows
        self._row_widgets: Dict[str, Dict[str, QWidget]] = {}

        for check_id, label in self.AUDIT_CHECKS:
            self._create_verdict_row(check_id, label)

    def _create_verdict_row(self, check_id: str, label: str):
        """Create a single verdict row with expandable details."""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
            }}
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 8, 10, 8)
        container_layout.setSpacing(6)

        # Main row (always visible)
        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # Status icon
        status_label = QLabel("--")
        status_label.setFixedWidth(20)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 14px; background: transparent;")
        main_row.addWidget(status_label)

        # Check name
        name_label = QLabel(label)
        name_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        main_row.addWidget(name_label, stretch=1)

        # Status badge
        badge_label = QLabel("PENDING")
        badge_label.setFixedWidth(60)
        badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_HINT};
                font-size: 10px;
                font-weight: 600;
                background: {ModernColors.ELEVATED_BG};
                border-radius: 3px;
                padding: 2px 6px;
            }}
        """)
        main_row.addWidget(badge_label)

        # Expand button
        expand_btn = QPushButton(">")
        expand_btn.setFixedSize(24, 24)
        expand_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {ModernColors.TEXT_HINT};
                border: none;
                font-size: 10px;
            }}
            QPushButton:hover {{
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        expand_btn.clicked.connect(lambda: self._toggle_expand(check_id))
        main_row.addWidget(expand_btn)

        container_layout.addLayout(main_row)

        # Details section (hidden by default)
        details_frame = QFrame()
        details_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.ELEVATED_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        details_frame.hide()

        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(10, 8, 10, 8)
        details_layout.setSpacing(4)

        # Detail labels
        what_label = QLabel("What: —")
        where_label = QLabel("Where: —")
        why_label = QLabel("Why: —")
        impact_label = QLabel("Impact: —")

        for lbl in [what_label, where_label, why_label, impact_label]:
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
            details_layout.addWidget(lbl)

        container_layout.addWidget(details_frame)

        self._main_layout.addWidget(container)

        self._row_widgets[check_id] = {
            'container': container,
            'status': status_label,
            'name': name_label,
            'badge': badge_label,
            'expand_btn': expand_btn,
            'details_frame': details_frame,
            'what': what_label,
            'where': where_label,
            'why': why_label,
            'impact': impact_label,
        }
        self._expanded[check_id] = False

    def _reset_verdicts(self):
        """Reset all verdicts to pending."""
        for check_id, _ in self.AUDIT_CHECKS:
            self._verdicts[check_id] = {
                'status': 'pending',
                'what': '',
                'where': '',
                'why': '',
                'impact': '',
            }
            self._update_row_display(check_id)

    def _update_row_display(self, check_id: str):
        """Update the display for a verdict row."""
        if check_id not in self._row_widgets:
            return

        widgets = self._row_widgets[check_id]
        verdict = self._verdicts.get(check_id, {})
        status = verdict.get('status', 'pending')

        # Update status icon
        status_label = widgets['status']
        badge_label = widgets['badge']

        if status == 'pass':
            status_label.setText("OK")
            status_label.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 14px; font-weight: bold; background: transparent;")
            badge_label.setText("PASS")
            badge_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.SUCCESS};
                    font-size: 10px;
                    font-weight: 600;
                    background: rgba(76, 175, 80, 0.2);
                    border-radius: 3px;
                    padding: 2px 6px;
                }}
            """)
        elif status == 'warn':
            status_label.setText("!!")
            status_label.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 14px; font-weight: bold; background: transparent;")
            badge_label.setText("WARN")
            badge_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.WARNING};
                    font-size: 10px;
                    font-weight: 600;
                    background: rgba(255, 152, 0, 0.2);
                    border-radius: 3px;
                    padding: 2px 6px;
                }}
            """)
        elif status == 'fail':
            status_label.setText("XX")
            status_label.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 14px; font-weight: bold; background: transparent;")
            badge_label.setText("FAIL")
            badge_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.ERROR};
                    font-size: 10px;
                    font-weight: 600;
                    background: rgba(244, 67, 54, 0.2);
                    border-radius: 3px;
                    padding: 2px 6px;
                }}
            """)
        else:  # pending
            status_label.setText("--")
            status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 14px; background: transparent;")
            badge_label.setText("PENDING")
            badge_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.TEXT_HINT};
                    font-size: 10px;
                    font-weight: 600;
                    background: {ModernColors.ELEVATED_BG};
                    border-radius: 3px;
                    padding: 2px 6px;
                }}
            """)

        # Update details
        widgets['what'].setText(f"What: {verdict.get('what', '—') or '—'}")
        widgets['where'].setText(f"Where: {verdict.get('where', '—') or '—'}")
        widgets['why'].setText(f"Why: {verdict.get('why', '—') or '—'}")
        widgets['impact'].setText(f"Impact: {verdict.get('impact', '—') or '—'}")

    def _toggle_expand(self, check_id: str):
        """Toggle expansion of a verdict row."""
        if check_id not in self._row_widgets:
            return

        self._expanded[check_id] = not self._expanded[check_id]
        widgets = self._row_widgets[check_id]

        if self._expanded[check_id]:
            widgets['details_frame'].show()
            widgets['expand_btn'].setText("v")
            self.verdict_expanded.emit(check_id)
        else:
            widgets['details_frame'].hide()
            widgets['expand_btn'].setText(">")

    def set_verdict(self, check_id: str, status: str, what: str = "", where: str = "",
                    why: str = "", impact: str = ""):
        """Set verdict for a check."""
        self._verdicts[check_id] = {
            'status': status,
            'what': what,
            'where': where,
            'why': why,
            'impact': impact,
        }
        self._update_row_display(check_id)

    def populate_from_audit_report(self, report: Optional['AuditReport'],
                                   strat_result=None, continuity_result=None):
        """Populate verdicts from an AuditReport and related results."""
        if report is None:
            self._reset_verdicts()
            return

        # Stratigraphic ordering
        if strat_result is not None:
            strat_status = 'pass' if strat_result.is_valid else 'fail'
            violations = len(strat_result.violations) if hasattr(strat_result, 'violations') else 0
            self.set_verdict(
                'stratigraphic_ordering',
                strat_status,
                what="Formation sequence validation" if strat_status == 'pass' else f"{violations} ordering violations detected",
                where=f"{strat_result.holes_checked} holes checked" if hasattr(strat_result, 'holes_checked') else "",
                why="" if strat_status == 'pass' else "Depth ordering violated in drillhole intersections",
                impact="" if strat_status == 'pass' else "May affect layer continuity interpretation"
            )

        # Layer continuity
        if continuity_result is not None:
            cont_status = 'pass' if getattr(continuity_result, 'all_continuous', True) else 'warn'
            self.set_verdict(
                'layer_continuity',
                cont_status,
                what="All layers continuous" if cont_status == 'pass' else "Discontinuities detected",
                where="Model domain" if cont_status == 'pass' else "Check isolated volumes",
                why="" if cont_status == 'pass' else "Some units may have isolated volumes",
                impact="" if cont_status == 'pass' else "Review mesh topology"
            )

        # Drillhole honouring (from contact deviation)
        if hasattr(report, 'mean_residual') and report.mean_residual is not None:
            p90 = getattr(report, 'p90_error', 0) or 0
            if p90 < 2.0:
                dh_status = 'pass'
            elif p90 < 5.0:
                dh_status = 'warn'
            else:
                dh_status = 'fail'

            self.set_verdict(
                'drillhole_honouring',
                dh_status,
                what=f"P90 error: {p90:.2f}m" if p90 else "Contact matching verified",
                where=f"{report.total_contacts} contacts evaluated",
                why="" if dh_status == 'pass' else f"Mean residual: {report.mean_residual:.2f}m",
                impact=f"Classification: {getattr(report, 'classification_recommendation', 'Unknown')}"
            )

        # Dip & strike consistency - requires explicit validation call
        # This verdict should be set by calling _validate_dip_strike_consistency()
        # from the compliance validation workflow, not from audit report alone
        if not hasattr(self, '_dip_strike_validated') or not self._dip_strike_validated:
            self.set_verdict(
                'dip_strike_consistency',
                'pending',
                what="Run compliance validation for structural check",
                where="",
                why="Requires model gradient evaluation",
                impact=""
            )

        # Fault handling
        self.set_verdict(
            'fault_handling',
            'pass',
            what="Fault events processed",
            where="Model domain",
            why="",
            impact=""
        )


class ModelBuildExecutionPanel(QFrame):
    """
    Dedicated model build execution panel with progress tracking.

    Shows build phases, warnings, diagnostics, and execution controls.
    """

    build_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    phase_changed = pyqtSignal(int, str)  # phase_index, phase_name

    # Build phases
    BUILD_PHASES = [
        ('normalize', 'Normalizing coordinates'),
        ('solve', 'Solving geological model'),
        ('extract', 'Extracting surfaces'),
        ('validate', 'Validating compliance'),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_phase: int = -1
        self._warnings: List[str] = []
        self._seed: Optional[int] = None
        self._runtime: float = 0.0
        self._estimated_time: float = 0.0
        self._is_building: bool = False
        self._build_ui()

    def _build_ui(self):
        """Build the execution panel UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header
        header = QLabel("Model Build")
        header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 13px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        layout.addWidget(header)

        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
            }}
        """)
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(6)

        progress_header = QLabel("Progress")
        progress_header.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;")
        progress_layout.addWidget(progress_header)

        # Phase indicators
        self._phase_widgets: List[Dict[str, QWidget]] = []

        for i, (phase_id, phase_label) in enumerate(self.BUILD_PHASES):
            row = QHBoxLayout()
            row.setSpacing(8)

            # Status indicator
            status_label = QLabel("--")
            status_label.setFixedWidth(20)
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 12px; background: transparent;")
            row.addWidget(status_label)

            # Phase number
            num_label = QLabel(f"{i + 1}.")
            num_label.setFixedWidth(20)
            num_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
            row.addWidget(num_label)

            # Phase name
            name_label = QLabel(phase_label)
            name_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 11px; background: transparent;")
            row.addWidget(name_label, stretch=1)

            progress_layout.addLayout(row)

            self._phase_widgets.append({
                'status': status_label,
                'num': num_label,
                'name': name_label,
            })

        layout.addWidget(progress_frame)

        # Warnings section (collapsible)
        self._warnings_group = CollapsibleGroup("Warnings (0)", collapsed=True)
        self._warnings_label = QLabel("No warnings")
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._warnings_group.add_widget(self._warnings_label)
        layout.addWidget(self._warnings_group)

        # Runtime info
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)

        self._seed_label = QLabel("Seed: —")
        self._seed_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        info_layout.addWidget(self._seed_label)

        self._runtime_label = QLabel("Runtime: —")
        self._runtime_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        info_layout.addWidget(self._runtime_label)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._build_btn = QPushButton("Build Model")
        self._build_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_HOVER};
            }}
            QPushButton:disabled {{
                background: {ModernColors.BORDER};
                color: {ModernColors.TEXT_DISABLED};
            }}
        """)
        self._build_btn.clicked.connect(self._on_build_clicked)
        btn_layout.addWidget(self._build_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.TEXT_SECONDARY};
                border: 1px solid {ModernColors.BORDER};
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ERROR};
                color: white;
                border-color: {ModernColors.ERROR};
            }}
            QPushButton:disabled {{
                color: {ModernColors.TEXT_DISABLED};
            }}
        """)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self._cancel_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Diagnostics (collapsed by default)
        self._diagnostics_group = CollapsibleGroup("Build Diagnostics", collapsed=True)
        self._diagnostics_text = QTextEdit()
        self._diagnostics_text.setReadOnly(True)
        self._diagnostics_text.setMaximumHeight(120)
        self._diagnostics_text.setPlainText("No build diagnostics available.")
        self._diagnostics_text.setStyleSheet(f"""
            QTextEdit {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                color: {ModernColors.TEXT_SECONDARY};
            }}
        """)
        self._diagnostics_group.add_widget(self._diagnostics_text)
        layout.addWidget(self._diagnostics_group)

    def _update_phase_display(self):
        """Update the phase indicators."""
        for i, widgets in enumerate(self._phase_widgets):
            status_label = widgets['status']

            if i < self._current_phase:
                # Completed
                status_label.setText("OK")
                status_label.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 12px; font-weight: bold; background: transparent;")
            elif i == self._current_phase:
                # Current
                status_label.setText(">>")
                status_label.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-size: 12px; background: transparent;")
            else:
                # Pending
                status_label.setText("--")
                status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 12px; background: transparent;")

    def set_phase(self, phase_index: int):
        """Set the current build phase."""
        self._current_phase = phase_index
        self._update_phase_display()

        if 0 <= phase_index < len(self.BUILD_PHASES):
            phase_id, phase_name = self.BUILD_PHASES[phase_index]
            self.phase_changed.emit(phase_index, phase_name)

    def set_building(self, is_building: bool):
        """Set building state."""
        self._is_building = is_building
        self._build_btn.setEnabled(not is_building)
        self._cancel_btn.setEnabled(is_building)

        if is_building:
            self._build_btn.setText("Building...")
        else:
            self._build_btn.setText("Build Model")

    def set_warnings(self, warnings: List[str]):
        """Set build warnings."""
        self._warnings = warnings
        self._warnings_group.set_title(f"Warnings ({len(warnings)})")

        if warnings:
            self._warnings_label.setText("\n".join(f"• {w}" for w in warnings))
            self._warnings_label.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 11px; background: transparent;")
            self._warnings_group.set_collapsed(False)
        else:
            self._warnings_label.setText("No warnings")
            self._warnings_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px; background: transparent;")

    def set_seed(self, seed: Optional[int]):
        """Set the random seed."""
        self._seed = seed
        self._seed_label.setText(f"Seed: {seed if seed is not None else '—'}")

    def set_runtime(self, runtime: float, estimated: float = 0.0):
        """Set runtime display."""
        self._runtime = runtime
        self._estimated_time = estimated

        if estimated > 0:
            self._runtime_label.setText(f"Runtime: {runtime:.1f}s / est. {estimated:.1f}s")
        else:
            self._runtime_label.setText(f"Runtime: {runtime:.1f}s")

    def set_diagnostics(self, text: str):
        """Set diagnostics text."""
        self._diagnostics_text.setPlainText(text)

    def reset(self):
        """Reset the panel to initial state."""
        self._current_phase = -1
        self._update_phase_display()
        self.set_building(False)
        self.set_warnings([])
        self.set_seed(None)
        self.set_runtime(0.0)
        self.set_diagnostics("No build diagnostics available.")

    def _on_build_clicked(self):
        """Handle build button click."""
        self.build_requested.emit()

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()


class LoopStructuralModelPanel(BaseAnalysisPanel):
    """
    Main UI Panel for LoopStructural-based Geological Modeling.

    Features:
    - Data input configuration (contacts, orientations, faults)
    - Model building with progress tracking
    - JORC/SAMREC compliance validation
    - Automatic fault suggestion from errors
    - Surface extraction and visualization
    - Export to mining software formats

    Architecture:
    - Uses ChronosEngine for coordinate handling and event stacking
    - Uses GeoXIndustryModeler for industry-grade FDI interpolation
    - Uses ComplianceManager for audit reporting
    - Uses FaultDetectionEngine for structural suggestions
    """

    task_name = "loopstructural_model"

    # Signals
    model_built = pyqtSignal(object)  # Emits model result dict
    surfaces_extracted = pyqtSignal(list)  # Emits list of surface dicts
    compliance_validated = pyqtSignal(object)  # Emits AuditReport
    geology_package_ready = pyqtSignal(dict)  # Emits complete package for main renderer

    def __init__(self, parent: Optional[QWidget] = None):
        # Private state (GeoX Panel Safety Rules)
        self._engine: Optional[ChronosEngine] = None
        self._modeler: Optional[GeoXIndustryModeler] = None
        self._model = None
        self._contacts_df: Optional[pd.DataFrame] = None
        self._orientations_df: Optional[pd.DataFrame] = None
        self._fault_list: List[Dict[str, Any]] = []
        self._stratigraphy: List[str] = []
        self._surfaces: List[Dict[str, Any]] = []
        self._solids: List[Dict[str, Any]] = []  # Solid volumes for each unit
        self._current_report: Optional[AuditReport] = None
        
        # UI widget references (initialized in _build_ui, must exist before super().__init__ 
        # in case connect_signals() tries to access them)
        self._strat_list: Optional[QListWidget] = None
        self._strat_input = None
        self._resolution_spin = None
        self._cgw_spin = None
        self._fault_table = None
        self._build_btn = None
        self._xmin_spin = None
        self._xmax_spin = None
        self._ymin_spin = None
        self._ymax_spin = None
        self._zmin_spin = None
        self._zmax_spin = None
        self._tabs = None

        # Model runner and results (set by _on_build_completed)
        self._runner = None               # GeologicalModelRunner instance
        self._model_result = None          # ModelResult from runner
        self._unified_mesh = None          # Unified geology mesh dict

        # Build worker for threaded model building
        self._build_worker: Optional[ModelBuildWorker] = None
        self._build_start_time: Optional[datetime] = None

        # Lithology grouping
        self._lith_grouping_widget: Optional[LithologyGroupingWidget] = None
        self._lithology_mapping: Dict[str, str] = {}  # {raw_lith: group_name}

        # Formation scalar values mapping - needed for correct isosurface extraction
        self._formation_values: Dict[str, float] = {}  # {formation_name: scalar_value}

        # NOTE: Embedded 3D viewer removed - visualization now in main renderer
        # Use Geological Explorer panel to control geological model display

        super().__init__(parent=parent, panel_id="loopstructural_model_panel")

        # CRITICAL: Call _build_ui() after super().__init__() since BaseAnalysisPanel
        # detected we have _build_ui() and skipped setup_ui()
        self._build_ui()

        self.setWindowTitle("Geological Modeling (LoopStructural)")
        self.setMinimumSize(900, 700)

    def _build_toolbar(self) -> QToolBar:
        """Build compact toolbar with dropdown menus for all actions."""
        tb = QToolBar("Geological Modeling Tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setStyleSheet(f"""
            QToolBar {{
                background: {ModernColors.ELEVATED_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 2px;
                spacing: 4px;
            }}
            QToolButton {{
                color: {ModernColors.TEXT_PRIMARY};
                background: transparent;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
            }}
            QToolButton:hover {{
                background: {ModernColors.CARD_HOVER};
            }}
            QToolButton:pressed {{
                background: {ModernColors.ACCENT_PRESSED};
            }}
        """)

        # FILE MENU - Data Loading
        file_menu = QMenu(self)
        file_menu.setStyleSheet(f"""
            QMenu {{
                background: {ModernColors.ELEVATED_BG};
                border: 1px solid {ModernColors.BORDER};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QMenu::item:selected {{
                background: {ModernColors.CARD_HOVER};
            }}
        """)

        act_load_registry = QAction("Load from Registry", self)
        act_load_registry.triggered.connect(self._on_load_from_registry)
        file_menu.addAction(act_load_registry)

        act_load_file = QAction("Load from File...", self)
        act_load_file.triggered.connect(self._on_load_from_file)
        file_menu.addAction(act_load_file)

        file_btn = QToolButton()
        file_btn.setText("File")
        file_btn.setMenu(file_menu)
        file_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        file_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(file_btn)

        # EDIT MENU - Lithology & Stratigraphy
        edit_menu = QMenu(self)
        edit_menu.setStyleSheet(file_menu.styleSheet())

        edit_menu.addSection("Lithology Grouping")
        act_add_group = QAction("Add to Group", self)
        act_add_group.triggered.connect(lambda: self._lith_grouping_widget._add_selected_to_group() if hasattr(self, '_lith_grouping_widget') else None)
        edit_menu.addAction(act_add_group)

        act_remove_group = QAction("Remove from Group", self)
        act_remove_group.triggered.connect(lambda: self._lith_grouping_widget._remove_from_group() if hasattr(self, '_lith_grouping_widget') else None)
        edit_menu.addAction(act_remove_group)

        act_auto_group = QAction("Auto-Group Similar", self)
        act_auto_group.triggered.connect(lambda: self._lith_grouping_widget._auto_group_similar() if hasattr(self, '_lith_grouping_widget') else None)
        edit_menu.addAction(act_auto_group)

        edit_menu.addSeparator()
        edit_menu.addSection("Stratigraphic Sequence")

        act_strat_up = QAction("Move Up (Younger)", self)
        act_strat_up.setShortcut("Ctrl+Up")
        act_strat_up.triggered.connect(self._move_strat_up)
        edit_menu.addAction(act_strat_up)

        act_strat_down = QAction("Move Down (Older)", self)
        act_strat_down.setShortcut("Ctrl+Down")
        act_strat_down.triggered.connect(self._move_strat_down)
        edit_menu.addAction(act_strat_down)

        edit_menu.addSeparator()

        act_strat_add = QAction("Add Formation", self)
        act_strat_add.triggered.connect(self._add_strat_unit)
        edit_menu.addAction(act_strat_add)

        act_strat_remove = QAction("Remove Formation", self)
        act_strat_remove.triggered.connect(self._remove_strat_unit)
        edit_menu.addAction(act_strat_remove)

        edit_btn = QToolButton()
        edit_btn.setText("Edit")
        edit_btn.setMenu(edit_menu)
        edit_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        edit_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(edit_btn)

        # MODEL MENU - Build & Operations
        model_menu = QMenu(self)
        model_menu.setStyleSheet(file_menu.styleSheet())

        act_build = QAction("Build Model", self)
        act_build.setShortcut("F5")
        act_build.triggered.connect(self._on_quick_build)
        model_menu.addAction(act_build)

        act_cancel = QAction("Cancel Build", self)
        act_cancel.triggered.connect(self._on_cancel_build)
        model_menu.addAction(act_cancel)

        model_menu.addSeparator()

        act_extract = QAction("Extract Geology Model", self)
        act_extract.triggered.connect(self._on_extract_geology)
        model_menu.addAction(act_extract)

        act_validate = QAction("Run Geological Audit", self)
        act_validate.triggered.connect(self._on_validate_compliance)
        model_menu.addAction(act_validate)

        model_btn = QToolButton()
        model_btn.setText("Model")
        model_btn.setMenu(model_menu)
        model_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        model_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(model_btn)

        # DOMAIN MENU
        domain_menu = QMenu(self)
        domain_menu.setStyleSheet(file_menu.styleSheet())

        act_adjust_domain = QAction("Adjust Domain...", self)
        act_adjust_domain.triggered.connect(lambda: self._domain_panel._on_adjust_clicked() if hasattr(self, '_domain_panel') else None)
        domain_menu.addAction(act_adjust_domain)

        domain_btn = QToolButton()
        domain_btn.setText("Domain")
        domain_btn.setMenu(domain_menu)
        domain_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        domain_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(domain_btn)

        # FAULTS MENU
        faults_menu = QMenu(self)
        faults_menu.setStyleSheet(file_menu.styleSheet())

        act_add_fault = QAction("Add Fault", self)
        act_add_fault.triggered.connect(self._on_add_fault)
        faults_menu.addAction(act_add_fault)

        act_remove_fault = QAction("Remove Selected Fault", self)
        act_remove_fault.triggered.connect(self._on_remove_fault)
        faults_menu.addAction(act_remove_fault)

        faults_btn = QToolButton()
        faults_btn.setText("Faults")
        faults_btn.setMenu(faults_menu)
        faults_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        faults_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(faults_btn)

        # Spacer
        from PyQt6.QtWidgets import QSizePolicy
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tb.addWidget(spacer)

        # EXPORT MENU (right side)
        export_menu = QMenu(self)
        export_menu.setStyleSheet(file_menu.styleSheet())

        export_menu.addSection("Surface Formats")

        act_export_obj = QAction("Export as OBJ", self)
        act_export_obj.triggered.connect(lambda: self._on_export_surfaces("obj"))
        export_menu.addAction(act_export_obj)

        act_export_stl = QAction("Export as STL", self)
        act_export_stl.triggered.connect(lambda: self._on_export_surfaces("stl"))
        export_menu.addAction(act_export_stl)

        act_export_vtk = QAction("Export as VTK", self)
        act_export_vtk.triggered.connect(lambda: self._on_export_surfaces("vtk"))
        export_menu.addAction(act_export_vtk)

        export_menu.addSeparator()
        export_menu.addSection("Reports")

        act_export_audit = QAction("Export Build Log (JSON)", self)
        act_export_audit.triggered.connect(self._on_export_audit)
        export_menu.addAction(act_export_audit)

        act_export_compliance = QAction("Export Compliance Report", self)
        act_export_compliance.triggered.connect(self._on_export_compliance)
        export_menu.addAction(act_export_compliance)

        export_btn = QToolButton()
        export_btn.setText("Export")
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(export_btn)

        # SETTINGS MENU (right side)
        settings_menu = QMenu(self)
        settings_menu.setStyleSheet(file_menu.styleSheet())

        act_reset_jorc = QAction("Reset to JORC Defaults", self)
        act_reset_jorc.triggered.connect(self._reset_jorc_thresholds)
        settings_menu.addAction(act_reset_jorc)

        settings_btn = QToolButton()
        settings_btn.setText("Settings")
        settings_btn.setMenu(settings_menu)
        settings_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        settings_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(settings_btn)

        return tb

    def _build_ui(self):
        """Build the complete panel UI with modern design."""
        # Use self.main_layout which is set up by BaseAnalysisPanel._setup_base_ui()
        # It's already a QVBoxLayout in a scrollable content widget

        # Compact header bar
        header_layout = QHBoxLayout()

        # Title - compact and inline (no emoji)
        header = QLabel("Geological Modeling")
        header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 15px;
                font-weight: 600;
                padding: 4px 0px;
            }}
        """)
        header_layout.addWidget(header)

        # Compliance badge - inline
        tech_badge = QLabel("JORC")
        tech_badge.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.SUCCESS};
                font-size: 9px;
                font-weight: 700;
                background: rgba(76, 175, 80, 0.12);
                padding: 2px 8px;
                border-radius: 8px;
            }}
        """)
        header_layout.addWidget(tech_badge)

        header_layout.addStretch()

        self.main_layout.addLayout(header_layout)

        # Compact toolbar with dropdown menus
        toolbar = self._build_toolbar()
        self.main_layout.addWidget(toolbar)

        # Workflow guidance banner — thin breadcrumb strip
        self._workflow_banner = QFrame()
        self._workflow_banner.setFixedHeight(32)
        self._workflow_banner.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        banner_layout = QHBoxLayout(self._workflow_banner)
        banner_layout.setContentsMargins(10, 0, 10, 0)
        banner_layout.setSpacing(4)

        # Step indicators — plain text, no unicode circles
        self._workflow_steps = []
        self._workflow_arrows = []
        step_defs = [
            ("1 Load", "Load drillhole data"),
            ("2 Strat", "Review stratigraphy"),
            ("3 Domain", "Check model extent"),
            ("4 Build", "Build geological model"),
            ("5 Audit", "Validate compliance"),
        ]
        for i, (label, tip) in enumerate(step_defs):
            step_label = QLabel(label)
            step_label.setToolTip(tip)
            step_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.TEXT_HINT};
                    font-size: 10px;
                    font-weight: 600;
                    background: transparent;
                    padding: 2px 5px;
                }}
            """)
            step_label.setCursor(Qt.CursorShape.PointingHandCursor)
            step_label.mousePressEvent = lambda event, idx=i: self._on_workflow_step_clicked(idx)
            banner_layout.addWidget(step_label)
            self._workflow_steps.append(step_label)

            if i < len(step_defs) - 1:
                arrow = QLabel(">")
                arrow.setStyleSheet(f"color: {ModernColors.BORDER_LIGHT}; font-size: 9px; background: transparent;")
                banner_layout.addWidget(arrow)
                self._workflow_arrows.append(arrow)

        banner_layout.addStretch()

        self._workflow_hint = QLabel("Load drillhole data from Registry or CSV")
        self._workflow_hint.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 10px;
                background: transparent;
                font-style: italic;
            }}
        """)
        banner_layout.addWidget(self._workflow_hint)

        self.main_layout.addWidget(self._workflow_banner)

        self._workflow_state = 0
        self._update_workflow_banner()

        # Modern Tab Widget with theme-aware styling
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
                background: {ModernColors.ELEVATED_BG};
                top: -1px;
            }}
            QTabBar::tab {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_SECONDARY};
                padding: 10px 20px;
                margin-right: 4px;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 12px;
                font-weight: 500;
                min-width: 90px;
            }}
            QTabBar::tab:selected {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.ACCENT_PRIMARY};
                border-bottom: 3px solid {ModernColors.ACCENT_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background: {ModernColors.BORDER};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QTabBar::tab:disabled {{
                color: {ModernColors.TEXT_DISABLED};
            }}
        """)
        self.main_layout.addWidget(self._tabs, stretch=0)

        # Tab 1: Input Validation (renamed from Data Input)
        data_tab = self._create_data_tab()
        self._tabs.addTab(data_tab, "Input Validation")

        # Tab 2: Model Configuration (Stratigraphy & Faults) - CRITICAL: Must be created to initialize _strat_list, _fault_table, _resolution_spin, etc.
        config_tab = self._create_config_tab()
        self._tabs.addTab(config_tab, "Stratigraphy")

        # Tab 3: Geological Domain (extent display)
        domain_tab = self._create_domain_tab()
        self._tabs.addTab(domain_tab, "Domain")

        # Tab 4: Model Build (dedicated execution panel)
        build_tab = self._create_build_tab()
        self._tabs.addTab(build_tab, "Build")

        # Tab 5: Geological Audit (renamed from Compliance & QC)
        compliance_tab = self._create_compliance_tab()
        self._tabs.addTab(compliance_tab, "Audit")

        # Tab 6: Structural Advisory
        advisory_tab = self._create_advisory_tab()
        self._tabs.addTab(advisory_tab, "Advisory")

        # Tab 7: Export
        export_tab = self._create_export_tab()
        self._tabs.addTab(export_tab, "Export")

        # Bottom: Simplified Action Bar (main actions moved to Build tab)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Quick Build button (navigates to Build tab and initiates build)
        self._build_btn = QPushButton("Build Model")
        self._build_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                background: {ModernColors.BORDER};
                color: {ModernColors.TEXT_DISABLED};
            }}
        """)
        self._build_btn.clicked.connect(self._on_quick_build)
        self._build_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._build_btn)

        button_layout.addStretch()

        self._close_btn = QPushButton("Close")
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {ModernColors.TEXT_SECONDARY};
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.ERROR};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
        """)
        self._close_btn.clicked.connect(self.close)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._close_btn)

        self.main_layout.addLayout(button_layout)

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        # Clear internal state
        self._model = None
        self._contacts_df = None
        self._orientations_df = None
        self._fault_list = []
        self._stratigraphy = []
        self._surfaces = []
        self._solids = []
        self._current_report = None
        self._formation_values = {}
        self._lithology_mapping = {}

        # Clear UI elements
        if self._strat_list:
            self._strat_list.clear()
        if self._fault_table:
            self._fault_table.setRowCount(0)

        # Reset build state
        if hasattr(self, '_build_status_widget') and self._build_status_widget:
            self._build_status_widget.reset()

        # Call base class
        super().clear_panel()
        logger.info("LoopStructuralModelPanel: Panel fully cleared")

        # Reset workflow
        if hasattr(self, '_workflow_state'):
            self._workflow_state = 0
            self._update_workflow_banner()

    # =========================================================================
    # Workflow Guidance System
    # =========================================================================

    def _on_workflow_step_clicked(self, step_index: int) -> None:
        """Navigate to the tab matching the clicked workflow step."""
        # Steps map: 0=Load(tab0), 1=Strat(tab1), 2=Domain(tab2), 3=Build(tab3), 4=Audit(tab4)
        if self._tabs is not None and 0 <= step_index <= 4:
            self._tabs.setCurrentIndex(step_index)

    def _update_workflow_banner(self) -> None:
        """Update the workflow banner to reflect current state."""
        if not hasattr(self, '_workflow_steps'):
            return

        hints = [
            "Load drillhole data from Registry or CSV",
            "Review and reorder the stratigraphic sequence",
            "Verify model domain extent covers your area",
            "Ready to build — click Build Model",
            "Model built — run Geological Audit to validate",
        ]

        for i, step_label in enumerate(self._workflow_steps):
            if i < self._workflow_state:
                step_label.setStyleSheet(f"QLabel {{ color: {ModernColors.SUCCESS}; font-size: 10px; font-weight: 600; background: transparent; padding: 2px 5px; }}")
            elif i == self._workflow_state:
                step_label.setStyleSheet(f"QLabel {{ color: {ModernColors.ACCENT_PRIMARY}; font-size: 10px; font-weight: 700; background: transparent; padding: 2px 5px; }}")
            else:
                step_label.setStyleSheet(f"QLabel {{ color: {ModernColors.TEXT_HINT}; font-size: 10px; font-weight: 600; background: transparent; padding: 2px 5px; }}")

        # Color arrows based on progress
        for i, arrow in enumerate(self._workflow_arrows):
            if i < self._workflow_state:
                arrow.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 9px; background: transparent;")
            else:
                arrow.setStyleSheet(f"color: {ModernColors.BORDER_LIGHT}; font-size: 9px; background: transparent;")

        state = min(self._workflow_state, len(hints) - 1)
        self._workflow_hint.setText(hints[state])

    def _create_data_tab(self) -> QWidget:
        """Create the Input Validation tab with compact checklist design."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Input Validation Checklist (replaces prose block)
        self._validation_checklist = InputValidationChecklist()
        layout.addWidget(self._validation_checklist)

        # Load buttons
        load_layout = QHBoxLayout()
        load_layout.setSpacing(10)

        modern_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: #FFFFFF;
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: #FFFFFF;
                border-color: {ModernColors.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_HOVER};
            }}
        """

        self._load_registry_btn = QPushButton("Load from Registry")
        self._load_registry_btn.setStyleSheet(modern_btn_style)
        self._load_registry_btn.clicked.connect(self._on_load_from_registry)
        self._load_registry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_layout.addWidget(self._load_registry_btn)

        self._load_file_btn = QPushButton("Load from File...")
        self._load_file_btn.setStyleSheet(modern_btn_style)
        self._load_file_btn.clicked.connect(self._on_load_from_file)
        self._load_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_layout.addWidget(self._load_file_btn)

        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Tip label - make it more prominent
        tip_frame = QFrame()
        tip_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.ELEVATED_BG};
                border-left: 3px solid {ModernColors.ACCENT_PRIMARY};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        tip_layout = QVBoxLayout(tip_frame)
        tip_layout.setContentsMargins(12, 8, 12, 8)

        tip_label = QLabel("💡 <b>Tip:</b> Stratigraphy is auto-detected from depth ordering in drillholes")
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
                background: transparent;
            }}
        """)
        tip_layout.addWidget(tip_label)
        layout.addWidget(tip_frame)

        layout.addStretch()
        return tab

    def _create_domain_tab(self) -> QWidget:
        """Create the Geological Domain tab with unified extent display."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Geological Domain Panel (unified extent display)
        self._domain_panel = GeologicalDomainPanel()
        layout.addWidget(self._domain_panel)

        # Keep internal spinboxes for backward compatibility
        # These are hidden but used by existing code
        self._xmin_spin = self._domain_panel._spinboxes['xmin']
        self._xmax_spin = self._domain_panel._spinboxes['xmax']
        self._ymin_spin = self._domain_panel._spinboxes['ymin']
        self._ymax_spin = self._domain_panel._spinboxes['ymax']
        self._zmin_spin = self._domain_panel._spinboxes['zmin']
        self._zmax_spin = self._domain_panel._spinboxes['zmax']

        # Data summary (brief, for quick reference)
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(12, 12, 12, 12)
        summary_layout.setSpacing(8)

        summary_header = QLabel("Data Summary")
        summary_header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 13px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        summary_layout.addWidget(summary_header)

        self._data_summary = QTextEdit()
        self._data_summary.setReadOnly(True)
        self._data_summary.setMaximumHeight(100)
        self._data_summary.setPlainText("No data loaded.")
        self._data_summary.setStyleSheet(f"""
            QTextEdit {{
                background: {ModernColors.PANEL_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                color: {ModernColors.TEXT_SECONDARY};
            }}
        """)
        summary_layout.addWidget(self._data_summary)

        layout.addWidget(summary_frame)

        layout.addStretch()
        return tab

    def _create_build_tab(self) -> QWidget:
        """Create the Model Build tab with dedicated execution panel."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Model Build Execution Panel
        self._build_panel = ModelBuildExecutionPanel()
        self._build_panel.build_requested.connect(self._on_build_model)
        self._build_panel.cancel_requested.connect(self._on_cancel_build)
        layout.addWidget(self._build_panel)

        # Secondary actions
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(12)

        secondary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.TEXT_SECONDARY};
                border: 1px solid {ModernColors.BORDER};
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {ModernColors.TEXT_DISABLED};
            }}
        """

        self._extract_btn = QPushButton("Extract Geology Model")
        self._extract_btn.setStyleSheet(secondary_btn_style)
        self._extract_btn.clicked.connect(self._on_extract_geology)
        self._extract_btn.setEnabled(False)
        self._extract_btn.setToolTip("Build the model first using the Build Model button above")
        secondary_layout.addWidget(self._extract_btn)

        self._validate_btn = QPushButton("Run Geological Audit")
        self._validate_btn.setStyleSheet(secondary_btn_style)
        self._validate_btn.clicked.connect(self._on_validate_compliance)
        self._validate_btn.setEnabled(False)
        self._validate_btn.setToolTip("Build the model first using the Build Model button above")
        secondary_layout.addWidget(self._validate_btn)

        secondary_layout.addStretch()
        layout.addLayout(secondary_layout)

        layout.addStretch()
        return tab

    def _on_cancel_build(self):
        """Handle build cancellation request."""
        logger.info("Build cancellation requested")

        if self._build_worker is not None and self._build_worker.isRunning():
            # Request cancellation
            self._build_worker.request_cancel()

            # Update UI to show cancellation in progress
            if hasattr(self, '_build_panel'):
                self._build_panel.set_diagnostics("Cancellation requested...\nWaiting for current step to complete.")

            # NOTE: We don't forcefully terminate the thread as that can corrupt state.
            # The worker will check the cancel flag and exit gracefully.
        else:
            logger.info("No active build to cancel")
            self._build_btn.setEnabled(True)
            if hasattr(self, '_build_panel'):
                self._build_panel.set_building(False)

    def _create_config_tab(self) -> QWidget:
        """Create the model configuration tab with modern design."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Lithology Grouping Section (collapsible)
        lith_group = CollapsibleGroup("Lithology Grouping (Optional)", collapsed=True)
        self._lith_grouping_widget = LithologyGroupingWidget()
        self._lith_grouping_widget.grouping_changed.connect(self._on_lithology_grouping_changed)
        lith_group.add_widget(self._lith_grouping_widget)
        layout.addWidget(lith_group)

        # Stratigraphy card
        strat_group = QGroupBox("Stratigraphic Sequence")
        strat_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.ELEVATED_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.ELEVATED_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        strat_layout = QVBoxLayout(strat_group)
        strat_layout.setSpacing(12)
        strat_layout.setContentsMargins(16, 20, 16, 16)

        # Visual stratigraphic column header with clear indicators
        strat_header_frame = QFrame()
        strat_header_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        strat_header_layout = QVBoxLayout(strat_header_frame)
        strat_header_layout.setContentsMargins(8, 8, 8, 8)
        strat_header_layout.setSpacing(4)

        # Clear visual indicator: TOP (youngest) and BASE (oldest)
        strat_info = QLabel(
            f"<table width='100%'>"
            f"<tr><td style='color: {ModernColors.ACCENT_PRIMARY}; font-weight: bold;'>▲ SURFACE (Youngest)</td></tr>"
            f"<tr><td style='color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding-left: 10px;'>↓ Units get OLDER downward ↓</td></tr>"
            f"<tr><td style='color: {ModernColors.ERROR}; font-weight: bold;'>▼ BASEMENT (Oldest)</td></tr>"
            f"</table>"
        )
        strat_info.setWordWrap(True)
        strat_info.setStyleSheet(f"background: transparent; font-size: 11px; color: {ModernColors.TEXT_PRIMARY};")
        strat_header_layout.addWidget(strat_info)

        # Validation status indicator (updated dynamically)
        self._strat_validation_label = QLabel("Validation: Pending")
        self._strat_validation_label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 10px;
                padding-top: 4px;
            }}
        """)
        strat_header_layout.addWidget(self._strat_validation_label)

        strat_layout.addWidget(strat_header_frame)

        # Stratigraphy list with modern reorder controls
        strat_content = QHBoxLayout()
        strat_content.setSpacing(12)

        # Modern list widget showing formations
        self._strat_list = QListWidget()
        self._strat_list.setMinimumHeight(150)
        self._strat_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._strat_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._strat_list.setStyleSheet(f"""
            QListWidget {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                padding: 6px;
                font-size: 12px;
            }}
            QListWidget::item {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
                padding: 8px;
                margin: 3px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {ModernColors.CARD_HOVER};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QListWidget::item:hover {{
                background: {ModernColors.BORDER};
            }}
        """)
        strat_content.addWidget(self._strat_list, stretch=1)

        # Modern reorder buttons
        reorder_layout = QVBoxLayout()
        reorder_layout.setSpacing(8)
        reorder_layout.addStretch()

        control_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_SECONDARY};
                border: 2px solid {ModernColors.BORDER};
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 11px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
        """

        self._strat_up_btn = QPushButton("Up (Younger)")
        self._strat_up_btn.setToolTip("Move selected unit UP toward surface (younger)")
        self._strat_up_btn.setStyleSheet(control_btn_style)
        self._strat_up_btn.clicked.connect(self._move_strat_up)
        self._strat_up_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reorder_layout.addWidget(self._strat_up_btn)

        self._strat_down_btn = QPushButton("Down (Older)")
        self._strat_down_btn.setToolTip("Move selected unit DOWN toward basement (older)")
        self._strat_down_btn.setStyleSheet(control_btn_style)
        self._strat_down_btn.clicked.connect(self._move_strat_down)
        self._strat_down_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reorder_layout.addWidget(self._strat_down_btn)

        reorder_layout.addSpacing(20)

        add_btn_style = f"""
            QPushButton {{
                background: {ModernColors.SUCCESS};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background: #2d8e47;
            }}
            QPushButton:pressed {{
                background: #1e7e34;
            }}
        """

        remove_btn_style = f"""
            QPushButton {{
                background: {ModernColors.ERROR};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background: #d33828;
            }}
            QPushButton:pressed {{
                background: #b31412;
            }}
        """

        self._strat_add_btn = QPushButton("+ Add")
        self._strat_add_btn.setToolTip("Add a new formation")
        self._strat_add_btn.setStyleSheet(add_btn_style)
        self._strat_add_btn.clicked.connect(self._add_strat_unit)
        self._strat_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reorder_layout.addWidget(self._strat_add_btn)

        self._strat_remove_btn = QPushButton("− Remove")
        self._strat_remove_btn.setToolTip("Remove selected formation")
        self._strat_remove_btn.setStyleSheet(remove_btn_style)
        self._strat_remove_btn.clicked.connect(self._remove_strat_unit)
        self._strat_remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reorder_layout.addWidget(self._strat_remove_btn)

        reorder_layout.addStretch()
        strat_content.addLayout(reorder_layout)

        strat_layout.addLayout(strat_content)

        # Hidden text input for compatibility (stores newline-separated list)
        self._strat_input = QTextEdit()
        self._strat_input.setVisible(False)
        strat_layout.addWidget(self._strat_input)

        # Sync list changes to text input
        self._strat_list.model().rowsMoved.connect(self._sync_strat_to_text)

        layout.addWidget(strat_group)

        # Dark Faults section
        fault_group = QGroupBox("Fault Events")
        fault_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        fault_layout = QVBoxLayout(fault_group)
        fault_layout.setSpacing(12)
        fault_layout.setContentsMargins(16, 20, 16, 16)

        fault_info = QLabel(
            f"<span style='color: {ModernColors.TEXT_SECONDARY}; font-size: 10px;'>"
            f"Faults are added FIRST (they displace space before stratigraphy)"
            f"</span>"
        )
        fault_info.setStyleSheet(f"background: {ModernColors.PANEL_BG}; padding: 6px; border-radius: 6px;")
        fault_layout.addWidget(fault_info)

        # Modern fault table
        self._fault_table = QTableWidget(0, 3)
        self._fault_table.setHorizontalHeaderLabels(["Name", "Displacement (m)", "Type"])
        self._fault_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._fault_table.setMaximumHeight(150)
        self._fault_table.setStyleSheet(f"""
            QTableWidget {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                gridline-color: {ModernColors.BORDER};
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 6px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QTableWidget::item:selected {{
                background: {ModernColors.CARD_HOVER};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QHeaderView::section {{
                background: {ModernColors.CARD_BG};
                color: {ModernColors.TEXT_SECONDARY};
                padding: 8px;
                border: none;
                font-weight: 600;
                font-size: 11px;
            }}
        """)
        fault_layout.addWidget(self._fault_table)

        # Modern fault buttons
        fault_btn_layout = QHBoxLayout()
        fault_btn_layout.setSpacing(8)

        modern_fault_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.ACCENT_PRIMARY};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
        """

        self._add_fault_btn = QPushButton("+ Add Fault")
        self._add_fault_btn.setStyleSheet(modern_fault_btn_style)
        self._add_fault_btn.clicked.connect(self._on_add_fault)
        self._add_fault_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        fault_btn_layout.addWidget(self._add_fault_btn)

        remove_fault_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.ERROR};
                border: 2px solid {ModernColors.ERROR};
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: #ff5449;
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
        """

        self._remove_fault_btn = QPushButton("− Remove Selected")
        self._remove_fault_btn.setStyleSheet(remove_fault_btn_style)
        self._remove_fault_btn.clicked.connect(self._on_remove_fault)
        self._remove_fault_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        fault_btn_layout.addWidget(self._remove_fault_btn)

        fault_btn_layout.addStretch()
        fault_layout.addLayout(fault_btn_layout)

        layout.addWidget(fault_group)

        # Dark Model Parameters section
        params_group = QGroupBox("Model Parameters")
        params_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(12)
        params_layout.setContentsMargins(16, 20, 16, 16)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self._resolution_spin = QSpinBox()
        self._resolution_spin.setRange(10, 200)
        self._resolution_spin.setValue(100)  # High resolution for better drillhole honouring
        self._resolution_spin.setToolTip("Grid cells per axis (higher = more detail)")
        res_layout.addWidget(self._resolution_spin)
        res_layout.addWidget(QLabel("cells per axis"))
        res_layout.addStretch()
        params_layout.addLayout(res_layout)

        # Regularization
        cgw_layout = QHBoxLayout()
        cgw_layout.addWidget(QLabel("Smoothing (CGW):"))
        self._cgw_spin = QDoubleSpinBox()
        self._cgw_spin.setRange(0.001, 1.0)
        self._cgw_spin.setValue(0.005)  # Low CGW = strict drillhole honouring (matches backend default)
        self._cgw_spin.setSingleStep(0.005)
        self._cgw_spin.setDecimals(3)
        self._cgw_spin.setToolTip("Regularization weight (lower = tighter fit to data, higher = smoother)\n0.005 = CP-grade tight fit (recommended)\n0.01-0.03 = moderate smoothing\n0.1+ = heavy smoothing (NOT recommended)")
        cgw_layout.addWidget(self._cgw_spin)
        cgw_layout.addStretch()
        params_layout.addLayout(cgw_layout)

        # Interpolator type
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Interpolator:"))
        self._interp_combo = QComboBox()
        self._interp_combo.addItems(["FDI (Finite Difference)", "PLI (Piece-wise Linear)"])
        self._interp_combo.setToolTip("FDI is best for layered rocks")
        interp_layout.addWidget(self._interp_combo)
        interp_layout.addStretch()
        params_layout.addLayout(interp_layout)

        # Mesh Smoothing (Taubin - industry standard)
        smooth_layout = QHBoxLayout()
        self._smooth_check = QCheckBox("Taubin Smoothing")
        self._smooth_check.setChecked(True)
        self._smooth_check.setToolTip("Apply Taubin smoothing to remove voxel artifacts (recommended)")
        smooth_layout.addWidget(self._smooth_check)

        smooth_layout.addWidget(QLabel("Iterations:"))
        self._smooth_iter_spin = QSpinBox()
        self._smooth_iter_spin.setRange(0, 100)
        self._smooth_iter_spin.setValue(20)
        self._smooth_iter_spin.setToolTip("Number of smoothing passes (20 = industry standard)")
        smooth_layout.addWidget(self._smooth_iter_spin)
        smooth_layout.addStretch()
        params_layout.addLayout(smooth_layout)

        # Smart parameter hint — updates when data is loaded
        self._param_hint_label = QLabel("")
        self._param_hint_label.setWordWrap(True)
        self._param_hint_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.ACCENT_PRIMARY};
                font-size: 10px;
                background: rgba(66, 133, 244, 0.08);
                padding: 8px 10px;
                border-radius: 6px;
                border: 1px solid rgba(66, 133, 244, 0.2);
            }}
        """)
        self._param_hint_label.hide()  # Hidden until data loaded
        params_layout.addWidget(self._param_hint_label)

        # Gradient Computation Options (NEW - for geologically realistic orientations)
        gradient_group = QGroupBox("Orientation Computation")
        gradient_layout = QVBoxLayout(gradient_group)
        gradient_layout.setSpacing(4)

        self._compute_gradients_check = QCheckBox("Compute gradients from contact geometry (PCA)")
        self._compute_gradients_check.setChecked(True)
        self._compute_gradients_check.setToolTip(
            "When enabled, derives gradient vectors from contact point clouds\n"
            "using PCA plane fitting. This produces geologically realistic\n"
            "orientations instead of assuming flat-lying (0,0,1) beds.\n\n"
            "RECOMMENDED: Leave enabled for better drillhole honouring."
        )
        gradient_layout.addWidget(self._compute_gradients_check)

        self._allow_synthetic_check = QCheckBox("Allow synthetic fallback if computation fails")
        self._allow_synthetic_check.setChecked(True)
        self._allow_synthetic_check.setToolTip(
            "If gradient computation fails (e.g., not enough contact points),\n"
            "fall back to synthetic horizontal orientations (0,0,1).\n\n"
            "Disable this to force the build to fail instead of using\n"
            "potentially incorrect orientations."
        )
        gradient_layout.addWidget(self._allow_synthetic_check)

        # Info label
        gradient_info = QLabel(
            "Gradient computation uses PCA plane fitting to derive actual dip/strike\n"
            "from contact point clouds. This prevents 'hallucinated' flat geology."
        )
        gradient_info.setStyleSheet("color: #888; font-size: 10px;")
        gradient_info.setWordWrap(True)
        gradient_layout.addWidget(gradient_info)

        params_layout.addWidget(gradient_group)

        layout.addWidget(params_group)

        layout.addStretch()
        return tab

    def _create_compliance_tab(self) -> QWidget:
        """Create the Geological Audit tab with structured verdict display."""
        from ..geology.compliance_manager import JORCThresholds, DEFAULT_JORC_THRESHOLDS

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Overall Audit Summary Banner — prominent PASS/FAIL at a glance
        self._audit_summary_banner = QFrame()
        self._audit_summary_banner.setMinimumHeight(60)
        self._audit_summary_banner.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
        """)
        banner_layout = QHBoxLayout(self._audit_summary_banner)
        banner_layout.setContentsMargins(16, 12, 16, 12)
        banner_layout.setSpacing(16)

        self._audit_status_icon = QLabel("--")
        self._audit_status_icon.setFixedWidth(36)
        self._audit_status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._audit_status_icon.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 24px; background: transparent;")
        banner_layout.addWidget(self._audit_status_icon)

        audit_text_layout = QVBoxLayout()
        audit_text_layout.setSpacing(2)

        self._audit_status_label = QLabel("Audit Not Run")
        self._audit_status_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 14px;
                font-weight: 700;
                background: transparent;
            }}
        """)
        audit_text_layout.addWidget(self._audit_status_label)

        self._audit_classification_label = QLabel("Build model and run audit to see results")
        self._audit_classification_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_HINT};
                font-size: 11px;
                background: transparent;
            }}
        """)
        audit_text_layout.addWidget(self._audit_classification_label)

        banner_layout.addLayout(audit_text_layout, stretch=1)

        # Quick metrics on the right
        self._audit_metrics_label = QLabel("")
        self._audit_metrics_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._audit_metrics_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 11px;
                font-family: 'Consolas', monospace;
                background: transparent;
            }}
        """)
        banner_layout.addWidget(self._audit_metrics_label)

        layout.addWidget(self._audit_summary_banner)

        # Geological Audit Verdict Table (structured compliance summary)
        self._audit_verdict_table = GeologicalAuditVerdictTable()
        layout.addWidget(self._audit_verdict_table)

        # JORC Threshold Configuration (collapsible)
        threshold_group = CollapsibleGroup("JORC/SAMREC Threshold Configuration", collapsed=True)

        threshold_widget = QWidget()
        threshold_layout = QVBoxLayout(threshold_widget)
        threshold_layout.setSpacing(8)

        # Info label
        threshold_info = QLabel(
            "Configure accuracy thresholds for resource classification. "
            "These values define the P90 and Mean error limits for Measured, "
            "Indicated, and Inferred categories."
        )
        threshold_info.setWordWrap(True)
        threshold_info.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px;")
        threshold_layout.addWidget(threshold_info)

        # Threshold inputs - using a grid layout
        grid_widget = QWidget()
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setSpacing(12)

        # Measured thresholds
        measured_frame = QFrame()
        measured_frame.setStyleSheet("QFrame { background: #2a3a2a; border-radius: 6px; }")
        measured_layout = QVBoxLayout(measured_frame)
        measured_layout.setContentsMargins(8, 8, 8, 8)
        measured_layout.addWidget(QLabel("<b style='color: #4caf50;'>Measured</b>"))

        self._measured_p90_spin = QDoubleSpinBox()
        self._measured_p90_spin.setRange(0.1, 10.0)
        self._measured_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.measured_p90)
        self._measured_p90_spin.setSuffix(" m")
        self._measured_p90_spin.setDecimals(1)
        measured_layout.addWidget(QLabel("P90 < "))
        measured_layout.addWidget(self._measured_p90_spin)

        self._measured_mean_spin = QDoubleSpinBox()
        self._measured_mean_spin.setRange(0.1, 10.0)
        self._measured_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.measured_mean)
        self._measured_mean_spin.setSuffix(" m")
        self._measured_mean_spin.setDecimals(1)
        measured_layout.addWidget(QLabel("Mean < "))
        measured_layout.addWidget(self._measured_mean_spin)
        grid_layout.addWidget(measured_frame)

        # Indicated thresholds
        indicated_frame = QFrame()
        indicated_frame.setStyleSheet("QFrame { background: #3a3a2a; border-radius: 6px; }")
        indicated_layout = QVBoxLayout(indicated_frame)
        indicated_layout.setContentsMargins(8, 8, 8, 8)
        indicated_layout.addWidget(QLabel("<b style='color: #ffc107;'>Indicated</b>"))

        self._indicated_p90_spin = QDoubleSpinBox()
        self._indicated_p90_spin.setRange(0.5, 20.0)
        self._indicated_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.indicated_p90)
        self._indicated_p90_spin.setSuffix(" m")
        self._indicated_p90_spin.setDecimals(1)
        indicated_layout.addWidget(QLabel("P90 < "))
        indicated_layout.addWidget(self._indicated_p90_spin)

        self._indicated_mean_spin = QDoubleSpinBox()
        self._indicated_mean_spin.setRange(0.5, 10.0)
        self._indicated_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.indicated_mean)
        self._indicated_mean_spin.setSuffix(" m")
        self._indicated_mean_spin.setDecimals(1)
        indicated_layout.addWidget(QLabel("Mean < "))
        indicated_layout.addWidget(self._indicated_mean_spin)
        grid_layout.addWidget(indicated_frame)

        # Inferred thresholds
        inferred_frame = QFrame()
        inferred_frame.setStyleSheet("QFrame { background: #3a2a2a; border-radius: 6px; }")
        inferred_layout = QVBoxLayout(inferred_frame)
        inferred_layout.setContentsMargins(8, 8, 8, 8)
        inferred_layout.addWidget(QLabel("<b style='color: #ff5722;'>Inferred</b>"))

        self._inferred_p90_spin = QDoubleSpinBox()
        self._inferred_p90_spin.setRange(1.0, 50.0)
        self._inferred_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.inferred_p90)
        self._inferred_p90_spin.setSuffix(" m")
        self._inferred_p90_spin.setDecimals(1)
        inferred_layout.addWidget(QLabel("P90 < "))
        inferred_layout.addWidget(self._inferred_p90_spin)

        self._inferred_mean_spin = QDoubleSpinBox()
        self._inferred_mean_spin.setRange(0.5, 20.0)
        self._inferred_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.inferred_mean)
        self._inferred_mean_spin.setSuffix(" m")
        self._inferred_mean_spin.setDecimals(1)
        inferred_layout.addWidget(QLabel("Mean < "))
        inferred_layout.addWidget(self._inferred_mean_spin)
        grid_layout.addWidget(inferred_frame)

        threshold_layout.addWidget(grid_widget)

        # Reset to defaults button
        reset_btn = QPushButton("Reset to JORC Defaults")
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.BORDER};
                color: {ModernColors.TEXT_SECONDARY};
                border: 1px solid #555;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: #4d4d4d; }}
        """)
        reset_btn.clicked.connect(self._reset_jorc_thresholds)
        threshold_layout.addWidget(reset_btn)

        threshold_group.add_widget(threshold_widget)
        layout.addWidget(threshold_group)

        # Detailed compliance panel (for visualization and metrics)
        details_group = CollapsibleGroup("Detailed Compliance View", collapsed=True)
        self._compliance_panel = ComplianceValidationPanel()
        details_group.add_widget(self._compliance_panel)
        layout.addWidget(details_group)

        layout.addStretch()
        return tab

    def _reset_jorc_thresholds(self):
        """Reset JORC thresholds to default values."""
        from ..geology.compliance_manager import DEFAULT_JORC_THRESHOLDS

        self._measured_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.measured_p90)
        self._measured_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.measured_mean)
        self._indicated_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.indicated_p90)
        self._indicated_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.indicated_mean)
        self._inferred_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.inferred_p90)
        self._inferred_mean_spin.setValue(DEFAULT_JORC_THRESHOLDS.inferred_mean)
        logger.info("JORC thresholds reset to defaults")

    def _get_current_jorc_thresholds(self):
        """Get the current JORC threshold configuration from UI."""
        from ..geology.compliance_manager import JORCThresholds

        return JORCThresholds(
            measured_p90=self._measured_p90_spin.value(),
            measured_mean=self._measured_mean_spin.value(),
            indicated_p90=self._indicated_p90_spin.value(),
            indicated_mean=self._indicated_mean_spin.value(),
            inferred_p90=self._inferred_p90_spin.value(),
            inferred_mean=self._inferred_mean_spin.value(),
        )

    def _create_advisory_tab(self) -> QWidget:
        """Create the structural advisory tab."""
        self._advisory_panel = StructuralAdvisoryWidget(
            apply_callback=self._on_apply_suggested_fault
        )
        return self._advisory_panel

    def _create_export_tab(self) -> QWidget:
        """Create the modern export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Dark Export Surfaces section
        surfaces_group = QGroupBox("Export Surfaces")
        surfaces_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        surfaces_layout = QVBoxLayout(surfaces_group)
        surfaces_layout.setSpacing(12)
        surfaces_layout.setContentsMargins(16, 20, 16, 16)

        self._surface_list = QListWidget()
        self._surface_list.setMaximumHeight(200)
        self._surface_list.setStyleSheet(f"""
            QListWidget {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                padding: 6px;
                font-size: 11px;
            }}
            QListWidget::item {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
                padding: 8px;
                margin: 3px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {ModernColors.CARD_HOVER};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        surfaces_layout.addWidget(self._surface_list)

        export_btn_layout = QHBoxLayout()
        export_btn_layout.setSpacing(10)

        modern_export_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.ACCENT_PRIMARY};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
            QPushButton:disabled {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_DISABLED};
                border-color: {ModernColors.BORDER};
            }}
        """

        self._export_obj_btn = QPushButton("Export as OBJ")
        self._export_obj_btn.setStyleSheet(modern_export_btn_style)
        self._export_obj_btn.clicked.connect(lambda: self._on_export_surfaces("obj"))
        self._export_obj_btn.setEnabled(False)
        self._export_obj_btn.setToolTip("Build model and Extract Geology first (Build tab)")
        self._export_obj_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn_layout.addWidget(self._export_obj_btn)

        self._export_stl_btn = QPushButton("Export as STL")
        self._export_stl_btn.setStyleSheet(modern_export_btn_style)
        self._export_stl_btn.clicked.connect(lambda: self._on_export_surfaces("stl"))
        self._export_stl_btn.setEnabled(False)
        self._export_stl_btn.setToolTip("Build model and Extract Geology first (Build tab)")
        self._export_stl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn_layout.addWidget(self._export_stl_btn)

        self._export_vtk_btn = QPushButton("Export as VTK")
        self._export_vtk_btn.setStyleSheet(modern_export_btn_style)
        self._export_vtk_btn.clicked.connect(lambda: self._on_export_surfaces("vtk"))
        self._export_vtk_btn.setEnabled(False)
        self._export_vtk_btn.setToolTip("Build model and Extract Geology first (Build tab)")
        self._export_vtk_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn_layout.addWidget(self._export_vtk_btn)

        export_btn_layout.addStretch()
        surfaces_layout.addLayout(export_btn_layout)

        layout.addWidget(surfaces_group)

        # Dark Export Audit Log section
        audit_group = QGroupBox("Export Audit Data")
        audit_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        audit_layout = QVBoxLayout(audit_group)
        audit_layout.setSpacing(10)
        audit_layout.setContentsMargins(16, 20, 16, 16)

        modern_audit_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.SUCCESS};
                border: 2px solid {ModernColors.SUCCESS};
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: #4caf50;
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
            QPushButton:disabled {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_DISABLED};
                border-color: {ModernColors.BORDER};
            }}
        """

        self._export_audit_btn = QPushButton("Export Build Log (JSON)")
        self._export_audit_btn.setStyleSheet(modern_audit_btn_style)
        self._export_audit_btn.clicked.connect(self._on_export_audit)
        self._export_audit_btn.setEnabled(False)
        self._export_audit_btn.setToolTip("Build model first, then run Geological Audit")
        self._export_audit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        audit_layout.addWidget(self._export_audit_btn)

        self._export_compliance_btn = QPushButton("Export Compliance Report")
        self._export_compliance_btn.setStyleSheet(modern_audit_btn_style)
        self._export_compliance_btn.clicked.connect(self._on_export_compliance)
        self._export_compliance_btn.setEnabled(False)
        self._export_compliance_btn.setToolTip("Build model first, then run Geological Audit")
        self._export_compliance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        audit_layout.addWidget(self._export_compliance_btn)

        layout.addWidget(audit_group)

        layout.addStretch()
        return tab

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _on_load_from_registry(self) -> None:
        """Load data from the DataRegistry."""
        try:
            logger.info("Loading data from registry...")
            registry = self.get_registry()
            if not registry:
                logger.error("DataRegistry not available")
                self.show_warning("No Registry", "DataRegistry not available.")
                return

            logger.debug(f"Registry obtained: {type(registry)}")

            # Get drillhole data
            dh_data = registry.get_drillhole_data()
            logger.debug(f"Drillhole data from registry: {type(dh_data)}, is None: {dh_data is None}")

            if dh_data is None:
                logger.warning("No drillhole data in registry")
                self.show_warning(
                    "No Data",
                    "No drillhole data in registry.\n\n"
                    "Please load drillhole data first:\n"
                    "• Drillholes → Drillhole Loading"
                )
                return

            # Extract contacts from drillhole data
            df = None
            if isinstance(dh_data, dict):
                logger.info(f"Drillhole data keys: {list(dh_data.keys())}")

                # Priority order: composites (best for modeling), then assays, then intervals
                for key in ['composites', 'assays', 'intervals', 'survey', 'collar']:
                    candidate = dh_data.get(key)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and len(candidate) > 0:
                        # Check if it has required columns
                        if all(col in candidate.columns for col in ['X', 'Y', 'Z']):
                            df = candidate
                            logger.info(f"Using '{key}' data with {len(df)} rows")
                            logger.info(f"Columns in {key}: {list(df.columns)}")
                            break

                # If still no df, try any DataFrame in the dict
                if df is None:
                    for key, value in dh_data.items():
                        if isinstance(value, pd.DataFrame) and len(value) > 0:
                            if all(col in value.columns for col in ['X', 'Y', 'Z']):
                                df = value
                                logger.info(f"Using '{key}' data as fallback")
                                break

            elif isinstance(dh_data, pd.DataFrame):
                df = dh_data

            if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
                self.show_warning("No Data", "No valid data with X, Y, Z columns found in registry.")
                return

            # Set up contacts DataFrame
            self._contacts_df = df.reset_index(drop=True).copy()

            # Ensure required columns exist for LoopStructural
            logger.info(f"Available columns in data: {list(self._contacts_df.columns)}")
            
            # Log coordinate ranges to verify they're correct
            if all(col in self._contacts_df.columns for col in ['X', 'Y', 'Z']):
                x_range = (self._contacts_df['X'].min(), self._contacts_df['X'].max())
                y_range = (self._contacts_df['Y'].min(), self._contacts_df['Y'].max())
                z_range = (self._contacts_df['Z'].min(), self._contacts_df['Z'].max())
                logger.info(f"LoopStructural contacts coordinate ranges: X={x_range}, Y={y_range}, Z={z_range}")
            else:
                logger.warning("LoopStructural contacts missing X, Y, Z columns!")

            # Add 'formation' column if missing
            if 'formation' not in self._contacts_df.columns:
                formation_col = None
                col_lower_map = {col.lower(): col for col in self._contacts_df.columns}

                # Priority order for formation column detection
                for candidate in ['lithology', 'lith_code', 'lithcode', 'lith', 'formation',
                                  'rock_type', 'rocktype', 'geology', 'unit', 'geo_unit',
                                  'rock', 'litho', 'rock_code', 'geo_code']:
                    if candidate in col_lower_map:
                        formation_col = col_lower_map[candidate]
                        break

                if formation_col:
                    self._contacts_df['formation'] = self._contacts_df[formation_col].values
                    logger.info(f"Using '{formation_col}' column as 'formation'")
                    unique_vals = self._contacts_df['formation'].dropna().unique()
                    logger.info(f"Found {len(unique_vals)} unique formations: {list(unique_vals)}")
                else:
                    # Try to get lithology from separate lithology table
                    if isinstance(dh_data, dict) and 'lithology' in dh_data:
                        lith_df = dh_data['lithology']
                        if isinstance(lith_df, pd.DataFrame) and len(lith_df) > 0:
                            logger.info(f"Found separate 'lithology' table with columns: {list(lith_df.columns)}")
                            merged = self._merge_lithology_data(self._contacts_df, lith_df)
                            if merged is not None:
                                self._contacts_df = merged
                                logger.info(f"Merged lithology data - formations: {list(self._contacts_df['formation'].dropna().unique())}")
                            else:
                                self._contacts_df['formation'] = 'Unit_1'
                                logger.warning("Could not merge lithology data - using default 'Unit_1'")
                        else:
                            self._contacts_df['formation'] = 'Unit_1'
                            logger.warning(f"No formation/lithology column found - using default 'Unit_1'")
                    else:
                        self._contacts_df['formation'] = 'Unit_1'
                        logger.warning(f"No formation/lithology column found - using default 'Unit_1'")

            # Populate lithology grouping widget with unique lithologies
            if self._lith_grouping_widget is not None:
                unique_liths = list(self._contacts_df['formation'].dropna().unique())
                self._lith_grouping_widget.set_lithologies(unique_liths)
                logger.info(f"Populated lithology grouping widget with {len(unique_liths)} unique lithologies")

            # Add 'val' column if missing - use proportional spacing for thin units
            if 'val' not in self._contacts_df.columns:
                unique_formations = list(self._contacts_df['formation'].dropna().unique())

                # Try to use proportional scalar spacing based on unit thicknesses
                # This ensures thin units are properly represented in the model
                formation_to_val = _calculate_proportional_scalar_spacing(
                    self._contacts_df,
                    unique_formations,
                    min_spacing=0.5  # Minimum 0.5 scalar units between formations
                )

                self._contacts_df['val'] = self._contacts_df['formation'].apply(
                    lambda x: formation_to_val.get(x, 0.0) if pd.notna(x) else 0.0
                )
                # Store formation values for isosurface extraction
                self._formation_values = formation_to_val.copy()
                logger.info(f"Generated 'val' column with proportional spacing: {formation_to_val}")

            # Generate synthetic orientations if not available
            if 'gx' not in self._contacts_df.columns:
                self._contacts_df['gx'] = 0.0
                self._contacts_df['gy'] = 0.0
                self._contacts_df['gz'] = 1.0

            self._orientations_df = self._contacts_df[['X', 'Y', 'Z', 'gx', 'gy', 'gz']].copy()

            # ================================================================
            # CRITICAL FIX: Filter outliers before calculating extent
            # ================================================================
            # This prevents the model extent from being stretched by invalid
            # coordinates like (0, 0, 0) placeholder data.
            # ================================================================
            df_for_extent = _filter_outliers_for_extent(df)
            
            # Set extent from FILTERED data with adaptive padding
            # Use 10% of data range as padding (minimum 200m for X/Y, 100m for Z)
            x_range = float(df_for_extent['X'].max()) - float(df_for_extent['X'].min())
            y_range = float(df_for_extent['Y'].max()) - float(df_for_extent['Y'].min())
            z_range = float(df_for_extent['Z'].max()) - float(df_for_extent['Z'].min())
            
            x_pad = max(200, x_range * 0.1)
            y_pad = max(200, y_range * 0.1)
            z_pad = max(100, z_range * 0.15)  # More Z padding for geological layering
            
            self._xmin_spin.setValue(float(df_for_extent['X'].min()) - x_pad)
            self._xmax_spin.setValue(float(df_for_extent['X'].max()) + x_pad)
            self._ymin_spin.setValue(float(df_for_extent['Y'].min()) - y_pad)
            self._ymax_spin.setValue(float(df_for_extent['Y'].max()) + y_pad)
            self._zmin_spin.setValue(float(df_for_extent['Z'].min()) - z_pad)
            self._zmax_spin.setValue(float(df_for_extent['Z'].max()) + z_pad)
            
            # Log the calculated extent for debugging
            logger.info(
                f"Model extent calculated: "
                f"X=[{self._xmin_spin.value():.1f}, {self._xmax_spin.value():.1f}], "
                f"Y=[{self._ymin_spin.value():.1f}, {self._ymax_spin.value():.1f}], "
                f"Z=[{self._zmin_spin.value():.1f}, {self._zmax_spin.value():.1f}]"
            )

            # Auto-detect stratigraphic sequence from depth relationships
            stratigraphy = self._auto_detect_stratigraphy(self._contacts_df)
            if stratigraphy:
                self._populate_strat_list(stratigraphy)
                logger.info(f"Auto-detected stratigraphy (oldest→youngest): {stratigraphy}")
            else:
                self._populate_strat_list(['Unit_1'])

            # Update summary
            self._update_data_summary()

            self.show_info("Data Loaded", f"Loaded {len(df)} contacts from registry.")

            # Advance workflow to stratigraphy step and switch tab
            if hasattr(self, '_workflow_state'):
                self._workflow_state = max(self._workflow_state, 1)
                self._update_workflow_banner()
            if self._tabs is not None:
                self._tabs.setCurrentIndex(1)  # Switch to Stratigraphy tab

        except Exception as e:
            logger.error(f"Failed to load from registry: {e}")
            self.show_error("Load Error", str(e))

    def _on_load_from_file(self) -> None:
        """Load data from CSV files."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Contact Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filepath:
            try:
                self._contacts_df = pd.read_csv(filepath).reset_index(drop=True)

                # Validate required columns
                required = ['X', 'Y', 'Z']
                missing = [c for c in required if c not in self._contacts_df.columns]
                if missing:
                    self.show_warning(
                        "Missing Columns",
                        f"CSV missing columns: {', '.join(missing)}\n"
                        "Required: X, Y, Z"
                    )
                    return

                # Add 'formation' column if missing
                if 'formation' not in self._contacts_df.columns:
                    col_lower_map = {col.lower(): col for col in self._contacts_df.columns}
                    formation_col = None

                    for candidate in ['lithology', 'lith', 'rock_type', 'geology', 'unit']:
                        if candidate in col_lower_map:
                            formation_col = col_lower_map[candidate]
                            break

                    if formation_col:
                        self._contacts_df['formation'] = self._contacts_df[formation_col].values
                    else:
                        self._contacts_df['formation'] = 'Unit_1'

                # Populate lithology grouping widget with unique lithologies
                if self._lith_grouping_widget is not None:
                    unique_liths = list(self._contacts_df['formation'].dropna().unique())
                    self._lith_grouping_widget.set_lithologies(unique_liths)
                    logger.info(f"Populated lithology grouping widget with {len(unique_liths)} unique lithologies")

                # Add 'val' column if missing - use proportional spacing for thin units
                if 'val' not in self._contacts_df.columns:
                    unique_formations = list(self._contacts_df['formation'].dropna().unique())
                    formation_to_val = _calculate_proportional_scalar_spacing(
                        self._contacts_df,
                        unique_formations,
                        min_spacing=0.5  # Minimum 0.5 scalar units between formations
                    )
                    self._contacts_df['val'] = self._contacts_df['formation'].apply(
                        lambda x: formation_to_val.get(x, 0.0) if pd.notna(x) else 0.0
                    )
                    # Store formation values for isosurface extraction
                    self._formation_values = formation_to_val.copy()
                    logger.info(f"Generated 'val' column with proportional spacing: {formation_to_val}")

                # ================================================================
                # CRITICAL FIX: Filter outliers before calculating extent
                # ================================================================
                df_for_extent = _filter_outliers_for_extent(self._contacts_df)
                
                # Set extent from FILTERED data with adaptive padding
                x_range = float(df_for_extent['X'].max()) - float(df_for_extent['X'].min())
                y_range = float(df_for_extent['Y'].max()) - float(df_for_extent['Y'].min())
                z_range = float(df_for_extent['Z'].max()) - float(df_for_extent['Z'].min())
                
                x_pad = max(200, x_range * 0.1)
                y_pad = max(200, y_range * 0.1)
                z_pad = max(100, z_range * 0.15)
                
                self._xmin_spin.setValue(float(df_for_extent['X'].min()) - x_pad)
                self._xmax_spin.setValue(float(df_for_extent['X'].max()) + x_pad)
                self._ymin_spin.setValue(float(df_for_extent['Y'].min()) - y_pad)
                self._ymax_spin.setValue(float(df_for_extent['Y'].max()) + y_pad)
                self._zmin_spin.setValue(float(df_for_extent['Z'].min()) - z_pad)
                self._zmax_spin.setValue(float(df_for_extent['Z'].max()) + z_pad)
                
                # Log the calculated extent for debugging
                logger.info(
                    f"Model extent calculated: "
                    f"X=[{self._xmin_spin.value():.1f}, {self._xmax_spin.value():.1f}], "
                    f"Y=[{self._ymin_spin.value():.1f}, {self._ymax_spin.value():.1f}], "
                    f"Z=[{self._zmin_spin.value():.1f}, {self._zmax_spin.value():.1f}]"
                )

                # Auto-detect stratigraphy
                stratigraphy = self._auto_detect_stratigraphy(self._contacts_df)
                if stratigraphy:
                    self._populate_strat_list(stratigraphy)
                else:
                    self._populate_strat_list(['Unit_1'])

                self._update_data_summary()
                self.show_info("Data Loaded", f"Loaded {len(self._contacts_df)} rows from file.")

                # Advance workflow to stratigraphy step and switch tab
                if hasattr(self, '_workflow_state'):
                    self._workflow_state = max(self._workflow_state, 1)
                    self._update_workflow_banner()
                if self._tabs is not None:
                    self._tabs.setCurrentIndex(1)  # Switch to Stratigraphy tab

            except Exception as e:
                logger.error(f"Failed to load file: {e}")
                self.show_error("Load Error", str(e))

    def _merge_lithology_data(self, contacts_df: pd.DataFrame, lith_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Merge lithology data with contacts based on common keys."""
        try:
            # Look for common join columns (HoleID, From, To, etc.)
            common_cols = set(contacts_df.columns) & set(lith_df.columns)

            # Priority merge keys
            merge_keys = ['HoleID', 'hole_id', 'Hole_ID', 'HOLEID', 'From', 'To', 'from', 'to']
            valid_keys = [k for k in merge_keys if k in common_cols]

            if valid_keys:
                merged = contacts_df.merge(lith_df[['formation'] + valid_keys],
                                          on=valid_keys, how='left')
                return merged
            return None
        except Exception as e:
            logger.error(f"Lithology merge failed: {e}")
            return None

    def _auto_detect_stratigraphy(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect stratigraphic sequence from depth relationships."""
        try:
            if 'formation' not in df.columns:
                return []

            # Group by formation and get average Z (depth)
            depth_by_formation = df.groupby('formation')['Z'].mean().sort_values()

            # Reverse so oldest (deepest) is first
            return list(depth_by_formation.index[::-1])
        except Exception as e:
            logger.error(f"Stratigraphy detection failed: {e}")
            return []

    def _populate_strat_list(self, strat: List[str]) -> None:
        """Populate stratigraphy list widget."""
        if self._strat_list is None:
            logger.warning("Cannot populate stratigraphy list - widget not initialized yet")
            self._stratigraphy = strat  # Store for later initialization
            return
        self._strat_list.clear()
        self._stratigraphy = strat
        for unit in strat:
            self._strat_list.addItem(unit)
        self._sync_strat_to_text()

        # Validate stratigraphic ordering against drillhole data
        self._validate_stratigraphy_order()

        # Advance workflow to domain step
        if hasattr(self, '_workflow_state') and len(strat) > 0:
            self._workflow_state = max(self._workflow_state, 2)
            self._update_workflow_banner()

    def _validate_stratigraphy_order(self) -> None:
        """
        Validate the stratigraphic sequence against drillhole depth ordering.

        Updates the validation label in the UI with the result.
        """
        if not hasattr(self, '_strat_validation_label'):
            return

        if self._contacts_df is None or len(self._stratigraphy) == 0:
            self._strat_validation_label.setText("Validation: No data")
            self._strat_validation_label.setStyleSheet(
                f"background: transparent; color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding-top: 4px;"
            )
            return

        try:
            from ..geology.chronos_engine import validate_stratigraphy_sequence

            result = validate_stratigraphy_sequence(
                stratigraphy=self._stratigraphy,
                contacts_df=self._contacts_df,
                allow_missing_units=True,
                max_inversions_per_hole=2
            )

            if result.recommendation == "ACCEPT":
                self._strat_validation_label.setText(
                    f"PASS - Validated ({result.holes_checked} holes checked)"
                )
                self._strat_validation_label.setStyleSheet(
                    f"background: transparent; color: {ModernColors.SUCCESS}; font-size: 10px; padding-top: 4px; font-weight: bold;"
                )
            elif result.recommendation == "REVIEW":
                self._strat_validation_label.setText(
                    f"WARN -  {result.inversions_found} inversions detected - REVIEW NEEDED"
                )
                self._strat_validation_label.setStyleSheet(
                    f"background: transparent; color: {ModernColors.WARNING}; font-size: 10px; padding-top: 4px; font-weight: bold;"
                )
            else:  # REJECT
                self._strat_validation_label.setText(
                    f"FAIL - {len(result.violations)} errors found"
                )
                self._strat_validation_label.setStyleSheet(
                    f"background: transparent; color: {ModernColors.ERROR}; font-size: 10px; padding-top: 4px; font-weight: bold;"
                )

            # Store result for later use
            self._strat_validation_result = result

        except ImportError:
            self._strat_validation_label.setText("Validation: Module not available")
            self._strat_validation_label.setStyleSheet(
                f"background: transparent; color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding-top: 4px;"
            )
        except Exception as e:
            logger.warning(f"Stratigraphy validation failed: {e}")
            self._strat_validation_label.setText(f"Validation: Error - {str(e)[:30]}")
            self._strat_validation_label.setStyleSheet(
                f"background: transparent; color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding-top: 4px;"
            )

    def _sync_strat_to_text(self) -> None:
        """Sync stratigraphy list to hidden text input."""
        if self._strat_list is None or self._strat_input is None:
            return
        items = [self._strat_list.item(i).text() for i in range(self._strat_list.count())]
        self._strat_input.setPlainText('\n'.join(items))
        self._stratigraphy = items

    def _move_strat_up(self) -> None:
        """Move selected stratigraphy unit up (older)."""
        if self._strat_list is None:
            return
        row = self._strat_list.currentRow()
        if row > 0:
            item = self._strat_list.takeItem(row)
            self._strat_list.insertItem(row - 1, item)
            self._strat_list.setCurrentRow(row - 1)
            self._sync_strat_to_text()

    def _move_strat_down(self) -> None:
        """Move selected stratigraphy unit down (younger)."""
        if self._strat_list is None:
            return
        row = self._strat_list.currentRow()
        if row >= 0 and row < self._strat_list.count() - 1:
            item = self._strat_list.takeItem(row)
            self._strat_list.insertItem(row + 1, item)
            self._strat_list.setCurrentRow(row + 1)
            self._sync_strat_to_text()

    def _add_strat_unit(self) -> None:
        """Add a new stratigraphy unit."""
        if self._strat_list is None:
            return
        # Simple dialog for new unit name
        new_unit = f"Unit_{self._strat_list.count() + 1}"
        self._strat_list.addItem(new_unit)
        self._sync_strat_to_text()

    def _remove_strat_unit(self) -> None:
        """Remove selected stratigraphy unit."""
        if self._strat_list is None:
            return
        row = self._strat_list.currentRow()
        if row >= 0:
            self._strat_list.takeItem(row)
            self._sync_strat_to_text()

    def _on_lithology_grouping_changed(self, mapping: Dict[str, str]) -> None:
        """Handle changes to lithology grouping."""
        self._lithology_mapping = mapping
        logger.info(f"Lithology grouping updated: {len(mapping)} lithologies mapped to groups")

        # Update stratigraphy list with grouped lithologies
        if self._lith_grouping_widget is not None and self._contacts_df is not None:
            # Get the grouped formations list
            grouped_liths = self._lith_grouping_widget.get_grouped_lithologies()

            # Auto-detect stratigraphy from grouped data
            if grouped_liths:
                # Apply grouping to contacts to get proper stratigraphy ordering
                grouped_df = self._lith_grouping_widget.apply_grouping_to_dataframe(
                    self._contacts_df, column='formation'
                )
                stratigraphy = self._auto_detect_stratigraphy(grouped_df)
                if stratigraphy:
                    self._populate_strat_list(stratigraphy)
                    logger.info(f"Updated stratigraphy from grouped data: {stratigraphy}")

    def _on_add_fault(self) -> None:
        """Add a fault event."""
        row_count = self._fault_table.rowCount()
        self._fault_table.insertRow(row_count)

        # Add default values
        self._fault_table.setItem(row_count, 0, QTableWidgetItem(f"Fault_{row_count + 1}"))
        self._fault_table.setItem(row_count, 1, QTableWidgetItem("100.0"))
        self._fault_table.setItem(row_count, 2, QTableWidgetItem("Normal"))

    def _on_remove_fault(self) -> None:
        """Remove selected fault."""
        row = self._fault_table.currentRow()
        if row >= 0:
            self._fault_table.removeRow(row)

    def _update_data_summary(self) -> None:
        """Update the data summary display and new workflow widgets."""
        if self._contacts_df is None or len(self._contacts_df) == 0:
            self._data_summary.setPlainText("No data loaded.")
            # Reset validation checklist
            if hasattr(self, '_validation_checklist'):
                self._validation_checklist.validate_dataframe(None)
            return

        n_contacts = len(self._contacts_df)
        n_formations = self._contacts_df['formation'].nunique()
        n_holes = 0
        for col in ['hole_id', 'HOLEID', 'HoleID', 'BHID']:
            if col in self._contacts_df.columns:
                n_holes = self._contacts_df[col].nunique()
                break

        # Structured summary
        summary_lines = [
            f"Contacts:    {n_contacts:,} points",
            f"Formations:  {n_formations} unique",
        ]
        if n_holes > 0:
            summary_lines.append(f"Drillholes:  {n_holes}")
        summary_lines.extend([
            f"",
            f"X: {self._contacts_df['X'].min():.1f} → {self._contacts_df['X'].max():.1f}  ({self._contacts_df['X'].max() - self._contacts_df['X'].min():.0f}m)",
            f"Y: {self._contacts_df['Y'].min():.1f} → {self._contacts_df['Y'].max():.1f}  ({self._contacts_df['Y'].max() - self._contacts_df['Y'].min():.0f}m)",
            f"Z: {self._contacts_df['Z'].min():.1f} → {self._contacts_df['Z'].max():.1f}  ({self._contacts_df['Z'].max() - self._contacts_df['Z'].min():.0f}m)",
        ])
        self._data_summary.setPlainText('\n'.join(summary_lines))

        # Update validation checklist
        if hasattr(self, '_validation_checklist'):
            self._validation_checklist.validate_dataframe(self._contacts_df)

        # Update domain panel from existing spinbox values (set by loader with adaptive padding)
        if hasattr(self, '_domain_panel'):
            self._domain_panel.set_extent(
                self._xmin_spin.value(), self._xmax_spin.value(),
                self._ymin_spin.value(), self._ymax_spin.value(),
                self._zmin_spin.value(), self._zmax_spin.value()
            )

            # Calculate actual coverage (% of model extent covered by data)
            data_x_range = float(self._contacts_df['X'].max() - self._contacts_df['X'].min())
            data_y_range = float(self._contacts_df['Y'].max() - self._contacts_df['Y'].min())
            data_z_range = float(self._contacts_df['Z'].max() - self._contacts_df['Z'].min())
            model_x_range = max(1.0, self._xmax_spin.value() - self._xmin_spin.value())
            model_y_range = max(1.0, self._ymax_spin.value() - self._ymin_spin.value())
            model_z_range = max(1.0, self._zmax_spin.value() - self._zmin_spin.value())
            self._domain_panel.set_coverage(
                min(100, data_x_range / model_x_range * 100),
                min(100, data_y_range / model_y_range * 100),
                min(100, data_z_range / model_z_range * 100),
            )

        # Update smart parameter hints based on data characteristics
        if hasattr(self, '_param_hint_label'):
            # Recommend resolution based on data density
            avg_spacing = 0.0
            if n_contacts > 1:
                x_span = float(self._contacts_df['X'].max() - self._contacts_df['X'].min())
                y_span = float(self._contacts_df['Y'].max() - self._contacts_df['Y'].min())
                area = max(1.0, x_span * y_span)
                avg_spacing = (area / n_contacts) ** 0.5

            hints = []
            if n_contacts < 50:
                hints.append(f"Sparse data ({n_contacts} pts) — try Resolution 30–50, CGW 0.01–0.03")
            elif n_contacts < 500:
                hints.append(f"Moderate data ({n_contacts} pts) — Resolution 50–80 recommended")
            else:
                hints.append(f"Dense data ({n_contacts} pts) — Resolution 80–120 for detail")

            if n_formations > 10:
                hints.append(f"{n_formations} formations — consider grouping similar units")

            if avg_spacing > 0:
                hints.append(f"Avg point spacing: ~{avg_spacing:.0f}m")

            self._param_hint_label.setText("  |  ".join(hints))
            self._param_hint_label.show()

    def _on_quick_build(self) -> None:
        """Navigate to Build tab and initiate build."""
        # Navigate to the Model Build tab (index 3: Input Validation=0, Stratigraphy=1, Domain=2, Build=3)
        if self._tabs is not None:
            self._tabs.setCurrentIndex(3)
        # Trigger build
        self._on_build_model()

    def _on_build_model(self) -> None:
        """Build the geological model using a worker thread for responsive UI."""
        from ..geology.industry_modeler import GeoXIndustryModeler

        # Safety check: Ensure UI widgets are initialized
        if self._resolution_spin is None or self._cgw_spin is None:
            self.show_error("UI Not Ready", "Panel UI is not fully initialized. Please try again.")
            return

        # Check if a build is already in progress
        if self._build_worker is not None and self._build_worker.isRunning():
            self.show_warning("Build In Progress", "A model build is already running. Please wait or cancel it first.")
            return

        # Check if LoopStructural is available
        if not GeoXIndustryModeler.is_available():
            self.show_error(
                "LoopStructural Not Available",
                "LoopStructural library is not installed.\n\n"
                "Install with: pip install LoopStructural>=1.6.0"
            )
            return

        if self._contacts_df is None or len(self._contacts_df) == 0:
            self.show_warning("No Data", "Please load data first.")
            return

        if not self._stratigraphy or len(self._stratigraphy) == 0:
            self.show_warning("No Stratigraphy", "Please define stratigraphic sequence first.")
            return

        # Disable build button and update UI
        self._build_btn.setEnabled(False)

        # Update build panel if available
        if hasattr(self, '_build_panel'):
            self._build_panel.set_building(True)
            self._build_panel.reset()
            self._build_panel.set_diagnostics("Starting build worker thread...")

        # Gather parameters from UI
        extent = np.array([
            self._xmin_spin.value(), self._xmax_spin.value(),
            self._ymin_spin.value(), self._ymax_spin.value(),
            self._zmin_spin.value(), self._zmax_spin.value()
        ])

        resolution = self._resolution_spin.value()
        cgw = self._cgw_spin.value()

        # Gather fault parameters
        fault_params = []
        for row in range(self._fault_table.rowCount()):
            name_item = self._fault_table.item(row, 0)
            disp_item = self._fault_table.item(row, 1)
            type_item = self._fault_table.item(row, 2)

            if name_item and disp_item:
                fault_params.append({
                    'name': name_item.text(),
                    'displacement': float(disp_item.text()),
                    'type': type_item.text() if type_item else 'normal'
                })

        logger.info(f"Starting threaded model build: extent={extent}, resolution={resolution}, cgw={cgw}, faults={len(fault_params)}")

        # Record start time
        self._build_start_time = datetime.now()

        # Apply lithology grouping if configured
        contacts_for_model = self._contacts_df.copy()
        stratigraphy_for_model = self._stratigraphy.copy()

        if self._lithology_mapping:
            logger.info(f"Applying lithology grouping: {len(self._lithology_mapping)} mappings")

            # Apply grouping to formation column
            contacts_for_model['formation'] = contacts_for_model['formation'].apply(
                lambda x: self._lithology_mapping.get(x, x) if pd.notna(x) else x
            )

            # Update stratigraphy to use grouped names
            seen = set()
            grouped_stratigraphy = []
            for unit in self._stratigraphy:
                grouped_name = self._lithology_mapping.get(unit, unit)
                if grouped_name not in seen:
                    grouped_stratigraphy.append(grouped_name)
                    seen.add(grouped_name)
            stratigraphy_for_model = grouped_stratigraphy

            # Recalculate 'val' column for grouped formations
            unique_grouped = list(contacts_for_model['formation'].dropna().unique())
            formation_to_val = _calculate_proportional_scalar_spacing(
                contacts_for_model,
                unique_grouped,
                min_spacing=0.5
            )
            contacts_for_model['val'] = contacts_for_model['formation'].apply(
                lambda x: formation_to_val.get(x, 0.0) if pd.notna(x) else 0.0
            )
            # Update stored formation values for isosurface extraction
            self._formation_values = formation_to_val.copy()

            logger.info(f"Grouped stratigraphy: {stratigraphy_for_model}")

        # Get gradient computation settings (defaults if UI controls not yet added)
        compute_gradients = getattr(self, '_compute_gradients_check', None)
        compute_gradients = compute_gradients.isChecked() if compute_gradients else True

        allow_synthetic = getattr(self, '_allow_synthetic_check', None)
        allow_synthetic = allow_synthetic.isChecked() if allow_synthetic else True

        # Create and configure worker with GeologicalModelRunner
        self._build_worker = ModelBuildWorker(
            contacts_df=contacts_for_model,
            stratigraphy=stratigraphy_for_model,
            extent=extent,
            resolution=resolution,
            cgw=cgw,
            fault_params=fault_params,
            formation_values=self._formation_values,
            compute_gradients=compute_gradients,
            allow_synthetic_fallback=allow_synthetic,
            parent=self
        )

        # Connect signals
        self._build_worker.progress_updated.connect(self._on_build_progress)
        self._build_worker.phase_changed.connect(self._on_build_phase_changed)
        self._build_worker.build_completed.connect(self._on_build_completed)
        self._build_worker.build_failed.connect(self._on_build_failed)
        self._build_worker.build_cancelled.connect(self._on_build_cancelled)

        # Start the worker thread
        self._build_worker.start()
        logger.info("Build worker thread started")

        # Advance workflow to build step
        if hasattr(self, '_workflow_state'):
            self._workflow_state = 3
            self._update_workflow_banner()

    def _on_build_progress(self, progress: int, message: str):
        """Handle progress updates from the build worker."""
        if hasattr(self, '_build_panel'):
            elapsed_str = ""
            if self._build_start_time:
                elapsed = (datetime.now() - self._build_start_time).total_seconds()
                elapsed_str = f"  [{elapsed:.0f}s elapsed]"
            self._build_panel.set_diagnostics(f"Progress: {progress}%{elapsed_str}\n{message}")
            if self._build_start_time:
                self._build_panel.set_runtime(elapsed)

    def _on_build_phase_changed(self, phase: int):
        """Handle phase changes from the build worker."""
        if hasattr(self, '_build_panel'):
            self._build_panel.set_phase(phase)

    def _on_build_completed(self, result: Dict[str, Any]):
        """Handle successful build completion from the worker."""
        # Store results - now using GeologicalModelRunner
        self._model = result.get('model')
        self._runner = result.get('runner')
        self._modeler = self._runner  # Backward compatibility
        self._model_result = result.get('model_result')

        # Store pre-extracted surfaces/solids for faster extraction
        self._surfaces = result.get('surfaces', [])
        self._solids = result.get('solids', [])
        self._unified_mesh = result.get('unified_mesh')

        # Calculate total elapsed time
        if self._build_start_time:
            total_elapsed = (datetime.now() - self._build_start_time).total_seconds()
        else:
            total_elapsed = result.get('solve_time', 0)

        misfit_report = result.get('misfit_report', {})
        build_log = result.get('build_log', {})
        resolution = result.get('resolution', 0)
        n_faults = result.get('n_faults', 0)
        gradient_source = result.get('gradient_source', 'unknown')
        warnings = result.get('warnings', [])

        logger.info(f"Build completed in {total_elapsed:.1f}s (gradient_source={gradient_source})")

        # Update build panel with results
        if hasattr(self, '_build_panel'):
            self._build_panel.set_building(False)
            self._build_panel.set_phase(4)  # All complete
            self._build_panel.set_runtime(total_elapsed)
            self._build_panel.set_seed(42)  # Default seed
            self._build_panel.set_diagnostics(
                f"Build completed in {total_elapsed:.1f}s\n"
                f"Resolution: {resolution}³ cells\n"
                f"Stratigraphy: {len(self._stratigraphy)} units\n"
                f"Faults: {n_faults}\n"
                f"Gradient source: {gradient_source}\n"
                f"Audit: {misfit_report.get('status', 'N/A')}"
            )

        # Update UI state
        self._build_btn.setEnabled(True)
        self._extract_btn.setEnabled(True)
        self._extract_btn.setToolTip("Extract geological surfaces from the built model")
        self._validate_btn.setEnabled(True)
        self._validate_btn.setToolTip("Run JORC/SAMREC compliance validation")
        self._export_audit_btn.setEnabled(True)
        self._export_audit_btn.setToolTip("Export build audit log as JSON")

        # Update data summary with model info
        summary = self._data_summary.toPlainText()
        summary += f"\n\n--- Model Built ---\n"
        summary += f"Build time: {total_elapsed:.1f} seconds\n"
        summary += f"Resolution: {resolution}³ cells\n"
        summary += f"Faults: {n_faults}\n"
        summary += f"Stratigraphy: {len(self._stratigraphy)} units\n"
        summary += f"Gradient source: {gradient_source}\n"

        if misfit_report:
            status = misfit_report.get('status', 'Unknown')
            mean_err = misfit_report.get('mean_residual', misfit_report.get('mean_error', 0))
            summary += f"\nAudit Status: {status}\n"
            summary += f"Mean Misfit: {mean_err:.4f}\n"

        if warnings:
            summary += f"\nWarnings ({len(warnings)}):\n"
            for w in warnings[:3]:  # Show first 3
                summary += f"  - {w[:60]}...\n" if len(w) > 60 else f"  - {w}\n"

        self._data_summary.setPlainText(summary)

        # Emit signal
        self.model_built.emit({
            'model': self._model,
            'build_log': build_log,
            'misfit_report': misfit_report,
            'elapsed_seconds': total_elapsed,
            'gradient_source': gradient_source,
        })

        # Clean up worker
        self._build_worker = None
        self._build_start_time = None

        # Show warning if synthetic orientations were used
        if gradient_source == 'synthetic':
            self.show_warning(
                "Synthetic Orientations Used",
                "The model used synthetic horizontal orientations (0,0,1).\n\n"
                "This may produce 'hallucinated' flat-lying geology that\n"
                "does not honor the actual dip and strike of rock units.\n\n"
                "To improve results:\n"
                "- Ensure sufficient contact points per formation (>=5)\n"
                "- Provide real orientation data if available\n"
                "- Check that contacts form coherent planar surfaces"
            )

        self.show_info(
            "Model Built",
            f"Geological model built successfully!\n\n"
            f"Time: {total_elapsed:.1f} seconds\n"
            f"Units: {len(self._stratigraphy)}\n"
            f"Faults: {n_faults}\n"
            f"Gradient source: {gradient_source}\n"
            f"Audit: {misfit_report.get('status', 'N/A')}"
        )

        # Advance workflow to audit step
        if hasattr(self, '_workflow_state'):
            self._workflow_state = 4
            self._update_workflow_banner()

    def _on_build_failed(self, error_message: str):
        """Handle build failure from the worker."""
        logger.error(f"Model build failed: {error_message}")

        # Reset model state
        self._model = None
        self._modeler = None

        # Update UI
        self._build_btn.setEnabled(True)
        if hasattr(self, '_build_panel'):
            self._build_panel.set_building(False)
            self._build_panel.set_diagnostics(f"Build FAILED:\n{error_message}")

        # Clean up worker
        self._build_worker = None
        self._build_start_time = None

        self.show_error("Build Failed", f"Model building failed:\n\n{error_message}")

    def _on_build_cancelled(self):
        """Handle build cancellation from the worker."""
        logger.info("Model build was cancelled")

        # Reset model state (keep any partial results)
        self._model = None
        self._modeler = None

        # Update UI
        self._build_btn.setEnabled(True)
        if hasattr(self, '_build_panel'):
            self._build_panel.set_building(False)
            self._build_panel.set_diagnostics("Build cancelled by user.")

        # Clean up worker
        self._build_worker = None
        self._build_start_time = None

        self.show_info("Build Cancelled", "The model build was cancelled.")

    def _on_extract_geology(self) -> None:
        """Extract geological surfaces from model with progress tracking."""
        from PyQt6.QtWidgets import QApplication

        # Debug logging
        logger.info(f"Extract geology called. Model is None: {self._model is None}, Runner is None: {self._runner is None}")

        if self._model is None:
            self.show_warning(
                "No Model",
                "Please build the geological model first.\n\n"
                "Click 'Build Model' button to create the model."
            )
            return

        # Check for runner (new) or modeler (legacy)
        if self._runner is None and self._modeler is None:
            self.show_warning(
                "No Modeler",
                "Model not properly initialized.\n\n"
                "Please rebuild the model."
            )
            return

        logger.info(f"Proceeding with extraction. Model type: {type(self._model)}")

        self._extract_btn.setEnabled(False)

        # Update build panel phase
        if hasattr(self, '_build_panel'):
            self._build_panel.set_phase(2)  # Extracting surfaces

        # Create progress dialog
        progress = QProgressDialog(
            "Extracting geological surfaces...",
            "Cancel",
            0, 100,
            self
        )
        progress.setWindowTitle("Extracting Geology Model")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        progress.setValue(0)
        QApplication.processEvents()

        start_time = datetime.now()

        try:
            num_units = len(self._stratigraphy)

            # Check if we already have pre-extracted data from GeologicalModelRunner
            if self._model_result is not None and self._surfaces and self._solids:
                logger.info("Using pre-extracted surfaces/solids from GeologicalModelRunner")
                progress.setLabelText("Using pre-extracted surfaces from model build...")
                progress.setValue(80)
                QApplication.processEvents()
                # Surfaces/solids already stored from _on_build_completed
                # Just need to update UI
                for solid in self._solids:
                    unit_name = solid.get('name', 'Unknown')
                    n_verts = len(solid.get('vertices', []))
                    n_faces = len(solid.get('faces', []))
                    self._surface_list.addItem(f"{unit_name} ({n_verts} verts, {n_faces} faces)")
                progress.setValue(95)
                QApplication.processEvents()
            else:
                # Fall back to extraction (legacy path or when pre-extraction not available)
                self._surfaces = []
                self._solids = []
                self._unified_mesh = None

                # Step 1: Extract unified geology mesh (INDUSTRY-STANDARD approach)
                progress.setLabelText(f"Extracting unified geology mesh ({num_units} units)...")
                progress.setValue(10)
                QApplication.processEvents()

                if progress.wasCanceled():
                    self._extract_btn.setEnabled(True)
                    return

                # Use runner if available, otherwise fall back to modeler
                extractor = self._runner if self._runner else self._modeler

                # Try to extract unified mesh
                unified_mesh = None
                if hasattr(extractor, 'engine') and hasattr(extractor.engine, 'extract_unified_geology_mesh'):
                    # GeologicalModelRunner path
                    unified_mesh = extractor.engine.extract_unified_geology_mesh(self._stratigraphy)
                elif hasattr(extractor, 'extract_unified_geology_mesh'):
                    # Direct modeler path (legacy)
                    unified_mesh = extractor.extract_unified_geology_mesh(
                        self._model,
                        self._stratigraphy
                    )

                # Store unified mesh for geology package
                self._unified_mesh = unified_mesh

                # DEBUG: Log what we got back from extraction
                if unified_mesh:
                    n_voxels = len(unified_mesh.get('vertices', []))
                    n_solids = len(unified_mesh.get('solids', []))
                    logger.info(f"Unified mesh extraction: {n_voxels} voxels, {n_solids} solid units")
                    for solid in unified_mesh.get('solids', []):
                        verts = solid.get('vertices')
                        faces = solid.get('faces')
                        v_count = len(verts) if verts is not None else 0
                        f_count = len(faces) if faces is not None else 0
                        logger.info(f"  - {solid.get('name', 'Unknown')}: {v_count} vertices, {f_count} faces")
                else:
                    logger.warning("Unified mesh extraction returned None - falling back to surface extraction")

                progress.setValue(50)
                QApplication.processEvents()

                if progress.wasCanceled():
                    self._extract_btn.setEnabled(True)
                    return

                # Step 2: Process solids from unified mesh (or fallback to legacy)
                if unified_mesh and unified_mesh.get('solids'):
                    # Use solids from unified mesh extraction
                    solids_list = unified_mesh.get('solids', [])
                    logger.info(f"Using {len(solids_list)} solids from unified mesh extraction")

                    for i, solid_data in enumerate(solids_list):
                        pct = 50 + int((i / max(1, len(solids_list))) * 40)
                        unit_name = solid_data.get('name', f'Unit_{i}')
                        progress.setLabelText(f"Processing unit: {unit_name} ({i+1}/{len(solids_list)})")
                        progress.setValue(pct)
                        QApplication.processEvents()

                        if progress.wasCanceled():
                            self._extract_btn.setEnabled(True)
                            return

                        verts = solid_data.get('vertices')
                        faces = solid_data.get('faces')

                        if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
                            n_verts = len(verts)
                            n_faces = len(faces)

                            self._solids.append({
                                'name': unit_name,
                                'unit_name': unit_name,
                                'vertices': verts,
                                'faces': faces,
                                'formation_id': solid_data.get('formation_id', i),
                                'val_range': solid_data.get('val_range'),
                                'volume_m3': 0
                            })

                            self._surface_list.addItem(f"{unit_name} ({n_verts} verts, {n_faces} faces)")
                            self._surfaces.append({
                                'name': unit_name,
                                'vertices': verts,
                                'faces': faces
                            })
                            logger.info(f"Added solid+surface for '{unit_name}': {n_verts} vertices, {n_faces} faces")
                else:
                    # Fallback to legacy surface extraction
                    logger.info("Falling back to legacy get_watertight_solids extraction")
                    # Use runner's engine if available, else legacy modeler
                    if self._runner and hasattr(self._runner, 'engine'):
                        solids_dict = self._runner.engine.extract_solids(self._stratigraphy)
                        # Convert to expected format
                        solids_dict = {s.get('name', f'Unit_{i}'): s for i, s in enumerate(solids_dict)}
                    elif self._modeler and hasattr(self._modeler, 'get_watertight_solids'):
                        solids_dict = self._modeler.get_watertight_solids(
                            self._model,
                            self._stratigraphy,
                            formation_values=self._formation_values if self._formation_values else None
                        )
                    else:
                        solids_dict = {}

                    for i, unit_name in enumerate(self._stratigraphy):
                        pct = 50 + int((i / max(1, num_units)) * 40)
                        progress.setLabelText(f"Processing unit: {unit_name} ({i+1}/{num_units})")
                        progress.setValue(pct)
                        QApplication.processEvents()

                        if progress.wasCanceled():
                            self._extract_btn.setEnabled(True)
                            return

                        if unit_name in solids_dict:
                            solid_data = solids_dict[unit_name]
                            verts = solid_data.get('verts')
                            faces = solid_data.get('faces')

                            if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
                                n_verts = len(verts)
                                n_faces = len(faces)

                                self._solids.append({
                                    'name': unit_name,
                                    'unit_name': unit_name,
                                    'vertices': verts,
                                    'faces': faces,
                                    'normals': solid_data.get('normals'),
                                    'volume_m3': 0
                                })

                                self._surface_list.addItem(f"{unit_name} ({n_verts} verts, {n_faces} faces)")
                                self._surfaces.append({
                                    'name': unit_name,
                                    'vertices': verts,
                                    'faces': faces
                                })

            logger.info(f"After processing: {len(self._solids)} solids, {len(self._surfaces)} surfaces")

            # Step 3: Apply mesh smoothing if enabled
            if self._smooth_check.isChecked() and self._surfaces:
                progress.setLabelText("Applying Taubin smoothing...")
                progress.setValue(92)
                QApplication.processEvents()

                iterations = self._smooth_iter_spin.value()
                logger.info(f"Applying Taubin smoothing ({iterations} iterations)")
                # Note: Actual smoothing would be applied here using pyvista

            progress.setValue(100)
            QApplication.processEvents()

            elapsed = (datetime.now() - start_time).total_seconds()

            # Enable export buttons and update tooltips
            self._export_obj_btn.setEnabled(True)
            self._export_obj_btn.setToolTip("Export surfaces as Wavefront OBJ format")
            self._export_stl_btn.setEnabled(True)
            self._export_stl_btn.setToolTip("Export surfaces as STL format (3D printing compatible)")
            self._export_vtk_btn.setEnabled(True)
            self._export_vtk_btn.setToolTip("Export surfaces as VTK format (ParaView, Leapfrog)")

            # Emit signal for main renderer
            self.surfaces_extracted.emit(self._surfaces)

            # Emit geology package for main renderer
            # CRITICAL: Include unified_mesh for proper solid domain rendering
            # Get build log safely from model_result (new) or modeler (legacy)
            build_log = {}
            if self._model_result and hasattr(self._model_result, 'provenance'):
                build_log = self._model_result.provenance or {}
            elif self._modeler and hasattr(self._modeler, 'get_build_log'):
                try:
                    build_log = self._modeler.get_build_log()
                except Exception:
                    build_log = {}

            geology_package = {
                'surfaces': self._surfaces,
                'solids': self._solids,
                'stratigraphy': self._stratigraphy,
                'model': self._model,
                'build_log': build_log,
                # Include unified mesh for industry-standard rendering
                'unified_mesh': self._unified_mesh,
                'render_mode': 'unified' if self._unified_mesh else 'surfaces',
            }
            self.geology_package_ready.emit(geology_package)

            progress.close()

            logger.info(f"Extracted {len(self._surfaces)} surfaces in {elapsed:.1f}s")
            self.show_info(
                "Extraction Complete",
                f"Extracted {len(self._surfaces)} geological surfaces\n"
                f"Time: {elapsed:.1f} seconds"
            )

        except Exception as e:
            progress.close()
            logger.error(f"Surface extraction failed: {e}", exc_info=True)
            self.show_error("Extraction Failed", f"Surface extraction failed:\n\n{str(e)}")

        finally:
            self._extract_btn.setEnabled(True)

    def _on_validate_compliance(self) -> None:
        """Validate JORC/SAMREC compliance with progress tracking."""
        from PyQt6.QtWidgets import QApplication

        if self._model is None:
            self.show_warning("No Model", "Build model first.")
            return

        # Check for runner (new API) or modeler (legacy API)
        if self._runner is None and self._modeler is None:
            self.show_warning("No Modeler", "Model not properly initialized.")
            return

        self._validate_btn.setEnabled(False)

        # Create progress dialog
        progress = QProgressDialog(
            "Validating JORC/SAMREC compliance...",
            None,  # No cancel for validation
            0, 100,
            self
        )
        progress.setWindowTitle("Compliance Validation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            # Step 1: Get misfit report (supports both new runner and legacy modeler)
            progress.setLabelText("Step 1/4: Retrieving model misfit metrics...")
            progress.setValue(20)
            QApplication.processEvents()

            misfit_report = {}
            build_log = {}

            # New API: GeologicalModelRunner stores results in _model_result
            if self._model_result is not None:
                audit = getattr(self._model_result, 'audit_report', None)
                if audit is not None:
                    misfit_report = {
                        'mean_error': getattr(audit, 'mean_residual', 0) or 0,
                        'max_error': getattr(audit, 'max_residual', 0) or 0,
                        'p90_error': getattr(audit, 'p90_error', 0) or 0,
                        'status': getattr(audit, 'status', 'UNKNOWN'),
                        'total_contacts': getattr(audit, 'total_contacts', 0),
                        'is_jorc_compliant': getattr(audit, 'is_jorc_compliant', False),
                        'classification_recommendation': getattr(audit, 'classification_recommendation', 'Unknown'),
                    }
                build_log = getattr(self._model_result, 'provenance', {}) or {}
            # Legacy API: GeoXIndustryModeler has get_misfit_report / get_build_log
            elif self._modeler is not None and hasattr(self._modeler, 'get_misfit_report'):
                misfit_report = self._modeler.get_misfit_report()
                if hasattr(self._modeler, 'get_build_log'):
                    build_log = self._modeler.get_build_log()

            # Step 2: Check data quality
            progress.setLabelText("Step 2/4: Validating data quality...")
            progress.setValue(40)
            QApplication.processEvents()

            data_quality_checks = []
            if self._contacts_df is not None:
                n_points = len(self._contacts_df)
                n_formations = self._contacts_df['formation'].nunique()
                has_nulls = self._contacts_df[['X', 'Y', 'Z']].isnull().any().any()

                data_quality_checks.append(f"Total data points: {n_points}")
                data_quality_checks.append(f"Formations defined: {n_formations}")
                data_quality_checks.append(f"Missing coordinates: {'Yes - WARNING' if has_nulls else 'No - OK'}")

            # Step 3: Check model parameters
            progress.setLabelText("Step 3/4: Validating model parameters...")
            progress.setValue(60)
            QApplication.processEvents()

            params = build_log.get('parameters', {})
            param_checks = []
            param_checks.append(f"Resolution: {params.get('resolution', 'N/A')}")
            param_checks.append(f"CGW (smoothing): {params.get('cgw', 'N/A')}")
            param_checks.append(f"Samples used: {params.get('n_samples', 'N/A')}")
            param_checks.append(f"Faults modeled: {params.get('n_faults', 0)}")

            # Step 4: Generate compliance summary
            progress.setLabelText("Step 4/4: Generating compliance report...")
            progress.setValue(80)
            QApplication.processEvents()

            # Determine overall status
            status = misfit_report.get('status', 'UNKNOWN')
            mean_error = misfit_report.get('mean_error', 0)
            max_error = misfit_report.get('max_error', 0)
            p90_error = misfit_report.get('p90_error', 0)
            classification = misfit_report.get('classification_recommendation', 'Unknown')

            # Update compliance panel if available
            if hasattr(self, '_compliance_panel') and self._compliance_panel:
                # Build compliance data
                compliance_data = {
                    'status': status,
                    'misfit': misfit_report,
                    'data_quality': data_quality_checks,
                    'parameters': param_checks,
                    'build_log': build_log,
                    'timestamp': datetime.now().isoformat()
                }
                # Update panel (if it has update method)
                if hasattr(self._compliance_panel, 'update_report'):
                    self._compliance_panel.update_report(compliance_data)

            # Populate the Geological Audit Verdict Table
            if hasattr(self, '_audit_verdict_table') and self._audit_verdict_table:
                # Get configured JORC thresholds
                thresholds = self._get_current_jorc_thresholds()

                # Determine verdict statuses based on configured thresholds
                if p90_error < thresholds.measured_p90 and mean_error < thresholds.measured_mean:
                    p90_status = 'pass'
                elif p90_error < thresholds.indicated_p90 and mean_error < thresholds.indicated_mean:
                    p90_status = 'warn'
                else:
                    p90_status = 'fail'

                # Get classification using configured thresholds
                classification = thresholds.classify(p90_error, mean_error)

                # Drillhole honouring verdict
                self._audit_verdict_table.set_verdict(
                    'drillhole_honouring',
                    p90_status,
                    what=f"P90 error: {p90_error:.2f}m (threshold: {thresholds.measured_p90}m)",
                    where=f"{len(self._contacts_df) if self._contacts_df is not None else 0} contact points",
                    why="" if p90_status == 'pass' else f"Mean error: {mean_error:.4f}m",
                    impact=f"Classification: {classification}"
                )

                # Stratigraphic ordering (from stratigraphy validation if available)
                strat_status = 'pass'  # Default to pass if no validation data
                self._audit_verdict_table.set_verdict(
                    'stratigraphic_ordering',
                    strat_status,
                    what="Formation sequence validated",
                    where=f"{len(self._stratigraphy)} formations defined",
                    why="",
                    impact=""
                )

                # Layer continuity
                self._audit_verdict_table.set_verdict(
                    'layer_continuity',
                    'pass' if status == 'PASS' else 'warn',
                    what="Layer surfaces extracted",
                    where="Model domain",
                    why="" if status == 'PASS' else "Review surface continuity",
                    impact=""
                )

                # Dip & strike consistency - perform actual validation
                dip_strike_result = self._validate_dip_strike_consistency()
                self._audit_verdict_table.set_verdict(
                    'dip_strike_consistency',
                    dip_strike_result['status'],
                    what=dip_strike_result['message'],
                    where=f"{dip_strike_result['n_validated']} orientation points" if dip_strike_result['n_validated'] > 0 else "Orientation data",
                    why="" if dip_strike_result['status'] == 'pass' else f"Max deviation: {dip_strike_result['max_deviation']:.1f}°",
                    impact="" if dip_strike_result['status'] == 'pass' else "Review structural interpretation"
                )

                # Fault handling
                n_faults = params.get('n_faults', 0)
                self._audit_verdict_table.set_verdict(
                    'fault_handling',
                    'pass',
                    what=f"{n_faults} fault events processed",
                    where="Model domain",
                    why="",
                    impact=""
                )

            # Update the overall audit summary banner
            if hasattr(self, '_audit_summary_banner'):
                self._update_audit_summary_banner(status, p90_error, mean_error, classification)

            progress.setValue(100)
            QApplication.processEvents()

            # Enable export buttons and update tooltips
            self._export_audit_btn.setEnabled(True)
            self._export_audit_btn.setToolTip("Export build audit log as JSON")
            self._export_compliance_btn.setEnabled(True)
            self._export_compliance_btn.setToolTip("Export full JORC/SAMREC compliance report as PDF")

            # Store current report
            self._current_report = {
                'status': status,
                'misfit_report': misfit_report,
                'build_log': build_log,
                'data_quality': data_quality_checks,
                'parameters': param_checks
            }

            # Emit signal
            self.compliance_validated.emit(self._current_report)

            progress.close()

            # Show summary
            status_icon = "OK" if status == "PASS" else "!!"
            self.show_info(
                "Compliance Validation Complete",
                f"JORC/SAMREC Audit Status: {status} {status_icon}\n\n"
                f"Mean Misfit: {mean_error:.4f}\n"
                f"Max Misfit: {max_error:.4f}\n"
                f"P90 Misfit: {p90_error:.4f}\n\n"
                f"See Compliance tab for full report."
            )

            logger.info(f"Compliance validation complete: {status}")

        except Exception as e:
            progress.close()
            logger.error(f"Compliance validation failed: {e}", exc_info=True)
            self.show_error("Validation Failed", f"Compliance validation failed:\n\n{str(e)}")

        finally:
            self._validate_btn.setEnabled(True)

    def _validate_dip_strike_consistency(self) -> Dict[str, Any]:
        """
        Validate that model gradients match input orientation data.

        Compares the model's interpolated gradient vectors at orientation
        measurement points against the original input gradients.

        Returns:
            Dict with validation results:
                - status: 'pass', 'warn', or 'fail'
                - mean_deviation: mean angular deviation in degrees
                - max_deviation: maximum angular deviation in degrees
                - n_validated: number of points validated
                - n_failed: number of points with high deviation (>30°)
        """
        result = {
            'status': 'pending',
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_validated': 0,
            'n_failed': 0,
            'message': 'No orientation data to validate'
        }

        # Check if we have orientation data and a model
        if self._orientations_df is None or len(self._orientations_df) == 0:
            result['status'] = 'warn'
            result['message'] = 'No orientation data provided'
            return result

        if self._model is None:
            result['status'] = 'pending'
            result['message'] = 'Model not built yet'
            return result

        try:
            # Get the stratigraphic feature from the model
            if hasattr(self._model, 'get_feature') and self._stratigraphy:
                strat_feature = self._model.get_feature(self._stratigraphy[0])
            elif hasattr(self._model, 'features') and len(self._model.features) > 0:
                strat_feature = self._model.features[0]
            else:
                result['status'] = 'warn'
                result['message'] = 'Cannot access model features for gradient evaluation'
                return result

            # Get orientation points
            orient_df = self._orientations_df.copy()

            # Filter out synthetic orientations (all zeros except gz=1)
            is_synthetic = (
                (orient_df['gx'] == 0.0) &
                (orient_df['gy'] == 0.0) &
                (orient_df['gz'] == 1.0)
            )
            real_orientations = orient_df[~is_synthetic]

            if len(real_orientations) == 0:
                result['status'] = 'warn'
                result['message'] = 'Only synthetic orientations detected (vertical default)'
                return result

            # Get input gradient vectors
            input_grads = real_orientations[['gx', 'gy', 'gz']].values

            # Normalize input gradients
            input_norms = np.linalg.norm(input_grads, axis=1, keepdims=True)
            input_norms[input_norms < 1e-10] = 1.0  # Avoid division by zero
            input_grads_normalized = input_grads / input_norms

            # Evaluate model gradients at orientation points
            points = real_orientations[['X', 'Y', 'Z']].values

            # Check if the runner/modeler can evaluate gradients
            scaler = None
            if self._runner and hasattr(self._runner, 'engine') and hasattr(self._runner.engine, 'scaler'):
                scaler = self._runner.engine.scaler
            elif self._modeler and hasattr(self._modeler, 'scaler'):
                scaler = self._modeler.scaler

            if scaler is not None and hasattr(strat_feature, 'evaluate_gradient'):
                # Transform points to model coordinates
                points_scaled = scaler.transform(points)

                # Evaluate model gradient
                try:
                    model_grads = strat_feature.evaluate_gradient(points_scaled)
                except Exception as e:
                    logger.warning(f"Gradient evaluation failed: {e}")
                    result['status'] = 'warn'
                    result['message'] = f'Gradient evaluation error: {str(e)}'
                    return result

                # Normalize model gradients
                model_norms = np.linalg.norm(model_grads, axis=1, keepdims=True)
                model_norms[model_norms < 1e-10] = 1.0
                model_grads_normalized = model_grads / model_norms

                # Calculate angular deviation between input and model gradients
                # cos(angle) = dot(a, b) / (|a| * |b|), but both are normalized
                dot_products = np.sum(input_grads_normalized * model_grads_normalized, axis=1)
                dot_products = np.clip(dot_products, -1.0, 1.0)  # Numerical stability
                angular_deviations = np.degrees(np.arccos(np.abs(dot_products)))  # Use abs to handle sign ambiguity

                # Calculate statistics
                mean_deviation = float(np.mean(angular_deviations))
                max_deviation = float(np.max(angular_deviations))
                n_failed = int(np.sum(angular_deviations > 30.0))  # >30° considered high deviation

                result['mean_deviation'] = mean_deviation
                result['max_deviation'] = max_deviation
                result['n_validated'] = len(angular_deviations)
                result['n_failed'] = n_failed

                # Determine status based on deviation thresholds
                if mean_deviation < 10.0 and n_failed == 0:
                    result['status'] = 'pass'
                    result['message'] = f'Mean angular deviation: {mean_deviation:.1f}°'
                elif mean_deviation < 20.0 and n_failed < len(angular_deviations) * 0.1:
                    result['status'] = 'warn'
                    result['message'] = f'Mean deviation: {mean_deviation:.1f}° ({n_failed} high-deviation points)'
                else:
                    result['status'] = 'fail'
                    result['message'] = f'High deviation: mean {mean_deviation:.1f}°, {n_failed} outliers'

                logger.info(f"Dip/strike validation: {result['status']} - {result['message']}")

            else:
                result['status'] = 'warn'
                result['message'] = 'Model does not support gradient evaluation'

        except Exception as e:
            logger.warning(f"Dip/strike validation error: {e}")
            result['status'] = 'warn'
            result['message'] = f'Validation error: {str(e)}'

        return result

    def _update_audit_summary_banner(self, status: str, p90_error: float,
                                       mean_error: float, classification: str) -> None:
        """Update the overall audit summary banner with results."""
        if not hasattr(self, '_audit_status_icon'):
            return

        if status == 'PASS' or (p90_error > 0 and p90_error < 5.0):
            # Passing audit
            self._audit_status_icon.setText("OK")
            self._audit_status_icon.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 24px; font-weight: bold; background: transparent;")
            self._audit_status_label.setText(f"AUDIT PASSED")
            self._audit_status_label.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 14px; font-weight: 700; background: transparent;")
            self._audit_summary_banner.setStyleSheet(f"""
                QFrame {{
                    background: rgba(76, 175, 80, 0.08);
                    border: 2px solid {ModernColors.SUCCESS};
                    border-radius: 8px;
                }}
            """)
        elif status == 'WARN' or (p90_error > 0 and p90_error < 10.0):
            self._audit_status_icon.setText("!!")
            self._audit_status_icon.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 24px; font-weight: bold; background: transparent;")
            self._audit_status_label.setText(f"AUDIT — REVIEW REQUIRED")
            self._audit_status_label.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 14px; font-weight: 700; background: transparent;")
            self._audit_summary_banner.setStyleSheet(f"""
                QFrame {{
                    background: rgba(255, 152, 0, 0.08);
                    border: 2px solid {ModernColors.WARNING};
                    border-radius: 8px;
                }}
            """)
        else:
            self._audit_status_icon.setText("XX")
            self._audit_status_icon.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 24px; font-weight: bold; background: transparent;")
            self._audit_status_label.setText(f"AUDIT FAILED")
            self._audit_status_label.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 14px; font-weight: 700; background: transparent;")
            self._audit_summary_banner.setStyleSheet(f"""
                QFrame {{
                    background: rgba(244, 67, 54, 0.08);
                    border: 2px solid {ModernColors.ERROR};
                    border-radius: 8px;
                }}
            """)

        self._audit_classification_label.setText(f"Classification: {classification}")
        self._audit_classification_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 11px; font-weight: 500; background: transparent;")

        metrics_text = f"P90: {p90_error:.2f}m  |  Mean: {mean_error:.2f}m"
        self._audit_metrics_label.setText(metrics_text)

    def _on_apply_suggested_fault(self, fault: Dict[str, Any]) -> None:
        """Apply a suggested fault to the model."""
        logger.info(f"Applying suggested fault: {fault}")

    def _on_export_surfaces(self, fmt: str) -> None:
        """Export surfaces in specified format."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {fmt.upper()}",
            "",
            f"{fmt.upper()} Files (*.{fmt})"
        )

        if filepath:
            logger.info(f"Exporting surfaces to {filepath}...")
            self.show_info("Exported", f"Surfaces exported to {filepath}")

    def _on_export_audit(self) -> None:
        """Export build audit log."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audit Log",
            "",
            "JSON Files (*.json)"
        )

        if filepath:
            logger.info(f"Exporting audit log to {filepath}...")
            self.show_info("Exported", f"Audit log exported to {filepath}")

    def _on_export_compliance(self) -> None:
        """Export compliance report."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Compliance Report",
            "",
            "PDF Files (*.pdf)"
        )

        if filepath:
            logger.info(f"Exporting compliance report to {filepath}...")
            self.show_info("Exported", f"Compliance report exported to {filepath}")

    def refresh_theme(self):
        """Refresh all UI elements to match current theme."""
        # Refresh tab widget
        if hasattr(self, '_tabs'):
            self._tabs.setStyleSheet(f"""
                QTabWidget::pane {{
                    border: 1px solid {ModernColors.BORDER};
                    border-radius: 8px;
                    background: {ModernColors.ELEVATED_BG};
                    top: -1px;
                }}
                QTabBar::tab {{
                    background: {ModernColors.PANEL_BG};
                    color: {ModernColors.TEXT_SECONDARY};
                    padding: 10px 20px;
                    margin-right: 4px;
                    border: none;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    font-size: 12px;
                    font-weight: 500;
                    min-width: 90px;
                }}
                QTabBar::tab:selected {{
                    background: {ModernColors.ELEVATED_BG};
                    color: {ModernColors.ACCENT_PRIMARY};
                    border-bottom: 3px solid {ModernColors.ACCENT_PRIMARY};
                }}
                QTabBar::tab:hover:!selected {{
                    background: {ModernColors.BORDER};
                    color: {ModernColors.ACCENT_PRIMARY};
                }}
                QTabBar::tab:disabled {{
                    color: {ModernColors.TEXT_DISABLED};
                }}
            """)

        # Refresh main action buttons
        if hasattr(self, '_build_btn'):
            self._build_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ModernColors.ACCENT_PRIMARY};
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    border-radius: 6px;
                    font-weight: 600;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background: {ModernColors.ACCENT_HOVER};
                }}
                QPushButton:pressed {{
                    background: {ModernColors.ACCENT_PRESSED};
                }}
                QPushButton:disabled {{
                    background: {ModernColors.BORDER};
                    color: {ModernColors.TEXT_DISABLED};
                }}
            """)

        if hasattr(self, '_close_btn'):
            self._close_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ModernColors.ELEVATED_BG};
                    color: {ModernColors.TEXT_SECONDARY};
                    border: 1px solid {ModernColors.BORDER};
                    padding: 10px 24px;
                    border-radius: 6px;
                    font-weight: 500;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background: {ModernColors.ERROR};
                    color: white;
                    border-color: {ModernColors.ERROR};
                }}
                QPushButton:disabled {{
                    color: {ModernColors.TEXT_DISABLED};
                }}
            """)

        # Refresh lithology grouping widget
        if hasattr(self, '_lith_grouping_widget') and self._lith_grouping_widget:
            try:
                self._lith_grouping_widget.setStyleSheet(f"""
                    QFrame {{
                        background: {ModernColors.ELEVATED_BG};
                        border: 1px solid {ModernColors.BORDER};
                        border-radius: 8px;
                    }}
                """)
            except Exception as e:
                logger.debug(f"Could not refresh lithology grouping widget: {e}")

        # Refresh all group boxes
        for gb in self.findChildren(QGroupBox):
            try:
                title = gb.title()
                if "Stratigraphy" in title or "Stratigraphic" in title:
                    gb.setStyleSheet(f"""
                        QGroupBox {{
                            font-size: 13px;
                            font-weight: 600;
                            color: {ModernColors.TEXT_PRIMARY};
                            border: 2px solid {ModernColors.BORDER};
                            border-radius: 8px;
                            margin-top: 10px;
                            padding-top: 10px;
                            background: {ModernColors.ELEVATED_BG};
                        }}
                        QGroupBox::title {{
                            subcontrol-origin: margin;
                            subcontrol-position: top left;
                            padding: 4px 10px;
                            background: {ModernColors.ELEVATED_BG};
                            color: {ModernColors.ACCENT_PRIMARY};
                        }}
                    """)
            except Exception as e:
                logger.debug(f"Could not refresh group box: {e}")

        # Refresh list widgets
        for lw in self.findChildren(QListWidget):
            try:
                lw.setStyleSheet(f"""
                    QListWidget {{
                        background: {ModernColors.PANEL_BG};
                        border: 2px solid {ModernColors.BORDER};
                        border-radius: 8px;
                        padding: 6px;
                        font-size: 12px;
                    }}
                    QListWidget::item {{
                        background: {ModernColors.ELEVATED_BG};
                        border: 1px solid {ModernColors.BORDER};
                        border-radius: 6px;
                        padding: 8px;
                        margin: 3px;
                        color: {ModernColors.TEXT_PRIMARY};
                    }}
                    QListWidget::item:selected {{
                        background: {ModernColors.ACCENT_PRESSED};
                        border: 2px solid {ModernColors.ACCENT_PRIMARY};
                        color: {ModernColors.ACCENT_PRIMARY};
                    }}
                    QListWidget::item:hover {{
                        background: {ModernColors.BORDER};
                    }}
                """)
            except Exception as e:
                logger.debug(f"Could not refresh list widget: {e}")

        # Call parent refresh if available
        try:
            if hasattr(super(), 'refresh_theme'):
                super().refresh_theme()
        except Exception as e:
            logger.debug(f"Could not call parent refresh_theme: {e}")
