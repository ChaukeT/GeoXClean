"""
ModelBuildWorker — threaded geological model building.

Moved unchanged from the monolithic loopstructural_panel.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from PyQt6.QtCore import pyqtSignal, QThread, QMutex
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


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
