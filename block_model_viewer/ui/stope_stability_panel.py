"""
Stope Stability Analysis Panel

Panel for Mathews Stability Graph analysis of underground stopes.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QCheckBox, QTabWidget, QWidget
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class StopeStabilityPanel(BaseAnalysisPanel):
    """
    Panel for Mathews Stability Graph analysis.
    
    Allows users to:
    - Define stope geometry
    - Set Mathews parameters
    - Run deterministic and probabilistic analysis
    - View stability graph position and recommendations
    """
    # PanelManager metadata
    PANEL_ID = "StopeStabilityPanel"
    PANEL_NAME = "StopeStability Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "stope_stability"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="stope_stability")
        
        self.current_result = None
        self._setup_ui()
        logger.info("Initialized Stope Stability panel")
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Mathews Stability Graph Analysis</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_input_tab(), "Stope Geometry")
        tabs.addTab(self._create_parameters_tab(), "Rock Mass Parameters")
        tabs.addTab(self._create_results_tab(), "Results")
        layout.addWidget(tabs)
        
        self.tabs = tabs
    
    def _create_input_tab(self) -> QWidget:
        """Create stope geometry input tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Geometry group
        geom_group = QGroupBox("Stope Geometry")
        geom_layout = QFormLayout(geom_group)
        
        self.span_spinbox = QDoubleSpinBox()
        self.span_spinbox.setRange(1.0, 100.0)
        self.span_spinbox.setValue(10.0)
        self.span_spinbox.setSuffix(" m")
        self.span_spinbox.setToolTip("Stope span (width)")
        geom_layout.addRow("Span:", self.span_spinbox)
        
        self.height_spinbox = QDoubleSpinBox()
        self.height_spinbox.setRange(1.0, 100.0)
        self.height_spinbox.setValue(20.0)
        self.height_spinbox.setSuffix(" m")
        self.height_spinbox.setToolTip("Stope height")
        geom_layout.addRow("Height:", self.height_spinbox)
        
        layout.addWidget(geom_group)
        
        # Link to UG stopes
        link_group = QGroupBox("Link to Underground Stopes")
        link_layout = QVBoxLayout(link_group)
        
        self.link_checkbox = QCheckBox("Use existing stope from UG module")
        link_layout.addWidget(self.link_checkbox)
        
        self.stope_combo = QComboBox()
        self.stope_combo.setEnabled(False)
        self.stope_combo.setToolTip("Select existing stope to use geometry")
        link_layout.addWidget(self.stope_combo)
        
        self.link_checkbox.stateChanged.connect(
            lambda state: self.stope_combo.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        layout.addWidget(link_group)
        
        layout.addStretch()
        return widget
    
    def _create_parameters_tab(self) -> QWidget:
        """Create rock mass parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Mathews parameters
        params_group = QGroupBox("Mathews Parameters")
        params_layout = QFormLayout(params_group)
        
        self.q_prime_spinbox = QDoubleSpinBox()
        self.q_prime_spinbox.setRange(0.1, 100.0)
        self.q_prime_spinbox.setValue(10.0)
        self.q_prime_spinbox.setToolTip("Modified Q-value (Q')")
        params_layout.addRow("Q' (Modified Q):", self.q_prime_spinbox)
        
        self.stress_factor_spinbox = QDoubleSpinBox()
        self.stress_factor_spinbox.setRange(0.1, 2.0)
        self.stress_factor_spinbox.setValue(1.0)
        self.stress_factor_spinbox.setToolTip("Stress factor (A)")
        params_layout.addRow("Stress Factor (A):", self.stress_factor_spinbox)
        
        self.joint_orientation_spinbox = QDoubleSpinBox()
        self.joint_orientation_spinbox.setRange(0.1, 2.0)
        self.joint_orientation_spinbox.setValue(1.0)
        self.joint_orientation_spinbox.setToolTip("Joint orientation factor (B)")
        params_layout.addRow("Joint Orientation (B):", self.joint_orientation_spinbox)
        
        self.gravity_spinbox = QDoubleSpinBox()
        self.gravity_spinbox.setRange(0.1, 2.0)
        self.gravity_spinbox.setValue(1.0)
        self.gravity_spinbox.setToolTip("Gravity factor (C)")
        params_layout.addRow("Gravity Factor (C):", self.gravity_spinbox)
        
        layout.addWidget(params_group)
        
        # Analysis type
        analysis_group = QGroupBox("Analysis Type")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.deterministic_btn = QPushButton("Run Deterministic Analysis")
        self.deterministic_btn.clicked.connect(self._run_deterministic)
        analysis_layout.addWidget(self.deterministic_btn)
        
        self.probabilistic_btn = QPushButton("Run Probabilistic Analysis (Monte Carlo)")
        self.probabilistic_btn.clicked.connect(self._run_probabilistic)
        analysis_layout.addWidget(self.probabilistic_btn)
        
        self.n_realizations_spinbox = QSpinBox()
        self.n_realizations_spinbox.setRange(10, 10000)
        self.n_realizations_spinbox.setValue(100)
        self.n_realizations_spinbox.setToolTip("Number of Monte Carlo realizations")
        analysis_layout.addWidget(QLabel("Number of Realizations:"))
        analysis_layout.addWidget(self.n_realizations_spinbox)
        
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Visualization buttons
        viz_layout = QHBoxLayout()
        
        self.visualize_btn = QPushButton("Visualize in 3D")
        self.visualize_btn.clicked.connect(self._visualize_results)
        viz_layout.addWidget(self.visualize_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        viz_layout.addWidget(self.export_btn)
        
        layout.addLayout(viz_layout)
        
        return widget
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect stope stability parameters."""
        return {
            'span': self.span_spinbox.value(),
            'height': self.height_spinbox.value(),
            'q_prime': self.q_prime_spinbox.value(),
            'stress_factor': self.stress_factor_spinbox.value(),
            'joint_orientation_factor': self.joint_orientation_spinbox.value(),
            'gravity_factor': self.gravity_spinbox.value(),
            'analysis_type': 'deterministic'
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if self.span_spinbox.value() <= 0 or self.height_spinbox.value() <= 0:
            self.show_error("Invalid Geometry", "Span and height must be positive.")
            return False
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle analysis results."""
        if payload.get('error'):
            self.show_error("Analysis Error", payload['error'])
            return
        
        self.current_result = payload.get('result')
        
        # Display results
        if self.current_result:
            result_text = self._format_results(self.current_result)
            self.results_text.setPlainText(result_text)
        
        # Update results tab
        self.tabs.setCurrentIndex(2)
        
        self.show_info("Success", "Stope stability analysis completed.")
    
    def _run_deterministic(self):
        """Run deterministic analysis."""
        self.task_name = "stope_stability"
        self.run_analysis()
    
    def _run_probabilistic(self):
        """Run probabilistic analysis."""
        self.task_name = "stope_stability_mc"
        params = self.gather_parameters()
        params['n_realizations'] = self.n_realizations_spinbox.value()
        params['analysis_type'] = 'probabilistic'
        
        if not self.validate_inputs():
            return
        
        if self.controller:
            self.controller.run_task("stope_stability_mc", params, self.handle_results)
    
    def _format_results(self, result: Dict[str, Any]) -> str:
        """Format results for display."""
        lines = []
        lines.append("=== Stope Stability Analysis Results ===\n")
        
        if 'stability_number' in result:
            lines.append(f"Stability Number (N): {result['stability_number']:.2f}")
        if 'factor_of_safety' in result:
            lines.append(f"Factor of Safety: {result['factor_of_safety']:.2f}")
        if 'stability_class' in result:
            lines.append(f"Stability Class: {result['stability_class']}")
        if 'probability_of_instability' in result:
            lines.append(f"Probability of Instability: {result['probability_of_instability']:.1%}")
        if 'recommended_support_class' in result:
            lines.append(f"Recommended Support: {result['recommended_support_class']}")
        if 'notes' in result:
            lines.append(f"\nNotes: {result['notes']}")
        
        # Monte Carlo results
        if 'summary_stats' in result:
            lines.append("\n=== Monte Carlo Summary ===")
            stats = result['summary_stats']
            if 'stability_number' in stats:
                sn_stats = stats['stability_number']
                lines.append(f"Mean N: {sn_stats.get('mean', 0):.2f}")
                lines.append(f"Std Dev: {sn_stats.get('std', 0):.2f}")
                lines.append(f"P10: {sn_stats.get('p10', 0):.2f}")
                lines.append(f"P50: {sn_stats.get('p50', 0):.2f}")
                lines.append(f"P90: {sn_stats.get('p90', 0):.2f}")
        
        return "\n".join(lines)
    
    def _visualize_results(self):
        """Request 3D visualization of results."""
        if not self.current_result:
            self.show_warning("No Results", "Please run analysis first.")
            return
        
        if self.controller and hasattr(self.controller, 'renderer'):
            try:
                # Request stope visualization
                # This would be handled by renderer adding a stope stability layer
                self.show_info("Visualization", "Stope stability visualization requested.")
            except Exception as e:
                logger.warning(f"Failed to visualize: {e}", exc_info=True)
    
    def _export_results(self):
        """Export results to CSV."""
        if not self.current_result:
            self.show_warning("No Results", "Please run analysis first.")
            return
        
        # Export via DataBridge
        try:
            from ..utils.data_bridge import stope_stability_result_to_dataframe
            df = stope_stability_result_to_dataframe(self.current_result)
            
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", "CSV Files (*.csv)"
            )
            
            if file_path:
                df.to_csv(file_path, index=False)
                self.show_info("Success", f"Results exported to {file_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            self.show_error("Export Error", str(e))

