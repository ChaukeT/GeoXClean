"""
Slope Stability Panel - UI for 2D/3D slope stability analysis (STEP 27).
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QMessageBox, QTextEdit, QDoubleSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QLineEdit, QSpinBox, QFormLayout
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class SlopeStabilityPanel(BaseAnalysisPanel):
    """Panel for slope stability analysis."""
    # PanelManager metadata
    PANEL_ID = "SlopeStabilityPanel"
    PANEL_NAME = "SlopeStability Panel"
    PANEL_CATEGORY = PanelCategory.GEOTECH
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "slope_stability"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="slope_stability")
        self.setWindowTitle("Slope Stability Analysis")
        # Sizing is handled by BaseAnalysisPanel._setup_panel_sizing()
        
        self.slope_sector = None
        self.material = None
        self.results_2d = []
        self.results_3d = []
        self._block_model = None  # Use _block_model instead of block_model property
        self.drillhole_data = None
        
        # Subscribe to block model and drillhole data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_data = self.registry.get_drillhole_data()
            if existing_data:
                self._on_drillhole_data_loaded(existing_data)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Slope Stability Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Automatically receive drillhole data when it's loaded from DataRegistry.
        
        Args:
            drillhole_data: Drillhole data dict with 'composites', 'assays', etc.
        """
        logger.info("Slope Stability Panel received drillhole data from DataRegistry")
        self.drillhole_data = drillhole_data
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        title = QLabel("Slope Stability Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        tabs = QTabWidget()
        tabs.addTab(self._create_slope_definition_tab(), "Slope Definition")
        tabs.addTab(self._create_material_tab(), "Material Properties")
        tabs.addTab(self._create_analysis_tab(), "Analysis")
        tabs.addTab(self._create_results_tab(), "Results")
        
        layout.addWidget(tabs)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
    
    def _create_slope_definition_tab(self) -> QWidget:
        """Create slope definition tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Slope Sector Geometry")
        form = QFormLayout()
        
        self.sector_id_edit = QLineEdit("SECTOR_1")
        form.addRow("Sector ID:", self.sector_id_edit)
        
        self.toe_x = QDoubleSpinBox()
        self.toe_x.setRange(-10000, 10000)
        form.addRow("Toe X:", self.toe_x)
        
        self.toe_y = QDoubleSpinBox()
        self.toe_y.setRange(-10000, 10000)
        form.addRow("Toe Y:", self.toe_y)
        
        self.toe_z = QDoubleSpinBox()
        self.toe_z.setRange(-10000, 10000)
        form.addRow("Toe Z:", self.toe_z)
        
        self.crest_x = QDoubleSpinBox()
        self.crest_x.setRange(-10000, 10000)
        form.addRow("Crest X:", self.crest_x)
        
        self.crest_y = QDoubleSpinBox()
        self.crest_y.setRange(-10000, 10000)
        form.addRow("Crest Y:", self.crest_y)
        
        self.crest_z = QDoubleSpinBox()
        self.crest_z.setRange(-10000, 10000)
        form.addRow("Crest Z:", self.crest_z)
        
        self.bench_height = QDoubleSpinBox()
        self.bench_height.setRange(1, 50)
        self.bench_height.setValue(10.0)
        form.addRow("Bench Height (m):", self.bench_height)
        
        self.berm_width = QDoubleSpinBox()
        self.berm_width.setRange(0, 20)
        self.berm_width.setValue(3.0)
        form.addRow("Berm Width (m):", self.berm_width)
        
        self.domain_code_edit = QLineEdit()
        form.addRow("Domain Code:", self.domain_code_edit)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        btn = QPushButton("Create Slope Sector")
        btn.clicked.connect(self._create_slope_sector)
        layout.addWidget(btn)
        
        layout.addStretch()
        return widget
    
    def _create_material_tab(self) -> QWidget:
        """Create material properties tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Material Properties")
        form = QFormLayout()
        
        self.material_name_edit = QLineEdit("Rock_Mass_1")
        form.addRow("Material Name:", self.material_name_edit)
        
        self.unit_weight = QDoubleSpinBox()
        self.unit_weight.setRange(15, 30)
        self.unit_weight.setValue(25.0)
        form.addRow("Unit Weight (kN/m³):", self.unit_weight)
        
        self.friction_angle = QDoubleSpinBox()
        self.friction_angle.setRange(0, 60)
        self.friction_angle.setValue(35.0)
        form.addRow("Friction Angle (°):", self.friction_angle)
        
        self.cohesion = QDoubleSpinBox()
        self.cohesion.setRange(0, 1000)
        self.cohesion.setValue(100.0)
        form.addRow("Cohesion (kPa):", self.cohesion)
        
        self.ru = QDoubleSpinBox()
        self.ru.setRange(0, 1)
        self.ru.setValue(0.0)
        form.addRow("Pore Pressure Ratio (ru):", self.ru)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        btn = QPushButton("Create Material")
        btn.clicked.connect(self._create_material)
        layout.addWidget(btn)
        
        layout.addStretch()
        return widget
    
    def _create_analysis_tab(self) -> QWidget:
        """Create analysis configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Analysis Configuration")
        form = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["2D", "3D"])
        form.addRow("Method:", self.method_combo)
        
        self.lem_method_combo = QComboBox()
        self.lem_method_combo.addItems(["Bishop", "Janbu"])
        form.addRow("LEM Method (2D):", self.lem_method_combo)
        
        self.n_surfaces = QSpinBox()
        self.n_surfaces.setRange(10, 500)
        self.n_surfaces.setValue(50)
        form.addRow("Number of Surfaces:", self.n_surfaces)
        
        self.n_slices = QSpinBox()
        self.n_slices.setRange(5, 100)
        self.n_slices.setValue(20)
        form.addRow("Number of Slices:", self.n_slices)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        btn = QPushButton("Run Analysis")
        btn.clicked.connect(self._run_analysis)
        layout.addWidget(btn)
        
        layout.addStretch()
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Rank", "FOS", "Type", "Status"])
        layout.addWidget(self.results_table)
        
        btn_layout = QHBoxLayout()
        btn = QPushButton("Render Critical Surface")
        btn.clicked.connect(self._render_critical_surface)
        btn_layout.addWidget(btn)
        
        btn2 = QPushButton("Export Results")
        btn2.clicked.connect(self._export_results)
        btn_layout.addWidget(btn2)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget
    
    def _create_slope_sector(self):
        """Create slope sector from UI inputs."""
        try:
            from ..geotech_common.slope_geometry import SlopeSector
            
            sector = SlopeSector(
                id=self.sector_id_edit.text(),
                toe_point=(self.toe_x.value(), self.toe_y.value(), self.toe_z.value()),
                crest_point=(self.crest_x.value(), self.crest_y.value(), self.crest_z.value()),
                height=abs(self.crest_z.value() - self.toe_z.value()),
                dip=0.0,  # Will be computed
                dip_direction=0.0,  # Will be computed
                bench_height=self.bench_height.value(),
                berm_width=self.berm_width.value(),
                overall_slope_angle=0.0,  # Will be computed
                domain_code=self.domain_code_edit.text() if self.domain_code_edit.text() else None
            )
            
            self.slope_sector = sector
            self.status_text.append(f"Created slope sector: {sector.id}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create slope sector: {e}")
    
    def _create_material(self):
        """Create material from UI inputs."""
        try:
            from ..geotech_common.material_properties import GeotechMaterial
            
            material = GeotechMaterial(
                name=self.material_name_edit.text(),
                unit_weight=self.unit_weight.value(),
                friction_angle=self.friction_angle.value(),
                cohesion=self.cohesion.value(),
                water_condition="dry" if self.ru.value() == 0 else "wet"
            )
            
            self.material = material
            self.status_text.append(f"Created material: {material.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create material: {e}")
    
    def _run_analysis(self):
        """Run slope stability analysis."""
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available")
            return
        
        if not self.slope_sector or not self.material:
            QMessageBox.warning(self, "Error", "Please define slope sector and material first")
            return
        
        method = self.method_combo.currentText()
        
        config = {
            "slope_sector": self.slope_sector,
            "material": self.material,
            "search_params": {"n_surfaces": self.n_surfaces.value()},
            "method": self.lem_method_combo.currentText(),
            "n_slices": self.n_slices.value(),
            "ru": self.ru.value(),
            "pore_pressure_mode": "ru_factor" if self.ru.value() > 0 else "none"
        }
        
        if method == "2D":
            self.controller.run_slope_lem_2d(config, self._on_analysis_complete)
        else:
            self.controller.run_slope_lem_3d(config, self._on_analysis_complete)
        
        self.status_text.append(f"Running {method} analysis...")
    
    def _on_analysis_complete(self, result: Dict[str, Any]):
        """Handle analysis completion."""
        results = result.get("results", [])
        critical_fos = result.get("critical_fos")
        
        if results:
            self.results_2d = results if isinstance(results[0], type(results[0])) else []
            self._update_results_table(results)
            self.status_text.append(f"Analysis complete: Critical FOS = {critical_fos:.3f}")
        else:
            QMessageBox.warning(self, "Warning", "No results generated")
    
    def _update_results_table(self, results: List[Any]):
        """Update results table."""
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            fos = result.fos if hasattr(result, 'fos') else 0.0
            
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{fos:.3f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem("2D" if hasattr(result, 'surface') and hasattr(result.surface, 'x_coords') else "3D"))
            
            if fos < 1.0:
                status = "FAILURE"
            elif fos < 1.3:
                status = "MARGINAL"
            else:
                status = "STABLE"
            
            self.results_table.setItem(i, 3, QTableWidgetItem(status))
    
    def _render_critical_surface(self):
        """Render critical failure surface."""
        if not self.controller or not self.controller.r:
            QMessageBox.warning(self, "Error", "Renderer not available")
            return
        
        if not self.results_2d:
            QMessageBox.warning(self, "Error", "No results to render")
            return
        
        renderer = self.controller.r
        critical_result = self.results_2d[0]
        
        if hasattr(critical_result, 'surface') and hasattr(critical_result.surface, 'x_coords'):
            # 2D surface
            renderer.render_failure_surface_2d(critical_result)
        else:
            # 3D surface
            renderer.render_failure_surface_3d(critical_result)
        
        self.status_text.append("Rendered critical failure surface")
    
    def _export_results(self):
        """Export results to CSV."""
        if not self.results_2d:
            QMessageBox.warning(self, "Error", "No results to export")
            return
        
        # Export via DataBridge
        try:
            from ..utils.data_bridge import lem_result_to_dataframe
            import pandas as pd
            
            dfs = []
            for result in self.results_2d:
                df = lem_result_to_dataframe(result)
                if not df.empty:
                    dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                from PyQt6.QtWidgets import QFileDialog
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
                if file_path:
                    combined_df.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Export", f"Results exported to {file_path}")
            else:
                QMessageBox.warning(self, "Export", "No valid results to export")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

