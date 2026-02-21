"""
Advanced Underground Panel (STEP 37)

SLOS, caving, dilution, and void tracking.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QDoubleSpinBox, QSpinBox, QLineEdit,
    QTextEdit, QMessageBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import pyqtSlot

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


class UGAdvancedPanel(BaseAnalysisPanel):
    """
    Advanced Underground Panel for SLOS, caving, dilution, and void tracking.
    """
    # PanelManager metadata
    PANEL_ID = "UGAdvancedPanel"
    PANEL_NAME = "UGAdvanced Panel"
    PANEL_CATEGORY = PanelCategory.PLANNING
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    PANEL_ID = "UGAdvancedPanel"  # STEP 40
    task_name = "ug_slos_generate_stopes"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="ug_advanced")
        self.current_stopes = []
        self.current_footprint = None
        self.current_schedule = None
        # block_model is a read-only property from BasePanel - use _block_model instead
        self._block_model = None
        
        # Subscribe to block model from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_loaded)
            
            # Prefer classified block model when available.
            existing_block_model = self.registry.get_classified_block_model()
            if not _has_model_data(existing_block_model):
                existing_block_model = self.registry.get_block_model()
            if _has_model_data(existing_block_model):
                self._on_block_model_loaded(existing_block_model)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    


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
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("UG Advanced Panel received block model from DataRegistry")
        # Use base class method to properly set the block model and trigger callbacks
        super().set_block_model(block_model)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Advanced Underground Module: SLOS stope design, cave planning, "
            "dilution/backfill analysis, and void tracking."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_slos_tab(), "SLOS Designer")
        tabs.addTab(self._build_cave_tab(), "Cave Planner")
        tabs.addTab(self._build_void_tab(), "Void/Backfill View")
        layout.addWidget(tabs)
    
    def _build_slos_tab(self) -> QWidget:
        """Build SLOS Designer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Template configuration
        template_group = QGroupBox("Stope Template")
        template_form = QFormLayout(template_group)
        
        self.template_id = QLineEdit("SLOS_Template_1")
        template_form.addRow("Template ID:", self.template_id)
        
        self.strike_length = QDoubleSpinBox()
        self.strike_length.setRange(10.0, 200.0)
        self.strike_length.setValue(50.0)
        self.strike_length.setSuffix(" m")
        template_form.addRow("Strike Length:", self.strike_length)
        
        self.dip_length = QDoubleSpinBox()
        self.dip_length.setRange(10.0, 200.0)
        self.dip_length.setValue(30.0)
        self.dip_length.setSuffix(" m")
        template_form.addRow("Dip Length:", self.dip_length)
        
        self.height = QDoubleSpinBox()
        self.height.setRange(5.0, 100.0)
        self.height.setValue(20.0)
        self.height.setSuffix(" m")
        template_form.addRow("Height:", self.height)
        
        self.level_spacing = QDoubleSpinBox()
        self.level_spacing.setRange(10.0, 100.0)
        self.level_spacing.setValue(30.0)
        self.level_spacing.setSuffix(" m")
        template_form.addRow("Level Spacing:", self.level_spacing)
        
        self.strike_panel_length = QDoubleSpinBox()
        self.strike_panel_length.setRange(10.0, 200.0)
        self.strike_panel_length.setValue(50.0)
        self.strike_panel_length.setSuffix(" m")
        template_form.addRow("Strike Panel Length:", self.strike_panel_length)
        
        layout.addWidget(template_group)
        
        # Generate button
        generate_btn = QPushButton("Generate Stopes")
        generate_btn.clicked.connect(self._on_generate_stopes)
        layout.addWidget(generate_btn)
        
        # Stopes table
        self.stopes_table = QTableWidget()
        self.stopes_table.setColumnCount(6)
        self.stopes_table.setHorizontalHeaderLabels([
            "Stope ID", "Level", "Center X", "Center Y", "Center Z", "Tonnes"
        ])
        layout.addWidget(QLabel("Generated Stopes:"))
        layout.addWidget(self.stopes_table)
        
        # Dilution configuration
        dilution_group = QGroupBox("Dilution Model")
        dilution_form = QFormLayout(dilution_group)
        
        self.overbreak = QDoubleSpinBox()
        self.overbreak.setRange(0.0, 5.0)
        self.overbreak.setValue(1.0)
        self.overbreak.setSuffix(" m")
        dilution_form.addRow("Overbreak:", self.overbreak)
        
        self.recovery = QDoubleSpinBox()
        self.recovery.setRange(0.0, 1.0)
        self.recovery.setValue(0.95)
        self.recovery.setDecimals(2)
        dilution_form.addRow("In-Stope Recovery:", self.recovery)
        
        layout.addWidget(dilution_group)
        
        # Schedule configuration
        schedule_group = QGroupBox("Schedule Configuration")
        schedule_form = QFormLayout(schedule_group)
        
        self.target_tonnes = QDoubleSpinBox()
        self.target_tonnes.setRange(1000.0, 1_000_000.0)
        self.target_tonnes.setValue(100_000.0)
        self.target_tonnes.setSuffix(" t/period")
        schedule_form.addRow("Target Tonnes/Period:", self.target_tonnes)
        
        self.max_concurrent = QSpinBox()
        self.max_concurrent.setRange(1, 20)
        self.max_concurrent.setValue(5)
        schedule_form.addRow("Max Concurrent Stopes:", self.max_concurrent)
        
        optimize_btn = QPushButton("Optimize SLOS Schedule")
        optimize_btn.clicked.connect(self._on_optimize_slos)
        schedule_form.addRow(optimize_btn)
        
        layout.addWidget(schedule_group)
        
        layout.addStretch()
        return widget
    
    def _build_cave_tab(self) -> QWidget:
        """Build Cave Planner tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Footprint configuration
        footprint_group = QGroupBox("Cave Footprint")
        footprint_form = QFormLayout(footprint_group)
        
        self.cell_size = QDoubleSpinBox()
        self.cell_size.setRange(10.0, 100.0)
        self.cell_size.setValue(25.0)
        self.cell_size.setSuffix(" m")
        footprint_form.addRow("Cell Size:", self.cell_size)
        
        build_footprint_btn = QPushButton("Build Cave Footprint")
        build_footprint_btn.clicked.connect(self._on_build_footprint)
        footprint_form.addRow(build_footprint_btn)
        
        layout.addWidget(footprint_group)
        
        # Draw rules
        draw_group = QGroupBox("Draw Rules")
        draw_form = QFormLayout(draw_group)
        
        self.max_draw_rate = QDoubleSpinBox()
        self.max_draw_rate.setRange(100_000.0, 50_000_000.0)
        self.max_draw_rate.setValue(10_000_000.0)
        self.max_draw_rate.setSuffix(" tpy")
        draw_form.addRow("Max Draw Rate:", self.max_draw_rate)
        
        self.max_draw_height = QDoubleSpinBox()
        self.max_draw_height.setRange(10.0, 500.0)
        self.max_draw_height.setValue(100.0)
        self.max_draw_height.setSuffix(" m")
        draw_form.addRow("Max Draw Height:", self.max_draw_height)
        
        simulate_btn = QPushButton("Simulate Cave Draw")
        simulate_btn.clicked.connect(self._on_simulate_cave)
        draw_form.addRow(simulate_btn)
        
        layout.addWidget(draw_group)
        
        # Results
        self.cave_results_table = QTableWidget()
        self.cave_results_table.setColumnCount(4)
        self.cave_results_table.setHorizontalHeaderLabels([
            "Period", "Tonnes", "Cells", "Status"
        ])
        layout.addWidget(QLabel("Cave Draw Results:"))
        layout.addWidget(self.cave_results_table)
        
        layout.addStretch()
        return widget
    
    def _build_void_tab(self) -> QWidget:
        """Build Void/Backfill View tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Void tracking table
        self.void_table = QTableWidget()
        self.void_table.setColumnCount(5)
        self.void_table.setHorizontalHeaderLabels([
            "Period", "Void Volume (m³)", "Backfilled (m³)", "Unfilled (m³)", "Backfill %"
        ])
        layout.addWidget(QLabel("Void and Backfill Tracking:"))
        layout.addWidget(self.void_table)
        
        # Visualization button
        visualize_btn = QPushButton("Visualize Void State in 3D")
        visualize_btn.clicked.connect(self._on_visualize_void)
        layout.addWidget(visualize_btn)
        
        layout.addStretch()
        return widget
    
    @pyqtSlot()
    def _on_generate_stopes(self):
        """Generate SLOS stopes."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        if not self.block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        from ..ug.slos.slos_geometry import StopeTemplate
        
        template = StopeTemplate(
            id=self.template_id.text(),
            strike_length_m=self.strike_length.value(),
            dip_length_m=self.dip_length.value(),
            height_m=self.height.value(),
            min_mining_width_m=5.0,
            crown_pillar_m=5.0,
            sill_pillar_m=5.0,
            orientation={'dip': 0.0, 'dip_azimuth': 0.0}
        )
        
        config = {
            "block_model": self.block_model,
            "template": template,
            "level_spacing_m": self.level_spacing.value(),
            "strike_panel_length_m": self.strike_panel_length.value()
        }
        
        self.show_progress("Generating SLOS stopes...")
        self.controller.ug_generate_slos_stopes(config, self._on_stopes_generated)
    
    @pyqtSlot(dict)
    def _on_stopes_generated(self, result: dict):
        """Handle stope generation completion."""
        self.hide_progress()
        
        if result.get("error"):
            self.show_error("Generation Error", result["error"])
            return
        
        stopes = result.get("result", {}).get("stopes", [])
        self.current_stopes = stopes
        
        # Update table
        self.stopes_table.setRowCount(len(stopes))
        for r, stope in enumerate(stopes):
            if isinstance(stope, dict):
                s = stope
            else:
                s = {
                    'id': getattr(stope, 'id', ''),
                    'level': getattr(stope, 'level', ''),
                    'center': getattr(stope, 'center', (0, 0, 0)),
                    'tonnes': getattr(stope, 'tonnes', 0.0)
                }
            
            center = s.get('center', (0, 0, 0))
            self.stopes_table.setItem(r, 0, QTableWidgetItem(str(s.get('id', ''))))
            self.stopes_table.setItem(r, 1, QTableWidgetItem(str(s.get('level', ''))))
            self.stopes_table.setItem(r, 2, QTableWidgetItem(f"{center[0]:.1f}"))
            self.stopes_table.setItem(r, 3, QTableWidgetItem(f"{center[1]:.1f}"))
            self.stopes_table.setItem(r, 4, QTableWidgetItem(f"{center[2]:.1f}"))
            self.stopes_table.setItem(r, 5, QTableWidgetItem(f"{s.get('tonnes', 0.0):,.0f}"))
        
        self.stopes_table.resizeColumnsToContents()
        self.show_info("Stopes Generated", f"Generated {len(stopes)} SLOS stopes")
    
    @pyqtSlot()
    def _on_optimize_slos(self):
        """Optimize SLOS schedule."""
        if not self.current_stopes:
            self.show_warning("No Stopes", "Please generate stopes first.")
            return
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            # Create periods (simplified - would get from UI)
            from ..mine_planning.scheduling.types import TimePeriod
            
            periods = [
                TimePeriod(id=f"Y{i+1:02d}", index=i, duration_days=365.0)
                for i in range(10)
            ]
            
            config = {
                "stopes": self.current_stopes,
                "periods": periods,
                "discount_rate": 0.10,
                "target_tonnes_per_period": self.target_tonnes.value(),
                "max_concurrent_stopes": self.max_concurrent.value()
            }
            
            self.show_progress("Optimizing SLOS schedule...")
            self.controller.ug_run_slos_schedule(config, self._on_slos_optimized)
        except Exception as e:
            logger.error(f"Error optimizing SLOS schedule: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("Optimization Error", f"Failed to optimize schedule:\n{e}")
    
    @pyqtSlot(dict)
    def _on_slos_optimized(self, result: dict):
        """Handle SLOS optimization completion."""
        self.hide_progress()
        
        if result.get("error"):
            self.show_error("Optimization Error", result["error"])
            return
        
        schedule_result = result.get("result")
        self.current_schedule = schedule_result
        
        npv = schedule_result.metadata.get("npv", 0.0) if hasattr(schedule_result, 'metadata') else 0.0
        self.show_info("Schedule Optimized", f"SLOS schedule optimized. NPV = ${npv:,.0f}")
    
    @pyqtSlot()
    def _on_build_footprint(self):
        """Build cave footprint."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        if not self.block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        try:
            config = {
                "block_model": self.block_model,
                "cell_size_m": self.cell_size.value()
            }
            
            self.show_progress("Building cave footprint...")
            self.controller.ug_build_cave_footprint(config, self._on_footprint_built)
        except Exception as e:
            logger.error(f"Error building footprint: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("Build Error", f"Failed to build footprint:\n{e}")
    
    @pyqtSlot(dict)
    def _on_footprint_built(self, result: dict):
        """Handle footprint build completion."""
        self.hide_progress()
        
        if result.get("error"):
            self.show_error("Build Error", result["error"])
            return
        
        footprint = result.get("result")
        self.current_footprint = footprint
        
        cell_count = len(footprint.cells) if hasattr(footprint, 'cells') else 0
        self.show_info("Footprint Built", f"Cave footprint built with {cell_count} cells")
    
    @pyqtSlot()
    def _on_simulate_cave(self):
        """Simulate cave draw."""
        if not self.current_footprint:
            self.show_warning("No Footprint", "Please build cave footprint first.")
            return
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            from ..mine_planning.scheduling.types import TimePeriod
            
            periods = [
                TimePeriod(id=f"Y{i+1:02d}", index=i, duration_days=365.0)
                for i in range(10)
            ]
            
            config = {
                "footprint": self.current_footprint,
                "periods": periods,
                "rule": {
                    "max_draw_rate_tpy": self.max_draw_rate.value(),
                    "max_draw_height_m": self.max_draw_height.value(),
                    "dilution_entry_height_ratio": 0.7,
                    "secondary_break_fraction": 0.1
                },
                "target_tonnes_per_period": 100_000.0
            }
            
            self.show_progress("Simulating cave draw...")
            self.controller.ug_run_cave_schedule(config, self._on_cave_simulated)
        except Exception as e:
            logger.error(f"Error simulating cave draw: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("Simulation Error", f"Failed to simulate cave draw:\n{e}")
    
    @pyqtSlot(dict)
    def _on_cave_simulated(self, result: dict):
        """Handle cave simulation completion."""
        self.hide_progress()
        
        if result.get("error"):
            self.show_error("Simulation Error", result["error"])
            return
        
        schedule_result = result.get("result")
        
        # Update table
        if hasattr(schedule_result, 'decisions'):
            decisions = schedule_result.decisions
        else:
            decisions = schedule_result.get("decisions", [])
        
        # Aggregate by period
        tonnes_by_period = {}
        for decision in decisions:
            period_id = decision.period_id if hasattr(decision, 'period_id') else decision.get('period_id', '')
            tonnes = decision.tonnes if hasattr(decision, 'tonnes') else decision.get('tonnes', 0.0)
            if period_id not in tonnes_by_period:
                tonnes_by_period[period_id] = {'tonnes': 0.0, 'count': 0}
            tonnes_by_period[period_id]['tonnes'] += tonnes
            tonnes_by_period[period_id]['count'] += 1
        
        self.cave_results_table.setRowCount(len(tonnes_by_period))
        for r, (period_id, data) in enumerate(tonnes_by_period.items()):
            self.cave_results_table.setItem(r, 0, QTableWidgetItem(str(period_id)))
            self.cave_results_table.setItem(r, 1, QTableWidgetItem(f"{data['tonnes']:,.0f}"))
            self.cave_results_table.setItem(r, 2, QTableWidgetItem(str(data['count'])))
            self.cave_results_table.setItem(r, 3, QTableWidgetItem("Active"))
        
        self.cave_results_table.resizeColumnsToContents()
        self.show_info("Simulation Complete", "Cave draw simulation completed")
    
    @pyqtSlot()
    def _on_visualize_void(self):
        """Visualize void state in 3D."""
        if not self.current_schedule:
            self.show_warning("No Schedule", "Please generate a schedule first.")
            return
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            # Extract void information from schedule
            # The schedule should contain void volumes per period
            void_data = {}
            if hasattr(self.current_schedule, 'void_volumes'):
                void_data = self.current_schedule.void_volumes
            elif isinstance(self.current_schedule, dict) and 'void_volumes' in self.current_schedule:
                void_data = self.current_schedule['void_volumes']
            
            if not void_data:
                self.show_info(
                    "No Void Data",
                    "Void visualization requires void volume data from the schedule.\n\n"
                    "This feature will visualize the void spaces created by mining operations."
                )
                return
            
            # Request visualization via controller
            config = {
                "schedule": self.current_schedule,
                "void_data": void_data,
                "layer_name": "Void State"
            }
            
            # Use controller method if available, otherwise show info
            if hasattr(self.controller, 'ug_visualize_void'):
                self.controller.ug_visualize_void(config, self._on_void_visualized)
            else:
                self.show_info(
                    "Visualization",
                    "Void visualization is being prepared.\n\n"
                    "This will show the void spaces created by mining operations in 3D."
                )
                logger.info("Void visualization requested - controller method may need to be implemented")
                
        except Exception as e:
            logger.error(f"Error visualizing void state: {e}", exc_info=True)
            self.show_error("Visualization Error", f"Failed to visualize void state:\n{e}")
    
    def _on_void_visualized(self, result: Dict[str, Any]):
        """Handle void visualization completion."""
        if result.get("error"):
            self.show_error("Visualization Error", result["error"])
        else:
            self.show_info("Success", "Void state visualized in 3D viewer")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

